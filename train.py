"""
Assignment 2: training pipeline.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download the Roboflow Universe classification dataset named in config.py.
2. Describe it: images per class and split, rotten share, image sizes.
3. Finetune yolo11n-cls from its pretrained checkpoint and copy the best
   weights to weights/best.pt.
4. Evaluate on the test split against the COCO detector used zero-shot and a
   majority-class baseline, per class and on the rotten/fresh decision.
5. Sweep the rotten-probability threshold on the validation split, pick the
   cheapest single threshold and the cheapest pass/review/fail band within
   review capacity, then report what they cost on the untouched test split.
6. Write everything app.py needs into artifacts/.

Thresholds are chosen on validation and only reported on test, so the test
number is not flattered by having been optimised on.

Resume after a crash in the evaluation steps: A2_REUSE_TRAINING=1 keeps the
weights and training.json the last full run wrote instead of training again.

Smoke test without Roboflow: set A2_SMOKE_DATASET to a local folder laid out
as <split>/<class>/<image>. It skips the download and trains for one epoch at
a small size, to check the pipeline runs. Never commit what a smoke run writes.
"""

import json
import os
import platform
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

import config
from evaluation import (
    Costs, band_rates, classification_metrics, cost_per_case, labels_and_scores,
    optimise_band, per_class_outcomes, policy_costs, records_to_arrays,
    review_share, roc_auc, staffing, status_quo_from_config, sweep_single, thresholds,
)

SMOKE_DATASET = os.environ.get("A2_SMOKE_DATASET")
EPOCHS = 1 if SMOKE_DATASET else config.EPOCHS
IMGSZ = 64 if SMOKE_DATASET else config.IMGSZ
SPLITS = {"train": "train", "val": "valid", "test": "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --- 1. Data -----------------------------------------------------------------

def download_dataset() -> Path:
    if SMOKE_DATASET:
        print(f"Smoke test: using local dataset {SMOKE_DATASET}")
        return Path(SMOKE_DATASET).resolve()
    from roboflow import Roboflow

    from api_key import load_api_key

    target = config.DATASET_DIR / f"{config.ROBOFLOW_PROJECT}-v{config.ROBOFLOW_VERSION}"
    if (target / "train").is_dir():
        print(f"Dataset already downloaded at {target}")
        return target
    rf = Roboflow(api_key=load_api_key())
    version = (rf.workspace(config.ROBOFLOW_WORKSPACE)
                 .project(config.ROBOFLOW_PROJECT)
                 .version(config.ROBOFLOW_VERSION))
    version.download(config.ROBOFLOW_FORMAT, location=str(target))
    return target


def split_dir(root: Path, split: str) -> Path:
    if split == "val" and not (root / "valid").is_dir() and (root / "val").is_dir():
        return root / "val"
    return root / SPLITS[split]


def class_names(root: Path) -> dict[int, str]:
    """Sorted training sub-folders: the same order Ultralytics assigns ids in."""
    folders = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
    return dict(enumerate(folders))


def split_images(root: Path, split: str, names: dict[int, str]) -> list[tuple[Path, int]]:
    """(image, class id) for every image in a split."""
    items = []
    for cid, name in names.items():
        folder = split_dir(root, split) / name
        if folder.is_dir():
            items += [(p, cid) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return items


def ensure_test_split(root: Path, names: dict[int, str]) -> None:
    """Some Universe versions ship without a test split. Carve one out of
    valid, per class and deterministically, so thresholds and the final
    number never come from the same images."""
    if (root / "test").is_dir() and any((root / "test").rglob("*.*")):
        return
    rng = np.random.default_rng(config.SEED)
    moved = 0
    for name in names.values():
        images = sorted((split_dir(root, "val") / name).glob("*"))
        (root / "test" / name).mkdir(parents=True, exist_ok=True)
        for i in rng.permutation(len(images))[: len(images) // 2]:
            shutil.move(images[i], root / "test" / name / images[i].name)
            moved += 1
    print(f"No test split shipped: moved {moved} valid images to test, stratified by class.")


def source_photo(path: Path) -> str:
    """The photo an image was made from, with Roboflow's suffix and the
    original dataset's augmentation prefixes stripped."""
    stem = path.name.split(".rf.")[0]
    return re.sub(config.AUGMENTATION_PREFIX, "", stem)


def leakage(root: Path, names: dict[int, str]) -> dict:
    """Share of val/test images whose source photo also appears in train."""
    train_sources = {(names[c], source_photo(p)) for p, c in split_images(root, "train", names)}
    out = {}
    for split in ("val", "test"):
        items = split_images(root, split, names)
        shared = sum((names[c], source_photo(p)) in train_sources for p, c in items)
        out[split] = {"images": len(items), "share_with_train": shared,
                      "share": shared / len(items) if items else 0.0,
                      "distinct_source_photos": len({(names[c], source_photo(p)) for p, c in items})}
    all_sources = {(names[c], source_photo(p)) for s in SPLITS for p, c in split_images(root, s, names)}
    out["distinct_source_photos"] = len(all_sources)
    return out


def regroup_by_source(root: Path, names: dict[int, str]) -> Path:
    """Re-split every downloaded image so all copies of one photo share a
    split. Per class, source photos are shuffled with a fixed seed and dealt
    into train/valid/test until each split holds its share of the images.
    Files are hard-linked, so this costs no disk space."""
    out = root.parent / f"{root.name}-by-source"
    if (out / ".done").exists():
        return out
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(config.SEED)
    shares = config.GROUPED_SPLIT_SHARES
    for cid, name in names.items():
        groups: dict[str, list[Path]] = {}
        for split in SPLITS:
            for path, c in split_images(root, split, names):
                if c == cid:
                    groups.setdefault(source_photo(path), []).append(path)
        keys = sorted(groups)
        order = [keys[i] for i in rng.permutation(len(keys))]
        total = sum(len(v) for v in groups.values())
        targets = {"test": shares["test"] * total, "valid": shares["valid"] * total}
        filled = {"test": 0, "valid": 0}
        for key in order:
            split = next((s for s in ("test", "valid") if filled[s] < targets[s]), "train")
            if split != "train":
                filled[split] += len(groups[key])
            folder = out / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for path in groups[key]:
                target = folder / path.name
                if target.exists():  # same file name in two published splits
                    target = folder / f"{path.parent.parent.name}_{path.name}"
                os.link(path, target)
    (out / ".done").write_text("ok")
    print(f"Re-split by source photo into {out}")
    return out


def class_roles(names: dict[int, str]) -> dict[str, dict]:
    """Which fruit each class is, and whether it counts as rotten."""
    roles = {}
    for name in names.values():
        if name in config.CLASS_OVERRIDES:
            fruit, rotten = config.CLASS_OVERRIDES[name]
        else:
            low = name.lower()
            rotten = any(w in low for w in config.ROTTEN_WORDS)
            fruit = low
            for w in (*config.ROTTEN_WORDS, *config.FRESH_WORDS):
                fruit = fruit.replace(w, "")
            fruit = fruit.strip(" _-").rstrip("s") or low
        roles[name] = {"fruit": fruit, "rotten": bool(rotten)}
    if not any(r["rotten"] for r in roles.values()) or all(r["rotten"] for r in roles.values()):
        raise SystemExit(f"Could not tell rotten from fresh classes in {list(names.values())}. "
                         "Set ROTTEN_WORDS or CLASS_OVERRIDES in config.py.")
    return roles


# --- 2. Describe -------------------------------------------------------------

def describe_dataset(root: Path, names: dict[int, str], roles: dict) -> dict:
    stats = {"splits": {}, "image_sizes": {}, "classes": roles}
    sizes: Counter = Counter()
    for split in SPLITS:
        items = split_images(root, split, names)
        per_class = Counter(names[c] for _, c in items)
        rotten = sum(roles[names[c]]["rotten"] for _, c in items)
        for path, _ in items:
            with Image.open(path) as im:
                sizes[f"{im.width}x{im.height}"] += 1
        stats["splits"][split] = {
            "images": len(items),
            "per_class": {name: per_class.get(name, 0) for name in names.values()},
            "rotten_images": rotten,
            "rotten_share": rotten / len(items) if items else 0.0,
        }
    stats["image_sizes"] = dict(sizes.most_common(10))
    stats["distinct_image_sizes"] = len(sizes)
    return stats


# --- 3. Finetune -------------------------------------------------------------

def finetune(root: Path) -> tuple[Path, dict]:
    reuse = config.ARTIFACTS_DIR / "training.json"
    if os.environ.get("A2_REUSE_TRAINING") and config.WEIGHTS_PATH.exists() and reuse.exists():
        print(f"A2_REUSE_TRAINING set: keeping {config.WEIGHTS_PATH} from the last run")
        return config.WEIGHTS_PATH, json.loads(reuse.read_text())
    model = YOLO(config.BASE_MODEL)
    started = time.time()
    model.train(
        data=str(root), epochs=EPOCHS, imgsz=IMGSZ, batch=config.BATCH,
        patience=config.PATIENCE, seed=config.SEED, deterministic=True,
        project=str(config.RUNS_DIR), name="finetune", exist_ok=True, plots=True,
    )
    best = Path(model.trainer.best)
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, config.WEIGHTS_PATH)

    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu"
    info = {
        "base_model": config.BASE_MODEL, "epochs_requested": EPOCHS,
        "epochs_run": int(model.trainer.epoch) + 1,
        "imgsz": IMGSZ, "batch": config.BATCH, "patience": config.PATIENCE, "seed": config.SEED,
        "device": device, "torch": torch.__version__, "python": platform.python_version(),
        "training_minutes": (time.time() - started) / 60,
        "smoke_test": bool(SMOKE_DATASET),
    }
    print(f"Best weights copied to {config.WEIGHTS_PATH}")
    return config.WEIGHTS_PATH, info


# --- 4. Evaluate -------------------------------------------------------------

def predict_records(model: YOLO, items: list[tuple[Path, int]], names: dict[int, str]) -> list[dict]:
    """Class probabilities for every image, re-ordered to the dataset's ids."""
    order = {v: k for k, v in model.names.items()}
    index = [order[names[c]] for c in range(len(names))]
    records = []
    results = model.predict(source=[str(p) for p, _ in items], imgsz=IMGSZ, stream=True, verbose=False)
    for (path, cid), result in zip(items, results):
        probs = result.probs.data.cpu().numpy()[index]
        records.append({"image": f"{names[cid]}/{path.name}", "true": cid,
                        "probs": [round(float(p), 5) for p in probs]})
    return records


def square_image(path: Path, size: int = 640) -> np.ndarray:
    """Letterbox an image onto a grey size x size canvas, as BGR for Ultralytics."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)[:, :, ::-1].copy()


def coco_zero_shot(items: list[tuple[Path, int]], names: dict[int, str], roles: dict,
                   fallback: int) -> np.ndarray:
    """What the COCO detector can say unaided: which fruit it sees.

    It has no idea of freshness, so every fruit it recognises is called the
    fresh class of that fruit; if it sees no known fruit, the majority class.
    """
    coco = YOLO(config.COCO_BASELINE)
    wanted = {cid: config.COCO_TO_FRUIT[n] for cid, n in coco.names.items() if n in config.COCO_TO_FRUIT}
    fresh_of = {}
    for cid, name in names.items():
        fruit = roles[name]["fruit"]
        if fruit not in fresh_of or not roles[name]["rotten"]:
            fresh_of[fruit] = cid
    preds = []
    for path, _ in items:
        # One fixed input shape: with ~1,450 different image sizes the CPU
        # backend otherwise caches kernels per shape until it runs out of RAM.
        result = coco.predict(square_image(path), imgsz=640, classes=list(wanted), conf=0.05,
                              verbose=False)[0]
        if len(result.boxes):
            top = int(result.boxes.cls[result.boxes.conf.argmax()])
            preds.append(fresh_of.get(wanted[top], fallback))
        else:
            preds.append(fallback)
    return np.array(preds, dtype=int)


def ultralytics_test_metrics(weights: Path, root: Path) -> dict:
    """Ultralytics' own test-split numbers, as a cross-check on ours."""
    metrics = YOLO(str(weights)).val(data=str(root), split="test", imgsz=IMGSZ, batch=config.BATCH,
                                     project=str(config.RUNS_DIR), name="test", exist_ok=True,
                                     plots=True, verbose=False)
    plots = config.ARTIFACTS_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for png in Path(metrics.save_dir).glob("*.png"):
        shutil.copy(png, plots / f"test_{png.name}")
    return {"top1": float(metrics.top1), "top5": float(metrics.top5)}


def binary_summary(y: np.ndarray, s: np.ndarray) -> dict:
    return {"images": int(len(y)), "rotten": int(y.sum()),
            "rotten_share": float(y.mean()) if len(y) else 0.0,
            "roc_auc": roc_auc(y, s),
            "recall_at_0.5": float((s[y] >= 0.5).mean()) if y.any() else 0.0,
            "false_alarm_at_0.5": float((s[~y] >= 0.5).mean()) if (~y).any() else 0.0}


# --- 5. Cost ------------------------------------------------------------------

def review_capacity_share() -> float:
    capacity = (config.REVIEWERS_AVAILABLE * config.PRODUCTIVE_HOURS_PER_REVIEWER
                * 3600 / config.REVIEW_SECONDS)
    return min(1.0, capacity / config.DAILY_VOLUME)


def choose_operating_point(y_val, s_val, y_test, s_test, costs: Costs, base_rate: float) -> dict:
    grid = thresholds(config.THRESHOLD_STEP)
    sweep_val = sweep_single(y_val, s_val, base_rate, costs, grid)
    single_t = min(sweep_val, key=lambda row: row["cost_per_1000"])["threshold"]
    capacity = review_capacity_share()
    band = optimise_band(y_val, s_val, base_rate, costs, grid, max_review_share=capacity)
    unconstrained = optimise_band(y_val, s_val, base_rate, costs, grid)

    test_rates = band_rates(y_test, s_test, band["lo"], band["hi"])
    share = review_share(test_rates, base_rate)
    return {
        "chosen_on": "val", "reported_on": "test", "base_rate": base_rate,
        "single_threshold": single_t,
        "lo": band["lo"], "hi": band["hi"],
        "review_capacity_share": capacity,
        "unconstrained_band": {"lo": unconstrained["lo"], "hi": unconstrained["hi"],
                               "review_share_val": unconstrained["review_share"],
                               "val_cost_per_1000": 1000 * unconstrained["cost_per_case"]},
        "val_cost_per_1000": 1000 * band["cost_per_case"],
        "test_cost_per_1000": 1000 * cost_per_case(test_rates, base_rate, costs),
        "test_review_share": share,
        "test_rates": test_rates,
        "staffing": staffing(share, config.DAILY_VOLUME, config.REVIEW_SECONDS,
                             config.PRODUCTIVE_HOURS_PER_REVIEWER),
        "policies_per_1000_test": {k: 1000 * v for k, v in policy_costs(
            y_test, s_test, base_rate, costs, single_t, band["lo"], band["hi"], status_quo_from_config(config)).items()},
        "daily_volume": config.DAILY_VOLUME,
    }


def sensitivity(y_val, s_val, y_test, s_test, costs: Costs) -> pd.DataFrame:
    """How the answer moves when the two least certain assumptions move."""
    grid = thresholds(0.02)
    rows = []
    for fn in (0.5, 1, 2, 3, 4, 6, 8, 12, 20):
        for base_rate in (0.01, 0.02, 0.04, 0.08, 0.15):
            c = Costs(**{**costs.__dict__, "fn": float(fn)})
            sweep = sweep_single(y_val, s_val, base_rate, c, grid)
            t = min(sweep, key=lambda row: row["cost_per_1000"])["threshold"]
            band = optimise_band(y_val, s_val, base_rate, c, grid, review_capacity_share())
            policies = policy_costs(y_test, s_test, base_rate, c, t, band["lo"], band["hi"],
                                    status_quo_from_config(config))
            rows.append({"cost_fn": fn, "base_rate": base_rate, "single_threshold": t,
                         "lo": band["lo"], "hi": band["hi"],
                         **{f"{k}_per_1000": 1000 * v for k, v in policies.items()}})
    return pd.DataFrame(rows)


def plot_cost_curve(sweep: pd.DataFrame, op: dict) -> None:
    """Cost against threshold on val and test, with the chosen points marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sweep["threshold"], sweep["val_cost_per_1000"], label="validation (thresholds chosen here)")
    ax.plot(sweep["threshold"], sweep["test_cost_per_1000"], label="test (reported)")
    t = op["single_threshold"]
    at_t = sweep.loc[(sweep["threshold"] - t).abs().idxmin(), "test_cost_per_1000"]
    ax.scatter([t], [at_t], marker="*", s=200, color="red", zorder=5, label=f"chosen model-alone threshold {t:.2f}")
    ax.axvline(op["lo"], ls=":", color="grey", label=f"review band {op['lo']:.2f}–{op['hi']:.2f}")
    ax.axvline(op["hi"], ls=":", color="grey")
    ax.axhline(op["test_cost_per_1000"], ls="--", color="green",
               label=f"model + review on test: €{op['test_cost_per_1000']:.1f}")
    ax.set_yscale("log")
    ax.set_xlabel("threshold on P(rotten)")
    ax.set_ylabel(f"€ per 1,000 pieces at {op['base_rate']:.0%} rotten (log)")
    ax.set_title("Expected cost against threshold, model deciding alone")
    ax.legend(fontsize=8)
    fig.tight_layout()
    (config.ARTIFACTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(config.ARTIFACTS_DIR / "plots" / "cost_vs_threshold.png", dpi=120)
    plt.close(fig)


def bootstrap_policies(records, rotten_ids, op, costs: Costs, rounds: int = 1000) -> dict:
    """95% interval of each policy's test cost, resampling whole source photos.

    Copies of one photo are not independent, so resampling images would
    understate the uncertainty."""
    y, s = labels_and_scores(records, rotten_ids)
    photos: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        photos.setdefault(f"{record['true']}/{source_photo(Path(record['image']))}", []).append(i)
    groups = list(photos.values())
    rng = np.random.default_rng(config.SEED)
    samples = []
    for _ in range(rounds):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        samples.append(policy_costs(y[idx], s[idx], op["base_rate"], costs,
                                    op["single_threshold"], op["lo"], op["hi"],
                                    status_quo_from_config(config)))
    return {k: {"low": 1000 * float(np.percentile([x[k] for x in samples], 2.5)),
                "high": 1000 * float(np.percentile([x[k] for x in samples], 97.5))}
            for k in samples[0]}


def stress_test(y, s, op, costs: Costs) -> list[dict]:
    """What if the model is less sure on real line fruit than on this data?

    Each rotten score is pulled towards 0.5 by shrinking its log-odds by a
    factor, keeping the thresholds fixed. Factor 1 is the model as measured."""
    logit = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1))
    rows = []
    for factor in (1.0, 0.75, 0.5, 0.35, 0.25):
        shrunk = 1 / (1 + np.exp(-logit * factor))
        alone = band_rates(y, shrunk, op["single_threshold"], op["single_threshold"])
        band = band_rates(y, shrunk, op["lo"], op["hi"])
        rows.append({
            "logit_factor": factor,
            "model_alone_per_1000": 1000 * cost_per_case(alone, op["base_rate"], costs),
            "model_plus_review_per_1000": 1000 * cost_per_case(band, op["base_rate"], costs),
            "model_alone_rotten_shipped": alone["pos"]["pass"],
            "band_rotten_shipped": band["pos"]["pass"] + band["pos"]["review"] * (1 - costs.reviewer_sensitivity),
            "band_review_share": review_share(band, op["base_rate"]),
        })
    return rows


def error_table(records, rotten_ids, names, op) -> pd.DataFrame:
    """Every test image the model gets wrong, by class or by final decision."""
    y, s = labels_and_scores(records, rotten_ids)
    rows = []
    for record, rotten, score in zip(records, y, s):
        predicted = int(np.argmax(record["probs"]))
        outcome = "fail" if score >= op["hi"] else "review" if score >= op["lo"] else "pass"
        costly = (rotten and outcome == "pass") or (not rotten and outcome == "fail")
        if predicted != record["true"] or costly or outcome == "review":
            rows.append({"image": record["image"], "photo": source_photo(Path(record["image"])),
                         "true": names[record["true"]], "predicted": names[predicted],
                         "p_rotten": round(float(score), 4), "outcome": outcome,
                         "costly_error": bool(costly)})
    return pd.DataFrame(rows)


# --- 6. Report ----------------------------------------------------------------

def frame_to_markdown(frame: pd.DataFrame) -> str:
    def cell(v):
        return f"{v:,.3f}" if isinstance(v, float) else str(v)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    rows += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(rows)


def markdown_summary(stats, classification, ultra, binary, op, sens, names) -> str:
    lines = ["# Pipeline output summary", "", "Generated by train.py. Numbers to quote in REPORT.md.", "",
             "## Dataset", ""]
    lines.append(frame_to_markdown(pd.DataFrame(
        [{"split": k, "images": v["images"], **v["per_class"], "rotten share": f"{v['rotten_share']:.1%}"}
         for k, v in stats["splits"].items()])))
    pl, gl = stats["published_leakage"], stats["leakage"]
    lines += ["", f"Published splits (not used): {stats['published_splits']}",
              f"Source photos: {pl['distinct_source_photos']}. Share of val/test images whose source photo is also "
              f"in train: published {pl['val']['share']:.1%} / {pl['test']['share']:.1%}; "
              f"split used {gl['val']['share']:.1%} / {gl['test']['share']:.1%}."]
    lines += ["", f"Image sizes (top): {stats['image_sizes']} ({stats['distinct_image_sizes']} distinct)", "",
              "## Classification, test split", ""]
    lines.append(frame_to_markdown(pd.DataFrame(
        [{"model": m, "accuracy": v["accuracy"], "fruit-type accuracy": v.get("fruit_type_accuracy", 0.0),
          **{f"macro {k}": x for k, x in v["macro"].items()}}
         for m, v in classification.items()])))
    lines += ["", f"Ultralytics' own test top-1 for the finetuned model: {ultra['top1']:.3f}"]
    for model, m in classification.items():
        lines += ["", f"### Per class: {model}", ""]
        lines.append(frame_to_markdown(pd.DataFrame([{"class": k, **v} for k, v in m["per_class"].items()])))
        lines += ["", "Confusion (rows true, columns predicted):", ""]
        lines.append(frame_to_markdown(pd.DataFrame(m["confusion"], columns=list(names.values()),
                                                    index=list(names.values())).reset_index()))
    lines += ["", "## Rotten vs fresh, test split", ""]
    lines.append(frame_to_markdown(pd.DataFrame([{"model": k, **v} for k, v in binary.items()])))
    p = op["policies_per_1000_test"]
    lines += ["", "## Operating point (chosen on val, reported on test)", "",
              f"- base rate assumed: {op['base_rate']:.1%}",
              f"- model-alone threshold: {op['single_threshold']:.2f}",
              f"- band: pass < {op['lo']:.2f} <= review < {op['hi']:.2f} <= fail",
              f"- review capacity: {op['review_capacity_share']:.1%} of pieces",
              f"- unconstrained band would be {op['unconstrained_band']['lo']:.2f}/"
              f"{op['unconstrained_band']['hi']:.2f}, reviewing {op['unconstrained_band']['review_share_val']:.1%}",
              f"- test review share: {op['test_review_share']:.1%} -> "
              f"{op['staffing']['cases_per_day']:.0f} pieces/day, "
              f"{op['staffing']['review_hours_per_day']:.1f} h/day, "
              f"{op['staffing']['reviewers_needed']:.2f} graders",
              "", "| policy | EUR per 1,000 | EUR per day |", "|---|---|---|"]
    ci = op["policies_per_1000_test_95ci"]
    lines[-1] = "|---|---|---|---|"
    lines[-2] = "| policy | EUR per 1,000 | 95% interval (photo bootstrap) | EUR per day |"
    for k, v in p.items():
        lines.append(f"| {k} | {v:,.1f} | {ci[k]['low']:,.1f}–{ci[k]['high']:,.1f} | "
                     f"{v * op['daily_volume'] / 1000:,.0f} |")
    e = op["test_errors"]
    lines += ["", f"Test errors: {e['misclassified_images']} misclassified images from "
              f"{e['misclassified_photos']} photos; {e['costly_images']} costly decisions "
              f"(rotten shipped or fresh binned) from {e['costly_photos']} photos. See errors_test.csv."]
    lines += ["", "## Stress test: model less sure than measured (thresholds fixed)", "",
              frame_to_markdown(pd.DataFrame(op["stress_test"]))]
    lines += ["", "## Sensitivity (thresholds re-chosen on val, cost on test)", "", frame_to_markdown(sens)]
    return "\n".join(lines) + "\n"


def save_json(name: str, obj) -> None:
    (config.ARTIFACTS_DIR / name).write_text(json.dumps(obj, indent=1))


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    root = download_dataset()
    names = class_names(root)
    ensure_test_split(root, names)
    roles = class_roles(names)
    rotten_ids = [c for c, n in names.items() if roles[n]["rotten"]]
    print(f"Classes: {roles}")

    print("\n== Describing the dataset")
    published = describe_dataset(root, names, roles)
    published_leakage = leakage(root, names)
    print(f"Published splits: {json.dumps(published_leakage)}")
    if config.SPLIT_BY_SOURCE_PHOTO:
        root = regroup_by_source(root, names)
    stats = describe_dataset(root, names, roles)
    stats.update(names=names, split_by_source_photo=config.SPLIT_BY_SOURCE_PHOTO,
                 published_splits=published["splits"], published_leakage=published_leakage,
                 leakage=leakage(root, names),
                 source={"workspace": config.ROBOFLOW_WORKSPACE, "project": config.ROBOFLOW_PROJECT,
                         "version": config.ROBOFLOW_VERSION})
    save_json("dataset_stats.json", stats)
    print(json.dumps(stats["splits"], indent=1))

    print("\n== Finetuning")
    weights, training = finetune(root)
    save_json("training.json", training)

    print("\n== Predicting on val and test")
    tuned = YOLO(str(weights))
    items = {split: split_images(root, split, names) for split in ("val", "test")}
    records = {split: predict_records(tuned, items[split], names) for split in items}
    for split, recs in records.items():
        save_json(f"records_{split}.json", recs)

    print("\n== Baselines and metrics on test")
    train_counts = Counter(c for _, c in split_images(root, "train", names))
    majority = train_counts.most_common(1)[0][0]
    true_test, probs_test = records_to_arrays(records["test"])
    coco_pred = coco_zero_shot(items["test"], names, roles, fallback=majority)
    majority_pred = np.full(len(true_test), majority)
    classification = {
        "finetuned": classification_metrics(true_test, probs_test.argmax(1), names),
        "coco_zero_shot": classification_metrics(true_test, coco_pred, names),
        "majority_class": classification_metrics(true_test, majority_pred, names),
    }
    ultra = ultralytics_test_metrics(weights, root)
    y_test, s_test = labels_and_scores(records["test"], rotten_ids)
    binary = {
        "finetuned": binary_summary(y_test, s_test),
        # Neither baseline has any notion of rot, so their rotten score is a
        # constant. (COCO's class label for unrecognised fruit falls back to the
        # majority class, which happens to be rotten; that is not a rot signal.)
        "coco_zero_shot": binary_summary(y_test, np.zeros(len(y_test))),
        "majority_class": binary_summary(y_test, np.full(len(y_test), float(majority in rotten_ids))),
    }
    fruit_of = np.array([roles[names[c]]["fruit"] for c in range(len(names))])
    probs_pred = probs_test.argmax(1)
    for model, pred in (("finetuned", probs_pred), ("coco_zero_shot", coco_pred),
                        ("majority_class", majority_pred)):
        classification[model]["fruit_type_accuracy"] = float((fruit_of[pred] == fruit_of[true_test]).mean())
    save_json("metrics.json", {"classification_test": classification, "ultralytics_test": ultra,
                               "binary_test": binary, "names": names, "rotten_ids": rotten_ids,
                               "majority_class": names[majority]})

    print("\n== Cost sweep and operating point")
    costs = Costs.from_config(config)
    y_val, s_val = labels_and_scores(records["val"], rotten_ids)
    grid = thresholds(config.THRESHOLD_STEP)
    sweep = pd.DataFrame({"threshold": grid})
    for split, (y, s) in (("val", (y_val, s_val)), ("test", (y_test, s_test))):
        rows = sweep_single(y, s, config.BASE_RATE, costs, grid)
        for key in ("cost_per_1000", "tpr", "fpr"):
            sweep[f"{split}_{key}"] = [r[key] for r in rows]
    sweep.to_csv(config.ARTIFACTS_DIR / "cost_sweep.csv", index=False)

    op = choose_operating_point(y_val, s_val, y_test, s_test, costs, config.BASE_RATE)
    op["per_class_outcomes_test"] = per_class_outcomes(true_test, s_test, op["lo"], op["hi"], names)
    op["costs"] = costs.__dict__
    op["policies_per_1000_test_95ci"] = bootstrap_policies(records["test"], rotten_ids, op, costs)
    op["stress_test"] = stress_test(y_test, s_test, op, costs)
    op["status_quo"] = status_quo_from_config(config)
    errors = error_table(records["test"], rotten_ids, names, op)
    errors.to_csv(config.ARTIFACTS_DIR / "errors_test.csv", index=False)
    op["test_errors"] = {
        "misclassified_images": int((errors["true"] != errors["predicted"]).sum()) if len(errors) else 0,
        "misclassified_photos": int(errors.loc[errors["true"] != errors["predicted"], "photo"].nunique())
        if len(errors) else 0,
        "costly_images": int(errors["costly_error"].sum()) if len(errors) else 0,
        "costly_photos": int(errors.loc[errors["costly_error"], "photo"].nunique()) if len(errors) else 0,
    }
    save_json("operating_point.json", op)
    plot_cost_curve(sweep, op)

    sens = sensitivity(y_val, s_val, y_test, s_test, costs)
    sens.to_csv(config.ARTIFACTS_DIR / "sensitivity.csv", index=False)

    summary = markdown_summary(stats, classification, ultra, binary, op, sens, names)
    (config.ARTIFACTS_DIR / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"Done. Commit {config.WEIGHTS_PATH.relative_to(config.ROOT)} and "
          f"{config.ARTIFACTS_DIR.relative_to(config.ROOT)}/.")


if __name__ == "__main__":
    main()
