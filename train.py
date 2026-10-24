"""
Assignment 2: training pipeline.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download the Roboflow Universe dataset named in settings.py.
2. Describe it from the labels: instances per class, split sizes, object size.
3. Finetune yolo11n from its COCO checkpoint and copy the best weights to weights/.
4. Evaluate on the test split against the pretrained COCO model zero-shot and
   a predict-nothing baseline.
5. Save every prediction on the validation and test splits, choose thresholds
   by sweeping cost on validation, and report what they cost on test.

Everything the dashboard shows is written to artifacts/, so app.py never
needs to train or even run the model over the test set.
"""

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

import costing
import settings as S
from api_key import load_api_key

SPLITS = {"train": "train", "val": "valid", "test": "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- 1. Data -----------------------------------------------------------------

def download_dataset() -> Path:
    location = S.DATASETS_DIR / f"{S.PROJECT}-{S.VERSION}"
    if not (location / "data.yaml").is_file():
        from roboflow import Roboflow

        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(S.WORKSPACE).project(S.PROJECT).version(S.VERSION)
        version.download(S.EXPORT_FORMAT, location=str(location), overwrite=True)
    return location


def write_data_yaml(location: Path) -> tuple[Path, list[str]]:
    """Roboflow's data.yaml uses relative paths that depend on where you run
    from. Write one with absolute paths so training and evaluation agree."""
    with (location / "data.yaml").open() as f:
        original = yaml.safe_load(f)
    names = list(original["names"].values()) if isinstance(original["names"], dict) else list(original["names"])
    data = {
        "path": str(location),
        "train": f"{SPLITS['train']}/images",
        "val": f"{SPLITS['val']}/images",
        "test": f"{SPLITS['test']}/images",
        "names": dict(enumerate(names)),
    }
    path = location / "data_abs.yaml"
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path, names


def split_images(location: Path, split: str) -> list[Path]:
    folder = location / SPLITS[split] / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_labels(image_path: Path) -> np.ndarray:
    """YOLO label file -> (N, 5) of class, cx, cy, w, h, all normalised."""
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label.is_file():
        return np.zeros((0, 5))
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
    return np.asarray(rows, dtype=float).reshape(-1, 5)


# --- 2. Describe -------------------------------------------------------------

def describe_dataset(location: Path, names: list[str]) -> dict:
    stats = {"splits": {}, "classes": {}}
    per_class = {n: {"instances": Counter(), "images": Counter(), "rel_area": []} for n in names}
    sizes = Counter()
    for split in SPLITS:
        images = split_images(location, split)
        empty = 0
        for path in images:
            labels = read_labels(path)
            with Image.open(path) as im:
                sizes[f"{im.width}x{im.height}"] += 1
            if len(labels) == 0:
                empty += 1
            for cls in set(labels[:, 0].astype(int)):
                per_class[names[cls]]["images"][split] += 1
            for cls, _, _, w, h in labels:
                per_class[names[int(cls)]]["instances"][split] += 1
                per_class[names[int(cls)]]["rel_area"].append(w * h)
        stats["splits"][split] = {"images": len(images), "images_without_labels": empty}
    for name, d in per_class.items():
        area = np.asarray(d["rel_area"])
        stats["classes"][name] = {
            "instances": {s: d["instances"][s] for s in SPLITS},
            "images": {s: d["images"][s] for s in SPLITS},
            "total_instances": int(sum(d["instances"].values())),
            "median_rel_area_pct": float(np.median(area) * 100) if len(area) else None,
            "p10_rel_area_pct": float(np.percentile(area, 10) * 100) if len(area) else None,
            "p90_rel_area_pct": float(np.percentile(area, 90) * 100) if len(area) else None,
        }
    stats["image_sizes"] = dict(sizes.most_common())
    return stats


# --- 3. Finetune -------------------------------------------------------------

def finetune(data_yaml: Path, device: str) -> tuple[Path, Path, float]:
    model = YOLO(S.BASE_MODEL)
    started = time.time()
    model.train(
        data=str(data_yaml),
        device=device,
        project=str(S.ROOT / "runs"),
        name="finetune",
        exist_ok=True,
        **S.TRAIN_ARGS,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    S.WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, S.WEIGHTS_PATH)
    return S.WEIGHTS_PATH, save_dir, time.time() - started


# --- 4. Evaluate --------------------------------------------------------------

def ultralytics_test_metrics(weights: Path, data_yaml: Path, names: list[str], device: str) -> dict:
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml), split="test", imgsz=S.TRAIN_ARGS["imgsz"], batch=S.TRAIN_ARGS["batch"],
        conf=S.MIN_CONF, device=device, plots=True, project=str(S.ROOT / "runs"),
        name="test_eval", exist_ok=True, verbose=False,
    )
    per_class = {}
    for i, cls in enumerate(m.box.ap_class_index):
        per_class[names[int(cls)]] = {
            "ap50": float(m.box.ap50[i]),
            "ap50_95": float(m.box.ap[i]),
            "precision_at_best_f1": float(m.box.p[i]),
            "recall_at_best_f1": float(m.box.r[i]),
        }
    for name in names:
        per_class.setdefault(name, {"ap50": 0.0, "ap50_95": 0.0,
                                    "precision_at_best_f1": 0.0, "recall_at_best_f1": 0.0})
    for plot in ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
        src = Path(m.save_dir) / plot
        if src.is_file():
            shutil.copy2(src, S.ARTIFACTS_DIR / f"test_{plot}")
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class}


def collect_predictions(model: YOLO, images: list[Path], device: str,
                        class_map: dict[int, int] | None = None) -> list[dict]:
    """Ground truth and every prediction down to MIN_CONF, in pixel xyxy.

    class_map translates the model's class ids into the dataset's. Predictions
    whose class has no counterpart are dropped, which is exactly what a model
    with no such class amounts to on this task.
    """
    records = []
    for path in images:
        with Image.open(path) as im:
            width, height = im.size
        labels = read_labels(path)
        gt = [[int(c), (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
              for c, cx, cy, w, h in labels]
        result = model.predict(str(path), conf=S.MIN_CONF, imgsz=S.TRAIN_ARGS["imgsz"],
                               device=device, max_det=300, verbose=False)[0]
        pred = []
        for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            cls = int(cls)
            if class_map is not None:
                if cls not in class_map:
                    continue
                cls = class_map[cls]
            pred.append([cls, round(conf, 5), *[round(v, 1) for v in box]])
        records.append({"file": path.name, "width": width, "height": height, "gt": gt, "pred": pred})
    return records


def coco_zero_shot(images: list[Path], device: str) -> dict:
    """The COCO checkpoint has no scoreboard classes. Record what it does see on
    the test frames, so the report can show that rather than assert it."""
    model = YOLO(S.BASE_MODEL)
    seen = Counter()
    for path in images:
        result = model.predict(str(path), conf=0.25, imgsz=S.TRAIN_ARGS["imgsz"], device=device, verbose=False)[0]
        seen.update(model.names[int(c)] for c in result.boxes.cls.tolist())
    return {"detections_at_conf_0.25": dict(seen.most_common()), "coco_classes_mapping_to_dataset": []}


def summarise_detection(records: list[dict], names: list[str]) -> dict:
    """mAP-style summary for a baseline that predicts no dataset class at all:
    AP is zero by construction. Count what it costs in missed boxes."""
    n_gt = Counter(int(g[0]) for r in records for g in r["gt"])
    n_pred = sum(len(r["pred"]) for r in records)
    return {
        "map50": 0.0, "map50_95": 0.0, "predictions_of_dataset_classes": n_pred,
        "per_class": {names[c]: {"ap50": 0.0, "recall": 0.0, "instances": n_gt[c]} for c in range(len(names))},
    }


# --- 5. Cost -----------------------------------------------------------------

def default_costs(**overrides) -> costing.Costs:
    values = dict(fp=S.COST_FP, fn=S.COST_FN, review=S.COST_REVIEW, tp=S.COST_TP,
                  base_rate=S.BASE_RATE, p_change=S.P_SCORE_CHANGE)
    values.update(overrides)
    return costing.Costs(**values)


def audited(records: list[dict], split: str) -> list[dict]:
    """Leave out the frames published as "no scoreboard" that show one."""
    drop = set(S.MISLABELLED_NULL_FRAMES.get(split, []))
    return [r for r in records if r["file"] not in drop]


def choose_thresholds(val_cases: costing.Cases, costs: costing.Costs, key: str = "cost_per_1000_upper") -> dict:
    """Choose on validation so the test split stays a fair check. Choosing and
    scoring on the same 24 frames would flatter the result.

    The default criterion is cost with the false-positive rate at its 95% upper
    bound: on 32 frames, "no wrong score seen yet" is weak evidence and a wrong
    score is the expensive mistake.
    """
    single = costing.pick_single(costing.sweep_single(val_cases, costs), key=key)
    band = costing.pick_band(costing.sweep_band(val_cases, costs), key=key)
    return {"t_single": single["t_high"], "t_low": band["t_low"], "t_high": band["t_high"]}


def price_policies(cases: costing.Cases, costs: costing.Costs, th: dict) -> dict:
    return {
        "do_nothing": costing.do_nothing(S.SCOUT_EUR_PER_FRAME),
        "human_everything": costing.human_checks_everything(cases, costs, S.MIN_CONF),
        "model_alone": costing.evaluate(cases, costs, th["t_single"], th["t_single"]),
        "with_review": costing.evaluate(cases, costs, th["t_low"], th["t_high"]),
        "predict_nothing": costing.predict_nothing(cases, costs),
    }


def false_positive_budget(policies: dict, costs: costing.Costs) -> dict:
    """How many wrong scores per 1,000 frames the review design can afford before
    it costs more than the scouts, against how many 0 observed errors rules out."""
    hybrid = policies["with_review"]
    without_fp = hybrid["cost_per_1000"] - costs.fp * hybrid["per_1000"]["fp_auto"]
    allowed = (policies["do_nothing"]["cost_per_1000"] - without_fp) / costs.fp
    published = hybrid["tp_auto"] + hybrid["fp_auto"]
    published_per_1000 = hybrid["per_1000"]["tp_auto"] + hybrid["per_1000"]["fp_auto"]
    # Rule of three: with 0 errors in n trials, the 95% upper bound on the rate is 3/n.
    upper_rate = (3 / published if hybrid["fp_auto"] == 0 else (hybrid["fp_auto"] + 3) / published) if published else 1.0
    return {
        "allowed_fp_per_1000_frames": allowed,
        "readings_published_on_test": published,
        "fp_published_on_test": hybrid["fp_auto"],
        "fp_rate_upper_95": upper_rate,
        "fp_per_1000_frames_upper_95": upper_rate * published_per_1000,
        "expected_cost_at_upper_bound": without_fp + costs.fp * upper_rate * published_per_1000,
    }


def staffing(review_share: float) -> dict:
    reviews_per_day = review_share * S.FRAMES_PER_DAY
    # Games cluster in evening windows, so size for the peak, not the daily total.
    peak_reviews_per_s = review_share * S.PEAK_CONCURRENT_GAMES / S.FRAME_INTERVAL_S
    return {
        "frames_per_day": S.FRAMES_PER_DAY,
        "reviews_per_day": reviews_per_day,
        "review_hours_per_day": reviews_per_day * S.REVIEW_SECONDS / 3600,
        "peak_reviews_per_minute": peak_reviews_per_s * 60,
        "peak_operators_busy": peak_reviews_per_s * S.REVIEW_SECONDS,
        "scout_hours_per_day_today": S.GAMES_PER_DAY * S.BROADCAST_MINUTES / 60,
        "scouts_at_peak_today": S.PEAK_CONCURRENT_GAMES,
    }


def sensitivity(val_cases: costing.Cases, test_cases: costing.Cases) -> list[dict]:
    """Re-choose the thresholds on validation under other assumptions and price them on test."""
    grid = [dict(fp=fp) for fp in (1.5, 15, 150, 1500)]
    grid += [dict(fn=S.P_SCORE_CHANGE * d) for d in (5, 80)]
    grid += [dict(base_rate=b) for b in (0.6, 0.95)]
    # A review also delays a score change by its 3 s. If that delay is worth far
    # more than the operator's time, auto-publishing starts to pay.
    grid += [dict(review=S.COST_REVIEW * m) for m in (0.5, 10, 100, 1000)]
    rows = []
    for change in grid:
        costs = default_costs(**change)
        th = choose_thresholds(val_cases, costs)
        p = price_policies(test_cases, costs, th)
        rows.append({
            "changed": {k: round(v, 4) for k, v in change.items()}, **th,
            "with_review_cost_per_1000": p["with_review"]["cost_per_1000"],
            "with_review_cost_per_1000_upper": p["with_review"]["cost_per_1000_upper"],
            "with_review_share": p["with_review"]["review_share"],
            "with_review_fp_published": p["with_review"]["fp_auto"],
            "model_alone_cost_per_1000": p["model_alone"]["cost_per_1000"],
            "do_nothing_cost_per_1000": p["do_nothing"]["cost_per_1000"],
            "human_everything_cost_per_1000": p["human_everything"]["cost_per_1000"],
        })
    return rows


def plot_cost_curve(sweeps: dict[str, list[dict]], th: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {"val (audited, used to choose)": dict(color="#888"),
              "test (audited)": dict(color="#1f5fa8", lw=2),
              "test (published labels)": dict(color="#1f5fa8", ls=":")}
    for label, sweep in sweeps.items():
        ax.plot([r["t_high"] for r in sweep], [r["cost_per_1000"] for r in sweep], label=label, **styles[label])
    val = sweeps["val (audited, used to choose)"]
    ax.plot([r["t_high"] for r in val], [r["cost_per_1000_upper"] for r in val], color="#c0392b", lw=1,
            label="val, false-positive rate at its 95% upper bound (criterion)")
    ax.axhline(1000 * S.SCOUT_EUR_PER_FRAME, color="#27ae60", ls="--", lw=1, label="do nothing (scouts)")
    ax.axvline(th["t_single"], color="#c0392b", ls="--", label=f"chosen single threshold {th['t_single']:.2f}")
    ax.axvspan(th["t_low"], th["t_high"], color="#f5b041", alpha=0.25,
               label=f"review band {th['t_low']:.2f}-{th['t_high']:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("confidence threshold for publishing a Team Points reading")
    ax.set_ylabel("expected cost per 1,000 frames (EUR)")
    ax.set_title("Model deciding alone: cost against threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_and_cost(location: Path, names: list[str], data_yaml: Path, weights: Path,
                      device: str, training: dict, stats: dict) -> dict:
    """Everything after training: test metrics, baselines, saved predictions, costing."""
    positive = names.index(S.POSITIVE_CLASS)
    finetuned_metrics = ultralytics_test_metrics(weights, data_yaml, names, device)
    model = YOLO(str(weights))
    val_records = collect_predictions(model, split_images(location, "val"), device)
    test_records = collect_predictions(model, split_images(location, "test"), device)

    # Zero-shot COCO: no COCO class corresponds to a scoreboard field, so the
    # class map is empty and every one of its boxes is dropped.
    coco_seen = coco_zero_shot(split_images(location, "test"), device)
    coco_records = collect_predictions(YOLO(S.BASE_MODEL), split_images(location, "test"), device, class_map={})
    nothing_records = [{**r, "pred": []} for r in test_records]

    with (S.ARTIFACTS_DIR / "predictions.json").open("w") as f:
        json.dump({"class_names": names, "positive_class": S.POSITIVE_CLASS,
                   "mislabelled_null_frames": S.MISLABELLED_NULL_FRAMES,
                   "val": val_records, "test": test_records}, f)

    def cases(records):
        return costing.build_cases(records, positive, S.IOU_MATCH)

    val_cases = cases(audited(val_records, "val"))
    test_views = {"audited": cases(audited(test_records, "test")), "published": cases(test_records)}
    costs = default_costs()
    th = choose_thresholds(val_cases, costs)
    op = {**th, "chosen_on": "val, audited"}
    for view, c in test_views.items():
        policies = price_policies(c, costs, th)
        policies["with_review_ci95"] = costing.bootstrap_cost(c, costs, th["t_low"], th["t_high"])
        policies["model_alone_ci95"] = costing.bootstrap_cost(c, costs, th["t_single"], th["t_single"])
        policies["fp_budget"] = false_positive_budget(policies, costs)
        policies["staffing"] = staffing(policies["with_review"]["review_share"])
        op[f"test_{view}"] = policies
    # The two choices the report argues against, priced on the same test frames.
    point = choose_thresholds(val_cases, costs, key="cost_per_1000")
    op["point_estimate_choice"] = {**point, "test_audited": price_policies(test_views["audited"], costs, point)}
    op["published_labels_choice"] = choose_thresholds(cases(val_records), costs)

    plot_cost_curve({
        "val (audited, used to choose)": costing.sweep_single(val_cases, costs),
        "test (audited)": costing.sweep_single(test_views["audited"], costs),
        "test (published labels)": costing.sweep_single(test_views["published"], costs),
    }, th, S.ARTIFACTS_DIR / "cost_curve.png")

    # Per class at the lower threshold: everything the pipeline acts on at all.
    pr_at_op = costing.per_class_pr(test_records, len(names), th["t_low"], S.IOU_MATCH)
    for row in pr_at_op:
        row["class"] = names[row["class_id"]]

    baselines = {
        "finetuned": {"map50": finetuned_metrics["map50"], "map50_95": finetuned_metrics["map50_95"],
                      "cost_model_alone_per_1000": op["test_published"]["model_alone"]["cost_per_1000"]},
        "coco_zero_shot": {**summarise_detection(coco_records, names), **coco_seen,
                           "cost_per_1000": costing.evaluate(cases(coco_records), costs, 0.5, 0.5)["cost_per_1000"]},
        "predict_nothing": {**summarise_detection(nothing_records, names),
                            "cost_per_1000": costing.predict_nothing(cases(nothing_records), costs)["cost_per_1000"]},
    }

    metrics = {
        "dataset": {"workspace": S.WORKSPACE, "project": S.PROJECT, "version": S.VERSION,
                    "url": f"https://universe.roboflow.com/{S.WORKSPACE}/{S.PROJECT}/dataset/{S.VERSION}",
                    "class_names": names, "positive_class": S.POSITIVE_CLASS,
                    "mislabelled_null_frames": S.MISLABELLED_NULL_FRAMES, **stats},
        "training": training,
        "finetuned_test": finetuned_metrics,
        "per_class_at_t_low": pr_at_op,
        "baselines": baselines,
        "costs": costs.__dict__,
        "scout_eur_per_frame": S.SCOUT_EUR_PER_FRAME,
        "operating_point": op,
        "sensitivity": sensitivity(val_cases, test_views["audited"]),
    }
    with (S.ARTIFACTS_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=float)

    examples = S.ARTIFACTS_DIR / "examples"
    shutil.rmtree(examples, ignore_errors=True)
    examples.mkdir()
    test_dir = location / SPLITS["test"] / "images"
    for name in S.EXAMPLE_FRAMES:
        if (test_dir / name).is_file():
            shutil.copy2(test_dir / name, examples / name)
    return metrics


def main() -> None:
    S.ARTIFACTS_DIR.mkdir(exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    location = download_dataset()
    data_yaml, names = write_data_yaml(location)
    print(f"Dataset at {location}; classes: {names}")

    stats = describe_dataset(location, names)
    print(json.dumps(stats, indent=2))

    weights, train_dir, train_seconds = finetune(data_yaml, device)
    for plot in ("results.png", "labels.jpg"):
        if (train_dir / plot).is_file():
            shutil.copy2(train_dir / plot, S.ARTIFACTS_DIR / f"train_{plot}")
    training = {"base_model": S.BASE_MODEL, "device": device, "seconds": round(train_seconds),
                "torch": torch.__version__, **S.TRAIN_ARGS}

    metrics = evaluate_and_cost(location, names, data_yaml, weights, device, training, stats)
    print(json.dumps({k: metrics[k] for k in ("finetuned_test", "baselines", "operating_point")},
                     indent=2, default=float))
    print(f"Saved {weights.relative_to(S.ROOT)} and artifacts/.")


if __name__ == "__main__":
    main()
