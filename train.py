"""
Assignment 2: training pipeline for solder-joint inspection.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download Roboflow Universe work-6qkmv/solder-2-hcmiq v37 (YOLO format).
2. Re-split it: Roboflow's unaugmented "valid" split becomes our held-out
   TEST set; a validation set for early stopping is carved from Roboflow's
   train split by *source image*, so augmented copies never straddle the line.
3. Describe the dataset (instances per class, split sizes, object sizes).
4. Finetune yolo11s from COCO-pretrained weights; copy best.pt to weights/.
5. Evaluate on the test split: AP50 / mAP50-95 per class, box-level confusion
   matrix, and an image-level pass/fail score for every test image.
6. Baselines on the same split: COCO yolo11s zero-shot, and predict-nothing.
7. Cost sweep: single threshold, and the (pass | review | fail) band.
8. Save everything app.py needs into artifacts/, plus artifacts/RESULTS.md.

Environment variables (all optional):
    EPOCHS=80        training epochs
    DEVICE=0|mps|cpu where to train (auto-detected otherwise)
    BATCH=16         batch size
    SKIP_TRAIN=1     reuse weights/best.pt and only redo evaluation + costing
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import config  # noqa: E402
import costing  # noqa: E402
from api_key import load_api_key  # noqa: E402

RAW = config.DATA_DIR / "raw"
SPLIT = config.DATA_DIR / "solder"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------- data
def download() -> Path:
    """Fetch the dataset version once; reuse it on later runs."""
    if (RAW / "data.yaml").exists():
        print(f"Dataset already downloaded at {RAW}")
        return RAW
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(config.RF_WORKSPACE).project(config.RF_PROJECT).version(config.RF_VERSION)
    version.download(config.RF_FORMAT, location=str(RAW), overwrite=True)
    return RAW


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT)


def label_for(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def source_id(img: Path) -> str:
    """Roboflow names augmented copies '<original>_jpg.rf.<hash>.jpg'."""
    return img.name.split(".rf.")[0]


def build_splits(raw: Path) -> tuple[Path, list[str]]:
    """train/val/test folders and a data.yaml for ultralytics."""
    meta = yaml.safe_load((raw / "data.yaml").read_text())
    names = list(meta["names"].values()) if isinstance(meta["names"], dict) else list(meta["names"])

    rf_train = images_in(raw / "train" / "images")
    rf_valid = images_in(raw / "valid" / "images")
    rf_test = images_in(raw / "test" / "images") if (raw / "test" / "images").exists() else []

    # Roboflow v37 has no test split. Its valid split is unaugmented, so it is
    # the cleanest held-out set available: use it as TEST (plus any test images
    # a future version might add).
    test = rf_valid + rf_test

    groups = defaultdict(list)
    for img in rf_train:
        groups[source_id(img)].append(img)
    keys = sorted(groups)
    random.Random(config.SEED).shuffle(keys)
    n_val = max(1, round(len(keys) * config.VAL_FRACTION))
    val_keys = set(keys[:n_val])
    train = [p for k in keys if k not in val_keys for p in groups[k]]
    val = [p for k in val_keys for p in groups[k]]

    if SPLIT.exists():
        shutil.rmtree(SPLIT)
    for split, files in (("train", train), ("val", val), ("test", test)):
        (SPLIT / split / "images").mkdir(parents=True)
        (SPLIT / split / "labels").mkdir(parents=True)
        for img in files:
            shutil.copy2(img, SPLIT / split / "images" / img.name)
            lbl = label_for(img)
            dst = SPLIT / split / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst)
            else:
                dst.write_text("")

    data_yaml = SPLIT / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(SPLIT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))
    print(f"Splits: train {len(train)}  val {len(val)}  test {len(test)} images "
          f"({len(keys) - n_val} / {n_val} source images in train / val)")
    return data_yaml, names


def read_labels(lbl: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO label rows as (class, x, y, w, h); polygons become their bounding box."""
    rows = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                rows.append((c, x, y, w, h))
            elif len(parts) > 5 and len(parts) % 2 == 1:
                c = int(parts[0])
                pts = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                rows.append((c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return rows


def defect_ids(names: list[str]) -> set[int]:
    return {i for i, n in enumerate(names) if n.strip().lower() not in config.GOOD_CLASSES}


def describe(names: list[str]) -> dict:
    defects = defect_ids(names)
    stats = {"names": names, "defect_class_ids": sorted(defects), "splits": {}}
    for split in ("train", "val", "test"):
        imgs = images_in(SPLIT / split / "images")
        inst = Counter()
        areas = defaultdict(list)
        n_def_img = n_empty = 0
        boxes_per_img = []
        for img in imgs:
            rows = read_labels(label_for(img))
            boxes_per_img.append(len(rows))
            if not rows:
                n_empty += 1
            if any(r[0] in defects for r in rows):
                n_def_img += 1
            for c, _, _, w, h in rows:
                inst[c] += 1
                areas[c].append(w * h)
        stats["splits"][split] = {
            "images": len(imgs),
            "images_with_defect": n_def_img,
            "images_without_labels": n_empty,
            "defect_image_share": n_def_img / max(len(imgs), 1),
            "boxes_per_image_mean": float(np.mean(boxes_per_img)) if imgs else 0.0,
            "instances": {names[c]: int(inst.get(c, 0)) for c in range(len(names))},
            "box_area_share_median": {
                names[c]: float(np.median(areas[c])) if areas[c] else None for c in range(len(names))
            },
            "box_area_share_p10_p90": {
                names[c]: [float(np.percentile(areas[c], 10)), float(np.percentile(areas[c], 90))]
                if areas[c] else None for c in range(len(names))
            },
        }
    return stats


# ----------------------------------------------------------------------------- model
def pick_device() -> str:
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def finetune(data_yaml: Path, device: str) -> dict:
    from ultralytics import YOLO

    epochs = int(os.environ.get("EPOCHS", config.EPOCHS))
    settings = {
        "model": config.BASE_MODEL, "epochs": epochs, "imgsz": config.IMGSZ,
        "patience": config.PATIENCE, "seed": config.SEED, "device": device,
        "batch": int(os.environ.get("BATCH", 16)),
    }
    print(f"Finetuning with {settings}")
    model = YOLO(config.BASE_MODEL)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=config.IMGSZ, patience=config.PATIENCE,
        seed=config.SEED, deterministic=True, device=device, batch=settings["batch"],
        workers=2 if device == "cpu" else 4, project=str(config.ROOT / "runs"),
        name="finetune", exist_ok=True, plots=True, verbose=False,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    config.WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, config.WEIGHTS)
    settings["epochs_run"] = int(getattr(model.trainer, "epoch", epochs - 1)) + 1
    settings["best_fitness"] = float(getattr(model.trainer, "best_fitness", 0) or 0)
    print(f"Saved {config.WEIGHTS}")
    return settings


def predict_all(model, images: list[Path], device: str) -> list:
    out = []
    for i in range(0, len(images), 32):
        batch = [str(p) for p in images[i:i + 32]]
        out.extend(model.predict(batch, conf=0.001, iou=0.6, imgsz=config.IMGSZ,
                                 device=device, verbose=False, max_det=100))
    return out


def test_records(model, names: list[str], device: str) -> list[dict]:
    """Ground truth + every prediction (conf >= 0.001) for each test image."""
    defects = defect_ids(names)
    imgs = images_in(SPLIT / "test" / "images")
    results = predict_all(model, imgs, device)
    records = []
    for img, res in zip(imgs, results):
        h, w = res.orig_shape
        gt = read_labels(label_for(img))
        gt_boxes = [[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                    for _, x, y, bw, bh in gt]
        b = res.boxes
        pcls = b.cls.int().tolist() if b is not None else []
        pconf = b.conf.tolist() if b is not None else []
        pbox = b.xyxy.tolist() if b is not None else []
        defect_conf = [c for k, c in zip(pcls, pconf) if k in defects]
        records.append({
            "image": img.name,
            "gt_cls": [g[0] for g in gt],
            "gt_boxes": gt_boxes,
            "pred_cls": pcls,
            "pred_conf": [round(c, 5) for c in pconf],
            "pred_boxes": [[round(v, 1) for v in bx] for bx in pbox],
            "label": bool(any(g[0] in defects for g in gt)),
            "score": float(max(defect_conf)) if defect_conf else 0.0,
        })
    return records


def coco_records(device: str) -> list[dict]:
    """COCO yolo11s zero-shot. It has no solder classes, so at image level its
    score is simply its most confident detection of anything."""
    from ultralytics import YOLO

    model = YOLO(config.COCO_MODEL)
    imgs = images_in(SPLIT / "test" / "images")
    results = predict_all(model, imgs, device)
    out = []
    for img, res in zip(imgs, results):
        conf = res.boxes.conf.tolist() if res.boxes is not None else []
        cls = res.boxes.cls.int().tolist() if res.boxes is not None else []
        top = Counter(model.names[c] for c, s in zip(cls, conf) if s >= 0.25).most_common(3)
        out.append({"image": img.name, "score": float(max(conf)) if conf else 0.0, "top": top})
    return out


def ultralytics_test_metrics(data_yaml: Path, names: list[str], device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(config.WEIGHTS))
    r = model.val(data=str(data_yaml), split="test", imgsz=config.IMGSZ, conf=0.001,
                  iou=0.6, device=device, project=str(config.ROOT / "runs"),
                  name="test_eval", exist_ok=True, plots=True, verbose=False)
    per_class = {}
    for i, c in enumerate(r.box.ap_class_index):
        per_class[names[int(c)]] = {
            "precision": float(r.box.p[i]), "recall": float(r.box.r[i]),
            "ap50": float(r.box.ap50[i]), "ap50_95": float(r.box.ap[i]),
        }
    return {"map50": float(r.box.map50), "map50_95": float(r.box.map),
            "precision": float(r.box.mp), "recall": float(r.box.mr), "per_class": per_class}


# ----------------------------------------------------------------------------- costing
def image_level(scores, labels) -> dict:
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    out = {"auc": costing.roc_auc(scores, labels)}
    neg = np.sort(scores[~labels])
    for fpr in (0.01, 0.05):
        key = f"recall_at_fpr_{fpr:g}"
        if len(neg) and labels.any():
            # smallest threshold that fails at most `fpr` of good windows
            t = neg[min(len(neg) - 1, int(np.floor((1 - fpr) * len(neg))))]
            out[key] = float((scores[labels] > t).mean())
        else:
            out[key] = float("nan")
    return out


def bootstrap_cost(scores, labels, lo, hi, costs, n=500) -> list[float]:
    """How much the chosen point's cost moves if the test set had come out
    differently: resample test images with replacement."""
    rng = np.random.default_rng(config.SEED)
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(scores), len(scores))
        vals.append(costing.cost_per_case(costing.route_counts(scores[idx], labels[idx], lo, hi), costs)["cost"])
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def sensitivity(scores, labels) -> list[dict]:
    """Re-pick the operating point under different assumptions."""
    rows = []
    for name, field, values in (
        ("base rate", "base_rate", [0.0005, 0.001, 0.002, 0.005, 0.02]),
        ("false-negative cost (EUR)", "fn", [15, 36, 72, 150, 300]),
        ("false-positive cost (EUR)", "fp", [0.5, 1.5, 3, 6]),
    ):
        for v in values:
            c = costing.Costs(**{field: v})
            single = min(costing.sweep_single(scores, labels, c), key=lambda r: r["cost"])
            band = costing.best_band(scores, labels, c)
            rows.append({"assumption": name, "value": v,
                         "best_single_threshold": single["threshold"],
                         "cost_per_1000_single": single["cost"] * 1000,
                         "band_lo": band["lo"], "band_hi": band["hi"],
                         "cost_per_1000_band": band["cost"] * 1000,
                         "review_share": band["review_share"]})
    return rows


def plot_cost_curve(sweep: list[dict], chosen: float, path: Path) -> None:
    df = pd.DataFrame(sweep)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.threshold, df.cost * 1000, lw=2)
    ax.axvline(chosen, color="crimson", ls="--", label=f"chosen = {chosen:.2f}")
    ax.set_xlabel("confidence threshold (fail if score ≥ t)")
    ax.set_ylabel("expected cost per 1,000 windows (EUR)")
    ax.set_title("Cost against threshold, model decides alone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def write_results_md(art: dict) -> None:
    s, names = art["dataset"], art["dataset"]["names"]
    L = ["# Results (written by train.py)\n"]
    L.append("## Dataset composition\n")
    L.append("| class | train | val | test | median box area (% of image, test) |")
    L.append("|---|---|---|---|---|")
    for n in names:
        med = s["splits"]["test"]["box_area_share_median"][n]
        L.append(f"| {n} | {s['splits']['train']['instances'][n]} | {s['splits']['val']['instances'][n]} | "
                 f"{s['splits']['test']['instances'][n]} | {'' if med is None else f'{med * 100:.1f}%'} |")
    L.append("")
    for sp in ("train", "val", "test"):
        d = s["splits"][sp]
        L.append(f"- **{sp}**: {d['images']} images, {d['images_with_defect']} contain a defect "
                 f"({d['defect_image_share']:.0%}), {d['images_without_labels']} unlabelled, "
                 f"{d['boxes_per_image_mean']:.1f} boxes per image")
    L.append("")
    m, b = art["metrics"], art["baselines"]
    L.append("## Finetuned model vs baselines (same test split)\n")
    L.append("| model | AP50 | mAP50-95 | image AUC | defect recall @1% false-fail | min cost / 1,000 windows |")
    L.append("|---|---|---|---|---|---|")
    for key, label in (("finetuned", "yolo11s finetuned"), ("coco_zero_shot", "yolo11s COCO zero-shot"),
                       ("predict_nothing", "predict nothing (pass everything)")):
        r = b[key]
        L.append(f"| {label} | {r['map50']:.3f} | {r['map50_95']:.3f} | {r['image']['auc']:.3f} | "
                 f"{r['image']['recall_at_fpr_0.01']:.3f} | €{r['best_cost'] * 1000:.2f} |")
    L.append("")
    L.append(f"COCO zero-shot most common labels on solder images: {b['coco_zero_shot']['typical_coco_labels']}\n")
    L.append("## Per class (test split)\n")
    L.append("| class | role | test instances | precision | recall | AP50 | mAP50-95 |")
    L.append("|---|---|---|---|---|---|---|")
    for n in names:
        pc = m["per_class"].get(n)
        role = "negative (good)" if n.lower() in config.GOOD_CLASSES else "positive (defect)"
        inst = s["splits"]["test"]["instances"][n]
        if pc:
            L.append(f"| {n} | {role} | {inst} | {pc['precision']:.3f} | {pc['recall']:.3f} | "
                     f"{pc['ap50']:.3f} | {pc['ap50_95']:.3f} |")
        else:
            L.append(f"| {n} | {role} | {inst} | – | – | – | – |")
    L.append(f"| **all** | | | {m['precision']:.3f} | {m['recall']:.3f} | {m['map50']:.3f} | {m['map50_95']:.3f} |")
    L.append("")
    cm, cm_names = np.array(art["confusion"]["matrix"]), names + ["background"]
    L.append(f"Box-level confusion matrix at conf ≥ {art['confusion']['conf']} "
             "(rows = true, columns = predicted):\n")
    L.append("| true \\ pred | " + " | ".join(cm_names) + " |")
    L.append("|---" * (len(cm_names) + 1) + "|")
    for i, n in enumerate(cm_names):
        L.append(f"| {n} | " + " | ".join(str(v) for v in cm[i]) + " |")
    L.append("")
    L.append("Image-level recall by defect type at the band's fail threshold "
             "(share of test windows containing that defect that the model fails on its own):\n")
    for n, v in art["recall_by_defect_type"].items():
        L.append(f"- {n}: {v['recall']:.3f} of {v['images']} windows")
    L.append("")
    op, alt = art["operating_point"], art["alternatives"]
    L.append("## Costing\n")
    L.append(f"Assumptions: {json.dumps(art['costs'])}\n")
    L.append(f"- Best single threshold: **{op['single']['threshold']:.2f}**, "
             f"€{op['single']['cost'] * 1000:.2f} per 1,000 windows "
             f"(90% bootstrap interval €{op['single']['ci'][0] * 1000:.2f}–€{op['single']['ci'][1] * 1000:.2f}); "
             f"recall {op['single']['recall_auto']:.3f}, false-fail rate {op['single']['false_fail_rate']:.4f}")
    L.append(f"- Best band: pass < **{op['band']['lo']:.2f}** ≤ review < **{op['band']['hi']:.2f}** ≤ fail, "
             f"€{op['band']['cost'] * 1000:.2f} per 1,000, review share {op['band']['review_share']:.2%} "
             f"(90% interval €{op['band']['ci'][0] * 1000:.2f}–€{op['band']['ci'][1] * 1000:.2f})")
    st = op["band"]["staffing"]
    L.append(f"- That sends **{st['reviews_per_day']:.0f} windows/day** to a person "
             f"({st['review_hours_per_day']:.1f} h at {config.REVIEW_SECONDS} s each) = "
             f"**{st['reviewers_needed']:.2f} reviewers** at {config.PRODUCTIVE_HOURS_PER_REVIEWER} productive h")
    L.append(f"- Test counts in the band (pos = defective, neg = good): {json.dumps(op['band']['counts'])}")
    L.append("")
    L.append("| policy | cost / 1,000 windows | per day |")
    L.append("|---|---|---|")
    per_day = config.BOARDS_PER_DAY * config.WINDOWS_PER_BOARD / 1000
    for label, v in (("do nothing (no inspection)", alt["do_nothing"]),
                     ("person checks every window", alt["human_checks_all"]),
                     ("fail everything", alt["fail_everything"]),
                     ("model decides alone", op["single"]["cost"]),
                     ("model + human review band", op["band"]["cost"])):
        L.append(f"| {label} | €{v * 1000:.2f} | €{v * 1000 * per_day:,.0f} |")
    L.append("")
    L.append("## Sensitivity: what moves the operating point\n")
    L.append("| assumption | value | best single t | €/1,000 | band lo–hi | €/1,000 | review share |")
    L.append("|---|---|---|---|---|---|---|")
    for r in art["sensitivity"]:
        L.append(f"| {r['assumption']} | {r['value']} | {r['best_single_threshold']:.2f} | "
                 f"{r['cost_per_1000_single']:.2f} | {r['band_lo']:.2f}–{r['band_hi']:.2f} | "
                 f"{r['cost_per_1000_band']:.2f} | {r['review_share']:.2%} |")
    L.append("")
    L.append(f"Training settings: {json.dumps(art['training'])}")
    (config.ARTIFACTS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main() -> None:
    config.ARTIFACTS.mkdir(exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    raw = download()
    data_yaml, names = build_splits(raw)
    stats = describe(names)
    print(json.dumps({k: v["instances"] for k, v in stats["splits"].items()}, indent=1))

    if os.environ.get("SKIP_TRAIN") == "1" and config.WEIGHTS.exists():
        print(f"SKIP_TRAIN=1: reusing {config.WEIGHTS}")
        prev = config.ARTIFACTS / "metrics.json"
        training = json.loads(prev.read_text())["training"] if prev.exists() else {"note": "reused weights"}
    else:
        training = finetune(data_yaml, device)

    from ultralytics import YOLO

    model = YOLO(str(config.WEIGHTS))
    metrics = ultralytics_test_metrics(data_yaml, names, device)
    records = test_records(model, names, device)
    scores = [r["score"] for r in records]
    labels = [r["label"] for r in records]

    costs = costing.Costs()
    sweep = costing.sweep_single(scores, labels, costs)
    single = min(sweep, key=lambda r: r["cost"])
    band = costing.best_band(scores, labels, costs)
    single["ci"] = bootstrap_cost(scores, labels, single["threshold"], single["threshold"], costs)
    band["ci"] = bootstrap_cost(scores, labels, band["lo"], band["hi"], costs)
    band["staffing"] = costing.staffing(band["review_share"])
    band["counts"] = costing.route_counts(scores, labels, band["lo"], band["hi"])

    conf_cm = round(band["hi"], 2) if band["hi"] > 0 else 0.25
    cm = costing.detection_confusion(records, len(names), conf_cm)

    recall_by_type = {}
    for c in sorted(defect_ids(names)):
        hit = [r["score"] >= band["hi"] for r in records if c in r["gt_cls"]]
        recall_by_type[names[c]] = {"images": len(hit), "recall": float(np.mean(hit)) if hit else float("nan")}

    # Baselines on the same test split.
    coco = coco_records(device)
    coco_scores = [r["score"] for r in coco]
    zeros = [0.0] * len(labels)
    baselines = {}
    for key, sc, det in (("finetuned", scores, metrics),
                         ("coco_zero_shot", coco_scores, {"map50": 0.0, "map50_95": 0.0}),
                         ("predict_nothing", zeros, {"map50": 0.0, "map50_95": 0.0})):
        best = min(costing.sweep_single(sc, labels, costs), key=lambda r: r["cost"])
        baselines[key] = {"map50": det["map50"], "map50_95": det["map50_95"],
                          "image": image_level(sc, labels), "best_cost": best["cost"],
                          "best_threshold": best["threshold"]}
    baselines["coco_zero_shot"]["note"] = (
        "COCO has no solder-joint classes, so every box is either unmatched or the "
        "wrong class: AP is 0 by construction. Image score = most confident COCO box.")
    baselines["coco_zero_shot"]["typical_coco_labels"] = Counter(
        lbl for r in coco for lbl, _ in r["top"]).most_common(5)

    art = {
        "dataset": stats, "training": training, "metrics": metrics, "baselines": baselines,
        "costs": costs.as_dict(), "operating_point": {"single": single, "band": band},
        "alternatives": costing.alternatives(costs), "sensitivity": sensitivity(scores, labels),
        "confusion": {"conf": conf_cm, "iou": 0.5, "matrix": cm.tolist(), "names": names + ["background"]},
        "recall_by_defect_type": recall_by_type,
        "source": {"workspace": config.RF_WORKSPACE, "project": config.RF_PROJECT,
                   "version": config.RF_VERSION},
    }

    (config.ARTIFACTS / "metrics.json").write_text(json.dumps(art, indent=1, default=float))
    (config.ARTIFACTS / "test_predictions.json").write_text(json.dumps(records))
    pd.DataFrame(sweep).to_csv(config.ARTIFACTS / "cost_sweep.csv", index=False)
    pd.DataFrame({"image": [r["image"] for r in records], "label": labels, "score": scores,
                  "coco_score": coco_scores}).to_csv(config.ARTIFACTS / "image_scores.csv", index=False)
    plot_cost_curve(sweep, single["threshold"], config.ARTIFACTS / "cost_curve.png")
    write_results_md(art)

    print(f"\nBest single threshold {single['threshold']:.2f}: €{single['cost'] * 1000:.2f} / 1,000 windows")
    print(f"Best band {band['lo']:.2f}-{band['hi']:.2f}: €{band['cost'] * 1000:.2f} / 1,000, "
          f"review {band['review_share']:.2%}, {band['staffing']['reviewers_needed']:.2f} reviewers")
    print(f"Wrote {config.ARTIFACTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
