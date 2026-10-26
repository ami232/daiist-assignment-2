"""
Assignment 2: training pipeline (fire and smoke detection).

Run with:  uv run python main.py train

What it does, in order:
1. Downloads the Roboflow dataset (public, version 2, not forked).
2. Finetunes yolo11n, starting from the pretrained COCO checkpoint.
3. Saves the best weights to weights/best.pt.
4. Evaluates on the TEST split: the finetuned model, the pretrained COCO
   model used zero-shot, and a "predict nothing" baseline.
5. Saves per-class results and the raw test predictions to outputs/.
   The cost sweep and the dashboard read these files later.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "datasets" / "fire-smoke-v2"

# Dataset (public Roboflow Universe dataset, used as it is, not forked)
WORKSPACE = "sayed-gamall"
PROJECT = "fire-smoke-detection-yolov11"
VERSION = 2

# Training settings
BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 20))
FRACTION = float(os.environ.get("FRACTION", 1.0))  # share of training images used
IMGSZ = 640
BATCH = 16
SEED = 0


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_dataset():
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    dataset = version.download("yolov11", location=str(DATA_DIR))
    return Path(dataset.location)


def train(data_yaml, device):
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        seed=SEED,
        patience=5,
        fraction=FRACTION,
        project=str(ROOT / "runs"),
        name="fire_smoke",
        exist_ok=True,
        workers=2,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    final = WEIGHTS_DIR / "best.pt"
    shutil.copy(best, final)
    return final


def per_class_results(metrics):
    rows = {}
    for i, c in enumerate(metrics.box.ap_class_index):
        rows[metrics.names[int(c)]] = {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "AP50": float(metrics.box.ap50[i]),
            "mAP50_95": float(metrics.box.ap[i]),
        }
    return rows


def evaluate_finetuned(weights, data_yaml, device):
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        device=device,
        conf=0.001,
        plots=False,
        project=str(ROOT / "runs"),
        name="test_eval",
        exist_ok=True,
    )
    return {
        "AP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "per_class": per_class_results(m),
    }


def read_labels(label_file):
    """Read one YOLO label file. Returns [class, x1, y1, x2, y2] (0-1 scale)."""
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        nums = [float(x) for x in parts[1:]]
        if len(nums) == 4:
            # box format: x_center, y_center, width, height
            xc, yc, w, h = nums
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        else:
            # outline format: a list of x, y points
            xs, ys = nums[0::2], nums[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append([cls, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)])
    return boxes


def collect_test_predictions(weights, data_root, device):
    """Run the model on every test image with a very low confidence cut-off,
    so any threshold can be applied later without running the model again."""
    model = YOLO(str(weights))
    images = sorted((data_root / "test" / "images").glob("*"))
    records = []
    results = model.predict(
        source=[str(p) for p in images],
        conf=0.001,
        iou=0.6,
        imgsz=IMGSZ,
        device=device,
        max_det=100,
        stream=True,
        verbose=False,
    )
    for r in results:
        stem = Path(r.path).stem
        preds = []
        for box, conf, cls in zip(
            r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
        ):
            preds.append([int(cls), round(conf, 4)] + [round(v, 4) for v in box])
        gt = read_labels(data_root / "test" / "labels" / f"{stem}.txt")
        records.append({"image": stem, "gt": gt, "pred": preds})
    names = [model.names[i] for i in sorted(model.names)]
    return names, records


def baselines(names, records):
    gt_counts = {n: 0 for n in names}
    for rec in records:
        for g in rec["gt"]:
            gt_counts[names[g[0]]] += 1

    # Baseline 1: the pretrained COCO model, zero-shot. COCO has no "fire" or
    # "smoke" class, so it cannot name either one correctly.
    coco = YOLO(BASE_MODEL)
    in_common = [n for n in coco.names.values() if n.lower() in ("fire", "smoke")]
    zero = {"precision": 0.0, "recall": 0.0, "AP50": 0.0, "mAP50_95": 0.0}
    coco_result = {
        "classes_in_common_with_dataset": in_common,
        "AP50": 0.0,
        "mAP50_95": 0.0,
        "per_class": {n: dict(zero) for n in names},
    }

    # Baseline 2: predict nothing. It never raises an alarm, so it misses everything.
    nothing_result = {
        "AP50": 0.0,
        "mAP50_95": 0.0,
        "per_class": {n: dict(zero) for n in names},
        "missed_objects": gt_counts,
    }
    return gt_counts, coco_result, nothing_result


def main():
    device = pick_device()
    print(f"Training on device: {device}")

    data_root = download_dataset()
    data_yaml = data_root / "data.yaml"

    weights = train(data_yaml, device)
    finetuned = evaluate_finetuned(weights, data_yaml, device)
    names, records = collect_test_predictions(weights, data_root, device)
    gt_counts, coco_result, nothing_result = baselines(names, records)

    fire_id = [i for i, n in enumerate(names) if n.lower() == "fire"]
    images_without_fire = None
    if fire_id:
        images_without_fire = sum(
            1 for rec in records if not any(g[0] == fire_id[0] for g in rec["gt"])
        )

    summary = {
        "dataset": {
            "workspace": WORKSPACE,
            "project": PROJECT,
            "version": VERSION,
            "classes": names,
        },
        "settings": {
            "base_model": BASE_MODEL,
            "epochs": EPOCHS,
            "image_size": IMGSZ,
            "batch": BATCH,
            "seed": SEED,
            "train_fraction": FRACTION,
            "device": str(device),
        },
        "test_split": {
            "images": len(records),
            "objects_per_class": gt_counts,
            "images_without_fire": images_without_fire,
        },
        "finetuned": finetuned,
        "coco_zero_shot": coco_result,
        "predict_nothing": nothing_result,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "train_results.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "test_predictions.json").write_text(
        json.dumps({"names": names, "images": records})
    )

    print("\n=== TEST RESULTS ===")
    print(f"Finetuned: AP50 {finetuned['AP50']:.3f}, mAP50-95 {finetuned['mAP50_95']:.3f}")
    for name, row in finetuned["per_class"].items():
        print(
            f"  {name}: precision {row['precision']:.3f}, recall {row['recall']:.3f}, "
            f"AP50 {row['AP50']:.3f}, mAP50-95 {row['mAP50_95']:.3f}"
        )
    print("COCO zero-shot: AP50 0.000 (no fire or smoke class)")
    print("Predict nothing: AP50 0.000")
    print(f"Test images with no fire in them: {images_without_fire}")
    print(f"Weights saved to {weights}")


# Called directly (not behind an if __name__ check) so that it runs however
# main.py chooses to launch this file.
main()

# Cost sweep and object-level evaluation, using the files main() just wrote.
from costing import run_costing
from evaluation import run_evaluation

run_costing()
run_evaluation()