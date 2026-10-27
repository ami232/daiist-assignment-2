"""Assignment 2: training pipeline. Run: uv run python main.py train

Downloads the dataset, finetunes YOLO11, evaluates on the test split against
two baselines, and saves everything the app and the cost sweep need.
"""
import json
import os
import platform
import runpy
import shutil
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from dataset import DATASET_DIR, ROOT, get_dataset

ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"

MODEL = "yolo11s.pt"
EPOCHS = int(os.environ.get("EPOCHS", 50))  # EPOCHS=1 for a quick smoke test
IMGSZ = 512
BATCH = 16
SEED = 0
PATIENCE = 15
PRED_CONF = 0.001  # keep almost every box, so any threshold can be swept later
IOU = 0.6


def test_images():
    return sorted((DATASET_DIR / "test" / "images").iterdir())


def read_gt(label_file):
    """Ground truth as [class, x1, y1, x2, y2], normalised to the image."""
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
        if len(vals) == 4:
            x, y, w, h = vals
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        else:  # polygon label: use its bounding box
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append([cls, x1, y1, x2, y2])
    return boxes


def train(data_yaml, device):
    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        seed=SEED, patience=PATIENCE, device=device,
        project=str(RUNS), name="train", exist_ok=True, plots=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS / "best.pt")
    return WEIGHTS / "best.pt"


def evaluate(weights, data_yaml, device):
    """Ultralytics test-split metrics. P and R are reported at the max-F1 point."""
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, conf=PRED_CONF,
        iou=IOU, device=device, project=str(RUNS), name="val_test",
        exist_ok=True, plots=True,
    )
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[model.names[int(c)]] = {
            "precision": float(m.box.p[i]),
            "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]),
            "ap50_95": float(m.box.ap[i]),
        }
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map),
            "per_class": per_class}


def save_predictions(weights, device):
    """Raw test predictions plus ground truth, for the cost sweep and the app."""
    model = YOLO(str(weights))
    out = {}
    for img in test_images():
        r = model.predict(str(img), conf=PRED_CONF, iou=IOU, imgsz=IMGSZ,
                          max_det=100, device=device, verbose=False)[0]
        preds = [
            [int(c), round(float(s), 4)] + [round(float(v), 4) for v in b]
            for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxyn)
        ]
        gt = read_gt(DATASET_DIR / "test" / "labels" / f"{img.stem}.txt")
        out[img.name] = {"pred": preds, "gt": gt}
    return out


def coco_zero_shot(device, our_names):
    """Pretrained COCO model, no finetuning, on the same test images."""
    model = YOLO(MODEL)
    shared = sorted(set(model.names.values()) & set(our_names))
    seen = {}
    for img in test_images():
        r = model.predict(str(img), conf=0.25, imgsz=IMGSZ, device=device,
                          verbose=False)[0]
        for c in r.boxes.cls:
            name = model.names[int(c)]
            seen[name] = seen.get(name, 0) + 1
    return {
        "shared_class_names": shared,
        "ap50": 0.0 if not shared else None,
        "note": ("COCO has no class with our names, so every detection is a "
                 "wrong class and AP50 is 0 by construction." if not shared
                 else "Some names match: map them by hand before scoring."),
        "detections_by_coco_class": seen,
    }


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    data_yaml = get_dataset()
    device = 0 if torch.cuda.is_available() else "cpu"

    start = time.time()
    weights = train(data_yaml, device)
    minutes = round((time.time() - start) / 60, 1)

    metrics = evaluate(weights, data_yaml, device)
    our_names = list(metrics["per_class"])
    baselines = {
        "coco_zero_shot": coco_zero_shot(device, our_names),
        "predict_nothing": {"ap50": 0.0, "recall": 0.0,
                            "note": "Predicts no objects: every positive is missed."},
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "baselines.json").write_text(json.dumps(baselines, indent=2))
    (ARTIFACTS / "predictions.json").write_text(
        json.dumps(save_predictions(weights, device)))
    (ARTIFACTS / "run_info.json").write_text(json.dumps({
        "model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
        "seed": SEED, "patience": PATIENCE, "train_minutes": minutes,
        "device": torch.cuda.get_device_name(0) if device == 0 else platform.processor(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
    }, indent=2))
    for png in (RUNS / "val_test").glob("confusion_matrix*.png"):
        shutil.copy(png, ARTIFACTS / png.name)  # runs/ is gitignored, artifacts/ is not
    runpy.run_path(str(ROOT / "cost_sweep.py"), run_name="__main__")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()