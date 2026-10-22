"""
Assignment 2: training pipeline.

Stage 1: download the dataset from Roboflow Universe.
Stage 2: describe the dataset (counts per class and split, object size).
Stage 3: finetune yolo11n on it.
Stage 4: evaluate on the test split against two baselines.

Quick timing test:      QUICK=1 uv run python main.py train
Reuse existing weights: SKIP_TRAINING=1 uv run python main.py train
"""

# Make Python trust the same certificates as the Mac. This must come first.
import truststore

truststore.inject_into_ssl()

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key
from costing import run_costing
from review import run_review

# The exact dataset version we use (also written in REPORT.md).
WORKSPACE = "defect-5hwax"
PROJECT = "fabric-defect-ieswu"
VERSION = 16
DATA_DIR = "data/fabric-defect"
OUT_DIR = Path("outputs")
WEIGHTS_DIR = Path("weights")

# Training settings. QUICK=1 is only a short test to measure speed.
QUICK = os.environ.get("QUICK") == "1"
SKIP_TRAINING = os.environ.get("SKIP_TRAINING") == "1"
MODEL = "yolo11n.pt"
EPOCHS = 1 if QUICK else 30
FRACTION = 0.05 if QUICK else 1.0   # share of the training images used
IMGSZ = 512                          # the dataset images are 512x512; defects are small, so no shrinking
BATCH = 16

# Use the fastest hardware available: NVIDIA GPU, then the Mac's GPU, then the CPU.
if torch.cuda.is_available():
    DEVICE = "0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def download_dataset():
    """Download the dataset in YOLO format and return the folder it was saved in."""
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    dataset = version.download("yolov11", location=DATA_DIR)
    return dataset.location


def describe_dataset(folder):
    """Count images and defects per split and class, and measure defect size.

    Each label line is: class x_center y_center width height (all 0 to 1),
    so width * height is the share of the photo the defect covers.
    """
    folder = Path(folder)
    with open(folder / "data.yaml") as f:
        config = yaml.safe_load(f)
    names = config["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    stats = {"class_names": names, "splits": {}}
    for split in ["train", "valid", "test"]:
        images = [p for p in (folder / split / "images").iterdir() if p.is_file()]
        counts = {name: 0 for name in names}
        areas = {name: [] for name in names}
        images_with_boxes = 0
        for label_file in (folder / split / "labels").glob("*.txt"):
            rows = [line.split() for line in label_file.read_text().splitlines() if line.strip()]
            if rows:
                images_with_boxes += 1
            for row in rows:
                name = names[int(row[0])]
                counts[name] += 1
                areas[name].append(float(row[3]) * float(row[4]))
        stats["splits"][split] = {
            "images": len(images),
            "images_with_defects": images_with_boxes,
            "images_without_defects": len(images) - images_with_boxes,
            "instances_per_class": counts,
            "median_size_pct_of_image": {
                name: round(100 * float(np.median(a)), 2) if a else None
                for name, a in areas.items()
            },
        }
    return stats


def print_stats(stats):
    for split, s in stats["splits"].items():
        print(f"\n{split.upper()}: {s['images']} images "
              f"({s['images_with_defects']} with defects, "
              f"{s['images_without_defects']} without)")
        for name in stats["class_names"]:
            print(f"  {name:15s} {s['instances_per_class'][name]:6d} instances, "
                  f"median size {s['median_size_pct_of_image'][name]}% of image")


def write_data_yaml(folder, class_names):
    """Write our own data file with absolute paths, so YOLO always finds the images."""
    folder = Path(folder).resolve()
    config = {
        "path": str(folder),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    yaml_path = folder / "data_abs.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    return yaml_path


def train_model(data_yaml):
    """Finetune the pretrained YOLO model and return the path of the best weights."""
    model = YOLO(MODEL)
    start = time.time()
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        fraction=FRACTION,
        seed=0,
        project=str(Path("runs").resolve()),
        name="quick" if QUICK else "train",
        exist_ok=True,
        plots=not QUICK,
        workers=2,
    )
    minutes = (time.time() - start) / 60
    print(f"\nTraining took {minutes:.1f} minutes for {EPOCHS} epoch(s) on {FRACTION:.0%} of the data.")
    return Path(model.trainer.best)


def evaluate_finetuned(weights, data_yaml, class_names):
    """Test our model on the test split: precision, recall and AP per class."""
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=0.001,
        plots=True,
        project=str(Path("runs").resolve()),
        name="test_eval",
        exist_ok=True,
        verbose=False,
    )
    per_class = {}
    for i, class_id in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[class_names[int(class_id)]] = {
            "precision": float(p),
            "recall": float(r),
            "ap50": float(ap50),
            "map50_95": float(ap),
        }
    overall = {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "ap50": float(m.box.map50),
        "map50_95": float(m.box.map),
    }
    OUT_DIR.mkdir(exist_ok=True)
    for png in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        source = Path(m.save_dir) / png
        if source.exists():
            shutil.copy(source, OUT_DIR / f"test_{png}")
    return {"overall": overall, "per_class": per_class}


def evaluate_coco_zero_shot(folder):
    """Baseline 1: the original COCO model, used as it is.

    COCO knows 80 everyday objects and none of them is a fabric defect, so its
    AP on our three classes is zero by construction. What we can measure is how
    often it flags a test image anyway, and with what (wrong) labels.
    """
    model = YOLO(MODEL)
    images = sorted(p for p in (Path(folder) / "test" / "images").iterdir() if p.is_file())
    boxes = 0
    images_flagged = 0
    labels_seen = {}
    for path in images:
        # One image per call, so the non-max-suppression time limit cannot drop predictions.
        result = model.predict(
            source=str(path), conf=0.25, imgsz=IMGSZ, device=DEVICE, verbose=False
        )[0]
        n = len(result.boxes)
        boxes += n
        images_flagged += 1 if n else 0
        for class_id in result.boxes.cls.tolist():
            label = result.names[int(class_id)]
            labels_seen[label] = labels_seen.get(label, 0) + 1
    return {
        "ap50_on_our_classes": 0.0,
        "note": "COCO has no defect classes, so AP is zero by construction.",
        "test_images": len(images),
        "images_with_any_box": images_flagged,
        "total_boxes": boxes,
        "labels_it_used": dict(sorted(labels_seen.items(), key=lambda kv: -kv[1])[:8]),
    }

def evaluate_predict_nothing(stats):
    """Baseline 2: a model that never reports a defect."""
    test = stats["splits"]["test"]
    return {
        "ap50": 0.0,
        "recall": 0.0,
        "test_images": test["images"],
        "defective_images_caught": 0,
        "defective_images_in_test": test["images_with_defects"],
        "note": "Scores zero on this test set because every test image has a defect.",
    }


def evaluate_all(weights, data_yaml, folder, stats):
    """Run the finetuned model and both baselines on the test split and save the results."""
    class_names = stats["class_names"]
    finetuned = evaluate_finetuned(weights, data_yaml, class_names)
    coco = evaluate_coco_zero_shot(folder)
    nothing = evaluate_predict_nothing(stats)

    results = {
        "weights": str(weights),
        "finetuned": finetuned,
        "baseline_coco_zero_shot": coco,
        "baseline_predict_nothing": nothing,
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(results, indent=2))

    print("\n=== TEST SPLIT RESULTS (finetuned model) ===")
    print(f"{'class':15s} {'precision':>9s} {'recall':>7s} {'AP50':>7s} {'mAP50-95':>9s}")
    for name, v in finetuned["per_class"].items():
        print(f"{name:15s} {v['precision']:9.3f} {v['recall']:7.3f} {v['ap50']:7.3f} {v['map50_95']:9.3f}")
    o = finetuned["overall"]
    print(f"{'ALL':15s} {o['precision']:9.3f} {o['recall']:7.3f} {o['ap50']:7.3f} {o['map50_95']:9.3f}")

    print("\n=== BASELINE 1: COCO model, zero-shot ===")
    print(f"AP50 on our classes: {coco['ap50_on_our_classes']} (zero by construction)")
    print(f"It put boxes on {coco['images_with_any_box']} of {coco['test_images']} test images "
          f"({coco['total_boxes']} boxes), using labels: {coco['labels_it_used']}")

    print("\n=== BASELINE 2: predict nothing ===")
    print(f"Defective test images caught: {nothing['defective_images_caught']} of "
          f"{nothing['defective_images_in_test']}  (AP50 {nothing['ap50']})")
    print("\nSaved outputs/test_metrics.json")


def main():
    print("Training device:", DEVICE)
    folder = download_dataset()
    print("Dataset saved in:", folder)

    stats = describe_dataset(folder)
    print_stats(stats)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    data_yaml = write_data_yaml(folder, stats["class_names"])

    if SKIP_TRAINING:
        final_weights = WEIGHTS_DIR / "best.pt"
        print("\nSKIP_TRAINING=1: using existing", final_weights)
    else:
        best = train_model(data_yaml)
        if QUICK:
            final_weights = best
            print("Quick test only: weights were NOT copied to weights/.")
        else:
            WEIGHTS_DIR.mkdir(exist_ok=True)
            shutil.copy(best, WEIGHTS_DIR / "best.pt")
            final_weights = WEIGHTS_DIR / "best.pt"
            print("Saved weights/best.pt")

    evaluate_all(final_weights, data_yaml, folder, stats)
    run_costing(final_weights, folder, DEVICE, IMGSZ, OUT_DIR)
    run_review(OUT_DIR)


if __name__ == "__main__":
    main()