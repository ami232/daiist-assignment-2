"""
Assignment 2: training pipeline. Run with `uv run python main.py train`.

Decision supported: can a forager eat this mushroom?
Positive class = Poisonous.

Steps, start to finish, with no manual intervention:

1. Download jamurdetection/jamur-detection version 2 from Roboflow Universe
   (the public dataset as it is, no fork).
2. Describe the dataset: instances per class per split, and object size
   relative to the image.
3. Finetune yolo11n from its COCO-pretrained checkpoint and copy the best
   weights to weights/best.pt.
4. Evaluate on the test split: AP50, mAP50-95, per-class precision, recall
   and AP, and the confusion matrix. Compare on the same test split against
   the COCO model zero-shot and a trivial predict-nothing baseline.
5. For every ground-truth mushroom, record the model's Poisonous confidence.
   Sweep the confidence threshold with the euro figures in costs.json,
   pick the cheapest single threshold and the cheapest review band on the
   VALIDATION split, then report what they cost on the TEST split (choosing
   on test would make the test cost look better than it really is).
6. Save everything the app needs to artifacts/, so the app never trains.
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent

# Dataset on Roboflow Universe.
WORKSPACE = "jamurdetection"
PROJECT = "jamur-detection"
VERSION = 2

# Training settings: Ultralytics defaults apart from the epoch count, chosen
# so a CPU-only GitHub runner finishes comfortably inside its time limit.
BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = 640
SEED = 0

DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
RUNS_DIR = ROOT / "runs"
BEST_WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

SPLITS = {"train": "train", "val": "valid", "test": "test"}  # name -> folder
MATCH_IOU = 0.5        # a predicted box "covers" a mushroom at IoU >= 0.5
SWEEP_CONF = 0.001     # keep nearly every box so the sweep sees the whole range
DEFAULT_CONF = 0.25    # Ultralytics' usual confidence: COCO baseline and confusion matrix
THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)  # single-threshold sweep
BAND_GRID = np.round(np.linspace(0, 1, 51), 2)    # coarser grid for the two-threshold search


# --------------------------------------------------------------------------- #
# 1-2. Data
# --------------------------------------------------------------------------- #

def download_dataset() -> Path:
    """Download the dataset once and make data.yaml use absolute paths."""
    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download("yolov11", location=str(DATA_DIR), overwrite=True)

    # Roboflow's exported paths assume a different folder layout.
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    config.update(path=str(DATA_DIR), train="train/images",
                  val="valid/images", test="test/images")
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data_yaml


def class_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def find_class(names: list[str], keyword: str) -> int:
    matches = [i for i, name in enumerate(names) if keyword in name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one class containing '{keyword}', got {names}")
    return matches[0]


def split_images(split: str) -> list[Path]:
    folder = DATA_DIR / SPLITS[split] / "images"
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def read_labels(image: Path) -> list[tuple[int, np.ndarray]]:
    """Ground truth for one image as [(class, box)], box = normalised x1 y1 x2 y2."""
    label_file = image.parent.parent / "labels" / f"{image.stem}.txt"
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, coords = int(parts[0]), np.array(parts[1:], dtype=float)
        if len(coords) == 4:  # x_center y_center width height
            x, y, w, h = coords
            box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        else:                 # polygon: use its bounding box
            xs, ys = coords[0::2], coords[1::2]
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        boxes.append((cls, box))
    return boxes


def describe_dataset(names: list[str]) -> dict:
    """Images and instances per class per split, and box area as a share of the image."""
    stats = {}
    for split in SPLITS:
        images = split_images(split)
        counts, areas, unlabelled = Counter(), [], 0
        for image in images:
            labels = read_labels(image)
            unlabelled += not labels
            for cls, box in labels:
                counts[names[cls]] += 1
                areas.append((box[2] - box[0]) * (box[3] - box[1]))
        areas = np.array(areas) if areas else np.zeros(1)
        stats[split] = {
            "images": len(images),
            "images_without_labels": unlabelled,
            "instances": {name: counts.get(name, 0) for name in names},
            "box_area_share_of_image": {
                "p10": float(np.percentile(areas, 10)),
                "median": float(np.median(areas)),
                "p90": float(np.percentile(areas, 90)),
            },
        }
    return stats


# --------------------------------------------------------------------------- #
# 3-4. Finetuning and standard evaluation
# --------------------------------------------------------------------------- #

def finetune(data_yaml: Path) -> Path:
    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, seed=SEED,
                project=str(RUNS_DIR), name="finetune", exist_ok=True)
    BEST_WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy2(model.trainer.best, BEST_WEIGHTS)
    return BEST_WEIGHTS


def evaluate_on_test(weights: Path, data_yaml: Path, names: list[str]) -> dict:
    """Ultralytics' test-split metrics: overall, per class, and confusion matrix.

    AP needs every box down to a very low confidence, but a confusion matrix
    built from those would count thousands of junk boxes as false positives.
    So the matrix comes from a second pass at Ultralytics' usual 0.25.
    """
    model = YOLO(str(weights))
    result = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                       project=str(RUNS_DIR), name="test", exist_ok=True)
    at_025 = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, conf=DEFAULT_CONF,
                       project=str(RUNS_DIR), name="test_conf025", exist_ok=True)
    box = result.box
    per_class = {
        names[int(cls)]: {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
        }
        for i, cls in enumerate(box.ap_class_index)
    }
    plots = {"BoxPR_curve.png": result, "confusion_matrix.png": at_025,
             "confusion_matrix_normalized.png": at_025}
    for plot, source_run in plots.items():
        source = Path(source_run.save_dir) / plot
        if source.exists():
            shutil.copy2(source, ARTIFACTS / plot)
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        # At confidence 0.25. Rows = predicted class, columns = true class;
        # the last row/column is background (missed mushroom / box on nothing).
        "confusion_matrix_conf": DEFAULT_CONF,
        "confusion_matrix": at_025.confusion_matrix.matrix.astype(int).tolist(),
        "confusion_labels": names + ["background"],
    }


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one x1 y1 x2 y2 box against many."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def coco_zero_shot(split: str) -> dict:
    """Pretrained COCO yolo11n with no finetuning.

    COCO has no mushroom class, let alone edible vs poisonous, so its AP on
    our classes is zero and it never flags anything as Poisonous: for costing
    it behaves exactly like the predict-nothing baseline. We still measure
    whether it finds the mushrooms at all, and what it thinks they are.
    """
    model = YOLO(BASE_MODEL)
    found, total, seen_as = 0, 0, Counter()
    for image in split_images(split):
        result = model.predict(str(image), conf=DEFAULT_CONF, imgsz=IMGSZ, verbose=False)[0]
        pred_boxes = result.boxes.xyxyn.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy().astype(int)
        for _, box in read_labels(image):
            total += 1
            overlaps = iou(box, pred_boxes)
            if len(overlaps) and overlaps.max() >= MATCH_IOU:
                found += 1
                seen_as[result.names[int(pred_cls[overlaps.argmax()])]] += 1
    return {
        "map50": 0.0,
        "map50_95": 0.0,
        "share_of_mushrooms_boxed_as_any_coco_class": found / total if total else 0.0,
        "coco_labels_it_used": dict(seen_as.most_common(5)),
    }


# --------------------------------------------------------------------------- #
# 5. One Poisonous confidence per ground-truth mushroom, then costing
# --------------------------------------------------------------------------- #

def poisonous_confidence(model: YOLO, split: str, names: list[str],
                         poison_idx: int) -> pd.DataFrame:
    """One row per true mushroom: its class and the model's Poisonous confidence.

    The confidence is the highest score of any Poisonous box covering the
    mushroom, or 0 if there is none (the model missed it or called it edible).
    """
    rows = []
    for image in split_images(split):
        boxes = model.predict(str(image), conf=SWEEP_CONF, imgsz=IMGSZ, verbose=False)[0].boxes
        pred_boxes = boxes.xyxyn.cpu().numpy()
        pred_cls = boxes.cls.cpu().numpy().astype(int)
        pred_conf = boxes.conf.cpu().numpy()
        for cls, box in read_labels(image):
            covers = (iou(box, pred_boxes) >= MATCH_IOU) & (pred_cls == poison_idx)
            rows.append({
                "image": image.name,
                "true_class": names[cls],
                "poisonous": cls == poison_idx,
                "poisonous_conf": float(pred_conf[covers].max(initial=0)),
            })
    return pd.DataFrame(rows)


def plot_cost_curve(val_sweep: list[dict], test_sweep: list[dict],
                    chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["threshold"] for r in val_sweep], [r["cost_per_1000"] for r in val_sweep],
            label="validation (threshold chosen here)")
    ax.plot([r["threshold"] for r in test_sweep], [r["cost_per_1000"] for r in test_sweep],
            label="test", alpha=0.6)
    ax.axvline(chosen, color="black", linestyle="--", label=f"chosen threshold = {chosen:.2f}")
    ax.set_xlabel("Poisonous confidence threshold (fail at or above)")
    ax.set_ylabel("Expected cost per 1,000 mushrooms (€)")
    ax.set_title("Cost against threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    # 1-2. Data
    data_yaml = download_dataset()
    names = class_names(data_yaml)
    poison_idx = find_class(names, "poison")
    stats = describe_dataset(names)
    print(json.dumps(stats, indent=2))

    # 3-4. Finetune and evaluate. REUSE_WEIGHTS=1 skips the finetune and reuses
    # the committed weights, e.g. to redo the costing after editing costs.json.
    if os.environ.get("REUSE_WEIGHTS") == "1" and BEST_WEIGHTS.exists():
        weights = BEST_WEIGHTS
    else:
        weights = finetune(data_yaml)
    finetuned = evaluate_on_test(weights, data_yaml, names)
    baselines = {"coco_zero_shot": coco_zero_shot("test"),
                 "predict_nothing": {"map50": 0.0, "map50_95": 0.0}}

    # 5. Costing: thresholds chosen on validation, reported on test
    model = YOLO(str(weights))
    val = poisonous_confidence(model, "val", names, poison_idx)
    test = poisonous_confidence(model, "test", names, poison_idx)
    val.to_csv(ARTIFACTS / "confidences_val.csv", index=False)
    test.to_csv(ARTIFACTS / "confidences_test.csv", index=False)

    costs = costing.load_costs()
    val_sweep = costing.sweep(val.poisonous_conf, val.poisonous, costs, THRESHOLDS)
    test_sweep = costing.sweep(test.poisonous_conf, test.poisonous, costs, THRESHOLDS)
    chosen = min(val_sweep, key=lambda r: r["cost_per_1000"])["threshold"]
    band = costing.best_band(val.poisonous_conf, val.poisonous, costs, BAND_GRID)
    pd.DataFrame(val_sweep).to_csv(ARTIFACTS / "cost_sweep_val.csv", index=False)
    pd.DataFrame(test_sweep).to_csv(ARTIFACTS / "cost_sweep_test.csv", index=False)
    plot_cost_curve(val_sweep, test_sweep, chosen, ARTIFACTS / "cost_vs_threshold.png")

    # Neither baseline ever flags Poisonous: every mushroom has confidence 0.
    flags_nothing = np.zeros(len(test))
    for baseline in baselines.values():
        baseline["test_cost_at_chosen_threshold"] = costing.expected_cost(
            flags_nothing, test.poisonous, chosen, chosen, costs)

    operating_point = {
        "positive_class": names[poison_idx],
        "costs": costs,
        "thresholds_chosen_on": "validation split",
        "single_threshold": chosen,
        "review_band": {"t_low": band["t_low"], "t_high": band["t_high"]},
        "test_policies": costing.policy_costs(
            test.poisonous_conf, test.poisonous, costs, chosen, band["t_low"], band["t_high"]),
    }
    metrics = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "classes": names, "stats": stats},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "seed": SEED},
        "finetuned": finetuned,
        "baselines": baselines,
    }

    # 6. Everything the app needs
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "operating_point.json").write_text(json.dumps(operating_point, indent=2))
    print(json.dumps({"test_map50": finetuned["map50"], **operating_point}, indent=2))


if __name__ == "__main__":
    main()
