"""
Assignment 2: training pipeline for helmet compliance detection.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download the dataset version named in decisions.py from Roboflow Universe.
2. Describe it from its label files: images and instances per class and split,
   and how big each head is relative to the image.
3. Finetune yolo11n from its pretrained COCO checkpoint, and copy the best
   weights to weights/best.pt.
4. Evaluate on the test split: box-level metrics per class and the confusion
   matrix, then a per-item score for every test image from the finetuned
   model and from the zero-shot COCO model.
5. If the costs in decisions.py are filled in, sweep the threshold and save
   the cost curve, the cheapest threshold and the cheapest review band.

Everything the dashboard shows is written to artifacts/, so app.py never has
to train or evaluate anything.

Set SMOKE_TEST=1 for a one-epoch run on a tenth of the data. It only proves the
pipeline runs end to end; never submit the weights it produces.
"""

import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import costing
import decisions
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / f"{decisions.ROBOFLOW_WORKSPACE}{decisions.ROBOFLOW_PROJECT}-v{decisions.ROBOFLOW_VERSION}"
RUNS_DIR = ROOT / "runs"
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"

# Training settings. Record these in REPORT.md alongside where you trained.
BASE_MODEL = "yolo11n.pt"
TRAIN_SETTINGS = {
    "epochs": 1 if SMOKE_TEST else 15,
    "patience": 15,          # stop early if val mAP has not improved for this many epochs
    "imgsz": 640,            # the dataset was exported at 640x640
    "batch": 16,
    "seed": 0,
    "deterministic": True,
    "fraction": 0.1 if SMOKE_TEST else 1.0,
}
SPLITS = {"train": "train", "val": "valid", "test": "test"}  # YOLO name -> Roboflow folder
COCO_PERSON_ID = 0  # "person" in the COCO class list
CM_CONF = 0.25  # confidence at which the box-level confusion matrix is counted


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- 1. Download -------------------------------------------------------------

def download_dataset() -> Path:
    """Download the dataset once, and return a data.yaml with absolute paths."""
    if not (DATA_DIR / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        version = (
            rf.workspace(decisions.ROBOFLOW_WORKSPACE)
            .project(decisions.ROBOFLOW_PROJECT)
            .version(decisions.ROBOFLOW_VERSION)
        )
        version.download("yolov11", location=str(DATA_DIR))

    # Roboflow's data.yaml uses relative paths that depend on where you run from.
    # Rewrite them as absolute so training works from any working directory.
    original = yaml.safe_load((DATA_DIR / "data.yaml").read_text())
    data_yaml = DATA_DIR / "data_abs.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(DATA_DIR),
        **{name: f"{folder}/images" for name, folder in SPLITS.items()},
        "names": dict(enumerate(original["names"])),
    }))
    return data_yaml


# --- 2. Describe -------------------------------------------------------------

def read_labels(label_file: Path) -> np.ndarray:
    """One row per box: class, x_center, y_center, width, height (relative)."""
    if not label_file.exists() or label_file.stat().st_size == 0:
        return np.zeros((0, 5))
    return np.loadtxt(label_file, ndmin=2)[:, :5]


def label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def describe_dataset(names: list[str]) -> dict:
    stats = {}
    for split, folder in SPLITS.items():
        images = sorted((DATA_DIR / folder / "images").glob("*"))
        boxes = [read_labels(label_path(image)) for image in images]
        all_boxes = np.vstack(boxes) if boxes else np.zeros((0, 5))
        per_class = {}
        for index, name in enumerate(names):
            own = all_boxes[all_boxes[:, 0] == index]
            area = own[:, 3] * own[:, 4]  # box area as a share of the image
            per_class[name] = {
                "instances": int(len(own)),
                "images_containing": int(sum((b[:, 0] == index).any() for b in boxes)),
                "box_area_share_p10": float(np.percentile(area, 10)) if len(area) else None,
                "box_area_share_median": float(np.median(area)) if len(area) else None,
                "box_area_share_p90": float(np.percentile(area, 90)) if len(area) else None,
            }
        stats[split] = {
            "images": len(images),
            "images_without_boxes": int(sum(len(b) == 0 for b in boxes)),
            "classes": per_class,
        }
    return stats


# --- 3. Finetune -------------------------------------------------------------

def finetune(data_yaml: Path, device: str) -> Path:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        device=device,
        project=str(RUNS_DIR),
        name="finetune",
        exist_ok=True,
        **TRAIN_SETTINGS,
    )
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS_PATH)
    return WEIGHTS_PATH


# --- 4. Evaluate -------------------------------------------------------------

def evaluate_boxes(weights: Path, data_yaml: Path, device: str) -> dict:
    """Box-level metrics on the test split, overall and per class."""
    model = YOLO(str(weights))
    common = dict(data=str(data_yaml), split="test", device=device,
                  project=str(RUNS_DIR), exist_ok=True, plots=True)
    # AP needs every detection down to a very low confidence ...
    metrics = model.val(name="test_eval", **common)
    # ... but the confusion matrix counts whatever val keeps, so it gets its
    # own pass at a usable threshold, or thousands of near-zero guesses show
    # up as false alarms.
    cm_metrics = model.val(name="test_cm", conf=CM_CONF, **common)
    copies = {
        "BoxPR_curve.png": metrics.save_dir,
        "confusion_matrix.png": cm_metrics.save_dir,
        "confusion_matrix_normalized.png": cm_metrics.save_dir,
    }
    for plot, folder in copies.items():
        source = Path(folder) / plot
        if source.exists():
            shutil.copy(source, ARTIFACTS / plot)
    names = [metrics.names[i] for i in range(len(metrics.names))]
    return {
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "AP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": metrics.summary(),
        # Rows are predicted class, columns true class; the last of each is
        # "background" (a missed box or a false alarm). Counted at CM_CONF.
        "confusion_matrix": {
            "labels": names + ["background"],
            "conf": CM_CONF,
            "matrix": cm_metrics.confusion_matrix.matrix.astype(int).tolist(),
        },
    }


def score_test_items(weights: Path, names: list[str], device: str) -> pd.DataFrame:
    """One row per test image: which classes it truly contains, and the
    highest confidence each model gives each class (0 if nothing found)."""
    finetuned = YOLO(str(weights))
    coco = YOLO(BASE_MODEL)
    rows = []
    for image in sorted((DATA_DIR / "test" / "images").glob("*")):
        row = {"image": image.name}
        true_classes = set(read_labels(label_path(image))[:, 0].astype(int))
        for index, name in enumerate(names):
            row[f"true_{name}"] = int(index in true_classes)

        result = finetuned.predict(image, conf=0.001, device=device, verbose=False)[0]
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for index, name in enumerate(names):
            mine = confidences[classes == index]
            row[f"score_{name}"] = float(mine.max()) if len(mine) else 0.0

        # COCO knows "person" but has no helmet class, so zero-shot the best
        # it can say is "there is a person here". Its score is its top
        # confidence for "person".
        result = coco.predict(image, conf=0.001, device=device, verbose=False)[0]
        coco_conf = result.boxes.conf.cpu().numpy()
        coco_cls = result.boxes.cls.cpu().numpy().astype(int)
        person = coco_cls == COCO_PERSON_ID
        row["score_coco_person"] = float(coco_conf[person].max()) if person.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Chance that a random violation outscores a random compliant item."""
    violation, compliant = y_score[y_true == 1], y_score[y_true == 0]
    if len(violation) == 0 or len(compliant) == 0:
        return None
    wins = (violation[:, None] > compliant[None, :]).sum() + 0.5 * (violation[:, None] == compliant[None, :]).sum()
    return float(wins / (len(violation) * len(compliant)))


def compare_with_baselines(y_true: np.ndarray, models: dict[str, np.ndarray], conf: float = 0.25) -> list[dict]:
    """Item-level comparison on the same test images, at a fixed threshold."""
    majority = int(y_true.mean() >= 0.5)
    candidates = {
        **models,
        "Trivial: predict nothing (pass every item)": np.zeros(len(y_true)),
        f"Trivial: majority class (flag {'every' if majority else 'no'} item)": np.full(len(y_true), float(majority)),
    }
    rows = []
    for name, y_score in candidates.items():
        flagged = y_score >= conf
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        tn = int((~flagged & (y_true == 0)).sum())
        rows.append({
            "model": name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": (tp + tn) / len(y_true),
            "roc_auc": roc_auc(y_true, y_score),
        })
    return rows


def save_json(name: str, payload) -> None:
    (ARTIFACTS / name).write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    device = pick_device()

    data_yaml = download_dataset()
    names = list(yaml.safe_load(data_yaml.read_text())["names"].values())
    save_json("dataset_stats.json", describe_dataset(names))

    weights = finetune(data_yaml, device)
    save_json("run.json", {
        "dataset": f"{decisions.ROBOFLOW_WORKSPACE}/{decisions.ROBOFLOW_PROJECT}/{decisions.ROBOFLOW_VERSION}",
        "base_model": BASE_MODEL,
        "settings": TRAIN_SETTINGS,
        "device": device,
        "machine": f"{platform.system()} {platform.machine()}",
        "smoke_test": SMOKE_TEST,
        "class_names": names,
    })
    save_json("box_metrics.json", evaluate_boxes(weights, data_yaml, device))

    scores = score_test_items(weights, names, device)
    scores.to_csv(ARTIFACTS / "test_scores.csv", index=False)

    if decisions.POSITIVE_CLASSES is None:
        print("\nPOSITIVE_CLASSES is not set in decisions.py: skipping the "
              "item-level comparison and the cost sweep.")
        return
    y_true, y_score = costing.item_level(scores, decisions.POSITIVE_CLASSES)
    save_json("baselines.json", compare_with_baselines(y_true, {
        "Finetuned yolo11n": y_score,
        "Zero-shot COCO yolo11n (person found)": scores["score_coco_person"].to_numpy(),
    }))

    missing = costing.missing_decisions()
    if missing:
        print(f"\nCosts not set in decisions.py ({', '.join(missing)}): skipping the cost sweep.")
        return
    sweep = costing.sweep(y_true, y_score)
    band = decisions.REVIEW_BAND or costing.cheapest_band(y_true, y_score)
    save_json("cost_sweep.json", {
        "sweep": sweep,
        "best_threshold": costing.best_threshold(sweep),
        "cheapest_band": costing.cheapest_band(y_true, y_score),
        "policies": costing.policy_comparison(y_true, y_score, band),
    })


if __name__ == "__main__":
    main()
