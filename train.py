import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key
from costing import best_band, load_costs, reference_policies, sweep_single_threshold

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
WEIGHTS_DIR = ROOT / "weights"
OUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"

WORKSPACE = "pothole-ztskn"
PROJECT = "newdataset-pmxsv"
VERSION = 1

BASE_MODEL = "yolo11n.pt"
COCO_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 15))
FRACTION = float(os.environ.get("FRACTION", 1.0))
IMGSZ = int(os.environ.get("IMGSZ", 512))
BATCH = int(os.environ.get("BATCH", 8))
PATIENCE = 5
SEED = 0
CONFUSION_CONF = 0.25

SPLITS = ("train", "valid", "test")


def download_dataset() -> Path:
    location = DATA_DIR / f"{PROJECT}-{VERSION}"
    if not (location / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov11", location=str(location))
    config = yaml.safe_load((location / "data.yaml").read_text())
    fixed = {
        "path": str(location),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": config["nc"],
        "names": config["names"],
    }
    data_yaml = location / "data_local.yaml"
    data_yaml.write_text(yaml.safe_dump(fixed))
    return data_yaml


def read_labels(label_file: Path) -> list[tuple[int, float]]:
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        values = list(map(float, parts[1:]))
        if len(values) == 4:
            area = values[2] * values[3]
        else:
            xs, ys = values[0::2], values[1::2]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        boxes.append((int(parts[0]), area))
    return boxes


def describe_dataset(data_yaml: Path) -> dict:
    config = yaml.safe_load(data_yaml.read_text())
    root, names = Path(config["path"]), config["names"]
    stats = {"names": names, "splits": {}, "duplicates_across_splits": 0}
    hashes = defaultdict(set)
    for split in SPLITS:
        instances, images_with, areas = Counter(), Counter(), defaultdict(list)
        image_files = sorted((root / split / "images").glob("*"))
        background = 0
        for image in image_files:
            hashes[hashlib.md5(image.read_bytes()).hexdigest()].add(split)
            boxes = read_labels(root / split / "labels" / f"{image.stem}.txt")
            if not boxes:
                background += 1
            for cls, area in boxes:
                instances[names[cls]] += 1
                areas[names[cls]].append(area)
            for cls in {cls for cls, _ in boxes}:
                images_with[names[cls]] += 1
        stats["splits"][split] = {
            "images": len(image_files),
            "background_images": background,
            "per_class": {
                name: {
                    "instances": instances[name],
                    "images": images_with[name],
                    "median_box_area_pct": round(100 * float(np.median(areas[name])), 2) if areas[name] else None,
                    "p10_box_area_pct": round(100 * float(np.percentile(areas[name], 10)), 2) if areas[name] else None,
                    "p90_box_area_pct": round(100 * float(np.percentile(areas[name], 90)), 2) if areas[name] else None,
                }
                for name in names
            },
        }
    stats["duplicates_across_splits"] = sum(1 for s in hashes.values() if len(s) > 1)
    return stats


def resumable_checkpoint() -> Path | None:
    last = RUNS_DIR / "finetune" / "weights" / "last.pt"
    if not last.exists():
        return None
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return last if checkpoint.get("epoch", -1) >= 0 else None


def train(data_yaml: Path) -> Path:
    checkpoint = resumable_checkpoint()
    if checkpoint:
        YOLO(str(checkpoint)).train(resume=True)
    else:
        YOLO(BASE_MODEL).train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            seed=SEED,
            fraction=FRACTION,
            project=str(RUNS_DIR),
            name="finetune",
            exist_ok=True,
        )
    WEIGHTS_DIR.mkdir(exist_ok=True)
    best = WEIGHTS_DIR / "best.pt"
    shutil.copy(RUNS_DIR / "finetune" / "weights" / "best.pt", best)
    shutil.copy(RUNS_DIR / "finetune" / "results.csv", OUT_DIR / "training_curve.csv")
    return best


def evaluate_detection(weights: Path, data_yaml: Path) -> dict:
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(RUNS_DIR),
        name="test_eval",
        exist_ok=True,
    )
    names = metrics.names
    per_class = []
    for i, cls in enumerate(metrics.box.ap_class_index):
        per_class.append({
            "class": names[int(cls)],
            "precision": round(float(metrics.box.p[i]), 4),
            "recall": round(float(metrics.box.r[i]), 4),
            "AP50": round(float(metrics.box.ap50[i]), 4),
            "AP50-95": round(float(metrics.box.ap[i]), 4),
        })
    matrix = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        conf=CONFUSION_CONF,
        project=str(RUNS_DIR),
        name="test_confusion",
        exist_ok=True,
    ).confusion_matrix.matrix
    labels = [names[i] for i in range(len(names))] + ["background"]
    return {
        "overall": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "rows_predicted": labels,
            "cols_actual": labels,
            "conf": CONFUSION_CONF,
            "values": matrix.astype(int).tolist(),
        },
    }


def test_images(data_yaml: Path) -> tuple[list[Path], dict]:
    config = yaml.safe_load(data_yaml.read_text())
    root = Path(config["path"]) / "test"
    images = sorted((root / "images").glob("*"))
    labels = {
        image.name: sorted({config["names"][cls] for cls, _ in read_labels(root / "labels" / f"{image.stem}.txt")})
        for image in images
    }
    return images, labels


def score_images(weights: Path, images: list[Path], labels: dict, positive: str) -> pd.DataFrame:
    model = YOLO(str(weights))
    names = model.names
    rows = []
    for image in images:
        result = model.predict(str(image), conf=0.001, imgsz=IMGSZ, verbose=False)[0]
        scores = defaultdict(float)
        for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            scores[names[int(cls)]] = max(scores[names[int(cls)]], conf)
        rows.append({
            "image": image.name,
            "labels": "|".join(labels[image.name]),
            "is_positive": positive in labels[image.name],
            "score": scores.get(positive, 0.0),
            **{f"score_{name}": scores.get(name, 0.0) for name in names.values()},
        })
    return pd.DataFrame(rows)


def coco_zero_shot(images: list[Path], labels: dict, positive: str, dataset_names: list[str]) -> dict:
    model = YOLO(COCO_MODEL)
    coco_names = {name.lower() for name in model.names.values()}
    fired = Counter()
    for image in images:
        if positive in labels[image.name]:
            result = model.predict(str(image), conf=0.25, imgsz=IMGSZ, verbose=False)[0]
            fired.update(model.names[int(c)] for c in result.boxes.cls.tolist())
    return {
        "dataset_classes_in_coco": sorted(n for n in dataset_names if n.lower() in coco_names),
        "coco_detections_on_pothole_images": dict(fired.most_common(10)),
    }


def image_level_metrics(scores: np.ndarray, is_positive: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    tp = int((flagged & is_positive).sum())
    fp = int((flagged & ~is_positive).sum())
    fn = int((~flagged & is_positive).sum())
    tn = int((~flagged & ~is_positive).sum())
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "accuracy": round((tp + tn) / len(scores), 4),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    costs = load_costs()
    positive = costs["positive_class"]

    data_yaml = download_dataset()
    dataset_stats = describe_dataset(data_yaml)
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2))

    weights = train(data_yaml)
    detection = evaluate_detection(weights, data_yaml)

    images, labels = test_images(data_yaml)
    image_scores = score_images(weights, images, labels, positive)
    image_scores.to_csv(OUT_DIR / "image_scores.csv", index=False)
    scores = image_scores["score"].to_numpy()
    is_positive = image_scores["is_positive"].to_numpy()

    zero = np.zeros_like(scores)
    coco = coco_zero_shot(images, labels, positive, dataset_stats["names"])
    names = [c["class"] for c in detection["per_class"]]
    baselines = {
        "finetuned": {"detection": detection["overall"], "image_level": image_level_metrics(scores, is_positive, 0.25)},
        "coco_zero_shot": {
            "detection": {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0},
            "per_class_AP50": {name: 0.0 for name in names},
            "image_level": image_level_metrics(zero, is_positive, 0.25),
            **coco,
        },
        "predict_nothing": {
            "detection": {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0},
            "image_level": image_level_metrics(zero, is_positive, 0.25),
        },
        "majority_class": {
            "image_level": image_level_metrics(
                np.ones_like(scores) if is_positive.mean() >= 0.5 else zero, is_positive, 0.25
            ),
        },
    }

    sweep = sweep_single_threshold(scores, is_positive, costs)
    sweep.to_csv(OUT_DIR / "cost_sweep.csv", index=False)
    chosen = sweep.loc[sweep["cost_per_1000"].idxmin()]
    band = best_band(scores, is_positive, costs)
    operating_point = {
        "costs": costs,
        "single_threshold": float(chosen["threshold"]),
        "single_threshold_cost_per_1000": float(chosen["cost_per_1000"]),
        "image_level_at_threshold": image_level_metrics(scores, is_positive, float(chosen["threshold"])),
        "band_low": band["low"],
        "band_high": band["high"],
        "band_cost_per_1000": band["cost_per_1000"],
        "band_review_share": band["review_share"],
        "band_reviewers_needed": band["reviewers_needed"],
        "policies": reference_policies(scores, is_positive, float(chosen["threshold"]), costs).to_dict("records"),
    }

    (OUT_DIR / "metrics.json").write_text(json.dumps({"detection": detection, "baselines": baselines}, indent=2))
    (OUT_DIR / "operating_point.json").write_text(json.dumps(operating_point, indent=2, default=float))
    print(json.dumps({"detection": detection["overall"], "operating_point": {k: v for k, v in operating_point.items() if k != "policies"}}, indent=2, default=float))


if __name__ == "__main__":
    main()
