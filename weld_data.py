"""
Data helpers shared by dataset_stats.py and train.py.

- downloads the public Roboflow dataset (never forked) into datasets/
- reads YOLO labels
- turns a model's predictions into WELD-LEVEL CASES, which is what the cost
  analysis works on

app.py does not import this file: it only needs the saved results.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent

# ---- The dataset (Roboflow Universe, public, used as it is) -----------------
WORKSPACE = "workspace-4nvpa"
PROJECT = "welding_defects-vwoj6"
VERSION = 1
EXPORT_FORMAT = "yolov11"
DATA_DIR = ROOT / "datasets" / f"{PROJECT}_v{VERSION}"

# ---- Which labelled classes are the POSITIVE class (a defective weld) --------
# Names are compared after lower-casing and dropping spaces, so "Bad Weld",
# "bad_weld" and "BadWeld" all match. Change this if the dataset stats show
# that "Defect" boxes are something other than a bad weld.
POSITIVE_CLASSES = {"badweld", "defect"}
NEGATIVE_CLASSES = {"goodweld"}

IMGSZ = int(os.environ.get("IMGSZ", "512"))   # the dataset is stretched to 512x512
MIN_CONF = 0.05      # the lowest score the pipeline keeps as a detection
MATCH_IOU = 0.5      # a detection "is" a labelled weld if they overlap this much

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def is_positive_name(name: str) -> bool:
    return norm_name(name) in POSITIVE_CLASSES


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    """Download the dataset once. Later runs reuse the copy in datasets/."""
    if (DATA_DIR / "data.yaml").is_file():
        print(f"Dataset already downloaded: {DATA_DIR}")
        return DATA_DIR

    from roboflow import Roboflow

    from api_key import load_api_key

    print(f"Downloading {WORKSPACE}/{PROJECT} version {VERSION} ({EXPORT_FORMAT}) ...")
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download(EXPORT_FORMAT, location=str(DATA_DIR))
    return DATA_DIR


def split_dirs(dataset_dir: Path) -> dict[str, tuple[Path, Path]]:
    """{'train': (images, labels), 'val': ..., 'test': ...} for the splits that exist."""
    out = {}
    for split, candidates in {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}.items():
        for name in candidates:
            images = dataset_dir / name / "images"
            if images.is_dir():
                out[split] = (images, dataset_dir / name / "labels")
                break
    return out


def read_class_names(dataset_dir: Path) -> list[str]:
    with (dataset_dir / "data.yaml").open() as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def prepare_data_yaml(dataset_dir: Path) -> tuple[Path, list[str]]:
    """Write a data yaml with absolute paths, so training works from any folder."""
    splits = split_dirs(dataset_dir)
    names = read_class_names(dataset_dir)
    cfg = {
        "path": str(dataset_dir.resolve()),
        "train": str(splits["train"][0].relative_to(dataset_dir)),
        "val": str(splits["val"][0].relative_to(dataset_dir)),
        "names": {i: n for i, n in enumerate(names)},
    }
    if "test" in splits:
        cfg["test"] = str(splits["test"][0].relative_to(dataset_dir))
    out = dataset_dir / "data_abs.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out, names


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


# ---------------------------------------------------------------------------
# Labels and boxes
# ---------------------------------------------------------------------------

def read_gt(label_path: Path) -> np.ndarray:
    """Labelled boxes of one image as rows [class, x1, y1, x2, y2], normalised 0-1."""
    if not label_path.is_file():
        return np.zeros((0, 5))
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:                       # x_center y_center w h
            xc, yc, w, h = vals
            rows.append([cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
        else:                                    # polygon: use its bounding box
            xs, ys = vals[0::2], vals[1::2]
            rows.append([cls, min(xs), min(ys), max(xs), max(ys)])
    return np.array(rows, dtype=float).reshape(-1, 5)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n,4) and every box in b (m,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def count_instances(labels_dir: Path, class_names: list[str]) -> dict[str, int]:
    counts = {n: 0 for n in class_names}
    if labels_dir.is_dir():
        for f in labels_dir.glob("*.txt"):
            for row in read_gt(f):
                counts[class_names[int(row[0])]] += 1
    return counts


# ---------------------------------------------------------------------------
# Predictions -> weld-level cases
# ---------------------------------------------------------------------------

def collect_cases(model, images_dir: Path, labels_dir: Path, gt_names: list[str],
                  is_positive_pred=is_positive_name, chunk: int = 32) -> pd.DataFrame:
    """One row per labelled weld, plus one row per spurious detection.

    DEFECT SCORE of a labelled weld = the highest confidence among the model's
    positive-class boxes (here: Bad Weld or Defect) that overlap that weld by
    IoU >= 0.5. If no such box exists the score is 0: the model did not flag it.

    A positive-class box that overlaps NO labelled weld is "spurious": a false
    alarm on the background. It is kept as its own row because it still costs a
    pull when it is above the threshold.

    is_positive_pred decides which of the MODEL's class names count as "the model
    says defective". For the COCO baseline we pass `lambda name: True`: any COCO
    box counts as a flag, which is the most generous reading possible.
    """
    rows = []
    images = list_images(images_dir)
    for start in range(0, len(images), chunk):
        batch = images[start:start + chunk]
        results = model.predict([str(p) for p in batch], conf=MIN_CONF, iou=0.6,
                                imgsz=IMGSZ, verbose=False)
        for path, r in zip(batch, results):
            gt = read_gt(labels_dir / f"{path.stem}.txt")
            boxes = r.boxes.xyxyn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            keep = np.array([is_positive_pred(r.names[c]) for c in classes], dtype=bool)
            pos_boxes, pos_confs = boxes[keep], confs[keep]

            iou = box_iou(gt[:, 1:5], pos_boxes)          # (labelled welds, positive boxes)
            for i, g in enumerate(gt):
                label = gt_names[int(g[0])]
                hit = iou[i] >= MATCH_IOU if iou.shape[1] else np.zeros(0, dtype=bool)
                score = float(pos_confs[hit].max()) if hit.any() else 0.0
                rows.append({"image": path.name, "kind": "weld", "label": label,
                             "is_positive": int(is_positive_name(label)), "score": score})

            if len(pos_boxes):
                best_overlap = iou.max(axis=0) if len(gt) else np.zeros(len(pos_boxes))
                for conf in pos_confs[best_overlap < MATCH_IOU]:
                    rows.append({"image": path.name, "kind": "spurious", "label": "spurious",
                                 "is_positive": 0, "score": float(conf)})
    return pd.DataFrame(rows, columns=["image", "kind", "label", "is_positive", "score"])
