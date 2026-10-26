"""
Dataset description and detection metrics, computed from the YOLO label files.

Every model (finetuned, COCO zero-shot, predict-nothing) is scored by these same
functions on the same images and labels, so the comparison is like for like.
Matching follows the Ultralytics validator: a prediction is correct when it has
the right class and overlaps an unmatched true box by at least the IoU threshold.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)            # AP50 ... AP95, as in COCO
CONF_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)  # per-class precision and recall
CM_GRID = np.round(np.arange(0.05, 0.9501, 0.05), 2)   # object-level confusion matrices
CM_IOU = 0.45                                          # the IoU Ultralytics uses for its matrix
SMALL_AREA = 0.01   # a box covering under 1% of the image counts as small
LARGE_AREA = 0.10   # over 10% counts as large

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ----------------------------------------------------------------- labels --

def split_images(dataset_dir: Path, split: str) -> list[Path]:
    folder = Path(dataset_dir) / SPLIT_DIRS[split] / "images"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_labels(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Classes (n,) and boxes (n, 4) as normalised x1, y1, x2, y2."""
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    classes, boxes = [], []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            values = [float(v) for v in parts[1:]]
            if len(values) == 4:                      # x_centre y_centre width height
                cx, cy, w, h = values
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            else:                                     # polygon x1 y1 x2 y2 ... -> its box
                xs, ys = values[0::2], values[1::2]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
            classes.append(int(float(parts[0])))
    box_array = np.clip(np.array(boxes, dtype=float).reshape(-1, 4), 0.0, 1.0)
    return np.array(classes, dtype=int), box_array


def load_ground_truth(dataset_dir: Path, split: str) -> list[dict]:
    truth = []
    for path in split_images(dataset_dir, split):
        classes, boxes = read_labels(path)
        truth.append({"path": path, "cls": classes, "boxes": boxes})
    return truth


# ---------------------------------------------------- dataset description --

def box_areas(truth: list[dict], n_classes: int) -> list[np.ndarray]:
    """Area of every true box as a share of its image, grouped by class."""
    areas: list[list[float]] = [[] for _ in range(n_classes)]
    for item in truth:
        b = item["boxes"]
        for c, a in zip(item["cls"], (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])):
            if 0 <= c < n_classes:
                areas[c].append(float(a))
    return [np.array(a) for a in areas]


def describe_split(truth: list[dict], names: list[str], positive_ids: list[int]) -> dict:
    n = len(truth)
    positive_images = sum(bool(np.isin(t["cls"], positive_ids).any()) for t in truth)
    per_class = []
    for name, a in zip(names, box_areas(truth, len(names))):
        per_class.append({
            "class": name,
            "instances": int(len(a)),
            "median_area_pct": round(float(np.median(a)) * 100, 2) if len(a) else None,
            "small_pct": round(float((a < SMALL_AREA).mean()) * 100, 1) if len(a) else None,
            "large_pct": round(float((a > LARGE_AREA).mean()) * 100, 1) if len(a) else None,
        })
    return {
        "images": n,
        "images_without_objects": int(sum(len(t["cls"]) == 0 for t in truth)),
        "positive_images": int(positive_images),
        "positive_image_pct": round(100 * positive_images / n, 1) if n else None,
        "objects_per_image": round(sum(len(t["cls"]) for t in truth) / n, 2) if n else None,
        "per_class": per_class,
    }


# ------------------------------------------------------- detection metrics --

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def _greedy_pairs(iou: np.ndarray, threshold: float, strict: bool = False) -> np.ndarray:
    """(true index, predicted index) pairs, best overlap first, each used once."""
    rows, cols = np.nonzero(iou > threshold if strict else iou >= threshold)
    pairs = np.stack([rows, cols], axis=1)
    if len(pairs) > 1:
        pairs = pairs[np.argsort(-iou[rows, cols], kind="stable")]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs.reshape(-1, 2)


def match_predictions(det: dict, truth: dict) -> np.ndarray:
    """(n_predictions, 10) booleans: is each prediction correct at IoU 0.50 ... 0.95."""
    correct = np.zeros((len(det["cls"]), len(IOU_THRESHOLDS)), dtype=bool)
    if len(det["cls"]) == 0 or len(truth["cls"]) == 0:
        return correct
    same_class = truth["cls"][:, None] == det["cls"][None, :]
    iou = box_iou(truth["boxes"], det["boxes"]) * same_class
    for k, threshold in enumerate(IOU_THRESHOLDS):
        pairs = _greedy_pairs(iou, threshold)
        correct[pairs[:, 1], k] = True
    return correct


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve, 101-point interpolation (COCO)."""
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    p = np.flip(np.maximum.accumulate(np.flip(p)))
    x = np.linspace(0, 1, 101)
    return float(_trapezoid(np.interp(x, r, p), x))


def confusion_matrix(dets: list[dict], truth: list[dict], n_classes: int,
                     conf_threshold: float, iou_threshold: float = CM_IOU) -> np.ndarray:
    """Rows: true class, then background. Columns: predicted class, then background.

    The last column counts true objects the model missed; the last row counts
    predictions that matched nothing (false alarms on background).
    """
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for d, t in zip(dets, truth):
        keep = d["conf"] >= conf_threshold
        pred_cls, pred_boxes = d["cls"][keep], d["boxes"][keep]
        pairs = _greedy_pairs(box_iou(t["boxes"], pred_boxes), iou_threshold, strict=True)
        for ti, pi in pairs:
            m[t["cls"][ti], pred_cls[pi]] += 1
        for ti in set(range(len(t["cls"]))) - set(pairs[:, 0].tolist()):
            m[t["cls"][ti], n_classes] += 1
        for pi in set(range(len(pred_cls))) - set(pairs[:, 1].tolist()):
            m[n_classes, pred_cls[pi]] += 1
    return m


def evaluate_detections(dets: list[dict], truth: list[dict], names: list[str],
                        with_curves: bool = True) -> dict:
    """Per-class AP50 and AP50-95, plus precision and recall at every threshold."""
    nc = len(names)
    n_true = np.zeros(nc, dtype=int)
    for t in truth:
        for c in t["cls"]:
            if 0 <= c < nc:
                n_true[c] += 1

    correct = [match_predictions(d, t) for d, t in zip(dets, truth)]
    correct = np.concatenate(correct) if correct else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate([d["conf"] for d in dets]) if dets else np.zeros(0)
    cls = np.concatenate([d["cls"] for d in dets]).astype(int) if dets else np.zeros(0, int)

    per_class, precision_grid, recall_grid = [], [], []
    for c in range(nc):
        order = np.argsort(-conf[cls == c], kind="stable")
        tp = correct[cls == c][order]
        conf_c = conf[cls == c][order]
        if n_true[c] and len(tp):
            tp_cum, fp_cum = np.cumsum(tp, axis=0), np.cumsum(~tp, axis=0)
            recall, precision = tp_cum / n_true[c], tp_cum / (tp_cum + fp_cum)
            aps = [average_precision(recall[:, k], precision[:, k]) for k in range(10)]
        else:
            aps = [0.0 if n_true[c] else np.nan] * 10

        # precision and recall at each confidence threshold, IoU 0.5
        n_kept = np.searchsorted(-conf_c, -CONF_GRID, side="right")
        tp_at = np.concatenate(([0], np.cumsum(tp[:, 0])))[n_kept] if len(tp) else np.zeros(len(CONF_GRID))
        with np.errstate(divide="ignore", invalid="ignore"):
            precision_at = np.where(n_kept > 0, tp_at / np.maximum(n_kept, 1), np.nan)
            recall_at = tp_at / n_true[c] if n_true[c] else np.full(len(CONF_GRID), np.nan)
        precision_grid.append(precision_at)
        recall_grid.append(recall_at)
        per_class.append({"class": names[c], "instances": int(n_true[c]),
                          "ap50": aps[0], "ap50_95": float(np.mean(aps))})

    scored = [p for p in per_class if p["instances"] > 0]
    result = {
        "per_class": per_class,
        "map50": float(np.mean([p["ap50"] for p in scored])) if scored else np.nan,
        "map50_95": float(np.mean([p["ap50_95"] for p in scored])) if scored else np.nan,
    }
    if with_curves:
        result["pr_grid"] = {"thresholds": CONF_GRID, "precision": precision_grid,
                             "recall": recall_grid, "iou": 0.5}
        result["confusion"] = {"thresholds": CM_GRID, "iou": CM_IOU,
                               "labels": list(names) + ["background"],
                               "matrices": [confusion_matrix(dets, truth, nc, t) for t in CM_GRID]}
    return result


# ------------------------------------------------------------ image level --

def image_level(dets: list[dict], truth: list[dict], positive_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per image: label 1 if it holds a positive object, and the model's score for that.

    The score is the highest confidence among the image's positive-class boxes,
    0 if there are none. One threshold on this score is one decision rule.
    """
    labels = np.array([int(np.isin(t["cls"], positive_ids).any()) for t in truth], dtype=int)
    scores = np.array([float(d["conf"][np.isin(d["cls"], positive_ids)].max(initial=0.0))
                       for d in dets], dtype=float)
    return labels, scores


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Chance that a random positive image scores above a random negative one."""
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, group, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    ranks = np.empty(len(scores))
    ranks[order] = ((last - counts + 1 + last) / 2)[group]   # tied scores share their average rank
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ------------------------------------------------------------------ drawing --

POSITIVE_RGB = (214, 39, 40)
OTHER_RGB = (31, 119, 180)
TRUTH_RGB = (255, 255, 255)


def draw(image_rgb: np.ndarray, boxes: np.ndarray, classes: np.ndarray, confs: np.ndarray,
         names: list[str], positive_ids: list[int], min_conf: float,
         truth: dict | None = None) -> np.ndarray:
    """Predicted boxes at or above min_conf (red if positive class) and, optionally, true boxes in white."""
    img = np.ascontiguousarray(image_rgb.copy())
    h, w = img.shape[:2]
    width = max(1, round(max(h, w) / 350))
    scale = max(0.4, max(h, w) / 1500)

    def box(xyxy, color, text, thickness):
        x1, y1, x2, y2 = (np.asarray(xyxy) * [w, h, w, h]).round().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y = max(y1, th + 4)
        cv2.rectangle(img, (x1, y - th - 4), (x1 + tw + 4, y), color, -1)
        ink = (0, 0, 0) if color == TRUTH_RGB else (255, 255, 255)
        cv2.putText(img, text, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)

    if truth is not None:
        for c, xyxy in zip(truth["cls"], truth["boxes"]):
            box(xyxy, TRUTH_RGB, f"true {names[c]}", 1)
    for i in np.argsort(confs):
        if confs[i] >= min_conf:
            c = int(classes[i])
            color = POSITIVE_RGB if c in positive_ids else OTHER_RGB
            box(boxes[i], color, f"{names[c]} {confs[i]:.2f}", width + 1)
    return img
