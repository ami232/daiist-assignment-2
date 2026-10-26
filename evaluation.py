"""
Assignment 2: evaluation shared by train.py and app.py.

Every model is scored the same way on the same test split: predicted boxes are
matched to labelled tomatoes, and the matches become AP50, per-class precision
and recall, and a confusion matrix. Boxes are normalised xyxy (0 to 1), one row
per box, with an `image` column and an integer `cls` column. Predictions also
carry a `conf` column.

A prediction finds a tomato when their IoU is at least IOU_MATCH.
"""

import numpy as np
import pandas as pd

IOU_MATCH = 0.5
BOX = ["x1", "y1", "x2", "y2"]
BACKGROUND = "background"


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n, 4) and every box in b (m, 4)."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one (prediction, truth) pairs, highest IoU first, ignoring class."""
    candidates = np.argwhere(iou >= IOU_MATCH)
    order = np.argsort(-iou[candidates[:, 0], candidates[:, 1]], kind="stable")
    pairs, used_pred, used_true = [], set(), set()
    for p, t in candidates[order]:
        if p not in used_pred and t not in used_true:
            pairs.append((int(p), int(t)))
            used_pred.add(p)
            used_true.add(t)
    return pairs


def average_precision(preds: pd.DataFrame, truths: pd.DataFrame, cls: int) -> float:
    """AP at IoU 0.5 for one class, with COCO-style 101-point interpolation."""
    truths = truths[truths["cls"] == cls]
    preds = preds[preds["cls"] == cls]
    if truths.empty:
        return float("nan")
    if preds.empty:
        return 0.0

    truth_boxes = {image: group[BOX].to_numpy() for image, group in truths.groupby("image")}
    confs, hits = [], []
    for image, group in preds.groupby("image"):
        group = group.sort_values("conf", ascending=False)
        boxes = truth_boxes.get(image, np.empty((0, 4)))
        taken = np.zeros(len(boxes), dtype=bool)
        iou = box_iou(group[BOX].to_numpy(), boxes)
        for row in iou:
            free = np.where(taken, 0.0, row)
            best = int(free.argmax()) if len(free) else -1
            hit = best >= 0 and free[best] >= IOU_MATCH
            if hit:
                taken[best] = True
            hits.append(hit)
        confs.extend(group["conf"])

    order = np.argsort(-np.asarray(confs), kind="stable")
    hits = np.asarray(hits, dtype=float)[order]
    tp = np.cumsum(hits)
    recall = tp / len(truths)
    precision = np.maximum.accumulate((tp / np.arange(1, len(tp) + 1))[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def confusion_matrix(
    preds: pd.DataFrame, truths: pd.DataFrame, names: list[str], conf: float
) -> pd.DataFrame:
    """Counts at one confidence threshold. Rows are the true class, columns the prediction.

    The background row counts predictions that matched no tomato. The background
    column counts tomatoes that no prediction found.
    """
    k = len(names)
    counts = np.zeros((k + 1, k + 1), dtype=int)
    preds = preds[preds["conf"] >= conf]
    pred_groups = dict(tuple(preds.groupby("image")))
    true_groups = dict(tuple(truths.groupby("image")))
    for image in set(pred_groups) | set(true_groups):
        p = pred_groups.get(image, preds.iloc[:0])
        t = true_groups.get(image, truths.iloc[:0])
        pairs = greedy_match(box_iou(p[BOX].to_numpy(), t[BOX].to_numpy()))
        p_cls, t_cls = p["cls"].to_numpy(), t["cls"].to_numpy()
        for i, j in pairs:
            counts[t_cls[j], p_cls[i]] += 1
        matched_pred = {i for i, _ in pairs}
        matched_true = {j for _, j in pairs}
        for i in range(len(p)):
            if i not in matched_pred:
                counts[k, p_cls[i]] += 1
        for j in range(len(t)):
            if j not in matched_true:
                counts[t_cls[j], k] += 1

    labels = names + [BACKGROUND]
    return pd.DataFrame(
        counts,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )


def summarise(
    preds: pd.DataFrame, truths: pd.DataFrame, names: list[str], conf: float
) -> dict:
    """Per-class AP50, precision and recall, plus mAP50 and the confusion matrix."""
    matrix = confusion_matrix(preds, truths, names, conf)
    per_class = []
    for c, name in enumerate(names):
        found = int(matrix.iloc[c, c])
        predicted = int(matrix.iloc[:, c].sum())
        actual = int(matrix.iloc[c].sum())
        per_class.append({
            "class": name,
            "tomatoes": actual,
            "ap50": average_precision(preds, truths, c),
            "precision": found / predicted if predicted else None,
            "recall": found / actual if actual else None,
        })
    return {
        "conf": conf,
        "map50": float(np.nanmean([row["ap50"] for row in per_class])),
        "per_class": per_class,
        "confusion": matrix.values.tolist(),
    }
