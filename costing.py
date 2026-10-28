"""
Cost and routing maths shared by train.py and app.py.

A case is one inspection window. The model gives it a score: the highest
confidence of any *defect* box in the image (0 if there is none). Two
thresholds route it:

    score <  lo         -> pass    (board moves on)
    lo <= score < hi    -> review  (a person looks at the window)
    score >= hi         -> fail    (board goes to rework)

With lo == hi there is no review band, and the model decides alone.

The test split has far more defects than the production line, so we do not
read costs off raw test counts. We take the *rates* the model achieves on
defective and on good windows separately, then weight them by the production
base rate. That keeps the error rates measured on real labels while making
the mix of cases realistic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import config


@dataclass
class Costs:
    fn: float = config.COST_FN
    fp: float = config.COST_FP
    tp: float = config.COST_TP
    tn: float = config.COST_TN
    review: float = config.COST_REVIEW
    human_sensitivity: float = config.HUMAN_SENSITIVITY
    human_false_fail: float = config.HUMAN_FALSE_FAIL
    base_rate: float = config.BASE_RATE

    def as_dict(self) -> dict:
        return asdict(self)


def route_counts(scores, labels, lo: float, hi: float) -> dict:
    """Counts of pass / review / fail, split by the true label."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    hi = max(hi, lo)
    passed = scores < lo
    failed = scores >= hi
    review = ~passed & ~failed
    out = {}
    for name, mask in (("pos", labels), ("neg", ~labels)):
        out[name] = {
            "pass": int((passed & mask).sum()),
            "review": int((review & mask).sum()),
            "fail": int((failed & mask).sum()),
            "n": int(mask.sum()),
        }
    return out


def _rates(counts: dict, side: str) -> tuple[float, float, float]:
    c = counts[side]
    n = max(c["n"], 1)
    return c["pass"] / n, c["review"] / n, c["fail"] / n


def cost_per_case(counts: dict, costs: Costs) -> dict:
    """Expected euros per window at the production base rate."""
    p = costs.base_rate
    s1, r1, f1 = _rates(counts, "pos")  # defective windows
    s0, r0, f0 = _rates(counts, "neg")  # good windows

    hs, hf = costs.human_sensitivity, costs.human_false_fail
    review_pos = costs.review + hs * costs.tp + (1 - hs) * costs.fn
    review_neg = costs.review + hf * costs.fp + (1 - hf) * costs.tn

    pos = s1 * costs.fn + r1 * review_pos + f1 * costs.tp
    neg = s0 * costs.tn + r0 * review_neg + f0 * costs.fp
    return {
        "cost": p * pos + (1 - p) * neg,
        "review_share": p * r1 + (1 - p) * r0,
        "fail_share": p * f1 + (1 - p) * f0,
        "escape_rate": p * (s1 + r1 * (1 - hs)),  # defects that reach the field side
        "recall_auto": f1,                          # defects the model fails by itself
        "false_fail_rate": f0,
    }


def sweep_single(scores, labels, costs: Costs, grid=None) -> list[dict]:
    """One threshold, no review band: cost at every threshold on the grid."""
    grid = np.round(np.linspace(0, 1, 101), 4) if grid is None else grid
    rows = []
    for t in grid:
        counts = route_counts(scores, labels, t, t)
        row = {"threshold": float(t), **cost_per_case(counts, costs)}
        row["tp"] = counts["pos"]["fail"]
        row["fn"] = counts["pos"]["pass"]
        row["fp"] = counts["neg"]["fail"]
        row["tn"] = counts["neg"]["pass"]
        rows.append(row)
    return rows


def best_band(scores, labels, costs: Costs, max_review_share: float | None = None,
              grid=None) -> dict:
    """Search every (lo, hi) pair with lo <= hi for the cheapest routing.

    max_review_share caps the share of cases sent to a person, which is how
    the staffing limit enters the decision.
    """
    grid = np.round(np.linspace(0, 1, 101), 4) if grid is None else grid
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            res = cost_per_case(route_counts(scores, labels, lo, hi), costs)
            if max_review_share is not None and res["review_share"] > max_review_share:
                continue
            if best is None or res["cost"] < best["cost"] - 1e-12:
                best = {"lo": float(lo), "hi": float(hi), **res}
    return best


def alternatives(costs: Costs) -> dict:
    """Policies that need no model, per window, at the production base rate."""
    p = costs.base_rate
    hs, hf = costs.human_sensitivity, costs.human_false_fail
    nothing = p * costs.fn + (1 - p) * costs.tn
    human_all = (
        costs.review
        + p * (hs * costs.tp + (1 - hs) * costs.fn)
        + (1 - p) * (hf * costs.fp + (1 - hf) * costs.tn)
    )
    fail_all = p * costs.tp + (1 - p) * costs.fp
    return {"do_nothing": nothing, "human_checks_all": human_all, "fail_everything": fail_all}


def staffing(review_share: float) -> dict:
    """Reviews per day and the people needed to do them."""
    windows = config.BOARDS_PER_DAY * config.WINDOWS_PER_BOARD
    reviews = windows * review_share
    hours = reviews * config.REVIEW_SECONDS / 3600
    return {
        "windows_per_day": windows,
        "reviews_per_day": reviews,
        "review_hours_per_day": hours,
        "reviewers_needed": hours / config.PRODUCTIVE_HOURS_PER_REVIEWER,
    }


def roc_auc(scores, labels) -> float:
    """Probability a random defective window outscores a random good one."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def iou(a, b) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), xyxy."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def detection_confusion(images: list[dict], n_classes: int, conf: float,
                        iou_thr: float = 0.5) -> np.ndarray:
    """Box-level confusion matrix at one confidence threshold.

    Rows are the true class, columns the predicted class; the last row and
    column are "background" (a missed box, or a box on nothing). Matching is
    class-agnostic and greedy by confidence, so a Bridge predicted as Good
    shows up as a confusion rather than as a miss plus a false alarm.
    """
    bg = n_classes
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for img in images:
        gt_boxes = np.asarray(img["gt_boxes"], dtype=float).reshape(-1, 4)
        gt_cls = list(img["gt_cls"])
        keep = [i for i, c in enumerate(img["pred_conf"]) if c >= conf]
        pb = np.asarray(img["pred_boxes"], dtype=float).reshape(-1, 4)[keep]
        pc = [img["pred_cls"][i] for i in keep]
        pconf = [img["pred_conf"][i] for i in keep]
        order = np.argsort(pconf)[::-1]
        ious = iou(pb, gt_boxes)
        used = set()
        for k in order:
            best_j, best_v = -1, iou_thr
            for j in range(len(gt_cls)):
                if j not in used and ious[k, j] >= best_v:
                    best_j, best_v = j, ious[k, j]
            if best_j >= 0:
                used.add(best_j)
                m[gt_cls[best_j], pc[k]] += 1
            else:
                m[bg, pc[k]] += 1
        for j, c in enumerate(gt_cls):
            if j not in used:
                m[c, bg] += 1
    return m


def per_class_pr(matrix: np.ndarray) -> list[dict]:
    """Precision and recall per class from a detection confusion matrix."""
    n = matrix.shape[0] - 1
    rows = []
    for c in range(n):
        tp = matrix[c, c]
        fn = matrix[c, :].sum() - tp
        fp = matrix[:, c].sum() - tp
        rows.append({
            "class_id": c,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        })
    return rows
