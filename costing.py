"""
Cost model and box matching shared by train.py and app.py.

Everything here is pure arithmetic on saved model scores, so the dashboard can
recompute costs, confusion matrices and per-class numbers live without ever
touching the model or retraining it.

Decision rule (image level, positive class = "garment has a defect"):
    score = highest confidence of any defect box the model draws on the photo
    score <  low            -> pass
    low <= score < high     -> human review
    score >= high           -> reject (sent to the rework station)
Model alone is the special case low == high.
"""

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def threshold_grid(step: float) -> np.ndarray:
    return np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)


# ---------------------------------------------------------------- rates


def exceed(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Share of scores >= each threshold."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.zeros_like(np.asarray(thresholds, dtype=float))
    return (scores[None, :] >= np.asarray(thresholds, dtype=float)[:, None]).mean(axis=1)


def garment_exceed_from_crops(crop_scores: np.ndarray, thresholds: np.ndarray, area_factor: float) -> np.ndarray:
    """P(a good garment's photo scores >= t), from defect-free crops.

    A crop covers 1/area_factor of a photo, so a photo has area_factor
    independent chances to raise a false alarm.
    """
    q = exceed(crop_scores, thresholds)
    return 1.0 - (1.0 - q) ** area_factor


def outcome_rates(pos_scores, neg_crop_scores, low, high, area_factor) -> dict:
    """Pass / review / reject shares for defective and for good garments."""
    t = np.array([low, high], dtype=float)
    e_pos = exceed(pos_scores, t)
    e_neg = garment_exceed_from_crops(neg_crop_scores, t, area_factor)
    return {
        "pos": {"pass": 1 - e_pos[0], "review": e_pos[0] - e_pos[1], "reject": e_pos[1]},
        "neg": {"pass": 1 - e_neg[0], "review": e_neg[0] - e_neg[1], "reject": e_neg[1]},
    }


# ---------------------------------------------------------------- costs


def review_cost_per_case(cfg: dict) -> float:
    h = cfg["human"]
    return h["review_minutes"] / 60.0 * h["wage_eur_per_hour"]


def policy_cost(pos_pass, pos_review, pos_reject, neg_pass, neg_review, neg_reject, cfg: dict, base_rate=None):
    """Expected cost per garment of a pass/review/reject policy. Works on arrays."""
    c, h = cfg["costs_eur"], cfg["human"]
    p = cfg["business"]["base_rate"] if base_rate is None else base_rate
    hr, hs = h["review_recall"], h["specificity"]
    reviewed_defect = hr * c["true_positive"] + (1 - hr) * c["false_negative"]
    reviewed_good = (1 - hs) * c["false_positive"] + hs * c["true_negative"]
    pos = pos_reject * c["true_positive"] + pos_pass * c["false_negative"] + pos_review * reviewed_defect
    neg = neg_reject * c["false_positive"] + neg_pass * c["true_negative"] + neg_review * reviewed_good
    review_share = p * pos_review + (1 - p) * neg_review
    return p * pos + (1 - p) * neg + review_share * review_cost_per_case(cfg) + c["model_per_garment"]


def review_share(rates: dict, base_rate: float) -> float:
    return base_rate * rates["pos"]["review"] + (1 - base_rate) * rates["neg"]["review"]


def reviewers_needed(share: float, cfg: dict, minutes: float | None = None) -> float:
    h = cfg["human"]
    minutes = h["review_minutes"] if minutes is None else minutes
    return cfg["business"]["garments_per_day"] * share * minutes / h["productive_minutes_per_day"]


def single_sweep(pos_scores, neg_crop_scores, cfg: dict, thresholds=None, base_rate=None) -> dict:
    """Model alone: cost per garment at every threshold."""
    ev = cfg["evaluation"]
    t = threshold_grid(ev["threshold_step"]) if thresholds is None else thresholds
    tpr = exceed(pos_scores, t)
    fpr = garment_exceed_from_crops(neg_crop_scores, t, ev["negative_area_factor"])
    zero = np.zeros_like(t)
    cost = policy_cost(1 - tpr, zero, tpr, 1 - fpr, zero, fpr, cfg, base_rate)
    return {"thresholds": t, "tpr": tpr, "fpr": fpr, "cost": cost}


def two_threshold_grid(pos_scores, neg_crop_scores, cfg: dict, base_rate=None) -> dict:
    """Cost, review share and reviewers needed for every (low, high) pair with low <= high."""
    ev = cfg["evaluation"]
    p = cfg["business"]["base_rate"] if base_rate is None else base_rate
    t = threshold_grid(ev["threshold_step"])
    e_pos = exceed(pos_scores, t)
    e_neg = garment_exceed_from_crops(neg_crop_scores, t, ev["negative_area_factor"])
    lo_pos, hi_pos = e_pos[:, None], e_pos[None, :]
    lo_neg, hi_neg = e_neg[:, None], e_neg[None, :]
    cost = policy_cost(1 - lo_pos, lo_pos - hi_pos, hi_pos, 1 - lo_neg, lo_neg - hi_neg, hi_neg, cfg, p)
    share = p * (lo_pos - hi_pos) + (1 - p) * (lo_neg - hi_neg)
    valid = t[:, None] <= t[None, :]
    cost = np.where(valid, cost, np.inf)
    return {"thresholds": t, "cost": cost, "review_share": np.where(valid, share, np.nan),
            "reviewers": reviewers_needed(np.where(valid, share, np.nan), cfg)}


def best_two_thresholds(grid: dict, cfg: dict) -> dict:
    """Cheapest (low, high), with and without the reviewer-capacity limit."""
    t, cost = grid["thresholds"], grid["cost"]
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    unconstrained = {"low": float(t[i]), "high": float(t[j]), "cost": float(cost[i, j]),
                     "reviewers": float(grid["reviewers"][i, j])}
    capped = np.where(grid["reviewers"] <= cfg["human"]["reviewers_available"], cost, np.inf)
    i, j = np.unravel_index(np.argmin(capped), capped.shape)
    constrained = {"low": float(t[i]), "high": float(t[j]), "cost": float(capped[i, j]),
                   "reviewers": float(grid["reviewers"][i, j])}
    return {"unconstrained": unconstrained, "constrained": constrained}


def strategy_costs(pos_scores, neg_crop_scores, cfg: dict, single_t: float, low: float, high: float,
                   base_rate=None) -> dict:
    """Expected cost per 1,000 garments of each way to run inspection."""
    c, h = cfg["costs_eur"], cfg["human"]
    p = cfg["business"]["base_rate"] if base_rate is None else base_rate
    af = cfg["evaluation"]["negative_area_factor"]
    fr, hs = h["full_inspection_recall"], h["specificity"]

    nothing = p * c["false_negative"] + (1 - p) * c["true_negative"]
    manual = (h["full_inspection_minutes"] / 60 * h["wage_eur_per_hour"]
              + p * (fr * c["true_positive"] + (1 - fr) * c["false_negative"])
              + (1 - p) * ((1 - hs) * c["false_positive"] + hs * c["true_negative"]))
    always_reject = p * c["true_positive"] + (1 - p) * c["false_positive"]

    alone = outcome_rates(pos_scores, neg_crop_scores, single_t, single_t, af)
    both = outcome_rates(pos_scores, neg_crop_scores, low, high, af)

    def cost_of(r):
        return float(policy_cost(r["pos"]["pass"], r["pos"]["review"], r["pos"]["reject"],
                                 r["neg"]["pass"], r["neg"]["review"], r["neg"]["reject"], cfg, p))

    both_share = review_share(both, p)
    return {
        "Do nothing (ship everything)": {"cost_per_1000": 1000 * nothing, "review_share": 0.0,
                                         "people": 0.0},
        "Person checks every garment": {"cost_per_1000": 1000 * manual, "review_share": 1.0,
                                        "people": reviewers_needed(1.0, cfg, h["full_inspection_minutes"])},
        "Model alone": {"cost_per_1000": 1000 * cost_of(alone), "review_share": 0.0, "people": 0.0},
        "Model + human review": {"cost_per_1000": 1000 * cost_of(both), "review_share": float(both_share),
                                 "people": reviewers_needed(both_share, cfg)},
        "Always reject (reference)": {"cost_per_1000": 1000 * always_reject, "review_share": 0.0,
                                      "people": 0.0},
    }


def expected_confusion(rates: dict, base_rate: float, n: int = 1000) -> np.ndarray:
    """2x3 matrix of expected garments: rows (defective, good), cols (pass, review, reject)."""
    cols = ["pass", "review", "reject"]
    return np.array([[n * base_rate * rates["pos"][k] for k in cols],
                     [n * (1 - base_rate) * rates["neg"][k] for k in cols]])


# ---------------------------------------------------------------- box matching


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_pairs(iou: np.ndarray, thr: float):
    pairs, used_g, used_p = [], set(), set()
    for g, p in sorted(zip(*np.nonzero(iou >= thr)), key=lambda gp: -iou[gp]):
        if g not in used_g and p not in used_p:
            pairs.append((g, p))
            used_g.add(g)
            used_p.add(p)
    return pairs, used_g, used_p


def class_confusion(images: list, conf: float, iou_thr: float, n_classes: int) -> np.ndarray:
    """(n+1)x(n+1) box confusion matrix; rows = true class, cols = predicted, last = background.

    images: [{"gt": [[cls,x1,y1,x2,y2],...], "pred": [[cls,conf,x1,y1,x2,y2],...]}, ...]
    """
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    bg = n_classes
    for im in images:
        gt = np.array(im["gt"], dtype=float).reshape(-1, 5)
        pr = np.array(im["pred"], dtype=float).reshape(-1, 6)
        pr = pr[pr[:, 1] >= conf]
        pairs, used_g, used_p = greedy_pairs(box_iou(gt[:, 1:], pr[:, 2:]), iou_thr)
        for g, p in pairs:
            cm[int(gt[g, 0]), int(pr[p, 0])] += 1
        for g in range(len(gt)):
            if g not in used_g:
                cm[int(gt[g, 0]), bg] += 1
        for p in range(len(pr)):
            if p not in used_p:
                cm[bg, int(pr[p, 0])] += 1
    return cm


def per_class_pr(cm: np.ndarray) -> list[dict]:
    rows = []
    for c in range(cm.shape[0] - 1):
        tp, n_pred, n_true = cm[c, c], cm[:, c].sum(), cm[c, :].sum()
        rows.append({"tp": int(tp), "fp": int(n_pred - tp), "fn": int(n_true - tp), "instances": int(n_true),
                     "precision": float(tp / n_pred) if n_pred else float("nan"),
                     "recall": float(tp / n_true) if n_true else float("nan")})
    return rows


def agnostic_ap50(images: list, iou_thr: float = 0.5) -> float:
    """Class-agnostic AP at IoU 0.5 (101-point interpolation): did it box *a* defect, any class."""
    records, n_gt = [], 0
    for im in images:
        gt = np.array(im["gt"], dtype=float).reshape(-1, 5)
        pr = np.array(im["pred"], dtype=float).reshape(-1, 6)
        n_gt += len(gt)
        pr = pr[np.argsort(-pr[:, 1])]
        iou = box_iou(pr[:, 2:], gt[:, 1:])
        taken = set()
        for k in range(len(pr)):
            hit = False
            if iou.shape[1]:
                for g in np.argsort(-iou[k]):
                    if iou[k, g] < iou_thr:
                        break
                    if g not in taken:
                        taken.add(g)
                        hit = True
                        break
            records.append((pr[k, 1], hit))
    if not records or n_gt == 0:
        return 0.0
    records.sort(key=lambda r: -r[0])
    hits = np.array([r[1] for r in records], dtype=float)
    tp = np.cumsum(hits)
    recall = tp / n_gt
    precision = tp / np.arange(1, len(hits) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    return float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in points]))


def image_scores(images: list) -> np.ndarray:
    """Highest box confidence per image (0 if the model drew nothing)."""
    return np.array([max((p[1] for p in im["pred"]), default=0.0) for im in images])
