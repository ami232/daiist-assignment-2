"""
Turn saved test-set predictions into decisions and euros.

train.py and app.py both use this, so the dashboard and the report cost the
same predictions the same way. Nothing here touches a model: it works on the
boxes train.py saved to artifacts/predictions.json.

The unit of decision is one Team Points reading (a box that would be cropped,
OCR'd and published as a team's score). Each one lands in one of three bands:

    conf >= t_high          published automatically
    t_low <= conf < t_high  sent to an operator, who confirms or rejects it
    conf < t_low            discarded

A real Team Points box that ends up with no prediction above t_low is a false
negative: the feed keeps the last score, which only costs anything if the
score had just changed (that cost is folded into Costs.fn).

Frames are what an operator looks at, so review load is counted in frames.
The pipeline only asks about a band reading that would change the published
score. A wrong reading almost always would, so a frame with a wrong reading in
the band always reaches the operator. A correct one only does when the score
has just changed, which happens with probability p_change per reading.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Costs:
    fp: float
    fn: float
    review: float
    tp: float
    base_rate: float
    p_change: float


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N, 4) and every box in b (M, 4), xyxy."""
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


def _split(image: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt = np.asarray(image["gt"], dtype=float).reshape(-1, 5)
    pred = np.asarray(image["pred"], dtype=float).reshape(-1, 6)
    return gt[:, 0].astype(int), gt[:, 1:], pred[:, 0].astype(int), pred[:, 1], pred[:, 2:]


def match_class(image: dict, cls: int, iou_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy same-class matching, highest confidence first.

    Matching in confidence order means the result at any threshold t is just
    the subset of predictions with conf >= t, so one match at the lowest
    confidence serves every threshold we later sweep.

    Returns (pred_conf, pred_is_tp, gt_best_conf). gt_best_conf is the
    confidence of the prediction that claimed that box, 0 if none did.
    """
    gt_cls, gt_box, pr_cls, pr_conf, pr_box = _split(image)
    gt_box = gt_box[gt_cls == cls]
    keep = pr_cls == cls
    pr_conf, pr_box = pr_conf[keep], pr_box[keep]
    order = np.argsort(-pr_conf)
    pr_conf, pr_box = pr_conf[order], pr_box[order]

    ious = iou_matrix(pr_box, gt_box)
    is_tp = np.zeros(len(pr_conf), dtype=bool)
    gt_conf = np.zeros(len(gt_box))
    taken = np.zeros(len(gt_box), dtype=bool)
    for i in range(len(pr_conf)):
        if len(gt_box) == 0:
            break
        candidates = np.where(~taken & (ious[i] >= iou_thr))[0]
        if len(candidates):
            j = candidates[np.argmax(ious[i, candidates])]
            taken[j] = True
            is_tp[i] = True
            gt_conf[j] = pr_conf[i]
    return pr_conf, is_tp, gt_conf


@dataclass
class Cases:
    """Flat arrays over every Team Points prediction and ground-truth box in the test set."""

    pred_img: np.ndarray
    pred_conf: np.ndarray
    pred_tp: np.ndarray
    gt_img: np.ndarray
    gt_conf: np.ndarray
    img_pos: np.ndarray  # frame contains at least one real Team Points box

    @property
    def n_pos(self) -> int:
        return int(self.img_pos.sum())

    @property
    def n_neg(self) -> int:
        return int((~self.img_pos).sum())


def build_cases(images: list[dict], cls: int, iou_thr: float) -> Cases:
    pred_img, pred_conf, pred_tp, gt_img, gt_conf, img_pos = [], [], [], [], [], []
    for idx, image in enumerate(images):
        conf, tp, gconf = match_class(image, cls, iou_thr)
        pred_img += [idx] * len(conf)
        pred_conf += conf.tolist()
        pred_tp += tp.tolist()
        gt_img += [idx] * len(gconf)
        gt_conf += gconf.tolist()
        img_pos.append(len(gconf) > 0)
    return Cases(
        np.asarray(pred_img, dtype=int),
        np.asarray(pred_conf, dtype=float),
        np.asarray(pred_tp, dtype=bool),
        np.asarray(gt_img, dtype=int),
        np.asarray(gt_conf, dtype=float),
        np.asarray(img_pos, dtype=bool),
    )


def per_image_outcomes(cases: Cases, t_low: float, t_high: float, p_change: float) -> dict[str, np.ndarray]:
    """Count each outcome per test frame for one pair of thresholds.

    frames_review is an expectation: 1 for a frame with a wrong reading in the
    band, else the chance that at least one of its correct band readings is a
    score change.
    """
    n = len(cases.img_pos)
    auto = cases.pred_conf >= t_high
    band = (cases.pred_conf >= t_low) & ~auto

    def per_img(mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.bincount(idx[mask], minlength=n).astype(float)

    tp_review = per_img(band & cases.pred_tp, cases.pred_img)
    fp_review = per_img(band & ~cases.pred_tp, cases.pred_img)
    return {
        "tp_auto": per_img(auto & cases.pred_tp, cases.pred_img),
        "fp_auto": per_img(auto & ~cases.pred_tp, cases.pred_img),
        "tp_review": tp_review,
        "fp_review": fp_review,
        "fn": per_img(cases.gt_conf < t_low, cases.gt_img),
        "frames_review": np.where(fp_review > 0, 1.0, 1 - (1 - p_change) ** tp_review),
    }


def per_image_cost(outcomes: dict[str, np.ndarray], costs: Costs) -> np.ndarray:
    return (
        costs.fp * outcomes["fp_auto"]
        + costs.fn * outcomes["fn"]
        + costs.tp * outcomes["tp_auto"]
        + costs.review * outcomes["frames_review"]
    )


def per_1000(values: np.ndarray, img_pos: np.ndarray, base_rate: float) -> float:
    """Reweight a per-frame quantity from the test mix to the assumed production mix."""
    pos = values[img_pos].mean() if img_pos.any() else 0.0
    neg = values[~img_pos].mean() if (~img_pos).any() else 0.0
    return 1000 * (base_rate * pos + (1 - base_rate) * neg)


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """95% upper bound on a rate after k events in n trials. With k = 0 it is
    close to the rule of three, 3/n."""
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (centre + margin) / (1 + z * z / n)))


def evaluate(cases: Cases, costs: Costs, t_low: float, t_high: float) -> dict:
    """Raw test counts and production-weighted figures per 1,000 frames.

    cost_per_1000_upper replaces the observed false-positive rate among
    published readings with its 95% upper bound. Zero wrong scores in 30
    published readings does not mean the rate is zero, and at EUR 150 each the
    difference decides whether to publish at all.
    """
    t_low = min(t_low, t_high)
    out = per_image_outcomes(cases, t_low, t_high, costs.p_change)
    cost = per_image_cost(out, costs)
    result = {key: int(val.sum()) for key, val in out.items()}
    result["frames_review"] = round(float(out["frames_review"].sum()), 2)
    result["per_1000"] = {key: per_1000(val, cases.img_pos, costs.base_rate) for key, val in out.items()}
    result["cost_per_1000"] = per_1000(cost, cases.img_pos, costs.base_rate)
    result["review_share"] = result["per_1000"]["frames_review"] / 1000
    published = result["tp_auto"] + result["fp_auto"]
    published_per_1000 = result["per_1000"]["tp_auto"] + result["per_1000"]["fp_auto"]
    result["fp_rate_upper"] = wilson_upper(result["fp_auto"], published)
    extra_fp = result["fp_rate_upper"] * published_per_1000 - result["per_1000"]["fp_auto"]
    result["cost_per_1000_upper"] = result["cost_per_1000"] + costs.fp * max(extra_fp, 0.0)
    result["t_low"], result["t_high"] = t_low, t_high
    return result


def do_nothing(scout_eur_per_frame: float) -> dict:
    """Today: no model, one scout per game keys in every score change. Priced as the
    scout's time over the broadcast, whatever is on screen."""
    return {"cost_per_1000": 1000 * scout_eur_per_frame, "review_share": 0.0, "fp_auto": 0}


def predict_nothing(cases: Cases, costs: Costs) -> dict:
    """The trivial model baseline with no scout behind it: no reading is ever made,
    so every real score is a false negative."""
    fn = np.bincount(cases.gt_img, minlength=len(cases.img_pos)).astype(float)
    return {
        "cost_per_1000": per_1000(costs.fn * fn, cases.img_pos, costs.base_rate),
        "review_share": 0.0,
        "fp_auto": 0,
    }


def human_checks_everything(cases: Cases, costs: Costs, min_conf: float) -> dict:
    """Every frame goes to an operator. Nothing wrong is ever published; scores the
    model never proposed are still missed."""
    out = per_image_outcomes(cases, min_conf, np.inf, costs.p_change)
    out["frames_review"] = np.ones(len(cases.img_pos))
    cost = per_image_cost(out, costs)
    return {
        "cost_per_1000": per_1000(cost, cases.img_pos, costs.base_rate),
        "review_share": 1.0,
        "fp_auto": 0,
    }


# Up to and including 1.00: an upper threshold of 1.00 means never publish automatically.
THRESHOLD_GRID = np.round(np.arange(0.01, 1.001, 0.01), 2)


def sweep_single(cases: Cases, costs: Costs, grid: np.ndarray = THRESHOLD_GRID) -> list[dict]:
    """The model decides alone: one threshold, no review band."""
    return [evaluate(cases, costs, t, t) for t in grid]


def pick_single(sweep: list[dict], key: str = "cost_per_1000", tolerance: float = 0.005) -> dict:
    """The cost curve on a small test set has flat stretches. Take the middle of
    the cheapest stretch rather than its arbitrary first point."""
    costs = np.array([row[key] for row in sweep])
    near = np.where(costs <= costs.min() * (1 + tolerance) + 1e-9)[0]
    return sweep[int(near[len(near) // 2])]


def sweep_band(cases: Cases, costs: Costs, grid: np.ndarray = THRESHOLD_GRID) -> list[dict]:
    rows = []
    for i, t_low in enumerate(grid):
        for t_high in grid[i:]:
            r = evaluate(cases, costs, t_low, t_high)
            rows.append({"t_low": t_low, "t_high": t_high, "cost_per_1000": r["cost_per_1000"],
                         "cost_per_1000_upper": r["cost_per_1000_upper"], "review_share": r["review_share"]})
    return rows


def pick_band(rows: list[dict], key: str = "cost_per_1000", tolerance: float = 0.005) -> dict:
    """Cheapest pair. Among near-ties: the fewest frames sent to a person, then
    the most conservative auto-publish threshold, then the middle of the flat
    stretch of lower thresholds, as pick_single does."""
    best = min(r[key] for r in rows)
    near = [r for r in rows if r[key] <= best * (1 + tolerance) + 1e-9]
    share = min(round(r["review_share"], 4) for r in near)
    near = [r for r in near if round(r["review_share"], 4) == share]
    t_high = max(r["t_high"] for r in near)
    near = sorted((r for r in near if r["t_high"] == t_high), key=lambda r: r["t_low"])
    return near[len(near) // 2]


def bootstrap_cost(cases: Cases, costs: Costs, t_low: float, t_high: float,
                   n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval on cost per 1,000 frames, resampling test frames within
    positive and negative groups so the base-rate weighting stays meaningful."""
    rng = np.random.default_rng(seed)
    cost = per_image_cost(per_image_outcomes(cases, t_low, t_high, costs.p_change), costs)
    pos, neg = cost[cases.img_pos], cost[~cases.img_pos]
    draws = []
    for _ in range(n_boot):
        p = rng.choice(pos, len(pos)).mean() if len(pos) else 0.0
        q = rng.choice(neg, len(neg)).mean() if len(neg) else 0.0
        draws.append(1000 * (costs.base_rate * p + (1 - costs.base_rate) * q))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def confusion_matrix(images: list[dict], n_classes: int, conf: float, iou_thr: float) -> np.ndarray:
    """(n+1) x (n+1), rows = actual, columns = predicted, last index = background.

    Matches across classes so a Team Points box predicted on top of a Shot Clock
    shows up as that confusion, not just as a miss and a stray box.
    """
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    bg = n_classes
    for image in images:
        gt_cls, gt_box, pr_cls, pr_conf, pr_box = _split(image)
        keep = pr_conf >= conf
        pr_cls, pr_box = pr_cls[keep], pr_box[keep]
        ious = iou_matrix(gt_box, pr_box)
        pairs = np.argwhere(ious >= iou_thr)
        pairs = pairs[np.argsort(-ious[pairs[:, 0], pairs[:, 1]])] if len(pairs) else pairs
        used_gt, used_pr = set(), set()
        for g, p in pairs:
            if g in used_gt or p in used_pr:
                continue
            used_gt.add(g)
            used_pr.add(p)
            cm[gt_cls[g], pr_cls[p]] += 1
        for g in range(len(gt_cls)):
            if g not in used_gt:
                cm[gt_cls[g], bg] += 1
        for p in range(len(pr_cls)):
            if p not in used_pr:
                cm[bg, pr_cls[p]] += 1
    return cm


def per_class_pr(images: list[dict], n_classes: int, conf: float, iou_thr: float) -> list[dict]:
    """Precision and recall per class at one confidence threshold."""
    rows = []
    for cls in range(n_classes):
        tp = fp = n_gt = 0
        for image in images:
            pc, is_tp, gconf = match_class(image, cls, iou_thr)
            keep = pc >= conf
            tp += int(is_tp[keep].sum())
            fp += int((~is_tp[keep]).sum())
            n_gt += len(gconf)
        rows.append({
            "class_id": cls,
            "instances": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": n_gt - tp,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / n_gt if n_gt else float("nan"),
        })
    return rows
