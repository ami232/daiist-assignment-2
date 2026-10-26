"""
Metrics and cost logic shared by train.py and app.py.

Both stages call the same functions, so the dashboard recomputes exactly the
numbers the training run reported, just at whatever thresholds you pick.

A "record" is one evaluated image:

    {"image": "name.jpg", "true": 3, "probs": [0.01, 0.02, ...]}

Decision rule for one image: score = probability the piece is rotten, i.e. the
summed probability of every rotten class. Then
    score <  lo        -> pass (ship it)
    lo <= score < hi   -> human review at the grading table
    score >= hi        -> fail (discard it)
A single-threshold policy is the special case lo == hi.
"""

from dataclasses import dataclass

import numpy as np

OUTCOMES = ("pass", "review", "fail")


# --- Class-level metrics --------------------------------------------------------

def class_confusion(true: np.ndarray, pred: np.ndarray, nc: int) -> np.ndarray:
    """Confusion matrix indexed [true, predicted]."""
    matrix = np.zeros((nc, nc), dtype=int)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def classification_metrics(true: np.ndarray, pred: np.ndarray, names: dict[int, str]) -> dict:
    """Accuracy, and precision / recall / F1 per class and macro-averaged."""
    matrix = class_confusion(true, pred, len(names))
    per_class = {}
    for c, name in names.items():
        tp = matrix[c, c]
        support, predicted = matrix[c].sum(), matrix[:, c].sum()
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        per_class[name] = {"support": int(support), "precision": float(p), "recall": float(r),
                           "f1": float(2 * p * r / (p + r)) if p + r else 0.0}
    present = [v for v in per_class.values() if v["support"] > 0]
    macro = {k: float(np.mean([v[k] for v in present])) if present else 0.0
             for k in ("precision", "recall", "f1")}
    return {"accuracy": float((true == pred).mean()) if len(true) else 0.0,
            "macro": macro, "per_class": per_class, "confusion": matrix.tolist()}


def records_to_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    true = np.array([r["true"] for r in records], dtype=int)
    probs = np.array([r["probs"] for r in records], dtype=float)
    return true, probs


# --- Binary decision: rotten or not ---------------------------------------------

def labels_and_scores(records: list[dict], rotten_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """y = piece is truly rotten; s = model's probability that it is."""
    true, probs = records_to_arrays(records)
    y = np.isin(true, rotten_ids)
    s = probs[:, rotten_ids].sum(1) if len(probs) else np.zeros(0)
    return y, np.clip(s, 0, 1)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Probability a random rotten piece outscores a random fresh one."""
    pos, neg = s[y], np.sort(s[~y])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    return float((below + 0.5 * ties).sum() / (len(pos) * len(neg)))


@dataclass
class Costs:
    fn: float
    fp: float
    tp: float
    tn: float
    review: float
    reviewer_sensitivity: float
    reviewer_specificity: float

    @classmethod
    def from_config(cls, config, **overrides) -> "Costs":
        values = dict(
            fn=config.COST_FN, fp=config.COST_FP, tp=config.COST_TP, tn=config.COST_TN,
            review=config.COST_REVIEW,
            reviewer_sensitivity=config.REVIEWER_SENSITIVITY,
            reviewer_specificity=config.REVIEWER_SPECIFICITY,
        )
        values.update(overrides)
        return cls(**values)


def outcome_masks(s: np.ndarray, lo: float, hi: float) -> dict[str, np.ndarray]:
    passed, failed = s < lo, s >= hi
    return {"pass": passed, "review": ~passed & ~failed, "fail": failed}


def band_rates(y: np.ndarray, s: np.ndarray, lo: float, hi: float) -> dict:
    """Share of rotten ("pos") and of fresh ("neg") pieces in each outcome."""
    masks = outcome_masks(s, lo, hi)
    rates = {}
    for label, group in (("pos", y), ("neg", ~y)):
        n = max(int(group.sum()), 1)
        rates[label] = {o: float((masks[o] & group).sum() / n) for o in OUTCOMES}
    return rates


def cost_per_case(rates: dict, base_rate: float, costs: Costs) -> float:
    """Expected euros per piece at the production base rate.

    Rates come from the evaluation split but are re-weighted by the assumed
    production base rate, because the dataset's own share of rotten fruit is
    not what a real line sees.
    """
    pos, neg = rates["pos"], rates["neg"]
    sens, spec = costs.reviewer_sensitivity, costs.reviewer_specificity
    on_pos = (pos["pass"] * costs.fn + pos["fail"] * costs.tp
              + pos["review"] * (costs.review + sens * costs.tp + (1 - sens) * costs.fn))
    on_neg = (neg["pass"] * costs.tn + neg["fail"] * costs.fp
              + neg["review"] * (costs.review + (1 - spec) * costs.fp + spec * costs.tn))
    return base_rate * on_pos + (1 - base_rate) * on_neg


def review_share(rates: dict, base_rate: float) -> float:
    return base_rate * rates["pos"]["review"] + (1 - base_rate) * rates["neg"]["review"]


def outcome_counts(rates: dict, base_rate: float, per: int = 1000) -> dict:
    """Expected pieces per `per` in each (actual class, outcome) cell."""
    return {
        "pos": {k: base_rate * per * v for k, v in rates["pos"].items()},
        "neg": {k: (1 - base_rate) * per * v for k, v in rates["neg"].items()},
    }


def per_class_outcomes(true: np.ndarray, s: np.ndarray, lo: float, hi: float, names: dict[int, str]) -> dict:
    """For each true class, the share sent to pass / review / fail."""
    masks = outcome_masks(s, lo, hi)
    out = {}
    for c, name in names.items():
        group = true == c
        n = max(int(group.sum()), 1)
        out[name] = {"n": int(group.sum()), **{o: float((masks[o] & group).sum() / n) for o in OUTCOMES}}
    return out


def thresholds(step: float) -> np.ndarray:
    return np.round(np.arange(0, 1 + step / 2, step), 4)


def sweep_single(y, s, base_rate, costs: Costs, grid) -> list[dict]:
    """Cost per 1,000 pieces for the model deciding alone at each threshold."""
    rows = []
    for t in grid:
        rates = band_rates(y, s, t, t)
        rows.append({
            "threshold": float(t),
            "cost_per_1000": 1000 * cost_per_case(rates, base_rate, costs),
            "tpr": rates["pos"]["fail"],
            "fpr": rates["neg"]["fail"],
        })
    return rows


def optimise_band(y, s, base_rate, costs: Costs, grid, max_review_share: float = 1.0) -> dict:
    """Search every lo <= hi pair; keep the cheapest within review capacity."""
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            rates = band_rates(y, s, lo, hi)
            share = review_share(rates, base_rate)
            if share > max_review_share + 1e-12:
                continue
            cost = cost_per_case(rates, base_rate, costs)
            if best is None or cost < best["cost_per_case"] - 1e-12:
                best = {"lo": float(lo), "hi": float(hi), "cost_per_case": cost,
                        "review_share": share}
    return best


def status_quo_from_config(config) -> dict:
    """Today's process: graders glancing at every piece on the belt."""
    labour = (config.STATUS_QUO_GRADERS * config.SHIFT_HOURS * config.REVIEWER_HOURLY_COST
              / config.DAILY_VOLUME)
    return {"labour_per_piece": labour, "sensitivity": config.STATUS_QUO_SENSITIVITY,
            "specificity": config.STATUS_QUO_SPECIFICITY}


def policy_costs(y, s, base_rate, costs: Costs, single_t: float, lo: float, hi: float,
                 status_quo: dict | None = None) -> dict:
    """The policies the brief asks to compare, plus today's process if given,
    in euros per piece."""
    ship_all = {"pos": {"pass": 1.0, "review": 0.0, "fail": 0.0},
                "neg": {"pass": 1.0, "review": 0.0, "fail": 0.0}}
    check_all = {"pos": {"pass": 0.0, "review": 1.0, "fail": 0.0},
                 "neg": {"pass": 0.0, "review": 1.0, "fail": 0.0}}
    out = {
        "do_nothing": cost_per_case(ship_all, base_rate, costs),
        "human_checks_all": cost_per_case(check_all, base_rate, costs),
        "model_alone": cost_per_case(band_rates(y, s, single_t, single_t), base_rate, costs),
        "model_plus_review": cost_per_case(band_rates(y, s, lo, hi), base_rate, costs),
    }
    if status_quo:
        sens, spec = status_quo["sensitivity"], status_quo["specificity"]
        out["status_quo"] = (status_quo["labour_per_piece"]
                             + base_rate * (sens * costs.tp + (1 - sens) * costs.fn)
                             + (1 - base_rate) * ((1 - spec) * costs.fp + spec * costs.tn))
    return out


def staffing(share: float, daily_volume: int, review_seconds: float, productive_hours: float) -> dict:
    cases = share * daily_volume
    hours = cases * review_seconds / 3600
    return {"cases_per_day": cases, "review_hours_per_day": hours,
            "reviewers_needed": hours / productive_hours}
