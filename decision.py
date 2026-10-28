"""
Cost model shared by train.py and app.py.

Each test image gets one score: the highest confidence of any disease box
(Anthracnose, Cercospora Spot, Stem End Rot, Sunblotch). An image with any
disease label is a positive (reject); an image with only Healthy is a negative
(export).

A policy has two thresholds, lo <= hi:
    score >= hi        -> reject
    lo <= score < hi   -> human review
    score < lo         -> pass (export)
A single threshold is the special case lo == hi (nothing goes to review).

Costs are computed from rates (TPR, FPR, ...) and then reweighted by the
assumed base rate, not by the test split's mix, because the test split is
about 74% diseased and a real packhouse sees far fewer.

Thresholds are chosen on the validation split and then evaluated on the test
split, so the reported cost is not the minimum of the same data it was tuned on.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
COSTS_FILE = ROOT / "costs.json"

DISEASE_CLASSES = ["Anthracnose", "Cercospora Spot", "Stem End Rot", "Sunblotch"]
HEALTHY_CLASS = "Healthy"
THRESHOLDS =np.round(np.linspace(0.0, 1.0, 101), 2)


def load_costs(path: Path = COSTS_FILE) -> dict:
    costs = json.loads(path.read_text())
    costs["review_cost_eur"] = (
        costs["review_seconds_per_case"] / 3600 * costs["reviewer_cost_eur_per_hour"]
    )
    return costs


def outcome_rates(scores: np.ndarray, labels: np.ndarray, lo: float, hi: float) -> dict:
    """Share of positives and of negatives that end up passed, reviewed, rejected."""
    pos, neg = scores[labels == 1], scores[labels == 0]

    def split(s):
        reject = np.mean(s >= hi) if len(s) else 0.0
        passed = np.mean(s < lo) if len(s) else 0.0
        return {"pass": passed, "review": 1 - reject - passed, "reject": reject}

    return {"pos": split(pos), "neg": split(neg)}


def expected_cost(rates: dict, costs: dict) -> dict:
    """Expected cost per case, plus the review share, at the assumed base rate."""
    p = costs["base_rate"]
    acc = costs["human_review_accuracy"]
    fn, fp, tp, tn = (costs[k] for k in ("cost_fn_eur", "cost_fp_eur", "cost_tp_eur", "cost_tn_eur"))
    review = costs["review_cost_eur"]

    pos, neg = rates["pos"], rates["neg"]
    pos_cost = (
        pos["pass"] * fn
        + pos["reject"] * tp
        + pos["review"] * (review + acc * tp + (1 - acc) * fn)
    )
    neg_cost = (
        neg["pass"] * tn
        + neg["reject"] * fp
        + neg["review"] * (review + acc * tn + (1 - acc) * fp)
    )
    return {
        "cost_per_case": p * pos_cost + (1 - p) * neg_cost,
        "review_share": p * pos["review"] + (1 - p) * neg["review"],
    }


def staffing(review_share: float, costs: dict) -> dict:
    per_day = review_share * costs["cases_per_day"]
    hours = per_day * costs["review_seconds_per_case"] / 3600
    return {
        "reviews_per_day": per_day,
        "review_hours_per_day": hours,
        "reviewers_needed": hours / costs["review_hours_per_person_per_day"],
    }


def single_threshold_sweep(scores: np.ndarray, labels: np.ndarray, costs: dict) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        rates = outcome_rates(scores, labels, t, t)
        cost = expected_cost(rates, costs)
        rows.append({
            "threshold": float(t),
            "recall": rates["pos"]["reject"],
            "false_positive_rate": rates["neg"]["reject"],
            "cost_per_1000": 1000 * cost["cost_per_case"],
        })
    return rows


def best_single_threshold(scores: np.ndarray, labels: np.ndarray, costs: dict) -> float:
    sweep = single_threshold_sweep(scores, labels, costs)
    return min(sweep, key=lambda r: r["cost_per_1000"])["threshold"]


def best_review_band(scores: np.ndarray, labels: np.ndarray, costs: dict) -> dict:
    """Search every (lo, hi) pair on the threshold grid for the cheapest policy."""
    best = None
    for i, lo in enumerate(THRESHOLDS):
        for hi in THRESHOLDS[i:]:
            cost = expected_cost(outcome_rates(scores, labels, lo, hi), costs)
            if best is None or cost["cost_per_case"] < best["cost_per_case"]:
                best = {"lo": float(lo), "hi": float(hi), **cost}
    return best


def alternatives(scores: np.ndarray, labels: np.ndarray, costs: dict,
                 threshold: float, lo: float, hi: float) -> dict:
    """Cost per 1,000 of the policies the review design is priced against.

    The thresholds are fixed beforehand (chosen on the validation split), so
    evaluating them on the test split is not optimistic.
    """
    everyone = {"pass": 0.0, "review": 1.0, "reject": 0.0}
    nobody = {"pass": 1.0, "review": 0.0, "reject": 0.0}
    reject_all = {"pass": 0.0, "review": 0.0, "reject": 1.0}

    def per_1000(rates):
        return 1000 * expected_cost(rates, costs)["cost_per_case"]

    return {
        "do_nothing (export everything)": per_1000({"pos": nobody, "neg": nobody}),
        "reject everything": per_1000({"pos": reject_all, "neg": reject_all}),
        "person checks everything": per_1000({"pos": everyone, "neg": everyone}),
        f"model alone (t={threshold:.2f})": per_1000(outcome_rates(scores, labels, threshold, threshold)),
        f"model + review band ({lo:.2f}-{hi:.2f})": per_1000(outcome_rates(scores, labels, lo, hi)),
    }


def confusion_3way(scores: np.ndarray, labels: np.ndarray, lo: float, hi: float) -> dict:
    """Counts on the test split: actual (diseased / healthy) x decision (pass / review / reject)."""
    out = {}
    for name, value in (("diseased", 1), ("healthy", 0)):
        s = scores[labels == value]
        out[name] = {
            "pass": int(np.sum(s < lo)),
            "review": int(np.sum((s >= lo) & (s < hi))),
            "reject": int(np.sum(s >= hi)),
        }
    return out
