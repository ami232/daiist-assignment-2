"""
Turning per-image scores into euros.

Every snapshot gets one score: the highest confidence the model gives to any
`head` (no-helmet) box in it, or 0 if it finds none. Two thresholds split the
score range into three outcomes:

    score <  t_low             -> PASS    (no action)
    t_low <= score < t_high    -> REVIEW  (a person looks at it)
    score >= t_high            -> FAIL    (alert: supervisor goes to the worker)

A binary model is the special case t_low == t_high (nothing goes to review).

Costs are computed per class rate, then re-weighted to the assumed production
base rate. That matters: the test split has far more violations than a real
site, so raw test-set counts would overstate how often alerts fire.

Assumption: a human reviewer gets the call right. A reviewed positive costs
the review plus the intervention (COST_TP); a reviewed negative costs only the
review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def outcome_rates(scores, labels, t_low: float, t_high: float) -> dict:
    """Share of positives / negatives landing in pass, review, fail."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    t_high = max(t_high, t_low)
    passed = scores < t_low
    failed = scores >= t_high
    review = ~passed & ~failed

    def share(mask, group):
        n = group.sum()
        return float((mask & group).sum() / n) if n else 0.0

    pos, neg = labels, ~labels
    return {
        "pos_pass": share(passed, pos), "pos_review": share(review, pos), "pos_fail": share(failed, pos),
        "neg_pass": share(passed, neg), "neg_review": share(review, neg), "neg_fail": share(failed, neg),
    }


def cost_per_case(r: dict, c_fn, c_fp, c_tp, c_tn, c_review, base_rate) -> float:
    p = base_rate
    pos = r["pos_pass"] * c_fn + r["pos_review"] * (c_review + c_tp) + r["pos_fail"] * c_tp
    neg = r["neg_pass"] * c_tn + r["neg_review"] * (c_review + c_tn) + r["neg_fail"] * c_fp
    return p * pos + (1 - p) * neg


def review_share(r: dict, base_rate) -> float:
    return base_rate * r["pos_review"] + (1 - base_rate) * r["neg_review"]


def evaluate(scores, labels, t_low, t_high, params: dict) -> dict:
    r = outcome_rates(scores, labels, t_low, t_high)
    return {
        **r,
        "cost_per_case": cost_per_case(r, **params),
        "review_share": review_share(r, params["base_rate"]),
    }


def expected_matrix_per_1000(r: dict, base_rate: float) -> pd.DataFrame:
    """Expected snapshots per 1,000 at the production base rate."""
    pos, neg = 1000 * base_rate, 1000 * (1 - base_rate)
    return pd.DataFrame(
        {
            "PASS": [pos * r["pos_pass"], neg * r["neg_pass"]],
            "REVIEW": [pos * r["pos_review"], neg * r["neg_review"]],
            "FAIL (alert)": [pos * r["pos_fail"], neg * r["neg_fail"]],
        },
        index=["Actual: violation", "Actual: compliant"],
    ).round(1)


THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def sweep_binary(scores, labels, params: dict, thresholds=THRESHOLDS) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        e = evaluate(scores, labels, t, t, params)
        rows.append({
            "threshold": float(t),
            "recall": e["pos_fail"],
            "false_alarm_rate": e["neg_fail"],
            "cost_per_1000": 1000 * e["cost_per_case"],
        })
    return pd.DataFrame(rows)


def best_band(scores, labels, params: dict, max_review_share: float | None,
              thresholds=THRESHOLDS) -> dict:
    """Cheapest (t_low, t_high) whose review load fits the people available."""
    best = None
    for i, lo in enumerate(thresholds):
        for hi in thresholds[i:]:
            e = evaluate(scores, labels, lo, hi, params)
            if max_review_share is not None and e["review_share"] > max_review_share:
                continue
            if best is None or e["cost_per_case"] < best["cost_per_case"] - 1e-12:
                best = {"t_low": float(lo), "t_high": float(hi), **e}
    return best


def alternatives(params: dict) -> dict:
    """Cost per case of the three non-model-band options (excluding model alone)."""
    p = params["base_rate"]
    return {
        "Do nothing (no cameras)": p * params["c_fn"] + (1 - p) * params["c_tn"],
        "Person checks every snapshot": params["c_review"] + p * params["c_tp"] + (1 - p) * params["c_tn"],
    }
