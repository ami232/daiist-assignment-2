"""
Turning per-image scores into euros.

Shared by train.ipynb (which picks the operating point) and app.ipynb (which
lets you move it), so both price a decision exactly the same way.

A "case" is one image. Its score is the highest confidence the model gives to
any `head` (bare head, no helmet) box in it. Two thresholds split cases into
three outcomes:

    score <  low            -> pass   (no alarm)
    low  <= score < high    -> review (a person looks at it)
    score >= high           -> fail   (alarm, the safety officer acts)

With low == high there is no review band and the model decides alone.

Rates are measured separately on positive and negative images and then
re-weighted by the base rate you assume for production, so the test split's own
class mix never leaks into the cost.
"""

import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)


def outcome_rates(scores, y_true, low, high):
    """Share of positives and of negatives that land in pass / review / fail."""
    scores, y_true = np.asarray(scores), np.asarray(y_true)
    low, high = min(low, high), max(low, high)

    def split(s):
        if len(s) == 0:
            return 0.0, 0.0, 0.0
        passed = float(np.mean(s < low))
        failed = float(np.mean(s >= high))
        return passed, 1.0 - passed - failed, failed

    return split(scores[y_true == 1]), split(scores[y_true == 0])


def band_cost(scores, y_true, low, high, costs, base_rate):
    """Expected cost per 1,000 cases and the share sent to review.

    Assumes the reviewer always gets it right, so a reviewed positive costs a
    true positive plus the review, and a reviewed negative a true negative plus
    the review.
    """
    (p_pass, p_rev, p_fail), (n_pass, n_rev, n_fail) = outcome_rates(scores, y_true, low, high)
    pos = p_pass * costs["fn"] + p_fail * costs["tp"] + p_rev * (costs["tp"] + costs["review"])
    neg = n_pass * costs["tn"] + n_fail * costs["fp"] + n_rev * (costs["tn"] + costs["review"])
    per_case = base_rate * pos + (1 - base_rate) * neg
    review_share = base_rate * p_rev + (1 - base_rate) * n_rev
    return 1000 * per_case, review_share


def sweep(scores, y_true, costs, base_rate, thresholds=THRESHOLDS):
    """Single-threshold sweep (model decides alone): TPR, FPR and cost per 1,000."""
    rows = []
    for t in thresholds:
        (p_pass, _, p_fail), (n_pass, _, n_fail) = outcome_rates(scores, y_true, t, t)
        cost, _ = band_cost(scores, y_true, t, t, costs, base_rate)
        rows.append({"threshold": float(t), "tpr": p_fail, "fpr": n_fail, "cost_per_1000": cost})
    return pd.DataFrame(rows)


def reviewers_needed(review_share, ops):
    """Full-time reviewers needed to clear the review queue each day."""
    minutes = review_share * ops["cases_per_day"] * ops["review_minutes"]
    return minutes / ops["reviewer_minutes_per_day"]


def best_band(scores, y_true, costs, base_rate, ops, thresholds=THRESHOLDS):
    """Cheapest (low, high) pair whose review load the available reviewers can clear."""
    best = None
    for i, low in enumerate(thresholds):
        for high in thresholds[i:]:
            cost, share = band_cost(scores, y_true, low, high, costs, base_rate)
            if reviewers_needed(share, ops) > ops["reviewers_available"]:
                continue
            if best is None or cost < best["cost_per_1000"]:
                best = {"low": float(low), "high": float(high), "cost_per_1000": cost, "review_share": share}
    return best


def alternatives(costs, base_rate):
    """Cost per 1,000 cases of the two no-model policies."""
    return {
        "do_nothing": 1000 * (base_rate * costs["fn"] + (1 - base_rate) * costs["tn"]),
        "person_checks_everything": 1000
        * (costs["review"] + base_rate * costs["tp"] + (1 - base_rate) * costs["tn"]),
    }
