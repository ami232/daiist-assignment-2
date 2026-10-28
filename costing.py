"""
Cost model for the escalation decision, shared by train.py and app.py.

A case is one frame sent to the model. Each frame has a score: the model's
highest confidence for Fire or Smoke. Two thresholds split frames three ways:

    score >= high          escalate automatically
    low <= score < high    send to an operator
    score < low            pass

With low == high there is no review band and the model decides alone.

The test split is about 80% positive, far above the real rate, so per-class
rates are measured on the test split and then reweighted to BASE_RATE rather
than counted directly.
"""

import numpy as np
import pandas as pd

# Euro cost of each outcome for a single frame
COSTS = {
    "fn": 50_000,  # fire/smoke missed: minutes of extra fire growth
    "fp": 1_000,   # false escalation: call-out fee, evacuation, lost trust
    "tp": 0,       # response cost is paid anyway once the fire is found
    "tn": 0,
}
REVIEW_COST = 1.0  # ~45 s of operator time per reviewed frame
BASE_RATE = 0.01   # share of frames sent to the model that show fire/smoke
# Most cases the review band may send to an operator. At 300 events a day,
# 30% is 90 reviews, about 70 minutes of operator time: what fits into
# existing shifts without hiring. Without a cap the cheapest band is
# "review almost everything", which is the fatigue problem the band avoids.
MAX_REVIEW_SHARE = 0.30

THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def outcome_rates(scores, labels, low, high):
    """Share of positives and of negatives landing in each of the three bands."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]

    def bands(s):
        escalate = (s >= high).mean()
        passed = (s < low).mean()
        return escalate, 1 - escalate - passed, passed

    tpr, pos_review, fnr = bands(pos)
    fpr, neg_review, tnr = bands(neg)
    return {"tpr": tpr, "fnr": fnr, "fpr": fpr, "tnr": tnr,
            "pos_review": pos_review, "neg_review": neg_review}


def cost_per_1000(scores, labels, low, high, base_rate=BASE_RATE,
                  costs=COSTS, review_cost=REVIEW_COST):
    """Expected outcomes and euro cost per 1,000 cases at the given base rate.

    A reviewed frame costs REVIEW_COST, and the operator is assumed to get it
    right, so a reviewed positive is escalated (a TP) and a reviewed negative
    is dismissed (a TN).
    """
    r = outcome_rates(scores, labels, low, high)
    n_pos, n_neg = 1000 * base_rate, 1000 * (1 - base_rate)
    counts = {
        "tp": n_pos * (r["tpr"] + r["pos_review"]),
        "fn": n_pos * r["fnr"],
        "fp": n_neg * r["fpr"],
        "tn": n_neg * (r["tnr"] + r["neg_review"]),
        "review": n_pos * r["pos_review"] + n_neg * r["neg_review"],
    }
    cost = sum(costs[k] * counts[k] for k in ("tp", "fn", "fp", "tn"))
    cost += review_cost * counts["review"]
    return counts | {"cost": cost, "review_share": counts["review"] / 1000}


def sweep(scores, labels, base_rate=BASE_RATE, costs=COSTS, thresholds=THRESHOLDS):
    """Cost against a single threshold, with no review band."""
    rows = [{"threshold": t} | cost_per_1000(scores, labels, t, t, base_rate, costs)
            for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold(sweep_df):
    """Lowest-cost threshold; ties go to the lower threshold, which misses less."""
    return sweep_df.loc[sweep_df["cost"].idxmin()]


def sweep_band(scores, labels, base_rate=BASE_RATE, costs=COSTS, thresholds=THRESHOLDS):
    """Cost for every (low, high) pair with low <= high: the three-outcome design."""
    rows = [{"low": lo, "high": hi} | cost_per_1000(scores, labels, lo, hi, base_rate, costs)
            for lo in thresholds for hi in thresholds if hi >= lo]
    return pd.DataFrame(rows)


def best_band(band_df, max_review_share=MAX_REVIEW_SHARE):
    """Cheapest (low, high) pair that keeps the review share under the cap."""
    allowed = band_df[band_df["review_share"] <= max_review_share]
    return allowed.loc[allowed["cost"].idxmin()]
