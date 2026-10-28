"""
Turns blister scores into decisions, costs and review workload.

Shared by train.py (to sweep and choose thresholds) and app.py (to recompute
everything live when the sliders move), so both always agree.

A blister's score is the highest confidence the model gives any empty-pocket
box in the image. Rates are measured on the evaluation images, then reweighted
to the production base rate in config.py, because the test split is far more
defective than a real line.
"""

import math

import numpy as np
import pandas as pd

import config

DECISIONS = ["pass", "review", "reject"]
# Starts at 0.01: a threshold of 0 would reject blisters with no detection at all.
GRID = np.round(np.arange(0.01, 1.0001, 0.01), 2)


def decide(scores, t_low, t_high):
    """Pass below t_low, reject at or above t_high, review in between."""
    scores = np.asarray(scores)
    return np.where(scores >= t_high, "reject", np.where(scores < t_low, "pass", "review"))


def outcome_counts(scores, defective, t_low, t_high):
    """2x3 table: actual (defective/good) by decision (pass/review/reject)."""
    decisions = decide(scores, t_low, t_high)
    defective = np.asarray(defective, bool)
    return pd.DataFrame(
        {d: [int(((decisions == d) & defective).sum()), int(((decisions == d) & ~defective).sum())]
         for d in DECISIONS},
        index=["defective", "good"],
    )


def rates(counts):
    """Each row divided by its total: P(decision | actual)."""
    return counts.div(counts.sum(axis=1).replace(0, 1), axis=0)


def cost_per_1000(counts, base_rate=None, cost_fn=None):
    """Expected euros per 1,000 blisters on a line with the given defect rate.

    base_rate and cost_fn default to config.py; the sensitivity analysis
    passes other values.
    """
    base_rate = config.BASE_RATE if base_rate is None else base_rate
    cost_fn = config.COST_FN if cost_fn is None else cost_fn
    r = rates(counts)
    defective = (r.at["defective", "pass"] * cost_fn
                 + r.at["defective", "review"] * (config.COST_REVIEW + config.COST_TP)
                 + r.at["defective", "reject"] * config.COST_TP)
    good = (r.at["good", "pass"] * config.COST_TN
            + r.at["good", "review"] * (config.COST_REVIEW + config.COST_TN)
            + r.at["good", "reject"] * config.COST_FP)
    return 1000 * (base_rate * defective + (1 - base_rate) * good)


def review_share(counts, base_rate=config.BASE_RATE):
    """Share of production blisters sent to a person."""
    r = rates(counts)
    return base_rate * r.at["defective", "review"] + (1 - base_rate) * r.at["good", "review"]


def per_1000(counts, base_rate=config.BASE_RATE):
    """The 2x3 table rescaled to 1,000 production blisters at the base rate."""
    r = rates(counts)
    return r.mul([1000 * base_rate, 1000 * (1 - base_rate)], axis=0)


def staffing(share):
    """Reviews per day and operators needed per shift for a given review share."""
    per_day = config.LINE_RATE_PER_MIN * 60 * config.HOURS_PER_DAY
    reviews = share * per_day
    shifts = config.HOURS_PER_DAY / config.SHIFT_HOURS
    hours_per_shift = reviews / shifts * config.REVIEW_SECONDS / 3600
    return {
        "blisters_per_day": per_day,
        "reviews_per_day": reviews,
        "review_hours_per_day": reviews * config.REVIEW_SECONDS / 3600,
        "operators_per_shift": hours_per_shift / config.PRODUCTIVE_HOURS_PER_SHIFT,
    }


def summarise(scores, defective, t_low, t_high):
    counts = outcome_counts(scores, defective, t_low, t_high)
    r = rates(counts)
    share = review_share(counts)
    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "cost_per_1000": cost_per_1000(counts),
        "review_share": share,
        "missed_defect_rate": float(r.at["defective", "pass"]),
        "good_rejected_rate": float(r.at["good", "reject"]),
        **staffing(share),
    }


def sweep_single(scores, defective):
    """Model decides alone (no review band): cost at every threshold."""
    rows = []
    for t in GRID:
        counts = outcome_counts(scores, defective, t, t)
        r = rates(counts)
        rows.append({
            "threshold": t,
            "cost_per_1000": cost_per_1000(counts),
            "recall": r.at["defective", "reject"],
            "false_positive_rate": r.at["good", "reject"],
        })
    return pd.DataFrame(rows)


def best_band(scores, defective, base_rate=None, cost_fn=None):
    """Cheapest (t_low, t_high) pair over the grid. Ties go to the narrower band.

    Same arithmetic as cost_per_1000, done for every pair at once.
    """
    base_rate = config.BASE_RATE if base_rate is None else base_rate
    cost_fn = config.COST_FN if cost_fn is None else cost_fn
    scores, defective = np.asarray(scores), np.asarray(defective, bool)
    n_def, n_good = max(defective.sum(), 1), max((~defective).sum(), 1)

    # Share of each class below t_low (passed) and at or above t_high (rejected).
    passed_def = (scores[defective][None, :] < GRID[:, None]).sum(1) / n_def
    passed_good = (scores[~defective][None, :] < GRID[:, None]).sum(1) / n_good
    rejected_def = (scores[defective][None, :] >= GRID[:, None]).sum(1) / n_def
    rejected_good = (scores[~defective][None, :] >= GRID[:, None]).sum(1) / n_good

    p_def, rj_def = passed_def[:, None], rejected_def[None, :]
    p_good, rj_good = passed_good[:, None], rejected_good[None, :]
    cost_def = (p_def * cost_fn + (1 - p_def - rj_def) * (config.COST_REVIEW + config.COST_TP)
                + rj_def * config.COST_TP)
    cost_good = (p_good * config.COST_TN + (1 - p_good - rj_good) * (config.COST_REVIEW + config.COST_TN)
                 + rj_good * config.COST_FP)
    cost = 1000 * (base_rate * cost_def + (1 - base_rate) * cost_good)
    cost[np.tril_indices(len(GRID), -1)] = np.inf  # t_low must not exceed t_high

    width = GRID[None, :] - GRID[:, None]
    order = np.lexsort((width.ravel(), np.round(cost, 9).ravel()))
    i, j = np.unravel_index(order[0], cost.shape)
    return float(GRID[i]), float(GRID[j])


def sensitivity(tune, test):
    """Re-choose the band on the tuning half for each FN cost and base rate, price it on the test half."""
    rows = []
    for cost_fn in config.SENSITIVITY_COST_FN:
        for base_rate in config.SENSITIVITY_BASE_RATE:
            t_low, t_high = best_band(tune.score.values, tune.defective.values, base_rate, cost_fn)
            counts = outcome_counts(test.score.values, test.defective.values, t_low, t_high)
            r = rates(counts)
            share = review_share(counts, base_rate)
            rows.append({
                "cost_fn": cost_fn, "base_rate": base_rate, "t_low": t_low, "t_high": t_high,
                "missed_defect_rate": float(r.at["defective", "pass"]),
                "good_rejected_rate": float(r.at["good", "reject"]),
                "review_share": share,
                "operators_per_shift": staffing(share)["operators_per_shift"],
                "cost_per_1000": cost_per_1000(counts, base_rate, cost_fn),
            })
    return pd.DataFrame(rows)


def alternatives(scores, defective, t_low, t_high, t_single):
    """The chosen design priced against the three alternatives in the brief."""
    n = len(scores)
    return {
        "Do nothing (pass everything)": summarise(np.zeros(n), defective, 0.5, 0.5),
        "Person checks everything": summarise(np.full(n, 0.5), defective, 0.0, 1.0),
        "Model decides alone": summarise(scores, defective, t_single, t_single),
        "Model + human review band": summarise(scores, defective, t_low, t_high),
    }


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)
