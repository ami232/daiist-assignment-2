"""
Business assumptions and cost arithmetic, shared by train.py and app.py.

Keeping them in one place means the dashboard and the training pipeline can
never disagree about what a mistake costs. Every figure is justified in
REPORT.md ("Business problem" and "Cost matrix and operating point").

One case = one person walking through the site entrance. The positive class is
"head" (no helmet). Each case has a "no helmet" score in [0, 1] from the model.
"""

import numpy as np
import pandas as pd

# --- Cost of each outcome, in EUR per case ----------------------------------------
HOURLY_COST = 26.51  # INE, average effective labour cost per hour, 2025-Q4
# Stopping someone: worker + guard each lose 2 minutes. Same whether they really
# had no helmet (TP) or did (FP).
STOP_COST = 2 * 2 / 60 * HOURLY_COST  # ~1.77
# Letting someone in without a helmet (FN): expected fine + expected accident.
P_INSPECTION, FINE = 1 / 250, 5_000            # LISOS art. 40.2, serious, minimum grade
P_ACCIDENT, ACCIDENT_COST = 1 / 5_000, 15_000  # head injury on a day without helmet


def fn_cost(p_inspection: float = P_INSPECTION, p_accident: float = P_ACCIDENT) -> float:
    return p_inspection * FINE + p_accident * ACCIDENT_COST


COSTS = {"tp": STOP_COST, "fp": STOP_COST, "fn": fn_cost(), "tn": 0.0}

# --- Production setting -------------------------------------------------------
BASE_RATE = 0.05     # share of entrances without a helmet
CASES_PER_DAY = 200  # 100 workers x 2 entrances

# --- Human review ---------------------------------------------------------------
# The gate guard looks at the photo on a screen. Assumed always right.
REVIEW_SECONDS = 15
REVIEW_COST = REVIEW_SECONDS / 3600 * HOURLY_COST  # ~0.11
# Busiest moment: the morning shift arrives together. The guard can only
# review as many people as fit in that window.
PEAK_ARRIVALS, PEAK_WINDOW_MIN = 100, 15
MAX_REVIEW_SHARE = min(1.0, PEAK_WINDOW_MIN * 60 / REVIEW_SECONDS / PEAK_ARRIVALS)  # 0.6

# Finer steps at the low end, because with FN ~13x dearer than FP the cheapest
# thresholds sit close to zero.
THRESHOLDS = np.round(np.concatenate([np.arange(0.0, 0.05, 0.001), np.arange(0.05, 1.0001, 0.01)]), 3)


def three_way_rates(scores: np.ndarray, positive: np.ndarray, low: float, high: float) -> dict:
    """Share of positives and of negatives that end up stopped, reviewed or passed.

    score >= high        -> stopped automatically
    low <= score < high  -> sent to the guard
    score < low          -> let in
    With low == high nobody is reviewed: that is the model deciding alone.
    """
    stop = scores >= high
    review = (scores >= low) & ~stop
    pos, neg = positive == 1, positive == 0
    return {
        "pos_stop": float(stop[pos].mean()), "pos_review": float(review[pos].mean()),
        "neg_stop": float(stop[neg].mean()), "neg_review": float(review[neg].mean()),
    }


def cost_per_case(r: dict, costs: dict = COSTS, base_rate: float = BASE_RATE,
                  review_cost: float = REVIEW_COST) -> float:
    """Expected cost per case at the production base rate.

    We work from the shares of positives and negatives in each outcome, which
    do not depend on how many positives the test split happens to contain
    (~27% there, 5% at our gate), and re-weight them by the base rate.
    A reviewed positive is caught by the guard and then stopped (review + TP);
    a reviewed negative only costs the review.
    """
    pos_pass = 1 - r["pos_stop"] - r["pos_review"]
    neg_pass = 1 - r["neg_stop"] - r["neg_review"]
    pos = r["pos_stop"] * costs["tp"] + r["pos_review"] * (review_cost + costs["tp"]) + pos_pass * costs["fn"]
    neg = r["neg_stop"] * costs["fp"] + r["neg_review"] * review_cost + neg_pass * costs["tn"]
    return base_rate * pos + (1 - base_rate) * neg


def review_share(r: dict, base_rate: float = BASE_RATE) -> float:
    return base_rate * r["pos_review"] + (1 - base_rate) * r["neg_review"]


def expected_counts(r: dict, n: int = 1000, base_rate: float = BASE_RATE) -> dict:
    """Expected confusion matrix per n cases at the base rate.

    "Stopped" counts both automatic stops and reviewed people the guard stops,
    since the guard is assumed always right.
    """
    pos, neg = n * base_rate, n * (1 - base_rate)
    return {
        "tp": pos * (r["pos_stop"] + r["pos_review"]),
        "fn": pos * (1 - r["pos_stop"] - r["pos_review"]),
        "fp": neg * r["neg_stop"],
        "tn": neg * (1 - r["neg_stop"]),
        "reviewed": pos * r["pos_review"] + neg * r["neg_review"],
    }


def sweep(scores: np.ndarray, positive: np.ndarray, costs: dict = COSTS,
          base_rate: float = BASE_RATE) -> pd.DataFrame:
    """Single threshold, no review: cost per 1,000 cases at every threshold."""
    rows = []
    for t in THRESHOLDS:
        r = three_way_rates(scores, positive, t, t)
        rows.append({"threshold": t, "tpr": r["pos_stop"], "fpr": r["neg_stop"],
                     "cost_per_1000": 1000 * cost_per_case(r, costs, base_rate)})
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> dict:
    row = curve.loc[curve["cost_per_1000"].idxmin()]
    return {k: float(row[k]) for k in ["threshold", "tpr", "fpr", "cost_per_1000"]}


def best_review_band(scores: np.ndarray, positive: np.ndarray,
                     max_share: float = MAX_REVIEW_SHARE) -> dict:
    """Cheapest (low, high) pair whose review share fits the guard's capacity."""
    best = None
    for i, low in enumerate(THRESHOLDS):
        for high in THRESHOLDS[i:]:
            r = three_way_rates(scores, positive, low, high)
            share = review_share(r)
            if share > max_share:
                continue
            cost = 1000 * cost_per_case(r)
            if best is None or cost < best["cost_per_1000"]:
                best = {"low": float(low), "high": float(high), "cost_per_1000": cost,
                        "review_share": share, **r}
    return best


def alternatives(scores: np.ndarray, positive: np.ndarray, band: dict, model_alone: dict) -> list[dict]:
    """Our design priced against doing nothing, a person checking everyone, and the model alone."""
    everyone = {"pos_stop": 0.0, "pos_review": 1.0, "neg_stop": 0.0, "neg_review": 1.0}
    nobody = {"pos_stop": 0.0, "pos_review": 0.0, "neg_stop": 0.0, "neg_review": 0.0}
    alone = three_way_rates(scores, positive, model_alone["threshold"], model_alone["threshold"])
    mixed = three_way_rates(scores, positive, band["low"], band["high"])
    out = []
    for name, r in [("do nothing", nobody), ("guard checks everyone", everyone),
                    ("model alone", alone), ("model + guard review", mixed)]:
        share = review_share(r)
        out.append({
            "option": name,
            "cost_per_1000": 1000 * cost_per_case(r),
            "missed_per_1000": expected_counts(r)["fn"],
            "review_share": share,
            "reviews_per_day": share * CASES_PER_DAY,
            "review_minutes_per_day": share * CASES_PER_DAY * REVIEW_SECONDS / 60,
            "peak_review_minutes": share * PEAK_ARRIVALS * REVIEW_SECONDS / 60,
            "fits_peak": share <= MAX_REVIEW_SHARE,
        })
    return out
