"""Cost model shared by train.ipynb and app.ipynb, so both stages price a decision the same way.

A case gets one score, the model's probability of any pneumonia. Two thresholds split it:

    score <  t_low           -> pass   (sent home)
    t_low <= score < t_high  -> review (a radiologist reads the X-ray)
    score >= t_high          -> fail   (treated as pneumonia)

With t_low == t_high there is no review band and the model decides alone.
"""

import numpy as np

# Outcome rates for the two policies that do not use the model at all
DO_NOTHING = {"pos_pass": 1, "pos_review": 0, "pos_fail": 0, "neg_pass": 1, "neg_review": 0, "neg_fail": 0}
CHECK_EVERYTHING = {"pos_pass": 0, "pos_review": 1, "pos_fail": 0, "neg_pass": 0, "neg_review": 1, "neg_fail": 0}


def outcome_rates(score, is_positive, t_low: float, t_high: float) -> dict:
    """Share of positives and of negatives that land in each outcome."""
    score, pos = np.asarray(score), np.asarray(is_positive, dtype=bool)
    passed, failed = score < t_low, score >= t_high
    review = ~passed & ~failed
    rates = {}
    for name, group in [("pos", pos), ("neg", ~pos)]:
        n = max(group.sum(), 1)
        rates[f"{name}_pass"] = float((passed & group).sum() / n)
        rates[f"{name}_review"] = float((review & group).sum() / n)
        rates[f"{name}_fail"] = float((failed & group).sum() / n)
    return rates


def cost_per_1000(rates: dict, a: dict) -> float:
    """Expected EUR per 1,000 production cases, reweighted to the assumed base rate.

    Rates are measured separately on positives and negatives, so they do not depend on how
    many positives the dataset happens to contain. The base rate then sets the real mix.
    A reviewed case costs the read plus whatever the radiologist's call leads to.
    """
    reviewed_pos = a["cost_review"] + a["reviewer_sensitivity"] * a["cost_tp"] + (1 - a["reviewer_sensitivity"]) * a["cost_fn"]
    reviewed_neg = a["cost_review"] + a["reviewer_specificity"] * a["cost_tn"] + (1 - a["reviewer_specificity"]) * a["cost_fp"]
    per_positive = rates["pos_pass"] * a["cost_fn"] + rates["pos_review"] * reviewed_pos + rates["pos_fail"] * a["cost_tp"]
    per_negative = rates["neg_fail"] * a["cost_fp"] + rates["neg_review"] * reviewed_neg + rates["neg_pass"] * a["cost_tn"]
    return 1000 * (a["base_rate"] * per_positive + (1 - a["base_rate"]) * per_negative)


def review_share(rates: dict, a: dict) -> float:
    """Share of production cases sent to a radiologist, at the assumed base rate."""
    return a["base_rate"] * rates["pos_review"] + (1 - a["base_rate"]) * rates["neg_review"]


def summarise(name: str, rates: dict, a: dict) -> dict:
    """One row of the comparison table: cost, errors and the staffing a policy needs."""
    share = review_share(rates, a)
    reviews_per_day = share * a["cases_per_day"]
    reviewer_minutes = reviews_per_day * a["minutes_per_review"]
    # Errors include the ones the radiologist makes on the cases sent for review
    missed = a["base_rate"] * (rates["pos_pass"] + rates["pos_review"] * (1 - a["reviewer_sensitivity"]))
    false_alarms = (1 - a["base_rate"]) * (rates["neg_fail"] + rates["neg_review"] * (1 - a["reviewer_specificity"]))
    return {
        "option": name,
        "cost_per_1000_eur": round(cost_per_1000(rates, a), 1),
        "missed_per_1000": round(1000 * missed, 1),
        "false_alarms_per_1000": round(1000 * false_alarms, 1),
        "review_share": round(share, 3),
        "reviews_per_day": round(reviews_per_day, 1),
        "radiologists_needed": round(reviewer_minutes / (60 * a["reviewer_hours_per_day"]), 2),
    }
