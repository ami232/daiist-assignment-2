"""
Cost assumptions and decision maths, shared by train.py and app.py.

The numbers below are YOUR business assumptions. Fill them in and justify
each one in REPORT.md ("Cost matrix and operating point").

Decision unit: one image = one fruit case on the sorting line.
Three outcomes per case, based on the model's highest "rotten" confidence:
    score >= t_high          -> FAIL   (fruit is discarded)
    t_low <= score < t_high  -> REVIEW (a person checks it)
    score <  t_low           -> PASS   (fruit is shipped)
With t_low == t_high there is no review: the model decides alone.
Assumption: the human reviewer is always right.
"""

import numpy as np

POSITIVE = "rotten"  # every class whose name contains this word is "positive"

# ---- Fill these in (euros per fruit unless stated) -------------------------
COST_FALSE_POSITIVE = 0.20
COST_FALSE_NEGATIVE = 2.00
HOURLY_WAGE = 12.50
REVIEW_SECONDS = 5
BASE_RATE = 0.03
CASES_PER_DAY = 100_000
MAX_REVIEW_SHARE = None
# -----------------------------------------------------------------------------


def costs_ready() -> bool:
    needed = (COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE, HOURLY_WAGE, REVIEW_SECONDS, BASE_RATE)
    return all(v is not None for v in needed)


def cost_review() -> float:
    """Euros for one human check of one fruit."""
    return HOURLY_WAGE * REVIEW_SECONDS / 3600


def outcome_rates(scores, labels, t_low: float, t_high: float) -> dict:
    """Share of rotten (pos) and fresh (neg) cases that are failed, reviewed or passed."""
    scores = np.asarray(scores, dtype=float)
    pos = np.asarray(labels, dtype=bool)
    neg = ~pos
    fail = scores >= t_high
    review = (scores >= t_low) & ~fail
    passed = ~fail & ~review

    def share(mask, group):
        return float(mask[group].mean()) if group.any() else 0.0

    return {
        "tp": share(fail, pos), "review_pos": share(review, pos), "fn": share(passed, pos),
        "fp": share(fail, neg), "review_neg": share(review, neg), "tn": share(passed, neg),
    }


def review_share(rates: dict, base_rate: float | None = None) -> float:
    """Share of all real cases sent to a person, at the real base rate."""
    p = BASE_RATE if base_rate is None else base_rate
    return p * rates["review_pos"] + (1 - p) * rates["review_neg"]


def cost_per_1000(rates: dict, base_rate: float | None = None) -> float:
    """Expected euros per 1,000 fruits, re-weighted to the real base rate.

    The test set is about 58% rotten, the real line is not, so we take the
    error RATES from the test set and apply them to the real base rate.
    """
    p = BASE_RATE if base_rate is None else base_rate
    rotten = p * (rates["fn"] * COST_FALSE_NEGATIVE + rates["review_pos"] * cost_review())
    fresh = (1 - p) * (rates["fp"] * COST_FALSE_POSITIVE + rates["review_neg"] * cost_review())
    return 1000 * (rotten + fresh)


def alternatives_per_1000(base_rate: float | None = None) -> dict:
    """The three designs the README asks you to price your model against."""
    p = BASE_RATE if base_rate is None else base_rate
    return {
        "do_nothing_ship_everything": 1000 * p * COST_FALSE_NEGATIVE,
        "person_checks_everything": 1000 * cost_review(),
        "discard_everything": 1000 * (1 - p) * COST_FALSE_POSITIVE,
    }