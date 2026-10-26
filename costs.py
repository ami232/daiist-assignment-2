"""
Cost model shared by train.py and app.py, so the threshold picked in training
and the numbers shown in the dashboard come from one definition.

Unit of decision: one photo of one drinks fridge in a convenience store,
taken by a fixed camera.
Positive class: the section has at least one empty gap (out-of-stock) that a
store associate should go and refill. The image-level score is the highest
confidence of any "empty" box the model draws on that photo.

Every euro figure below is an assumption argued in REPORT.md. Change it here
and re-run `main.py train`; the app picks the new values up from artifacts/.
"""

import numpy as np

POSITIVE_CLASS = "empty"

STAFF_COST_PER_HOUR = 14.0  # Spanish retail associate, wage + social security

COSTS = {
    # Missed gap: ~0.70 EUR margin on a ~2 EUR drink x 40% of customers who
    # leave instead of taking another drink x 6 customers wanting it in the
    # ~2 h before staff notice the gap themselves.
    "fn": 0.70 * 0.4 * 6,
    # False alarm: an associate walks to a full fridge for nothing (~2 min).
    "fp": STAFF_COST_PER_HOUR * 2 / 60,
    # Correct alert: the associate still spends time refilling, but that work
    # was needed anyway, so it is not a cost of the model.
    "tp": 0.0,
    "tn": 0.0,
    # A person looks at the photo on a screen (~10 s) and decides. Assumed correct.
    "review": STAFF_COST_PER_HOUR * 10 / 3600,
}

# How often a photographed section really has a gap in production. The
# dataset is collected to show gaps, so its own rate is far higher than this.
BASE_RATE = 0.10

# Operating volume, used to turn the review share into people.
OPERATIONS = {
    "stores": 200,
    "cameras_per_store": 6,  # one per drinks fridge
    "photos_per_camera_per_day": 16,  # one an hour, 16 h opening
    "seconds_per_review": 10,
    "review_hours_per_person_per_day": 6.0,  # 8 h shift minus breaks/other work
}


def photos_per_day(ops: dict = OPERATIONS) -> int:
    return ops["stores"] * ops["cameras_per_store"] * ops["photos_per_camera_per_day"]


def outcome_rates(y_true: np.ndarray, scores: np.ndarray, low: float, high: float) -> dict:
    """Rates of each (truth, decision) cell, conditional on the truth.

    score >= high  -> fail   (send an associate to refill)
    score <  low   -> pass   (do nothing)
    otherwise      -> review (a person looks at the photo)
    A single threshold t is the special case low == high == t.
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    fail = scores >= high
    review = (scores >= low) & ~fail
    passed = ~fail & ~review
    pos, neg = y_true, ~y_true
    n_pos, n_neg = max(pos.sum(), 1), max(neg.sum(), 1)
    return {
        "tp": (fail & pos).sum() / n_pos,
        "fn": (passed & pos).sum() / n_pos,
        "rev_pos": (review & pos).sum() / n_pos,
        "fp": (fail & neg).sum() / n_neg,
        "tn": (passed & neg).sum() / n_neg,
        "rev_neg": (review & neg).sum() / n_neg,
    }


def expected_cost(rates: dict, base_rate: float = BASE_RATE, costs: dict = COSTS) -> dict:
    """Expected euros per photo and review share at the assumed base rate.

    Rates are measured on the test set per class, then re-weighted to the
    production base rate, so the dataset's own (inflated) gap rate does not
    leak into the cost.
    """
    p, n = base_rate, 1 - base_rate
    per_photo = (
        p * (rates["tp"] * costs["tp"] + rates["fn"] * costs["fn"] + rates["rev_pos"] * costs["review"])
        + n * (rates["fp"] * costs["fp"] + rates["tn"] * costs["tn"] + rates["rev_neg"] * costs["review"])
    )
    review_share = p * rates["rev_pos"] + n * rates["rev_neg"]
    return {"cost_per_photo": per_photo, "cost_per_1000": 1000 * per_photo, "review_share": review_share}


def reviewers_needed(review_share: float, ops: dict = OPERATIONS) -> dict:
    reviews = review_share * photos_per_day(ops)
    hours = reviews * ops["seconds_per_review"] / 3600
    return {
        "reviews_per_day": reviews,
        "review_hours_per_day": hours,
        "people": hours / ops["review_hours_per_person_per_day"],
    }


def alternatives(base_rate: float = BASE_RATE, costs: dict = COSTS) -> dict:
    """Cost per 1,000 photos of the three policies the design is priced against."""
    return {
        "do nothing (no alerts)": 1000 * base_rate * costs["fn"],
        "person checks every photo": 1000 * costs["review"],
    }
