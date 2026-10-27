"""
Cost model shared by train.py (to pick the thresholds) and app.py (to let the
user move them). Keeping it in one file means the dashboard can never disagree
with the numbers in REPORT.md.

The unit of decision is one camera snapshot of a work zone. There are two
models, each with its own costs: the hard hat model (a snapshot is positive when
a worker has no hard hat, a `head` box) and the vest model (positive when a
worker has no safety vest). A snapshot's score is the highest confidence among
the positive-class detections, or 0 if there are none.

Three outcomes per snapshot:
- pass   (score <  low):  nothing happens.
- review (low <= score < high): a remote safety reviewer looks at the image and
  calls the foreman if it really is a violation.
- fail   (score >= high): the foreman is sent straight to the zone.

With low == high the review band is empty and the model decides alone.
"""

import numpy as np

# Euro cost of each outcome for one snapshot. Justified in REPORT.md.
COSTS = {
    "fn": 300.0,      # violation missed: expected fine + injury exposure
    "fp": 10.0,       # foreman walks to a zone where everyone is fine
    "tp": 10.0,       # same walk, but this time it fixes a real violation
    "tn": 0.0,        # nothing happens
    "review": 0.10,   # 15 s of a remote reviewer at about 24 EUR/h
}
BASE_RATE = 0.05      # share of production snapshots with a worker without a hat

# Missing vest: same fine, but a much smaller injury risk per snapshot, because
# the vest only matters where vehicles move. People also take vests off more
# often than hats (heat, jackets), so violations are more common.
VEST_COSTS = {**COSTS, "fn": 75.0}
VEST_BASE_RATE = 0.10

TASK_COSTS = {
    "helmet": (COSTS, BASE_RATE),
    "vest": (VEST_COSTS, VEST_BASE_RATE),
}
HUMAN_MISS = 0.05     # share of real violations a reviewer still waves through
# A person screening every snapshot sees a violation in only 1 of 20, and people
# miss far more targets when they are rare (Wolfe et al., 2005, Nature 435).
HUMAN_MISS_CHECK_ALL = 0.15

# Operating assumptions for the staffing maths.
DAILY_SNAPSHOTS = 1200        # 10 cameras x 1 snapshot / 5 min x 10 h shift
REVIEW_SECONDS = 15
REVIEWER_PRODUCTIVE_HOURS = 6.5

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def outcome_shares(scores, labels, low, high):
    """Share of positives and of negatives that land in pass / review / fail."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    shares = {}
    for name, mask in (("pos", labels), ("neg", ~labels)):
        s = scores[mask]
        n = max(len(s), 1)
        fail = (s >= high).sum() / n
        review = ((s >= low) & (s < high)).sum() / n
        shares[name] = {"pass": 1 - fail - review, "review": review, "fail": fail}
    return shares


def cost_per_case(shares, costs=COSTS, base_rate=BASE_RATE, human_miss=HUMAN_MISS):
    """Expected euros per snapshot at the production base rate.

    The test split has a different share of positives from production, so the
    per-class rates are measured on the split and re-weighted by base_rate.
    """
    pos, neg = shares["pos"], shares["neg"]
    reviewed_pos = costs["review"] + (1 - human_miss) * costs["tp"] + human_miss * costs["fn"]
    pos_cost = pos["pass"] * costs["fn"] + pos["review"] * reviewed_pos + pos["fail"] * costs["tp"]
    neg_cost = neg["pass"] * costs["tn"] + neg["review"] * (costs["review"] + costs["tn"]) + neg["fail"] * costs["fp"]
    return base_rate * pos_cost + (1 - base_rate) * neg_cost


def review_share(shares, base_rate=BASE_RATE):
    return base_rate * shares["pos"]["review"] + (1 - base_rate) * shares["neg"]["review"]


def evaluate(scores, labels, low, high, costs=COSTS, base_rate=BASE_RATE, human_miss=HUMAN_MISS):
    """Everything the report and dashboard show for one pair of thresholds."""
    shares = outcome_shares(scores, labels, low, high)
    per_case = cost_per_case(shares, costs, base_rate, human_miss)
    rev = review_share(shares, base_rate)
    reviews_per_day = rev * DAILY_SNAPSHOTS
    reviewer_hours = reviews_per_day * REVIEW_SECONDS / 3600
    return {
        "low": float(low),
        "high": float(high),
        "shares": shares,
        "cost_per_1000": 1000 * per_case,
        "review_share": rev,
        "reviews_per_day": reviews_per_day,
        "reviewer_hours_per_day": reviewer_hours,
        "reviewers_needed": reviewer_hours / REVIEWER_PRODUCTIVE_HOURS,
    }


def single_threshold_sweep(scores, labels, costs=COSTS, base_rate=BASE_RATE):
    """Cost per 1,000 snapshots when the model decides alone at each threshold."""
    return [
        {"threshold": float(t), **evaluate(scores, labels, t, t, costs, base_rate)}
        for t in THRESHOLDS
    ]


def best_single_threshold(scores, labels, costs=COSTS, base_rate=BASE_RATE):
    sweep = single_threshold_sweep(scores, labels, costs, base_rate)
    return min(sweep, key=lambda row: row["cost_per_1000"])


def best_band(scores, labels, costs=COSTS, base_rate=BASE_RATE, human_miss=HUMAN_MISS):
    """Search every (low, high) pair with low <= high for the cheapest band."""
    best = None
    for low in THRESHOLDS:
        for high in THRESHOLDS[THRESHOLDS >= low]:
            row = evaluate(scores, labels, low, high, costs, base_rate, human_miss)
            if best is None or row["cost_per_1000"] < best["cost_per_1000"]:
                best = row
    return best


def alternatives(costs=COSTS, base_rate=BASE_RATE, human_miss=HUMAN_MISS_CHECK_ALL):
    """Cost per 1,000 snapshots for the two policies that need no model."""
    do_nothing = base_rate * costs["fn"] + (1 - base_rate) * costs["tn"]
    reviewed_pos = costs["review"] + (1 - human_miss) * costs["tp"] + human_miss * costs["fn"]
    human_all = base_rate * reviewed_pos + (1 - base_rate) * (costs["review"] + costs["tn"])
    hours = DAILY_SNAPSHOTS * REVIEW_SECONDS / 3600
    return {
        "do_nothing": {"cost_per_1000": 1000 * do_nothing, "review_share": 0.0, "reviewers_needed": 0.0},
        "human_checks_all": {
            "cost_per_1000": 1000 * human_all,
            "review_share": 1.0,
            "reviewers_needed": hours / REVIEWER_PRODUCTIVE_HOURS,
        },
    }
