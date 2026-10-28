"""
Assignment 2: the cost model, shared by train.py and app.py.

Every euro figure and every assumption about the gate lives here, so the cost
sweep that train.py saves and the numbers the dashboard shows always come from
the same place. The reasoning behind each figure is in REPORT.md.

A case is one gate pass (one image). The positive class is "worker without a
hard hat": an image is positive if it has at least one `head` label. The model
gives each image a score, its highest `head` confidence (0 if it found none).

Two thresholds split the score range into three outcomes:

    score <  low            -> pass   (worker walks in)
    low  <= score < high    -> review (snapshot goes to the supervisor)
    score >= high           -> fail   (worker is stopped)

A single threshold is the same thing with low == high, so nothing goes to
review. The test set has about 20% positives, but the gate sees BASE_RATE, so
costs are computed from the rates within each true class and reweighted with
BASE_RATE, never from raw test set counts.
"""

import numpy as np
import pandas as pd

# How often a worker shows up without a hard hat at the gate.
BASE_RATE = 0.03

# Euros per case for each outcome when the model decides alone.
COST_FP = 0.70  # 1 min of the worker + 30 s of the supervisor at 26.51 EUR/h (INE Q4 2025)
COST_FN = 25.0  # injury, fine and site stoppage risk plus a safety culture allowance, see REPORT.md
COST_TP = 2.50  # worker walks back for a hard hat, about 5 min, plus 30 s of the supervisor
COST_TN = 0.0   # worker with a hard hat walks in

# Human review of a snapshot: the supervisor looks at it for about 15 s while
# the worker waits at the gate for the verdict.
REVIEW_SECONDS = 15
SUPERVISOR_HOURLY_EUR = 30.0
WORKER_HOURLY_EUR = 26.51  # INE Q4 2025, average labour cost per hour worked
REVIEW_COST = REVIEW_SECONDS / 3600 * (SUPERVISOR_HOURLY_EUR + WORKER_HOURLY_EUR)
REVIEWER_ACCURACY = 0.95  # share of reviewed cases the supervisor gets right

# Gate volume, used to turn shares into people.
CASES_PER_DAY = 600
SHIFT_HOURS = 8

# A person checking everyone has to stand at the gate for the whole shift.
GUARD_COST_PER_CASE = SHIFT_HOURS * WORKER_HOURLY_EUR / CASES_PER_DAY

THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)


def outcome_shares(
    scores: np.ndarray, labels: np.ndarray, low: float, high: float
) -> dict[str, float]:
    """Share of positives and of negatives that pass, go to review, or fail."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    high = max(high, low)
    passed = scores < low
    failed = scores >= high
    review = ~passed & ~failed
    pos, neg = labels, ~labels
    return {
        "pos_pass": passed[pos].mean(),
        "pos_review": review[pos].mean(),
        "pos_fail": failed[pos].mean(),
        "neg_pass": passed[neg].mean(),
        "neg_review": review[neg].mean(),
        "neg_fail": failed[neg].mean(),
    }


def outcome_counts(
    scores: np.ndarray, labels: np.ndarray, low: float, high: float
) -> pd.DataFrame:
    """Test set confusion matrix: true class in rows, pass/review/fail in columns."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    high = max(high, low)
    outcome = np.where(scores < low, "pass", np.where(scores >= high, "fail", "review"))
    table = pd.crosstab(
        pd.Series(np.where(labels, "no hard hat", "hard hat"), name="true"),
        pd.Series(outcome, name="decision"),
    )
    return table.reindex(
        index=["no hard hat", "hard hat"], columns=["pass", "review", "fail"], fill_value=0
    )


def cost_per_case(
    shares: dict[str, float],
    base_rate: float = BASE_RATE,
    review_cost: float | None = None,
) -> float:
    """Expected euros per gate pass at the gate's base rate.

    A reviewed case costs review_cost (REVIEW_COST unless given), and then the
    person's decision: right with probability REVIEWER_ACCURACY, wrong otherwise.
    """
    review_cost = REVIEW_COST if review_cost is None else review_cost
    right, wrong = REVIEWER_ACCURACY, 1 - REVIEWER_ACCURACY
    positive = (
        shares["pos_pass"] * COST_FN
        + shares["pos_fail"] * COST_TP
        + shares["pos_review"] * (review_cost + right * COST_TP + wrong * COST_FN)
    )
    negative = (
        shares["neg_pass"] * COST_TN
        + shares["neg_fail"] * COST_FP
        + shares["neg_review"] * (review_cost + right * COST_TN + wrong * COST_FP)
    )
    return base_rate * positive + (1 - base_rate) * negative


# The two reference policies with no model, as outcome shares.
NOBODY_CHECKED = dict(pos_pass=1, pos_review=0, pos_fail=0, neg_pass=1, neg_review=0, neg_fail=0)
EVERYONE_CHECKED = dict(pos_pass=0, pos_review=1, pos_fail=0, neg_pass=0, neg_review=1, neg_fail=0)


def best_band(
    scores: np.ndarray, labels: np.ndarray, base_rate: float = BASE_RATE
) -> tuple[float, float, float]:
    """Cheapest (low, high) pair on the THRESHOLDS grid, and its cost per case."""
    best = (np.inf, 0.0, 0.0)
    for i, low in enumerate(THRESHOLDS):
        for high in THRESHOLDS[i:]:
            cost = cost_per_case(outcome_shares(scores, labels, low, high), base_rate)
            if cost < best[0] - 1e-12:
                best = (cost, float(low), float(high))
    return best[1], best[2], best[0]


def review_share(shares: dict[str, float], base_rate: float = BASE_RATE) -> float:
    """Share of all gate passes sent to the supervisor, at the gate's base rate."""
    return base_rate * shares["pos_review"] + (1 - base_rate) * shares["neg_review"]


def sweep(
    scores: np.ndarray, labels: np.ndarray, base_rate: float = BASE_RATE
) -> pd.DataFrame:
    """Cost per 1,000 gate passes at every single threshold (no review band)."""
    rows = []
    for t in THRESHOLDS:
        shares = outcome_shares(scores, labels, t, t)
        rows.append({
            "threshold": t,
            "recall": shares["pos_fail"],
            "false_positive_rate": shares["neg_fail"],
            "cost_per_1000": 1000 * cost_per_case(shares, base_rate),
        })
    return pd.DataFrame(rows)
