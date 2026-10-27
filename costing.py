"""
Business assumptions and cost formulas, shared by train.py and app.py.

Everything that is a *decision* (the euro costs, the base rate, the daily
volume, how many reviewers exist) lives here, in one place, so the training
pipeline and the dashboard can never disagree about them. REPORT.md explains
where every number comes from.

The setting: the packing line of a supermarket distribution centre. Each piece
of fruit (apple, banana or orange) is photographed before it goes into a retail
pack. The model gives a "rotten score" = P(rotten). Three outcomes:

    score <  t_low             -> PASS   (goes into the pack)
    t_low <= score < t_high    -> REVIEW (a person looks at the photo)
    score >= t_high            -> FAIL   (discarded)

Positive class = ROTTEN.
"""

import numpy as np

# --------------------------------------------------------------------------
# Cost of each outcome, in euros per piece of fruit
# --------------------------------------------------------------------------
# False negative: a rotten piece goes into a pack and reaches a customer.
# The customer returns / throws away the whole pack (about 1 kg, ~2 EUR retail)
# and mould spreads to the other pieces in the pack.
COST_FN = 2.00

# False positive: a good piece is thrown away. We lose what we paid for it
# (wholesale, roughly 1.2 EUR/kg and ~5 pieces per kg -> ~0.25 EUR).
COST_FP = 0.25

# True positive / true negative: no extra cost. Discarding a rotten piece
# loses nothing we had not already lost; passing a good piece is the normal case.
COST_TP = 0.00
COST_TN = 0.00

# Human review: an operator looks at the photo on a screen.
# ~18 EUR/hour fully loaded and ~10 seconds per check -> 18 / 360 = 0.05 EUR.
# We assume the person decides correctly, so a reviewed case costs only this.
SECONDS_PER_REVIEW = 10
REVIEWER_COST_PER_HOUR = 18.0
COST_REVIEW = REVIEWER_COST_PER_HOUR * SECONDS_PER_REVIEW / 3600

# --------------------------------------------------------------------------
# Operating assumptions
# --------------------------------------------------------------------------
# Base rate: share of incoming fruit that is actually rotten in production.
# The dataset is about half rotten, which is NOT what a real line sees.
BASE_RATE = 0.05

# Pieces of fruit photographed per day on one packing line.
DAILY_VOLUME = 20_000

# People available for review: one quality controller, one 8-hour shift.
REVIEWERS = 1
HOURS_PER_SHIFT = 8
REVIEW_CAPACITY_PER_DAY = REVIEWERS * HOURS_PER_SHIFT * 3600 // SECONDS_PER_REVIEW


def outcome_rates(scores, is_rotten, t_low, t_high):
    """Share of rotten and of fresh items that end up in PASS / REVIEW / FAIL.

    scores    : rotten score of each image (0..1)
    is_rotten : True/False, the real label of each image
    Returns two dicts, one for rotten items and one for fresh items.
    """
    scores = np.asarray(scores, dtype=float)
    is_rotten = np.asarray(is_rotten, dtype=bool)

    def split(s):
        n = max(len(s), 1)
        return {
            "pass": float(np.sum(s < t_low) / n),
            "review": float(np.sum((s >= t_low) & (s < t_high)) / n),
            "fail": float(np.sum(s >= t_high) / n),
        }

    return split(scores[is_rotten]), split(scores[~is_rotten])


def expected_cost(scores, is_rotten, t_low, t_high, base_rate=BASE_RATE):
    """Expected cost per piece of fruit, re-weighted to the production base rate.

    We measure *rates* on the test images (how often a rotten image passes, how
    often a fresh image is failed...), then combine them with the base rate we
    expect in production instead of the dataset's own 50/50 mix.

    A single threshold t is the special case t_low == t_high == t (no review).
    """
    rot, fresh = outcome_rates(scores, is_rotten, t_low, t_high)
    cost_rotten = rot["pass"] * COST_FN + rot["review"] * COST_REVIEW + rot["fail"] * COST_TP
    cost_fresh = fresh["pass"] * COST_TN + fresh["review"] * COST_REVIEW + fresh["fail"] * COST_FP
    return base_rate * cost_rotten + (1 - base_rate) * cost_fresh


def review_share(scores, is_rotten, t_low, t_high, base_rate=BASE_RATE):
    """Share of all production cases that would be sent to a person."""
    rot, fresh = outcome_rates(scores, is_rotten, t_low, t_high)
    return base_rate * rot["review"] + (1 - base_rate) * fresh["review"]


def staff_needed(share, daily_volume=DAILY_VOLUME):
    """How many full-time reviewers a given review share needs."""
    seconds = share * daily_volume * SECONDS_PER_REVIEW
    return seconds / (HOURS_PER_SHIFT * 3600)
