"""
The cost maths, shared by train.py and app.py so both compute it the same way.

Each test image has a score (the model's highest confidence for any defect
class in it) and a label (1 if it really contains a bad weld or defect).

A band (low, high) turns a score into one of three outcomes:
    score <  low          -> pass
    low <= score < high   -> send to a person
    score >= high         -> fail
A single threshold t is just the band (t, t): nothing goes to a person.

Costs are reweighted from the test set's mix of good and bad welds to the
production base rate in config.py. We measure the outcome rates separately
for bad and good welds on the test set, then weight them by how often each
actually occurs on the line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg

THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)


def _split(scores: np.ndarray, low: float, high: float) -> tuple[float, float, float]:
    """Share of these scores that pass, go to review, and fail."""
    if len(scores) == 0:
        return (np.nan, np.nan, np.nan)
    passed = float(np.mean(scores < low))
    failed = float(np.mean(scores >= high))
    reviewed = float(np.mean((scores >= low) & (scores < high)))  # counted directly, so never -0.0
    return passed, reviewed, failed


def outcome_rates(scores, labels, low, high):
    """(pass, review, fail) shares for bad welds and for good welds."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    low, high = min(low, high), max(low, high)
    return _split(scores[labels == 1], low, high), _split(scores[labels == 0], low, high)


def cost_per_case(scores, labels, low, high, base_rate=None, costs=None) -> float:
    """Expected euros per case at the production base rate."""
    base_rate = cfg.BASE_RATE if base_rate is None else base_rate
    c = cfg.costs() if costs is None else costs
    (p_pass, p_rev, p_fail), (n_pass, n_rev, n_fail) = outcome_rates(scores, labels, low, high)

    # A reviewed case costs the check, then whatever the person decides.
    reviewed_bad = c["review"] + (1 - c["human_miss"]) * c["tp"] + c["human_miss"] * c["fn"]
    reviewed_good = (
        c["review"]
        + (1 - c["human_false_alarm"]) * c["tn"]
        + c["human_false_alarm"] * c["fp"]
    )

    bad = p_pass * c["fn"] + p_fail * c["tp"] + p_rev * reviewed_bad
    good = n_pass * c["tn"] + n_fail * c["fp"] + n_rev * reviewed_good
    return base_rate * bad + (1 - base_rate) * good


def review_share(scores, labels, low, high, base_rate=None) -> float:
    """Share of production cases sent to a person."""
    base_rate = cfg.BASE_RATE if base_rate is None else base_rate
    (_, p_rev, _), (_, n_rev, _) = outcome_rates(scores, labels, low, high)
    return base_rate * p_rev + (1 - base_rate) * n_rev


def staffing(share: float) -> dict[str, float]:
    """How many checks a day a review share means, and how many people that takes."""
    checks = cfg.DAILY_VOLUME * share
    hours = checks * cfg.REVIEW_SECONDS / 3600
    return {
        "checks_per_day": checks,
        "hours_per_day": hours,
        "inspectors_needed": hours / cfg.INSPECTOR_HOURS_PER_DAY,
    }


def sweep(scores, labels, base_rate=None, costs=None) -> pd.DataFrame:
    """Model-alone cost at every threshold (pass below t, fail at or above t)."""
    rows = []
    for t in THRESHOLDS:
        (p_pass, _, p_fail), (n_pass, _, n_fail) = outcome_rates(scores, labels, t, t)
        rows.append(
            {
                "threshold": float(t),
                "cost_per_1000": 1000 * cost_per_case(scores, labels, t, t, base_rate, costs),
                "recall_bad": p_fail,
                "missed_bad": p_pass,
                "false_alarm_rate_good": n_fail,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(scores, labels, base_rate=None, costs=None) -> float:
    table = sweep(scores, labels, base_rate, costs)
    return float(table.loc[table["cost_per_1000"].idxmin(), "threshold"])


def best_band(scores, labels, base_rate=None, costs=None) -> tuple[float, float]:
    """Cost-minimising (low, high). Ties go to the band that sends fewer cases to a person."""
    best, best_key = (0.0, 0.0), (np.inf, np.inf)
    for i, low in enumerate(THRESHOLDS):
        for high in THRESHOLDS[i:]:
            key = (
                round(cost_per_case(scores, labels, low, high, base_rate, costs), 9),
                review_share(scores, labels, low, high, base_rate),
            )
            if key < best_key:
                best, best_key = (float(low), float(high)), key
    return best


def operating_point(scores, labels) -> dict:
    """The threshold and band to use: config overrides if set, else the cost minimum."""
    threshold = (
        cfg.CHOSEN_THRESHOLD if cfg.CHOSEN_THRESHOLD is not None else best_threshold(scores, labels)
    )
    band = tuple(cfg.CHOSEN_BAND) if cfg.CHOSEN_BAND is not None else best_band(scores, labels)
    return {
        "threshold": float(threshold),
        "band": [float(band[0]), float(band[1])],
        "chosen_by": "config.py override"
        if (cfg.CHOSEN_THRESHOLD is not None or cfg.CHOSEN_BAND is not None)
        else "cost-minimising sweep",
    }


def alternatives(scores, labels, threshold, band) -> pd.DataFrame:
    """The design priced against doing nothing, checking everything, and the model alone."""
    options = [
        ("Do nothing (pass every weld)", 1.0 + 1e-9, 1.0 + 1e-9),
        ("A person checks every weld", -1e-9, 1.0 + 1e-9),
        (f"Model alone (threshold {threshold:.2f})", threshold, threshold),
        (f"Model + human review ({band[0]:.2f}–{band[1]:.2f})", band[0], band[1]),
    ]
    rows = []
    for name, low, high in options:
        share = review_share(scores, labels, low, high)
        staff = staffing(share)
        rows.append(
            {
                "Strategy": name,
                "Cost per 1,000 welds (€)": round(1000 * cost_per_case(scores, labels, low, high), 2),
                "Cost per day (€)": round(cfg.DAILY_VOLUME * cost_per_case(scores, labels, low, high), 2),
                "Sent to a person": f"{share:.1%}",
                "Checks per day": round(staff["checks_per_day"], 1),
                "Inspectors needed": round(staff["inspectors_needed"], 2),
            }
        )
    return pd.DataFrame(rows)


def outcome_table(scores, labels, low, high) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 2x3 confusion matrix: raw test-set counts, and expected per 1,000 in production."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    low, high = min(low, high), max(low, high)
    counts, expected = [], []
    for name, value, weight in (
        ("Actually bad", 1, cfg.BASE_RATE),
        ("Actually good", 0, 1 - cfg.BASE_RATE),
    ):
        s = scores[labels == value]
        counts.append(
            {
                "": name,
                "Pass": int(np.sum(s < low)),
                "Send to a person": int(np.sum((s >= low) & (s < high))),
                "Fail": int(np.sum(s >= high)),
            }
        )
        rates = _split(s, low, high)
        expected.append(
            {
                "": name,
                "Pass": round(1000 * weight * rates[0], 1),
                "Send to a person": round(1000 * weight * rates[1], 1),
                "Fail": round(1000 * weight * rates[2], 1),
            }
        )
    return pd.DataFrame(counts), pd.DataFrame(expected)


def roc_auc(scores, labels) -> float:
    """Chance a random bad weld scores higher than a random good one (ties count half)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    bad, good = scores[labels == 1], scores[labels == 0]
    if len(bad) == 0 or len(good) == 0:
        return float("nan")
    diff = bad[:, None] - good[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))
