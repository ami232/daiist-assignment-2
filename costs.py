"""
Cost assumptions and the pass / review / fail rule.

Shared by train.ipynb and app.py, so the notebook and the dashboard always use
the same numbers. Every figure here is justified in REPORT.md.
"""

import numpy as np

NEGATIVE = "notumor"  # the healthy class; the other three classes are tumours

# Euros per scan
COST_FN = 30_000   # tumour passed as normal: delayed diagnosis, worse prognosis, liability
COST_FP = 600      # healthy scan sent for urgent referral: consult + extra imaging
COST_REVIEW = 50   # one remote neuroradiologist read
BASE_RATE = 0.05   # share of the hospital's brain MRIs that really show a tumour

# Workload behind the review band
SCANS_PER_DAY = 20
MINUTES_PER_READ = 15
READER_HOURS_PER_DAY = 6

GRID = np.round(np.arange(0, 1.01, 0.01), 2)  # thresholds we sweep


def route(score, t_low, t_high):
    """Pass below t_low, fail at or above t_high, review in between."""
    return np.where(score >= t_high, "fail", np.where(score >= t_low, "review", "pass"))


def outcome_rates(df, t_low, t_high):
    """Share of tumour scans and of healthy scans sent to each outcome."""
    out = route(df["score"].to_numpy(), t_low, t_high)
    tumour = (df["true"] != NEGATIVE).to_numpy()
    return {o: ((out[tumour] == o).mean(), (out[~tumour] == o).mean())
            for o in ("pass", "review", "fail")}


def cost_per_1000(df, t_low, t_high, base_rate=BASE_RATE, cost_fn=COST_FN):
    """Expected euros per 1,000 scans at the hospital's base rate, not the dataset's.

    The neuroradiologist is assumed to get reviewed scans right, so a review
    costs only its fee. Correct outcomes (tumour referred, healthy passed) cost
    nothing extra.
    """
    r = outcome_rates(df, t_low, t_high)
    tumour = r["pass"][0] * cost_fn + r["review"][0] * COST_REVIEW
    healthy = r["review"][1] * COST_REVIEW + r["fail"][1] * COST_FP
    return 1000 * (base_rate * tumour + (1 - base_rate) * healthy)


def review_share(df, t_low, t_high, base_rate=BASE_RATE):
    """Share of the hospital's scans that go to the neuroradiologist."""
    r = outcome_rates(df, t_low, t_high)
    return base_rate * r["review"][0] + (1 - base_rate) * r["review"][1]


def best_band(df, base_rate=BASE_RATE, cost_fn=COST_FN):
    """Cheapest (t_low, t_high) pair on the grid, and its cost per 1,000 scans."""
    pairs = [(lo, hi) for lo in GRID for hi in GRID if hi >= lo]
    costs = [cost_per_1000(df, lo, hi, base_rate, cost_fn) for lo, hi in pairs]
    best = int(np.argmin(costs))
    return pairs[best], costs[best]


def best_single(df, base_rate=BASE_RATE, cost_fn=COST_FN):
    """Model decides alone: one threshold, no review band."""
    costs = [cost_per_1000(df, t, t, base_rate, cost_fn) for t in GRID]
    best = int(np.argmin(costs))
    return GRID[best], costs[best]
