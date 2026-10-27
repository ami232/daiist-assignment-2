"""Cost model. Pure functions on a DataFrame with columns: label (0/1), score (0..1).

score = highest confidence of any POSITIVE-class (diseased) box in the image.
Decision rule: score < lo -> pass, lo <= score < hi -> human review, score >= hi -> fail.
lo == hi means no review band (a plain single threshold).

Rates come from the test set; the PRODUCTION base rate is applied on top, so the
test set's class balance does not distort the cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

N = 1000  # report everything per 1,000 cases


def _split(scores: pd.Series, lo: float, hi: float):
    s = np.asarray(scores)
    if len(s) == 0:
        return 0.0, 0.0, 0.0
    p_pass = float((s < lo).mean())
    p_fail = float((s >= hi).mean())
    return p_pass, 1.0 - p_pass - p_fail, p_fail


def evaluate(df: pd.DataFrame, lo: float, hi: float, base_rate: float | None = None) -> dict:
    base = C.BASE_RATE if base_rate is None else base_rate
    lo, hi = min(lo, hi), max(lo, hi)
    c, rv = C.COSTS, C.REVIEW
    a = rv["accuracy"]

    pos_pass, pos_rev, pos_fail = _split(df.loc[df.label == 1, "score"], lo, hi)
    neg_pass, neg_rev, neg_fail = _split(df.loc[df.label == 0, "score"], lo, hi)
    pos_n, neg_n = N * base, N * (1 - base)

    # raw 2x3 grid, per 1,000 cases: rows = actual (healthy, diseased); cols = pass/review/fail
    grid = np.array([
        [neg_n * neg_pass, neg_n * neg_rev, neg_n * neg_fail],
        [pos_n * pos_pass, pos_n * pos_rev, pos_n * pos_fail],
    ])

    # reviewed cases are resolved by a person who is right with probability `a`
    tp = grid[1, 2] + grid[1, 1] * a
    fn = grid[1, 0] + grid[1, 1] * (1 - a)
    fp = grid[0, 2] + grid[0, 1] * (1 - a)
    tn = grid[0, 0] + grid[0, 1] * a
    reviews = grid[:, 1].sum()

    cost = tp * c["tp"] + fn * c["fn"] + fp * c["fp"] + tn * c["tn"] + reviews * rv["cost_per_case"]
    return {
        "cost_per_1000": float(cost),
        "review_share": float(reviews / N),
        "grid": grid,
        "tp": float(tp), "fn": float(fn), "fp": float(fp), "tn": float(tn),
    }


def sweep(df: pd.DataFrame, base_rate: float | None = None, steps: int = 201):
    """Single-threshold sweep (no review band). Returns thresholds, costs, argmin."""
    ts = np.linspace(0, 1, steps)
    costs = np.array([evaluate(df, t, t, base_rate)["cost_per_1000"] for t in ts])
    # With a small, near-perfect test set many thresholds tie for the minimum. argmin would
    # return the first tie (often ~0.01), which is meaningless. Take the MIDDLE of the longest
    # flat minimum-cost stretch instead: it is the threshold furthest from either cliff.
    ok = costs <= costs.min() + max(1e-9, 0.01 * abs(costs.min()))
    runs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(ok) - 1):
            runs.append((start, i if v else i - 1))
            start = None
    s, e = max(runs, key=lambda r: r[1] - r[0])
    return ts, costs, float((ts[s] + ts[e]) / 2)


def bootstrap(df: pd.DataFrame, base_rate: float | None = None, n: int = 200, seed: int = 0):
    """Resample the scored images with replacement and re-run the sweep each time.

    Returns (best thresholds, minimum costs). The spread shows how much the 'optimal'
    threshold and its cost depend on which few images happened to be in the sample.
    """
    rng = np.random.default_rng(seed)
    ths, mins = [], []
    for _ in range(n):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        if s.label.nunique() < 2:
            continue
        _, c, t = sweep(s, base_rate, steps=101)
        ths.append(t)
        mins.append(float(c.min()))
    return np.array(ths), np.array(mins)


def staffing(review_share: float) -> dict:
    reviews_day = review_share * C.CASES_PER_DAY
    hours = reviews_day * C.REVIEW["seconds"] / 3600
    people = hours / C.SHIFT_HOURS
    return {
        "reviews_per_day": reviews_day,
        "person_hours": hours,
        "people_needed": people,
        "feasible": people <= C.REVIEWERS_AVAILABLE,
    }


def max_review_share() -> float:
    """Largest share of cases the available staff can physically check."""
    capacity = C.REVIEWERS_AVAILABLE * C.SHIFT_HOURS * 3600 / C.REVIEW["seconds"]
    return min(1.0, capacity / C.CASES_PER_DAY)


def best_band(df: pd.DataFrame, base_rate: float | None = None, step: float = 0.02,
              cap: float | None = None):
    """Grid-search (lo, hi) minimising cost, subject to review share <= cap."""
    cap = max_review_share() if cap is None else cap
    grid = np.arange(0, 1 + 1e-9, step)
    best = None
    for lo in grid:
        for hi in grid[grid >= lo]:
            r = evaluate(df, lo, hi, base_rate)
            if r["review_share"] > cap:
                continue
            if best is None or r["cost_per_1000"] < best[2]["cost_per_1000"]:
                best = (float(lo), float(hi), r)
    return best


def strategies(df: pd.DataFrame, base_rate: float | None = None) -> pd.DataFrame:
    """Price the four options the assignment asks for, per 1,000 cases."""
    base = C.BASE_RATE if base_rate is None else base_rate
    c, rv = C.COSTS, C.REVIEW
    a = rv["accuracy"]
    pos_n, neg_n = N * base, N * (1 - base)

    do_nothing = pos_n * c["fn"] + neg_n * c["tn"]
    check_all = (N * rv["cost_per_case"]
                 + pos_n * (a * c["tp"] + (1 - a) * c["fn"])
                 + neg_n * (a * c["tn"] + (1 - a) * c["fp"]))
    _, costs, t_best = sweep(df, base)
    alone = float(costs.min())
    lo, hi, r = best_band(df, base)

    rows = [
        ("Do nothing (ship everything)", do_nothing, 0.0),
        ("Person checks everything", check_all, 1.0),
        (f"Model alone (threshold {t_best:.2f})", alone, 0.0),
        (f"Model + review (lo {lo:.2f}, hi {hi:.2f})", r["cost_per_1000"], r["review_share"]),
    ]
    out = pd.DataFrame(rows, columns=["strategy", "cost_per_1000_eur", "review_share"])
    out["people_needed"] = [staffing(s)["people_needed"] for s in out.review_share]
    return out


def image_metrics(labels, scores, thr: float = 0.5) -> dict:
    """Plain image-level numbers at one threshold (used for the baselines table)."""
    y = np.asarray(labels).astype(int)
    p = (np.asarray(scores) >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return {
        "threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / max(1, len(y)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }
