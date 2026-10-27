"""
Cost model for the weld-inspection decision.

Every euro figure here is defended in REPORT.md ("Cost matrix and operating
point" and "What a person still has to check"). If you change a number, change
it there too.

One CASE is one weld seam. The positive class is a DEFECTIVE weld. Each case
gets a defect score between 0 and 1 (how confident the model is that it is
defective) and one of three decisions:

    score <  t_low             -> PASS            (bracket carries on)
    t_low <= score < t_high    -> HUMAN REVIEW    (an inspector looks at the image)
    score >= t_high            -> FAIL            (bracket goes to rework)

A single threshold (no review band) is the special case t_low == t_high. That is
"the model decides alone".

The dataset is enriched with defects compared with a real line, so nothing here
uses the dataset's own defect share. We measure the model's rates on the
labelled welds (true-positive rate, false-positive rate) and then re-weight them
with the ASSUMED production base rate.

This file imports only numpy and pandas, so the dashboard can use it without
loading the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    # Euro cost per case, by outcome (see REPORT.md section "Cost matrix").
    fp_cost: float = 2.0      # good weld flagged: pull, re-check, sometimes regrind
    tp_cost: float = 6.0      # defect caught: handling + grind out + re-weld
    fn_cost: float = 140.0    # defect ships: expected cost of an escape
    tn_cost: float = 0.0      # good weld passed: nothing happens

    # How often a weld really is defective on the line (NOT the dataset's share).
    base_rate: float = 0.01

    # Human review.
    review_seconds: float = 20.0          # one stored image, one good/bad call
    labour_eur_per_hour: float = 25.0     # INE, cost per effective hour, Q1 2026
    reviewer_sensitivity: float = 0.85    # share of real defects a reviewer catches
    reviewer_false_alarm: float = 0.03    # share of good welds a reviewer wrongly fails
    productive_hours_per_reviewer: float = 6.0   # of an 8 h shift

    # Volume.
    welds_per_day: int = 9000

    @property
    def review_cost(self) -> float:
        """Euro cost of one review: seconds x hourly labour cost."""
        return self.review_seconds * self.labour_eur_per_hour / 3600.0


# A small team is realistic; a team of eight is not. The chosen operating point
# is the cheapest one that needs no more than this many reviewers.
MAX_REVIEWERS = 2.0

# Candidate thresholds. 0.05 is the lowest score the pipeline keeps (MIN_CONF).
# 1.00 means "never flag anything": it lets a useless model fall back to doing nothing.
THRESHOLDS = np.round(np.arange(0.05, 1.001, 0.01), 2)


# --------------------------------------------------------------------------
# From a table of cases to the three score arrays
# --------------------------------------------------------------------------

def split_scores(cases: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (defective welds, good welds, spurious detections) as score arrays.

    cases has one row per labelled weld (kind == "weld") and one row per
    detection that matches no labelled weld (kind == "spurious"). A spurious
    detection is a false alarm on the background: it still costs a pull.
    """
    welds = cases[cases["kind"] == "weld"]
    pos = welds.loc[welds["is_positive"] == 1, "score"].to_numpy(dtype=float)
    neg = welds.loc[welds["is_positive"] == 0, "score"].to_numpy(dtype=float)
    spurious = cases.loc[cases["kind"] == "spurious", "score"].to_numpy(dtype=float)
    return pos, neg, spurious


# --------------------------------------------------------------------------
# Counting and costing one operating point
# --------------------------------------------------------------------------

def outcome_counts(pos, neg, spurious, t_low: float, t_high: float) -> dict:
    """Counts of labelled welds by decision. Spurious detections are separate."""
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    pos_flag = int((pos >= t_low).sum())
    pos_fail = int((pos >= t_high).sum())
    neg_flag = int((neg >= t_low).sum())
    neg_fail = int((neg >= t_high).sum())
    sp_flag = int((spurious >= t_low).sum())
    sp_fail = int((spurious >= t_high).sum())
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "pos_pass": len(pos) - pos_flag,
        "pos_review": pos_flag - pos_fail,
        "pos_fail": pos_fail,
        "neg_pass": len(neg) - neg_flag,
        "neg_review": neg_flag - neg_fail,
        "neg_fail": neg_fail,
        "spurious_review": sp_flag - sp_fail,
        "spurious_fail": sp_fail,
    }


def evaluate(pos, neg, spurious, t_low: float, t_high: float, a: Assumptions) -> dict:
    """Expected cost, review load and final outcomes for one pair of thresholds.

    Rates are measured on the labelled welds, then weighted by the assumed
    production base rate b:

        cost per case = b * cost per defective weld + (1 - b) * cost per good weld
    """
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    c = outcome_counts(pos, neg, spurious, t_low, t_high)
    P, N = max(c["n_pos"], 1), max(c["n_neg"], 1)
    b, s, fa, rc = a.base_rate, a.reviewer_sensitivity, a.reviewer_false_alarm, a.review_cost

    # Share of defective welds going to each decision.
    tpr_fail, tpr_rev, tpr_pass = c["pos_fail"] / P, c["pos_review"] / P, c["pos_pass"] / P
    # Share of good welds going to each decision. Spurious detections are added to
    # the flagged shares because each one triggers the same action.
    fpr_fail = (c["neg_fail"] + c["spurious_fail"]) / N
    fpr_rev = (c["neg_review"] + c["spurious_review"]) / N
    tn_pass = c["neg_pass"] / N
    fpr_rev_real = c["neg_review"] / N      # real good welds sent to review (no spurious boxes)

    # A reviewed defective weld is caught with probability s (then it is reworked:
    # TP cost) and missed otherwise (it ships: FN cost). A reviewed good weld is
    # wrongly failed with probability fa (FP cost).
    cost_defective = (
        tpr_fail * a.tp_cost
        + tpr_rev * (rc + s * a.tp_cost + (1 - s) * a.fn_cost)
        + tpr_pass * a.fn_cost
    )
    cost_good = (
        fpr_fail * a.fp_cost
        + fpr_rev * (rc + fa * a.fp_cost + (1 - fa) * a.tn_cost)
        + tn_pass * a.tn_cost
    )
    cost_per_case = b * cost_defective + (1 - b) * cost_good

    review_share = b * tpr_rev + (1 - b) * fpr_rev
    reviews_per_day = review_share * a.welds_per_day
    reviewers = reviews_per_day * a.review_seconds / 3600.0 / a.productive_hours_per_reviewer

    # Final outcome per 1,000 production welds, after any human review.
    per_1000 = {
        "TP": 1000 * b * (tpr_fail + tpr_rev * s),
        "FN": 1000 * b * (tpr_pass + tpr_rev * (1 - s)),
        "FP": 1000 * (1 - b) * (fpr_fail + fpr_rev * fa),
        # Only REAL good welds count as true negatives. Spurious boxes that a reviewer
        # clears are not extra good welds, so they are left out here. They still cost
        # review time, and spurious boxes that get failed count as false positives.
        "TN": 1000 * (1 - b) * (tn_pass + fpr_rev_real * (1 - fa)),
    }
    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "counts": c,
        "tpr_fail": tpr_fail, "tpr_review": tpr_rev, "tpr_pass": tpr_pass,
        "fpr_fail": fpr_fail, "fpr_review": fpr_rev,
        "cost_per_case": cost_per_case,
        "cost_per_1000": 1000 * cost_per_case,
        "cost_per_day": cost_per_case * a.welds_per_day,
        "review_share": review_share,
        "reviews_per_day": reviews_per_day,
        "reviewers": reviewers,
        "per_1000": per_1000,
    }


# --------------------------------------------------------------------------
# Reference options that need no model
# --------------------------------------------------------------------------

def cost_do_nothing(a: Assumptions) -> dict:
    """Nobody looks. Every defect ships."""
    cpc = a.base_rate * a.fn_cost + (1 - a.base_rate) * a.tn_cost
    return {"cost_per_case": cpc, "cost_per_1000": 1000 * cpc,
            "cost_per_day": cpc * a.welds_per_day, "review_share": 0.0, "reviewers": 0.0}


def cost_human_everything(a: Assumptions) -> dict:
    """A person looks at every weld."""
    b, s, fa = a.base_rate, a.reviewer_sensitivity, a.reviewer_false_alarm
    cpc = (
        a.review_cost
        + b * (s * a.tp_cost + (1 - s) * a.fn_cost)
        + (1 - b) * (fa * a.fp_cost + (1 - fa) * a.tn_cost)
    )
    reviewers = a.welds_per_day * a.review_seconds / 3600.0 / a.productive_hours_per_reviewer
    return {"cost_per_case": cpc, "cost_per_1000": 1000 * cpc,
            "cost_per_day": cpc * a.welds_per_day, "review_share": 1.0, "reviewers": reviewers}


def cost_flag_everything(a: Assumptions) -> dict:
    """Every weld is pulled for rework without looking. Shown only for context."""
    b = a.base_rate
    cpc = b * a.tp_cost + (1 - b) * a.fp_cost
    return {"cost_per_case": cpc, "cost_per_1000": 1000 * cpc,
            "cost_per_day": cpc * a.welds_per_day, "review_share": 0.0, "reviewers": 0.0}


def ideal_posterior_threshold(a: Assumptions) -> float:
    """For a perfectly calibrated model: flag when P(defective) exceeds this."""
    return a.fp_cost / (a.fn_cost - a.tp_cost + a.fp_cost)


# --------------------------------------------------------------------------
# Sweeps and searches
# --------------------------------------------------------------------------

def sweep_single(pos, neg, spurious, a: Assumptions, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Cost against threshold for the model deciding alone (no review band)."""
    rows = []
    for t in thresholds:
        e = evaluate(pos, neg, spurious, t, t, a)
        rows.append({
            "threshold": float(t),
            "cost_per_1000": e["cost_per_1000"],
            "tpr": e["tpr_fail"],
            "fpr": e["fpr_fail"],
            "FN_per_1000": e["per_1000"]["FN"],
            "FP_per_1000": e["per_1000"]["FP"],
        })
    return pd.DataFrame(rows)


def best_single(pos, neg, spurious, a: Assumptions, thresholds=THRESHOLDS) -> dict:
    """The single threshold with the lowest expected cost."""
    sweep = sweep_single(pos, neg, spurious, a, thresholds)
    t = float(sweep.loc[sweep["cost_per_1000"].idxmin(), "threshold"])
    return {"threshold": t, "sweep": sweep}


def best_band(pos, neg, spurious, a: Assumptions, max_reviewers: float | None = None,
              thresholds=THRESHOLDS) -> dict:
    """The (t_low, t_high) pair with the lowest expected cost.

    If max_reviewers is given, pairs that would need more reviewers than that
    are skipped. t_low == t_high (no review band) is always allowed, so a
    solution always exists.
    """
    best = None
    for i, lo in enumerate(thresholds):
        for hi in thresholds[i:]:
            e = evaluate(pos, neg, spurious, lo, hi, a)
            if max_reviewers is not None and e["reviewers"] > max_reviewers:
                continue
            if best is None or e["cost_per_case"] < best["cost_per_case"] - 1e-12:
                best = e
    return best


def sensitivity_table(val_cases: pd.DataFrame, test_cases: pd.DataFrame, base: Assumptions,
                      fn_costs, base_rates, max_reviewers: float | None = MAX_REVIEWERS) -> pd.DataFrame:
    """How the answer moves when the two shakiest assumptions move.

    For every (false-negative cost, base rate) pair: choose the thresholds on the
    validation cases, then report what they cost on the test cases.
    """
    vp, vn, vs = split_scores(val_cases)
    tp_, tn_, ts_ = split_scores(test_cases)
    rows = []
    for fn in fn_costs:
        for br in base_rates:
            a = replace(base, fn_cost=float(fn), base_rate=float(br))
            t = best_single(vp, vn, vs, a)["threshold"]
            band = best_band(vp, vn, vs, a, max_reviewers)
            alone = evaluate(tp_, tn_, ts_, t, t, a)
            both = evaluate(tp_, tn_, ts_, band["t_low"], band["t_high"], a)
            rows.append({
                "fn_cost_eur": fn,
                "base_rate_pct": br * 100,
                "do_nothing": cost_do_nothing(a)["cost_per_1000"],
                "human_everything": cost_human_everything(a)["cost_per_1000"],
                "model_alone": alone["cost_per_1000"],
                "model_alone_threshold": t,
                "model_plus_review": both["cost_per_1000"],
                "band_low": band["t_low"],
                "band_high": band["t_high"],
                "review_share_pct": both["review_share"] * 100,
                "reviewers": both["reviewers"],
            })
    return pd.DataFrame(rows)


def assumptions_to_dict(a: Assumptions) -> dict:
    d = asdict(a)
    d["review_cost"] = a.review_cost
    return d
