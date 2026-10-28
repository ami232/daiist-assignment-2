"""
Cost model for Assignment 2, shared by train.py and app.py.

There is exactly one definition of "what does a mistake cost" and "what does
the escalation band do", so the sweep saved by the training stage and the
numbers the dashboard recomputes when you move a slider can never disagree.

The decision: a greenhouse scout photographs a plant that looks wrong. The
grower has to decide whether that bed gets the FUNGICIDE application today.
Fungal diseases (gray mold, powdery mildew, anthracnose, leaf spot, rust, leaf
mold, blossom blight) need it. Spider mites (a pest) and bacterial angular leaf
spot do not: spraying fungicide on them wastes product and leaves the real
problem spreading for another week.

Positive = the photo shows a fungal disease. The model's score for a photo is
the highest confidence of any detection belonging to a fungal class (0 if none).

Three outcomes from two thresholds, t_low <= t_high:

    score <  t_low            -> PASS    (no fungicide; follow the pest/bacterial protocol)
    t_low <= score < t_high   -> REVIEW  (photo goes to the agronomist)
    score >= t_high           -> FAIL    (fungicide application scheduled for that bed)

Every figure in COST_MODEL is justified in REPORT.md. Change them there and
here together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The assumptions. All money in euros, per single case (= one scouting photo).
# ---------------------------------------------------------------------------
COST_MODEL: dict = {
    "POSITIVE_NAME": "fungal disease present",
    "NEGATIVE_NAME": "not fungal (pest / bacterial)",
    # Cost of each cell of the confusion matrix for the FINAL decision
    # (after the agronomist, if any, has given their verdict).
    "C_FN": 60.0,   # fungal infection not sprayed: ~a week of spread in the bed, ~10 kg fruit at ~4 EUR/kg + removal labour
    "C_FP": 12.0,   # fungicide on a bed that did not need it: ~5 EUR product + 15 min labour + real problem untreated a week
    "C_TP": 9.0,    # fungicide on a bed that needed it: product + labour, money well spent
    "C_TN": 0.0,    # nothing sprayed, nothing paid
    # Agronomist review of one photo in the band.
    "C_REVIEW": 1.50,        # ~2 min of an agronomist at ~45 EUR/h
    "REVIEW_ACCURACY": 0.95,  # share of band cases the agronomist gets right from a photo
    # How often the positive class really occurs in production. Scouts only
    # photograph plants that look wrong, and roughly 4 in 10 of those turn out
    # fungal. The test split does NOT have this base rate (see REPORT.md), so
    # every cost is reweighted.
    "BASE_RATE": 0.40,
    # Volume and capacity, to turn "share sent for review" into people.
    "CASES_PER_DAY": 300,           # 2 scouts x ~150 photos in a 2 ha greenhouse
    "REVIEW_SECONDS": 120,          # time for the agronomist to judge one photo
    "REVIEWER_HOURS_PER_DAY": 1.0,  # review time the agronomist can give per day
    "REVIEWERS_AVAILABLE": 1,       # one agronomist for the site
}


# ---------------------------------------------------------------------------
# Rates from the test-split scores. Everything else is arithmetic on these.
# ---------------------------------------------------------------------------
def split_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    return scores[labels == 1], scores[labels == 0]


def band_fractions(pos: np.ndarray, neg: np.ndarray, t_low: float, t_high: float) -> dict:
    """Share of positives / negatives landing in pass, review and fail."""
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)

    def frac(arr: np.ndarray) -> tuple[float, float, float]:
        if arr.size == 0:
            return 0.0, 0.0, 0.0
        p = float(np.mean(arr < t_low))
        f = float(np.mean(arr >= t_high))
        return p, max(0.0, 1.0 - p - f), f

    pos_pass, pos_rev, pos_fail = frac(pos)
    neg_pass, neg_rev, neg_fail = frac(neg)
    return {
        "pos_pass": pos_pass, "pos_review": pos_rev, "pos_fail": pos_fail,
        "neg_pass": neg_pass, "neg_review": neg_rev, "neg_fail": neg_fail,
    }


def expected_cost_per_case(fr: dict, cm: dict = COST_MODEL) -> dict:
    """Expected cost of one production case under the band `fr`, at BASE_RATE."""
    pi = cm["BASE_RATE"]
    a = cm["REVIEW_ACCURACY"]

    # Final-decision rates after review: review cases are resolved by the person.
    tpr = fr["pos_fail"] + a * fr["pos_review"]           # positives acted on
    fnr = fr["pos_pass"] + (1 - a) * fr["pos_review"]     # positives missed
    fpr = fr["neg_fail"] + (1 - a) * fr["neg_review"]     # negatives acted on
    tnr = fr["neg_pass"] + a * fr["neg_review"]           # negatives left alone

    review_share = pi * fr["pos_review"] + (1 - pi) * fr["neg_review"]
    decision_cost = (
        pi * (tpr * cm["C_TP"] + fnr * cm["C_FN"])
        + (1 - pi) * (fpr * cm["C_FP"] + tnr * cm["C_TN"])
    )
    review_cost = review_share * cm["C_REVIEW"]
    return {
        "cost_per_case": decision_cost + review_cost,
        "cost_per_1000": 1000 * (decision_cost + review_cost),
        "decision_cost_per_1000": 1000 * decision_cost,
        "review_cost_per_1000": 1000 * review_cost,
        "review_share": review_share,
        "fail_share": pi * fr["pos_fail"] + (1 - pi) * fr["neg_fail"],
        "pass_share": pi * fr["pos_pass"] + (1 - pi) * fr["neg_pass"],
        "TPR": tpr, "FNR": fnr, "FPR": fpr, "TNR": tnr,
        # Expected confusion matrix per 1000 production cases, final decision.
        "TP_per_1000": 1000 * pi * tpr,
        "FN_per_1000": 1000 * pi * fnr,
        "FP_per_1000": 1000 * (1 - pi) * fpr,
        "TN_per_1000": 1000 * (1 - pi) * tnr,
    }


def three_way_matrix_per_1000(fr: dict, cm: dict = COST_MODEL) -> pd.DataFrame:
    """Rows: true class. Columns: model outcome. Per 1000 production cases."""
    pi = cm["BASE_RATE"]
    return pd.DataFrame(
        {
            "PASS": [1000 * pi * fr["pos_pass"], 1000 * (1 - pi) * fr["neg_pass"]],
            "REVIEW": [1000 * pi * fr["pos_review"], 1000 * (1 - pi) * fr["neg_review"]],
            "FAIL": [1000 * pi * fr["pos_fail"], 1000 * (1 - pi) * fr["neg_fail"]],
        },
        index=[cm["POSITIVE_NAME"], cm["NEGATIVE_NAME"]],
    ).round(1)


def workload(review_share: float, cm: dict = COST_MODEL) -> dict:
    cases = review_share * cm["CASES_PER_DAY"]
    hours = cases * cm["REVIEW_SECONDS"] / 3600
    people = hours / cm["REVIEWER_HOURS_PER_DAY"]
    return {
        "review_cases_per_day": cases,
        "review_hours_per_day": hours,
        "people_needed": people,
        "capacity_cases_per_day": cm["REVIEWERS_AVAILABLE"] * cm["REVIEWER_HOURS_PER_DAY"] * 3600 / cm["REVIEW_SECONDS"],
        "fits_available_people": people <= cm["REVIEWERS_AVAILABLE"] + 1e-9,
    }


def evaluate_band(scores, labels, t_low: float, t_high: float, cm: dict = COST_MODEL) -> dict:
    pos, neg = split_scores(scores, labels)
    fr = band_fractions(pos, neg, t_low, t_high)
    out = {"t_low": float(min(t_low, t_high)), "t_high": float(max(t_low, t_high))}
    out.update(fr)
    out.update(expected_cost_per_case(fr, cm))
    out.update(workload(out["review_share"], cm))
    return out


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def threshold_grid(n: int = 99) -> np.ndarray:
    return np.round(np.linspace(0.01, 0.99, n), 4)


def sweep_single_threshold(scores, labels, cm: dict = COST_MODEL, grid=None) -> pd.DataFrame:
    """Model decides alone: one threshold, no review band (t_low == t_high)."""
    grid = threshold_grid() if grid is None else grid
    rows = [evaluate_band(scores, labels, t, t, cm) for t in grid]
    df = pd.DataFrame(rows)
    df.insert(0, "threshold", grid)
    return df


def sweep_band(scores, labels, cm: dict = COST_MODEL, grid=None) -> pd.DataFrame:
    """Every (t_low, t_high) pair with t_low <= t_high."""
    grid = np.round(np.linspace(0.02, 0.98, 49), 4) if grid is None else grid
    rows = []
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            rows.append(evaluate_band(scores, labels, lo, hi, cm))
    return pd.DataFrame(rows)


def choose_operating_point(single: pd.DataFrame, band: pd.DataFrame, cm: dict = COST_MODEL) -> dict:
    """Cheapest single threshold; cheapest band overall; cheapest band that
    the available people can actually review."""
    best_single = single.loc[single["cost_per_1000"].idxmin()]
    best_band = band.loc[band["cost_per_1000"].idxmin()]
    feasible = band[band["fits_available_people"]]
    best_feasible = feasible.loc[feasible["cost_per_1000"].idxmin()] if len(feasible) else best_band
    return {
        "model_alone": {"threshold": float(best_single["threshold"]),
                        "cost_per_1000": float(best_single["cost_per_1000"])},
        "band_unconstrained": {"t_low": float(best_band["t_low"]), "t_high": float(best_band["t_high"]),
                               "cost_per_1000": float(best_band["cost_per_1000"]),
                               "review_share": float(best_band["review_share"]),
                               "people_needed": float(best_band["people_needed"])},
        "band_chosen": {"t_low": float(best_feasible["t_low"]), "t_high": float(best_feasible["t_high"]),
                        "cost_per_1000": float(best_feasible["cost_per_1000"]),
                        "review_share": float(best_feasible["review_share"]),
                        "people_needed": float(best_feasible["people_needed"])},
    }


def alternatives(scores, labels, chosen: dict, cm: dict = COST_MODEL) -> pd.DataFrame:
    """Price the design against the three alternatives in the brief."""
    pi = cm["BASE_RATE"]
    a = cm["REVIEW_ACCURACY"]

    rows = []
    # 1. Do nothing: no photo is ever acted on; every fungal case goes unsprayed.
    rows.append({"design": "Do nothing (never spray on the scout's photo)", "cost_per_1000": 1000 * pi * cm["C_FN"],
                 "review_share": 0.0, "people_needed": 0.0})
    # 2. The agronomist checks every photo: review cost on every case, human accuracy.
    cost = cm["C_REVIEW"] + pi * (a * cm["C_TP"] + (1 - a) * cm["C_FN"]) + (1 - pi) * ((1 - a) * cm["C_FP"])
    wl = workload(1.0, cm)
    rows.append({"design": "Agronomist checks every photo", "cost_per_1000": 1000 * cost,
                 "review_share": 1.0, "people_needed": wl["people_needed"]})
    # 3. Model alone at its cheapest threshold.
    t = chosen["model_alone"]["threshold"]
    e = evaluate_band(scores, labels, t, t, cm)
    rows.append({"design": f"Model alone (t = {t:.2f})", "cost_per_1000": e["cost_per_1000"],
                 "review_share": 0.0, "people_needed": 0.0})
    # 4. Model + review band, the chosen design.
    b = chosen["band_chosen"]
    e = evaluate_band(scores, labels, b["t_low"], b["t_high"], cm)
    rows.append({"design": f"Model + review band [{b['t_low']:.2f}, {b['t_high']:.2f})",
                 "cost_per_1000": e["cost_per_1000"], "review_share": e["review_share"],
                 "people_needed": e["people_needed"]})
    df = pd.DataFrame(rows)
    df["cost_per_day"] = df["cost_per_1000"] * cm["CASES_PER_DAY"] / 1000
    return df


def precision_recall_at(scores, labels, t: float) -> dict:
    """Plain case-level precision/recall on the TEST split (its own base rate)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pred = s >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def roc_auc(scores, labels) -> float:
    """Rank-based AUC (probability a random positive outscores a random negative)."""
    pos, neg = split_scores(scores, labels)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[order]
    ranks = np.empty(order.size, dtype=float)
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1   # average rank for ties
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
