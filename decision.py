"""Image-level policies. Confidence scores are not calibrated probabilities."""
import numpy as np
import pandas as pd

ACTIONS = ["Pass", "Human review", "Automatic incident"]


def validate_scores(frame):
    if frame.empty or not {0, 1}.issubset(set(frame.positive)):
        raise ValueError("Scores must contain both positive and negative images.")
    if not frame.positive.isin([0, 1]).all():
        raise ValueError("Labels must be binary.")
    if not np.isfinite(frame.anomaly_score).all() or not frame.anomaly_score.between(0, 1).all():
        raise ValueError("Scores must be finite and between zero and one.")


def evaluate(frame, lower, upper, assumptions):
    validate_scores(frame)
    if not 0 <= lower <= upper <= 1.000001:
        raise ValueError("Use 0 <= lower threshold <= upper threshold <= 1.")
    score = frame.anomaly_score.to_numpy()
    action = np.where(score < lower, 0, np.where(score < upper, 1, 2))
    counts = np.zeros((2, 3), dtype=int)
    np.add.at(counts, (frame.positive.to_numpy(dtype=int), action), 1)
    rates = counts / counts.sum(axis=1, keepdims=True)
    p = float(assumptions["production_prevalence"])
    if not 0 < p < 1:
        raise ValueError("Production prevalence must be between zero and one.")
    weighted = rates * np.array([1 - p, p])[:, None]
    review = float(weighted[:, 1].sum())
    automatic = float(weighted[:, 2].sum())
    incidents = automatic + float(weighted[1, 1])
    missed = float(weighted[1, 0])
    review_cost = assumptions["review_seconds"] * assumptions["labour_eur_per_hour"] / 3600
    cost = review * review_cost + incidents * assumptions["incident_cost_eur"] + missed * assumptions["missed_anomaly_cost_eur"]
    return {
        "counts": counts, "rates": rates, "weighted_counts_per_1000": 1000 * weighted,
        "cost_per_1000": 1000 * cost, "review_share": review,
        "automatic_share": automatic, "incident_share": incidents,
        "missed_share": missed, "positive_escalation": 1 - rates[1, 0],
    }


def optimize(frame, assumptions):
    """Exact search over observed score boundaries, using validation only."""
    validate_scores(frame)
    scores = frame.anomaly_score.to_numpy()
    y = frame.positive.to_numpy()
    thresholds = np.unique(np.r_[0.0, scores, np.nextafter(scores.max(), np.inf)])
    positive = np.sort(scores[y == 1])
    negative = np.sort(scores[y == 0])
    tp = 1 - np.searchsorted(positive, thresholds, side="left") / len(positive)
    fp = 1 - np.searchsorted(negative, thresholds, side="left") / len(negative)
    p = assumptions["production_prevalence"]
    cr = assumptions["review_seconds"] * assumptions["labour_eur_per_hour"] / 3600
    ci, cm = assumptions["incident_cost_eur"], assumptions["missed_anomaly_cost_eur"]
    candidates, auto = [], []
    for i, low in enumerate(thresholds):
        review = p * (tp[i] - tp[i:]) + (1 - p) * (fp[i] - fp[i:])
        incidents = p * tp[i] + (1 - p) * fp[i:]
        cost = 1000 * (cr * review + ci * incidents + cm * p * (1 - tp[i]))
        j = int(np.argmin(cost))
        candidates.append((float(cost[j]), float(low), float(thresholds[i + j])))
        auto.append((float(cost[0]), float(low), float(low)))
    best = min(candidates)
    automatic = min(auto)
    return {
        "lower_threshold": best[1], "upper_threshold": best[2],
        "assumptions": dict(assumptions),
        "automatic_lower_threshold": automatic[1],
    }


def comparisons(frame, assumptions, policy):
    rows = []
    never = 1.000001
    automatic_threshold = policy["automatic_lower_threshold"]
    for title, lo, hi in [
        ("Do nothing", never, never),
        ("Review every photograph", 0, never),
        ("Model alone: automatic pass/fail", automatic_threshold, automatic_threshold),
        ("Frozen hybrid policy", policy["lower_threshold"], policy["upper_threshold"]),
    ]:
        r = evaluate(frame, lo, hi, assumptions)
        rows.append({"Policy": title, "Cost / 1,000 (€)": r["cost_per_1000"],
                     "Photo review (%)": 100 * r["review_share"],
                     "Automatic incident (%)": 100 * r["automatic_share"]})
    return pd.DataFrame(rows)
