"""
The business side of the assignment, in one place.

Both train.ipynb (to choose the thresholds) and app.ipynb (to recompute
everything when the sliders move) import this file, so there is a single
definition of "what a decision costs". If the two had their own copies, the
dashboard could show a different cost from the one the threshold was chosen on.

Unit of decision: one camera snapshot. Positive class: the snapshot has at least
one head without a hard hat. The model's score for a snapshot is the confidence of
its most confident `head` box (0 if it found none).

Three zones, set by two thresholds:
    score <  low           -> pass (do nothing)
    low <= score < high    -> a person reviews the snapshot
    score >= high          -> alert, the zone supervisor goes to check
low == high is the model deciding alone; low = 0, high > 1 is a person checking everything.
"""

import numpy as np

# ---- scenario (see REPORT.md, "Cost matrix and operating point") ------------------------
CAMERAS = 20
SNAPSHOT_EVERY_MIN = 10
HOURS_PER_DAY = 10
SNAPSHOTS_PER_DAY = CAMERAS * (60 // SNAPSHOT_EVERY_MIN) * HOURS_PER_DAY   # 1,200
BASE_RATE = 0.05            # share of snapshots with someone without a hard hat

# ---- cost of each outcome, in euros per snapshot -----------------------------------------
C_TP = 3.0                  # supervisor walks over and fixes it
C_FP = 3.0                  # supervisor walks over for nothing
C_FN = 25.0                 # someone stays without a helmet until the next snapshot
C_TN = 0.0

# ---- the person in the loop ---------------------------------------------------------------
SECONDS_PER_REVIEW = 20
REVIEWER_EUR_PER_HOUR = 27.0
C_REVIEW = REVIEWER_EUR_PER_HOUR * SECONDS_PER_REVIEW / 3600             # 0.15 EUR
HUMAN_RECALL = 0.90
HUMAN_SPECIFICITY = 0.98
REVIEW_HOURS_AVAILABLE = 1.0   # per day, from the site's safety officer

PER = 1000
NEVER = 1.01                # a threshold no score reaches


def params():
    """Everything above as a dict, so it can be saved next to the results."""
    return {
        "cameras": CAMERAS, "snapshot_every_min": SNAPSHOT_EVERY_MIN, "hours_per_day": HOURS_PER_DAY,
        "snapshots_per_day": SNAPSHOTS_PER_DAY, "base_rate": BASE_RATE,
        "c_tp": C_TP, "c_fp": C_FP, "c_fn": C_FN, "c_tn": C_TN,
        "seconds_per_review": SECONDS_PER_REVIEW, "reviewer_eur_per_hour": REVIEWER_EUR_PER_HOUR,
        "c_review": C_REVIEW, "human_recall": HUMAN_RECALL, "human_specificity": HUMAN_SPECIFICITY,
        "review_hours_available": REVIEW_HOURS_AVAILABLE,
    }


def confusion(truth, score, cut):
    """2x2 table at one threshold. Alert when score >= cut."""
    truth = np.asarray(truth).astype(bool)
    alert = np.asarray(score) >= cut
    tp, fn = int((alert & truth).sum()), int((~alert & truth).sum())
    fp, tn = int((alert & ~truth).sum()), int((~alert & ~truth).sum())
    nan = float("nan")
    return {
        "cut": cut, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else nan,
        "specificity": tn / (tn + fp) if tn + fp else nan,
        "precision": tp / (tp + fp) if tp + fp else nan,
        "accuracy": (tp + tn) / len(truth) if len(truth) else nan,
    }


def auc(truth, score):
    """ROC-AUC from ranks (Mann-Whitney): P(a random positive scores above a random negative)."""
    truth = np.asarray(truth).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = truth.sum(), (~truth).sum()
    if n_pos == 0 or n_neg == 0 or np.unique(score).size == 1:
        return float("nan")
    order = score.argsort()
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for value in np.unique(score):            # ties share their average rank
        tied = score == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def zone_rates(score, low, high):
    """Share of snapshots that pass, go to review, and raise an alert."""
    score = np.asarray(score)
    if score.size == 0:
        return 0.0, 0.0, 0.0
    return float((score < low).mean()), float(((score >= low) & (score < high)).mean()), float((score >= high).mean())


def design_cost(pos_scores, neg_scores, low, high, p=BASE_RATE, c_fn=C_FN, c_fp=C_FP, c_tp=C_TP,
                c_tn=C_TN, c_review=C_REVIEW, hr=HUMAN_RECALL, hs=HUMAN_SPECIFICITY,
                snapshots_per_day=SNAPSHOTS_PER_DAY, seconds_per_review=SECONDS_PER_REVIEW):
    """Expected cost of one design per 1,000 snapshots at base rate p.

    The zone rates are measured separately on positive and negative snapshots, because
    those are properties of the model and do not change with the mix. They are then
    re-weighted with the real base rate p, which is not the one in the dataset.
    A snapshot sent to review gets the person's error rates, at the price of a review.
    """
    a, b, c = zone_rates(pos_scores, low, high)     # positives: pass, review, alert
    d, e, f = zone_rates(neg_scores, low, high)     # negatives: pass, review, alert

    fn = p * (a + b * (1 - hr))                      # missed by the model, or by the person
    tp = p * (c + b * hr)
    fp = (1 - p) * (f + e * (1 - hs))
    tn = (1 - p) * (d + e * hs)
    review = p * b + (1 - p) * e
    cost = fn * c_fn + tp * c_tp + fp * c_fp + tn * c_tn + review * c_review
    alerts_true, alerts_false = p * c, (1 - p) * f
    nan = float("nan")
    return {
        "low": low, "high": high,
        "eur_per_1000": cost * PER,
        "eur_per_day": cost * snapshots_per_day,
        "fn_per_1000": fn * PER, "fp_per_1000": fp * PER, "tp_per_1000": tp * PER,
        "review_share": review,
        "reviews_per_day": review * snapshots_per_day,
        "review_hours_per_day": review * snapshots_per_day * seconds_per_review / 3600,
        "recall_system": tp / (tp + fn) if tp + fn else nan,
        "alert_precision": alerts_true / (alerts_true + alerts_false) if alerts_true + alerts_false else nan,
    }


def split_scores(df, score_col="score"):
    """Positive and negative snapshot scores from a scores table."""
    return (df.loc[df["positive"] == 1, score_col].to_numpy(),
            df.loc[df["positive"] == 0, score_col].to_numpy())
