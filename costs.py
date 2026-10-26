"""
Assignment 2: the cost of the model's decisions, shared by train.py and app.py.

Ripe is the positive class. Each labelled test tomato gets one score: the
model's highest "ripe" confidence on a box that overlaps it (IoU at least
IOU_MATCH), or 0 if no ripe box does. Two thresholds then split tomatoes three
ways: at or above `upper` the model calls it ripe, below `lower` it calls it
not ripe, and in between a person checks it. With lower == upper nothing goes
to a person.

Costs are per real tomato, so boxes that match no labelled tomato are not
priced. A person who checks a tomato is assumed to get it right.
"""

import numpy as np
import pandas as pd

from evaluation import BOX, IOU_MATCH, box_iou

# The cost matrix and volumes argued for in REPORT.md.
POSITIVE_CLASS = "ripe"
FN_COST = 0.30  # euros: a ripe tomato not called ripe is shipped far and spoils
FP_COST = 0.15  # euros: a half-ripe or unripe tomato sold locally as ripe
BASE_RATE = 0.35  # share of tomatoes that are ripe, as in the test set
TOMATOES_PER_DAY = 100_000
REVIEW_SECONDS = 3
REVIEWER_EUR_PER_HOUR = 13
SHIFT_HOURS = 8

REVIEW_COST = REVIEW_SECONDS / 3600 * REVIEWER_EUR_PER_HOUR  # euros per tomato checked
THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)
OUTCOMES = ["called ripe", "sent to a person", "called not ripe"]


def tomato_scores(preds: pd.DataFrame, truths: pd.DataFrame, positive_cls: int) -> pd.DataFrame:
    """One row per labelled tomato: whether it is ripe, and its ripe score."""
    ripe_boxes = dict(tuple(preds[preds["cls"] == positive_cls].groupby("image")))
    scores = pd.Series(0.0, index=truths.index)
    for image, tomatoes in truths.groupby("image"):
        boxes = ripe_boxes.get(image)
        if boxes is None:
            continue
        iou = box_iou(tomatoes[BOX].to_numpy(), boxes[BOX].to_numpy())
        overlapping = np.where(iou >= IOU_MATCH, boxes["conf"].to_numpy()[None, :], 0.0)
        scores.loc[tomatoes.index] = overlapping.max(axis=1)
    return truths[["image", "cls"]].assign(positive=truths["cls"] == positive_cls, score=scores)


def rates(tomatoes: pd.DataFrame, lower, upper) -> dict:
    """Shares of ripe and not-ripe tomatoes on each side of the two thresholds.

    fn: ripe tomatoes called not ripe. fp: not-ripe tomatoes called ripe.
    review_pos and review_neg: ripe and not-ripe tomatoes sent to a person.
    lower and upper may be arrays of the same shape, for sweeping.
    """
    positive = tomatoes["positive"].to_numpy(dtype=bool)
    scores = tomatoes["score"].to_numpy()
    pos, neg = np.sort(scores[positive]), np.sort(scores[~positive])

    def below(sorted_scores, threshold):
        return np.searchsorted(sorted_scores, threshold, side="left") / len(sorted_scores)

    return {
        "fn": below(pos, lower),
        "fp": 1 - below(neg, upper),
        "review_pos": below(pos, upper) - below(pos, lower),
        "review_neg": below(neg, upper) - below(neg, lower),
    }


def review_share(r: dict):
    """Share of all tomatoes, at BASE_RATE, that a person checks."""
    return BASE_RATE * r["review_pos"] + (1 - BASE_RATE) * r["review_neg"]


def expected_cost(r: dict):
    """Expected euros per tomato at BASE_RATE: the model's mistakes plus the checking."""
    mistakes = BASE_RATE * r["fn"] * FN_COST + (1 - BASE_RATE) * r["fp"] * FP_COST
    return mistakes + review_share(r) * REVIEW_COST


def best_thresholds(tomatoes: pd.DataFrame) -> tuple[float, float]:
    """The (lower, upper) pair on the THRESHOLDS grid with the lowest expected cost."""
    lower, upper = np.meshgrid(THRESHOLDS, THRESHOLDS, indexing="ij")
    cost = expected_cost(rates(tomatoes, lower, upper))
    cost[lower > upper] = np.inf
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return float(THRESHOLDS[i]), float(THRESHOLDS[j])


def reviewers_needed(share: float) -> float:
    """People on SHIFT_HOURS shifts needed to check `share` of TOMATOES_PER_DAY."""
    return share * TOMATOES_PER_DAY * REVIEW_SECONDS / 3600 / SHIFT_HOURS


def decision_table(tomatoes: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Test tomatoes counted by true class and by what happens to them."""
    outcome = np.select(
        [tomatoes["score"] >= upper, tomatoes["score"] < lower], [OUTCOMES[0], OUTCOMES[2]], OUTCOMES[1]
    )
    truth = np.where(tomatoes["positive"], "ripe", "not ripe")
    table = pd.crosstab(pd.Series(truth, name="true"), pd.Series(outcome, name="outcome"))
    return table.reindex(index=["ripe", "not ripe"], columns=OUTCOMES, fill_value=0)
