"""
The business rule for grading boards, shared by train.py and app.py so that the
training evaluation and the dashboard always apply exactly the same logic.

See REPORT.md, "Business problem" and "Dataset".
"""

import numpy as np
from PIL import Image

# Defects that disqualify a board from the premium grade (the positive class).
POSITIVE_CLASSES = {
    "0 Blue_Stain",
    "1 Crack",
    "2 Dead_Knot",
    "3 Knot_missing",
    "5 Marrow",
    "7 knot_with_crack",
    "9 resin",
}
# Defects that are acceptable in the premium grade.
ACCEPTABLE_CLASSES = {"4 Live_Knot", "6 Quartzity"}
# Too rare to learn or evaluate (2 examples in train, none in test).
EXCLUDED_CLASSES = {"8 overgrown"}


def to_landscape(image: Image.Image) -> Image.Image:
    """Turn a portrait image 90° counter-clockwise so the board lies horizontally,
    the orientation the model was trained on. Landscape images are returned unchanged."""
    if image.height > image.width:
        return image.transpose(Image.Transpose.ROTATE_90)
    return image


def board_score(result, positive_ids: set[int] | None) -> float:
    """
    The board's score: the highest confidence among detections of disqualifying
    defects, or 0 if there are none. `result` is one Ultralytics prediction.
    With positive_ids=None every detection counts (used for the COCO baseline,
    whose classes have nothing to do with wood).
    """
    best = 0.0
    for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        if positive_ids is None or int(cls) in positive_ids:
            best = max(best, conf)
    return best


def positive_ids(names: dict[int, str]) -> set[int]:
    """Class ids of the disqualifying defects, given a model's or dataset's id -> name map."""
    return {i for i, n in names.items() if n in POSITIVE_CLASSES}


def decide(score: float, lo: float, hi: float) -> str:
    """The three outcomes for one board."""
    if score < lo:
        return "PREMIUM"
    if score < hi:
        return "REVIEW"
    return "STANDARD"


def share_below(sorted_scores: np.ndarray, t) -> np.ndarray:
    """Share of boards whose score is strictly below each threshold in t."""
    return np.searchsorted(sorted_scores, t, side="left") / len(sorted_scores)


def outcome_shares(pos: np.ndarray, neg: np.ndarray, lo, hi) -> dict:
    """
    For defective boards (pos = their sorted scores) and good boards (neg), the share
    that ends up in each outcome: PREMIUM (score < lo), REVIEW (lo <= score < hi),
    STANDARD (score >= hi).
    """
    pos_lo, pos_hi = share_below(pos, lo), share_below(pos, hi)
    neg_lo, neg_hi = share_below(neg, lo), share_below(neg, hi)
    return {"pos_premium": pos_lo, "pos_review": pos_hi - pos_lo, "pos_standard": 1 - pos_hi,
            "neg_premium": neg_lo, "neg_review": neg_hi - neg_lo, "neg_standard": 1 - neg_hi}


def expected_cost(pos, neg, lo, hi, base_rate: float, cost_fn: float, cost_fp: float,
                  cost_review: float):
    """
    Expected cost per board, and share of boards sent to review, for thresholds lo <= hi.

    A defective board sent to PREMIUM is a false negative (cost_fn); a good board sent to
    STANDARD is a false positive (cost_fp); every board in REVIEW costs the review, and I
    assume the grader then decides correctly. Correct automatic decisions cost 0.

    The model's error rates come from an evaluation split, but they are weighted by the
    production base rate, not by that split's mix. With lo == hi there is no review band:
    the model decides alone.
    """
    s = outcome_shares(pos, neg, lo, hi)
    cost = (base_rate * (s["pos_premium"] * cost_fn + s["pos_review"] * cost_review)
            + (1 - base_rate) * (s["neg_standard"] * cost_fp + s["neg_review"] * cost_review))
    review_share = base_rate * s["pos_review"] + (1 - base_rate) * s["neg_review"]
    return cost, review_share
