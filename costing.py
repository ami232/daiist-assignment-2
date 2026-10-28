import numpy as np


def route(scores, low, high):
    scores = np.asarray(scores, dtype=float)
    alerted = scores >= high
    passed = scores < low
    reviewed = ~alerted & ~passed
    return passed, reviewed, alerted


def outcome_rates(scores, is_fall, low, high):
    is_fall = np.asarray(is_fall, dtype=bool)
    passed, reviewed, alerted = route(scores, low, high)
    falls, normal = max(is_fall.sum(), 1), max((~is_fall).sum(), 1)
    return {
        "fall_missed": (passed & is_fall).sum() / falls,
        "fall_reviewed": (reviewed & is_fall).sum() / falls,
        "fall_alerted": (alerted & is_fall).sum() / falls,
        "normal_passed": (passed & ~is_fall).sum() / normal,
        "normal_reviewed": (reviewed & ~is_fall).sum() / normal,
        "normal_alerted": (alerted & ~is_fall).sum() / normal,
    }


def cost_per_1000(scores, is_fall, low, high, costs, base_rate):
    r = outcome_rates(scores, is_fall, low, high)
    p = base_rate
    cost_fall = (
        r["fall_missed"] * costs["fn"]
        + r["fall_alerted"] * costs["tp"]
        + r["fall_reviewed"] * (costs["tp"] + costs["review"])
    )
    cost_normal = (
        r["normal_alerted"] * costs["fp"]
        + r["normal_passed"] * costs["tn"]
        + r["normal_reviewed"] * (costs["tn"] + costs["review"])
    )
    review_share = p * r["fall_reviewed"] + (1 - p) * r["normal_reviewed"]
    return 1000 * (p * cost_fall + (1 - p) * cost_normal), review_share


def confusion_at(scores, is_fall, threshold):
    is_fall = np.asarray(is_fall, dtype=bool)
    flagged = np.asarray(scores, dtype=float) >= threshold
    return {
        "tp": int((flagged & is_fall).sum()),
        "fp": int((flagged & ~is_fall).sum()),
        "fn": int((~flagged & is_fall).sum()),
        "tn": int((~flagged & ~is_fall).sum()),
    }