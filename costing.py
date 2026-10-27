"""
Turns model scores into a costed three-way decision: pass, human review, fail.

One case is one shelf slot (a labelled product or missing box). Its score is
the model's highest `missing` confidence on that slot. A `missing` prediction
that matches no labelled slot is a phantom: it can only ever be a false alarm,
so it adds review or false-positive cost on top of the slots.

Shared by train.py and app.py so both use the same maths.
"""

import numpy as np

import config

PASS, REVIEW, FAIL = 0, 1, 2
OUTCOMES = ["pass", "review", "fail"]
REQUIRED = ["COST_FN", "COST_FP", "COST_TP", "COST_TN", "COST_REVIEW", "BASE_RATE",
            "IMAGES_PER_DAY", "SLOTS_PER_IMAGE", "SECONDS_PER_REVIEW"]


def check_config() -> None:
    missing = [name for name in REQUIRED if getattr(config, name) is None]
    if missing:
        raise SystemExit(f"Fill in config.py first: {', '.join(missing)}")


def route(scores, t_low, t_high):
    scores = np.asarray(scores)
    return np.where(scores >= t_high, FAIL, np.where(scores >= t_low, REVIEW, PASS))


def outcome_shares(scores, labels, t_low, t_high):
    """Share of real gaps, and of filled slots, landing in each outcome."""
    routed = route(scores, t_low, t_high)
    labels = np.asarray(labels)

    def shares(mask):
        if not mask.any():
            return np.zeros(3)
        return np.array([(routed[mask] == k).mean() for k in (PASS, REVIEW, FAIL)])

    return shares(labels == 1), shares(labels == 0)


def phantom_shares(phantoms, n_slots, t_low, t_high):
    """Phantom detections sent to review, and to fail, per real slot."""
    routed = route(phantoms, t_low, t_high)
    return (routed == REVIEW).sum() / n_slots, (routed == FAIL).sum() / n_slots


def cost_per_1000(scores, labels, t_low, t_high, phantoms=(), base_rate=None):
    """Expected euros per 1,000 slots, reweighted to the assumed base rate.

    Assumes the reviewer is always right: a reviewed gap costs a review plus
    a restock, a reviewed full slot or phantom costs only the review.
    """
    pi = config.BASE_RATE if base_rate is None else base_rate
    pos, neg = outcome_shares(scores, labels, t_low, t_high)
    ph_review, ph_fail = phantom_shares(phantoms, len(labels), t_low, t_high)
    pos_cost = pos @ [config.COST_FN, config.COST_REVIEW + config.COST_TP, config.COST_TP]
    neg_cost = neg @ [config.COST_TN, config.COST_REVIEW + config.COST_TN, config.COST_FP]
    ph_cost = ph_review * config.COST_REVIEW + ph_fail * config.COST_FP
    return float(1000 * (pi * pos_cost + (1 - pi) * neg_cost + ph_cost))


def review_share(scores, labels, t_low, t_high, phantoms=(), base_rate=None):
    """Reviews per slot (phantoms included), at the assumed base rate."""
    pi = config.BASE_RATE if base_rate is None else base_rate
    pos, neg = outcome_shares(scores, labels, t_low, t_high)
    ph_review, _ = phantom_shares(phantoms, len(labels), t_low, t_high)
    return float(pi * pos[REVIEW] + (1 - pi) * neg[REVIEW] + ph_review)


def reviews_per_day(share):
    return share * config.IMAGES_PER_DAY * config.SLOTS_PER_IMAGE


def reviewers_needed(share):
    return reviews_per_day(share) * config.SECONDS_PER_REVIEW / config.REVIEWER_SECONDS_PER_DAY


def confusion_table(scores, labels, t_low, t_high, phantoms=()):
    """Counts on the evaluated split. Rows: filled slot / gap / phantom. Columns: pass/review/fail."""
    routed = route(scores, t_low, t_high)
    labels = np.asarray(labels)
    rows = [[int(((labels == y) & (routed == k)).sum()) for k in (PASS, REVIEW, FAIL)]
            for y in (0, 1)]
    ph = route(phantoms, t_low, t_high)
    rows.append([0, int((ph == REVIEW).sum()), int((ph == FAIL).sum())])
    return rows


def binary_metrics(scores, labels, t, phantoms=()):
    scores, labels = np.asarray(scores), np.asarray(labels)
    flagged = scores >= t
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum()) + int((np.asarray(phantoms) >= t).sum())
    fn = int((~flagged & (labels == 1)).sum())
    tn = int((~flagged & (labels == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0}


def sweep(scores, labels, thresholds, phantoms=(), base_rate=None):
    """Cost at each single threshold, with no review band."""
    return [cost_per_1000(scores, labels, t, t, phantoms, base_rate) for t in thresholds]


def best_band(scores, labels, thresholds, phantoms=(), max_reviewers=None, base_rate=None):
    """Cheapest (t_low, t_high) pair, optionally capped by review staff."""
    best = None
    for i, lo in enumerate(thresholds):
        for hi in thresholds[i:]:
            share = review_share(scores, labels, lo, hi, phantoms, base_rate)
            if max_reviewers is not None and reviewers_needed(share) > max_reviewers:
                continue
            cost = cost_per_1000(scores, labels, lo, hi, phantoms, base_rate)
            if best is None or cost < best[0]:
                best = (cost, float(lo), float(hi))
    return best[1], best[2]


def alternatives(scores, labels, t_single, t_low, t_high, phantoms=(), base_rate=None):
    """The design priced against doing nothing, checking everything, and the model alone."""
    inf = float("inf")
    # COST_FN already prices a gap left until the next manual round, so never
    # flagging is today's process, not a worst case.
    options = {
        "Do nothing (manual rounds only)": (inf, inf),
        "Person checks every slot": (0.0, inf),
        "Model decides alone": (t_single, t_single),
        "Model + review band": (t_low, t_high),
    }
    rows = []
    for name, (lo, hi) in options.items():
        # Checking everything means looking at every slot, not at phantom boxes.
        ph = () if name.startswith("Person") else phantoms
        share = review_share(scores, labels, lo, hi, ph, base_rate)
        rows.append({"option": name,
                     "cost_per_1000": round(cost_per_1000(scores, labels, lo, hi, ph, base_rate), 2),
                     "review_share": round(share, 4),
                     "reviews_per_day": round(reviews_per_day(share)),
                     "reviewers": round(reviewers_needed(share), 2)})
    return rows
