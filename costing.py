"""
Turn the model's Poisonous confidence into decisions, euros and review workload.

Shared by train.py (to sweep and pick the thresholds) and app.py (to
recompute everything when the sliders move), so both always agree.

Every ground-truth mushroom gets one number: the highest confidence of any
"Poisonous" box covering it (0 if the model drew no Poisonous box there).
Two thresholds split that confidence into three outcomes:

    conf <  t_low             -> PASS   (treated as edible: it gets eaten)
    t_low <= conf < t_high    -> REVIEW (a person checks it)
    conf >= t_high            -> FAIL   (treated as poisonous: thrown away)

Positive class = Poisonous, so:
    poisonous + FAIL = true positive     poisonous + PASS = false negative
    edible    + PASS = true negative     edible    + FAIL = false positive

With a single threshold (no review band) t_low == t_high.
"""

import json
from pathlib import Path

import numpy as np

COSTS_FILE = Path(__file__).resolve().parent / "costs.json"
COST_KEYS = ("fn", "fp", "tp", "tn", "review", "base_rate",
             "cases_per_day", "review_minutes", "reviewer_hours_per_day")


def load_costs(path: Path = COSTS_FILE) -> dict:
    """Read the euro figures, base rate and workload assumptions from costs.json."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in COST_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{path.name} is missing: {', '.join(missing)}")
    return {k: float(raw[k]) for k in COST_KEYS}


def confusion_counts(conf, poisonous, t_low: float, t_high: float) -> dict:
    """How many mushrooms of each true class pass, go to review, or fail."""
    conf = np.asarray(conf, dtype=float)
    poisonous = np.asarray(poisonous, dtype=bool)
    passed = conf < t_low
    failed = conf >= t_high
    review = ~passed & ~failed
    return {
        "tp": int((failed & poisonous).sum()),
        "fn": int((passed & poisonous).sum()),
        "fp": int((failed & ~poisonous).sum()),
        "tn": int((passed & ~poisonous).sum()),
        "review_poisonous": int((review & poisonous).sum()),
        "review_edible": int((review & ~poisonous).sum()),
    }


def expected_cost(conf, poisonous, t_low: float, t_high: float, costs: dict) -> dict:
    """Expected euros, review share and review workload at the given thresholds.

    Rates are measured within each true class, then reweighted by the assumed
    base rate, so the answer reflects how often poisonous mushrooms really
    turn up rather than the class mix of the split. A reviewed mushroom costs
    the review fee, and the person is assumed to judge it correctly.
    """
    c = confusion_counts(conf, poisonous, t_low, t_high)
    n_pos = max(c["tp"] + c["fn"] + c["review_poisonous"], 1)
    n_neg = max(c["tn"] + c["fp"] + c["review_edible"], 1)
    pi = costs["base_rate"]

    cost_if_poisonous = (c["fn"] * costs["fn"] + c["tp"] * costs["tp"]
                         + c["review_poisonous"] * costs["review"]) / n_pos
    cost_if_edible = (c["fp"] * costs["fp"] + c["tn"] * costs["tn"]
                      + c["review_edible"] * costs["review"]) / n_neg
    review_share = pi * c["review_poisonous"] / n_pos + (1 - pi) * c["review_edible"] / n_neg

    reviews_per_day = review_share * costs["cases_per_day"]
    review_hours = reviews_per_day * costs["review_minutes"] / 60
    return {
        **c,
        "cost_per_1000": 1000 * (pi * cost_if_poisonous + (1 - pi) * cost_if_edible),
        "review_share": review_share,
        "reviews_per_day": reviews_per_day,
        "reviewers_needed": review_hours / costs["reviewer_hours_per_day"],
    }


def sweep(conf, poisonous, costs: dict, thresholds) -> list[dict]:
    """Single-threshold sweep: cost at every t, with no review band."""
    return [
        {"threshold": float(t), **expected_cost(conf, poisonous, t, t, costs)}
        for t in thresholds
    ]


def best_band(conf, poisonous, costs: dict, grid) -> dict:
    """Cheapest (t_low, t_high) pair on the grid, with t_low <= t_high."""
    best = None
    for i, low in enumerate(grid):
        for high in grid[i:]:
            result = expected_cost(conf, poisonous, low, high, costs)
            if best is None or result["cost_per_1000"] < best["cost_per_1000"]:
                best = {"t_low": float(low), "t_high": float(high), **result}
    return best


def policy_costs(conf, poisonous, costs: dict, t_single: float,
                 t_low: float, t_high: float) -> dict:
    """The design priced against the three alternatives the brief asks for."""
    return {
        # No model, no check: every mushroom is eaten.
        "do_nothing": expected_cost(conf, poisonous, np.inf, np.inf, costs),
        # Every mushroom goes to a person.
        "person_checks_all": expected_cost(conf, poisonous, -np.inf, np.inf, costs),
        "model_alone": expected_cost(conf, poisonous, t_single, t_single, costs),
        "model_plus_review": expected_cost(conf, poisonous, t_low, t_high, costs),
    }
