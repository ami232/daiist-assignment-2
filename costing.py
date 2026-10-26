"""
From image scores to euros: the cost of each threshold, the review band, and
the people it takes.

Pure numpy, no model involved. train.py calls run() at the end of training and
app.py calls the same functions every time a slider moves. After editing the
business figures in config.py, refresh the saved outputs without retraining:

    uv run python costing.py

Decision rule for one image with score s (its highest positive-class confidence):
    pass    if s <  t_low
    review  if t_low <= s < t_high      (a person decides, assumed correctly)
    fail    if s >= t_high
A single threshold t is the special case t_low = t_high = t (nobody reviews).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

import config as cfg

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


@dataclass(frozen=True)
class Assumptions:
    base_rate: float
    cost_fp: float
    cost_fn: float
    cost_tp: float
    cost_tn: float
    seconds_per_review: float
    reviewer_cost_per_hour: float
    cases_per_day: float
    productive_hours: float
    reviewers_available: float

    @classmethod
    def from_config(cls) -> "Assumptions":
        return cls(
            base_rate=cfg.BASE_RATE,
            cost_fp=cfg.COST_FALSE_POSITIVE,
            cost_fn=cfg.COST_FALSE_NEGATIVE,
            cost_tp=cfg.COST_TRUE_POSITIVE,
            cost_tn=cfg.COST_TRUE_NEGATIVE,
            seconds_per_review=cfg.SECONDS_PER_REVIEW,
            reviewer_cost_per_hour=cfg.REVIEWER_COST_PER_HOUR,
            cases_per_day=cfg.CASES_PER_DAY,
            productive_hours=cfg.PRODUCTIVE_HOURS_PER_REVIEWER,
            reviewers_available=cfg.REVIEWERS_AVAILABLE,
        )

    @property
    def cost_review(self) -> float:
        """Euros of a person's time to check one case."""
        return self.seconds_per_review / 3600 * self.reviewer_cost_per_hour

    @property
    def review_capacity_per_day(self) -> float:
        return self.reviewers_available * self.productive_hours * 3600 / self.seconds_per_review


# ------------------------------------------------------------- core maths --

def share_at_or_above(scores: np.ndarray, thresholds) -> np.ndarray:
    """Fraction of scores >= each threshold (thresholds may be any shape)."""
    s = np.sort(np.asarray(scores, dtype=float))
    t = np.asarray(thresholds, dtype=float)
    if len(s) == 0:
        return np.zeros_like(t)
    return 1.0 - np.searchsorted(s, t, side="left") / len(s)


def outcome_rates(labels, scores, t_low, t_high) -> dict:
    """Of the positive and the negative images, the share sent to pass, review and fail."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    pos_lo, pos_hi = share_at_or_above(pos, t_low), share_at_or_above(pos, t_high)
    neg_lo, neg_hi = share_at_or_above(neg, t_low), share_at_or_above(neg, t_high)
    return {
        "pos_pass": 1 - pos_lo, "pos_review": pos_lo - pos_hi, "pos_fail": pos_hi,
        "neg_pass": 1 - neg_lo, "neg_review": neg_lo - neg_hi, "neg_fail": neg_hi,
    }


def cost_per_case(rates: dict, a: Assumptions):
    """Expected euros per case at the assumed base rate, not the test set's own mix.

    Rates are measured separately on positive and negative images, then weighted
    by how often each occurs in production. A reviewed case costs the person's
    time plus the cost of the correct outcome.
    """
    positive = (rates["pos_fail"] * a.cost_tp + rates["pos_pass"] * a.cost_fn
                + rates["pos_review"] * (a.cost_review + a.cost_tp))
    negative = (rates["neg_fail"] * a.cost_fp + rates["neg_pass"] * a.cost_tn
                + rates["neg_review"] * (a.cost_review + a.cost_tn))
    return a.base_rate * positive + (1 - a.base_rate) * negative


def review_share(rates: dict, a: Assumptions):
    return a.base_rate * rates["pos_review"] + (1 - a.base_rate) * rates["neg_review"]


def staffing(share: float, a: Assumptions) -> dict:
    reviews = share * a.cases_per_day
    hours = reviews * a.seconds_per_review / 3600
    people = hours / a.productive_hours
    return {"reviews_per_day": reviews, "review_hours_per_day": hours,
            "people_needed": people, "people_available": a.reviewers_available,
            "fits_team": bool(people <= a.reviewers_available + 1e-9)}


def decision_counts(labels, scores, t_low, t_high) -> dict:
    """Raw counts on the evaluated split: rows = truth, columns = decision."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    decision = np.where(scores >= t_high, "fail", np.where(scores >= t_low, "review", "pass"))
    return {truth: {d: int(((labels == flag) & (decision == d)).sum()) for d in ("pass", "review", "fail")}
            for truth, flag in (("positive", 1), ("negative", 0))}


# ------------------------------------------------------------ the choices --

def sweep(labels, scores, a: Assumptions) -> dict:
    """Model decides alone (no review) at every threshold on the grid."""
    rates = outcome_rates(labels, scores, GRID, GRID)
    recall, false_alarms = rates["pos_fail"], rates["neg_fail"]
    flagged = a.base_rate * recall + (1 - a.base_rate) * false_alarms
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, a.base_rate * recall / flagged, np.nan)
    return {"threshold": GRID, "cost_per_1000": 1000 * cost_per_case(rates, a),
            "recall": recall, "false_alarm_rate": false_alarms,
            "precision_at_base_rate": precision}


def best_threshold(labels, scores, a: Assumptions) -> tuple[float, float]:
    s = sweep(labels, scores, a)
    i = int(np.argmin(s["cost_per_1000"]))
    return float(GRID[i]), float(s["cost_per_1000"][i])


def best_band(labels, scores, a: Assumptions, respect_capacity: bool = True) -> tuple[float, float, float]:
    """Cheapest (t_low, t_high) pair on the grid; optionally within the review team's capacity."""
    low, high = np.meshgrid(GRID, GRID, indexing="ij")
    rates = outcome_rates(labels, scores, low, high)
    cost = cost_per_case(rates, a)
    share = review_share(rates, a)
    allowed = high >= low
    if respect_capacity:
        allowed &= share * a.cases_per_day <= a.review_capacity_per_day + 1e-9
    objective = np.where(allowed, cost + 1e-9 * share, np.inf)  # ties: less review wins
    i, j = np.unravel_index(int(np.argmin(objective)), objective.shape)
    return float(GRID[i]), float(GRID[j]), float(1000 * cost[i, j])


def evaluate_band(labels, scores, a: Assumptions, t_low: float, t_high: float) -> dict:
    rates = outcome_rates(labels, scores, t_low, t_high)
    share = float(review_share(rates, a))
    cost = float(cost_per_case(rates, a))
    return {
        "cost_per_1000": 1000 * cost,
        "cost_per_day": cost * a.cases_per_day,
        "review_share_pct": 100 * share,
        "positives_missed_pct": 100 * float(rates["pos_pass"]),
        "negatives_flagged_pct": 100 * float(rates["neg_fail"]),
        "counts": decision_counts(labels, scores, t_low, t_high),
        "staffing": staffing(share, a),
    }


def strategy_table(labels, scores, a: Assumptions, t_single: float, band: tuple[float, float]) -> list[dict]:
    """The four ways to run the decision, priced on the same images."""
    rows = []
    for name, low, high in (
        ("Do nothing: every case passes", np.inf, np.inf),
        ("A person checks every case", -np.inf, np.inf),
        (f"Model decides alone (fail at >= {t_single:.2f})", t_single, t_single),
        (f"Model + person (pass < {band[0]:.2f}, fail >= {band[1]:.2f})", band[0], band[1]),
    ):
        result = evaluate_band(labels, scores, a, low, high)
        rows.append({
            "strategy": name,
            "cost_per_1000": result["cost_per_1000"],
            "cost_per_day": result["cost_per_day"],
            "review_share_pct": result["review_share_pct"],
            "people_needed": result["staffing"]["people_needed"],
            "fits_team": result["staffing"]["fits_team"],
            "positives_missed_pct": result["positives_missed_pct"],
            "negatives_flagged_pct": result["negatives_flagged_pct"],
        })
    return rows


def sensitivity(val: tuple, test: tuple, a: Assumptions,
                factors=(0.25, 0.5, 1.0, 2.0, 4.0)) -> list[dict]:
    """Re-choose the operating point when the base rate or the miss cost is off by a factor."""
    rows = []
    for fb in factors:
        for ffn in factors:
            b = replace(a, base_rate=min(a.base_rate * fb, 1.0), cost_fn=a.cost_fn * ffn)
            t, _ = best_threshold(*val, b)
            low, high, _ = best_band(*val, b)
            result = evaluate_band(*test, b, low, high)
            rows.append({
                "base_rate": b.base_rate, "cost_fn": b.cost_fn,
                "best_single_threshold": t, "pass_below": low, "fail_at_or_above": high,
                "review_share_pct": result["review_share_pct"],
                "cost_per_1000": result["cost_per_1000"],
                "do_nothing_per_1000": 1000 * (b.base_rate * b.cost_fn + (1 - b.base_rate) * b.cost_tn),
            })
    return rows


# --------------------------------------------------------------- outputs --

def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), 6) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def load_scores(outputs: Path = OUTPUTS) -> dict:
    data = json.loads((outputs / "image_scores.json").read_text())
    return {model: {split: (np.array(v["labels"], dtype=int), np.array(v["scores"], dtype=float))
                    for split, v in splits.items()}
            for model, splits in data["models"].items()}


def plot_cost_curve(path: Path, curves: dict, a: Assumptions, t_single: float, band: tuple) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nothing = 1000 * (a.base_rate * a.cost_fn + (1 - a.base_rate) * a.cost_tn)
    everyone = 1000 * (a.cost_review + a.base_rate * a.cost_tp + (1 - a.base_rate) * a.cost_tn)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axvspan(band[0], band[1], color="tab:orange", alpha=0.12, label="Review band (chosen)")
    ax.plot(GRID, curves["finetuned_test"], lw=2.2, color="tab:blue", label="Finetuned, test")
    ax.plot(GRID, curves["finetuned_val"], lw=1.2, ls="--", color="tab:blue", label="Finetuned, validation (used to choose)")
    ax.plot(GRID, curves["coco_zero_shot_test"], lw=1.2, color="tab:green", label="COCO zero-shot, test")
    ax.axhline(nothing, color="grey", ls=":", label="Do nothing")
    ax.axhline(everyone, color="grey", ls="-.", label="Person checks everything")
    ax.set_ylim(bottom=0)
    ax.axvline(t_single, color="black", lw=1)
    ax.annotate(f"chosen t = {t_single:.2f}", (t_single, 0), xytext=(4, 6),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("Confidence threshold: fail at or above")
    ax.set_ylabel("Expected cost per 1,000 cases (EUR)")
    ax.set_title(f"Cost against threshold at a {a.base_rate:.1%} base rate")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(outputs: Path = OUTPUTS, verbose: bool = True) -> dict:
    """Choose the operating point on validation, report it on test, save everything."""
    scores = load_scores(outputs)
    a = Assumptions.from_config()
    val, test = scores["finetuned"]["val"], scores["finetuned"]["test"]

    t_single, _ = best_threshold(*val, a)
    low, high, _ = best_band(*val, a)

    curves = {"finetuned_val": sweep(*val, a)["cost_per_1000"]}
    test_sweep = sweep(*test, a)
    curves["finetuned_test"] = test_sweep["cost_per_1000"]
    for model in ("coco_zero_shot", "predict_nothing"):
        curves[f"{model}_test"] = sweep(*scores[model]["test"], a)["cost_per_1000"]

    write_csv(outputs / "cost_sweep.csv", [
        {"threshold": float(t), **{k: float(v[i]) for k, v in curves.items()},
         "finetuned_test_recall": float(test_sweep["recall"][i]),
         "finetuned_test_false_alarm_rate": float(test_sweep["false_alarm_rate"][i]),
         "finetuned_test_precision_at_base_rate": float(test_sweep["precision_at_base_rate"][i])}
        for i, t in enumerate(GRID)])

    strategies = strategy_table(*test, a, t_single, (low, high))
    write_csv(outputs / "strategy_comparison.csv", strategies)
    sens = sensitivity(val, test, a)
    write_csv(outputs / "sensitivity.csv", sens)
    plot_cost_curve(outputs / "cost_curve.png", curves, a, t_single, (low, high))

    operating_point = {
        "placeholders": cfg.PLACEHOLDERS,
        "assumptions": {**asdict(a), "cost_review": a.cost_review,
                        "review_capacity_per_day": a.review_capacity_per_day},
        "chosen_on": "validation split",
        "reported_on": "test split",
        "model_alone": {"threshold": t_single, "test": evaluate_band(*test, a, t_single, t_single)},
        "with_review": {"pass_below": low, "fail_at_or_above": high,
                        "test": evaluate_band(*test, a, low, high)},
        "strategies": strategies,
    }
    save_json(outputs / "operating_point.json", operating_point)

    if verbose:
        print(f"\nCost of being wrong at a {a.base_rate:.1%} base rate "
              f"(FP {a.cost_fp:g} EUR, FN {a.cost_fn:g} EUR, review {a.cost_review:.2f} EUR/case)")
        print(f"  model alone: fail at >= {t_single:.2f}  |  with review: pass < {low:.2f}, fail >= {high:.2f}")
        for row in strategies:
            print(f"  {row['strategy']:<48} {row['cost_per_1000']:>10.2f} EUR per 1,000"
                  f"   review {row['review_share_pct']:5.1f}%   people {row['people_needed']:.2f}")
        if cfg.PLACEHOLDERS:
            print("  WARNING: config.PLACEHOLDERS is True, so these figures are placeholders.")
    return operating_point


if __name__ == "__main__":
    run()
