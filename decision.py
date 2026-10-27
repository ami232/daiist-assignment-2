"""
Turns model scores into pass / human review / send-a-truck decisions and puts a
euro cost on them. Used by train.py (to pick the thresholds) and by app.py (to
recompute everything when the sliders move).

Run it on its own to redo the costing after editing assumptions.json, without
retraining:

    uv run python decision.py

The idea in one paragraph: each image gets a score, the model's probability
that the road needs treatment (Snow + Slush). Below the low threshold we do
nothing, above the high threshold we send a truck, in between an operator looks
at the image and makes the right call. We measure on the validation split how
often snowy and clear roads land in each of the three outcomes, then weight
those shares by how often snowy roads really happen (the base rate), because
the dataset has far more snow than real life.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSUMPTIONS_FILE = ROOT / "assumptions.json"

# Thresholds we try: 0.00, 0.01, ..., 1.00
GRID = np.round(np.linspace(0, 1, 101), 2)
NEVER = np.inf  # a threshold no score reaches


def load_assumptions() -> dict:
    return json.loads(ASSUMPTIONS_FILE.read_text())


def review_cost(a: dict) -> float:
    """Euros of operator time to look at one image."""
    return a["review_seconds"] / 3600 * a["operator_eur_per_hour"]


def review_capacity_per_day(a: dict) -> float:
    """How many images the operators can check in a day."""
    return a["operators"] * a["operator_hours_per_day"] * 3600 / a["review_seconds"]


def outcome_shares(scores, y, low, high):
    """Share of truly snowy (y=1) and truly fine (y=0) images that end up in
    each outcome: [pass, review, truck]."""
    scores, y = np.asarray(scores), np.asarray(y)

    def shares(s):
        return np.array([(s < low).mean(), ((s >= low) & (s < high)).mean(), (s >= high).mean()])

    return shares(scores[y == 1]), shares(scores[y == 0])


def evaluate(scores, y, low, high, a: dict) -> dict:
    """Expected cost per 1,000 real images, and the review load, for one pair
    of thresholds. low == high means the model decides alone."""
    pos, neg = outcome_shares(scores, y, low, high)
    truck, missed, nothing = (
        a["cost_truck_run_eur"],
        a["cost_missed_snow_eur"],
        a["cost_nothing_needed_eur"],
    )
    r = review_cost(a)
    # A reviewed image costs the operator's time, then the right action is taken.
    cost_if_snowy = pos[0] * missed + pos[1] * (r + truck) + pos[2] * truck
    cost_if_fine = neg[0] * nothing + neg[1] * (r + nothing) + neg[2] * truck
    p = a["base_rate"]
    review_share = p * pos[1] + (1 - p) * neg[1]
    reviews_per_day = review_share * a["images_per_day"]
    return {
        "low": float(low),
        "high": float(high),
        "cost_per_1000": 1000 * (p * cost_if_snowy + (1 - p) * cost_if_fine),
        "review_share": review_share,
        "reviews_per_day": reviews_per_day,
        "operators_needed": reviews_per_day * a["review_seconds"] / 3600 / a["operator_hours_per_day"],
        "missed_snowy_per_1000": 1000 * p * pos[0],
        "needless_trucks_per_1000": 1000 * (1 - p) * neg[2],
        "snowy_caught_share": pos[1] + pos[2],
    }


def sweep_single_threshold(scores, y, a: dict) -> pd.DataFrame:
    """The model decides alone: send a truck if score >= t."""
    return pd.DataFrame([evaluate(scores, y, t, t, a) for t in GRID]).rename(
        columns={"low": "threshold"}
    ).drop(columns="high")


def best_single_threshold(scores, y, a: dict) -> float:
    sweep = sweep_single_threshold(scores, y, a)
    return float(sweep.loc[sweep["cost_per_1000"].idxmin(), "threshold"])


def best_band(scores, y, a: dict) -> tuple[float, float]:
    """Cheapest (low, high) pair whose review load fits the operators we have."""
    capacity = review_capacity_per_day(a)
    best = None
    for i, low in enumerate(GRID):
        for high in GRID[i:]:
            e = evaluate(scores, y, low, high, a)
            if e["reviews_per_day"] <= capacity and (best is None or e["cost_per_1000"] < best["cost_per_1000"]):
                best = e
    return best["low"], best["high"]


def break_even(a: dict, train_positive_share: float) -> dict:
    """The textbook answer, used as a cross-check on the sweep.

    Send a truck when  P(snowy) x cost of a miss  >  cost of a truck run
    (more precisely, when the expected cost of acting is below that of not
    acting). The model was trained on a dataset that is mostly snow, so its raw
    score overstates P(snowy) in real life; we convert the break-even
    probability back into the model's score scale.
    """
    gain = a["cost_missed_snow_eur"] - a["cost_truck_run_eur"]
    loss = a["cost_truck_run_eur"] - a["cost_nothing_needed_eur"]
    p_star = loss / (loss + gain)
    odds_real = p_star / (1 - p_star)
    prior_shift = (train_positive_share / (1 - train_positive_share)) / (
        a["base_rate"] / (1 - a["base_rate"])
    )
    odds_model = odds_real * prior_shift
    return {
        "break_even_real_probability": p_star,
        "break_even_model_score": odds_model / (1 + odds_model),
    }


def compare_policies(test: pd.DataFrame, chosen: dict, a: dict) -> pd.DataFrame:
    """The five ways of running this, priced on the test split."""
    y, s = test["needs_treatment"], test["p_treat"]
    rows = {
        # Reference: trucks go exactly where needed. The gap to this row is the cost of mistakes.
        "Perfect model (reference)": evaluate(y, y, 0.5, 0.5, a),
        "Do nothing (never send a truck)": evaluate(s, y, NEVER, NEVER, a),
        "Send a truck everywhere (majority class)": evaluate(s, y, 0, 0, a),
        "Operator checks every image": evaluate(s, y, 0, NEVER, a),
        "Pretrained model, zero-shot, alone": evaluate(
            test["zero_shot_score"], y, chosen["zero_shot_threshold"], chosen["zero_shot_threshold"], a
        ),
        "Finetuned model alone": evaluate(s, y, chosen["threshold"], chosen["threshold"], a),
        "Finetuned model + operator review": evaluate(s, y, chosen["low"], chosen["high"], a),
    }
    table = pd.DataFrame(rows).T
    table.index.name = "policy"
    return table


def choose_operating_point(valid: pd.DataFrame, a: dict, train_positive_share: float) -> dict:
    """Everything is chosen on the validation split; the test split only reports."""
    y = valid["needs_treatment"]
    t = best_single_threshold(valid["p_treat"], y, a)
    low, high = best_band(valid["p_treat"], y, a)
    zs = best_single_threshold(valid["zero_shot_score"], y, a)
    return {
        "threshold": t,
        "low": low,
        "high": high,
        "zero_shot_threshold": zs,
        "review_capacity_per_day": review_capacity_per_day(a),
        **break_even(a, train_positive_share),
    }


def sensitivity(valid: pd.DataFrame, test: pd.DataFrame, a: dict) -> pd.DataFrame:
    """Re-choose the model-alone threshold on validation with one assumption
    changed at a time, and price it on test. Shows which assumption moves the
    answer and how far."""
    changes = [
        ("cost_missed_snow_eur", [a["cost_missed_snow_eur"] * m for m in (0.25, 0.5, 1, 2, 4)]),
        ("cost_truck_run_eur", [a["cost_truck_run_eur"] * m for m in (0.5, 1, 2, 4)]),
        ("base_rate", sorted({0.05, 0.10, a["base_rate"], 0.25, 0.40})),
    ]
    rows = []
    for key, values in changes:
        for value in values:
            b = {**a, key: value}
            t = best_single_threshold(valid["p_treat"], valid["needs_treatment"], b)
            e = evaluate(test["p_treat"], test["needs_treatment"], t, t, b)
            nothing = evaluate(test["p_treat"], test["needs_treatment"], NEVER, NEVER, b)["cost_per_1000"]
            everywhere = evaluate(test["p_treat"], test["needs_treatment"], 0, 0, b)["cost_per_1000"]
            rows.append(
                {
                    "assumption": key,
                    "value": value,
                    "is_current": value == a[key],
                    "best_threshold": t,
                    "cost_per_1000": e["cost_per_1000"],
                    "do_nothing_cost_per_1000": nothing,
                    "truck_everywhere_cost_per_1000": everywhere,
                    "missed_snowy_per_1000": e["missed_snowy_per_1000"],
                    "needless_trucks_per_1000": e["needless_trucks_per_1000"],
                }
            )
    return pd.DataFrame(rows)


def plot_cost_curve(valid_sweep, test_sweep, chosen: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(valid_sweep["threshold"], valid_sweep["cost_per_1000"], label="validation (used to choose)")
    ax.plot(test_sweep["threshold"], test_sweep["cost_per_1000"], label="test (reported)", alpha=0.7)
    ax.axvline(chosen["threshold"], color="red", linestyle="--", label=f"chosen threshold {chosen['threshold']:.2f}")
    ax.set_xlabel("Threshold on the model's snow/slush probability")
    ax.set_ylabel("Expected cost per 1,000 images (EUR)")
    ax.set_title("Cost against threshold (model decides alone)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(a: dict | None = None) -> dict:
    """Recompute the whole costing from the saved scores."""
    a = a or load_assumptions()
    valid = pd.read_csv(RESULTS / "scores_valid.csv")
    test = pd.read_csv(RESULTS / "scores_test.csv")
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())
    chosen = choose_operating_point(valid, a, stats["train_positive_share"])

    valid_sweep = sweep_single_threshold(valid["p_treat"], valid["needs_treatment"], a)
    test_sweep = sweep_single_threshold(test["p_treat"], test["needs_treatment"], a)
    pd.concat(
        [valid_sweep.assign(split="valid"), test_sweep.assign(split="test")]
    ).to_csv(RESULTS / "cost_sweep.csv", index=False)

    policies = compare_policies(test, chosen, a)
    policies.to_csv(RESULTS / "policies.csv")
    sensitivity(valid, test, a).to_csv(RESULTS / "sensitivity.csv", index=False)
    (RESULTS / "operating_point.json").write_text(json.dumps(chosen, indent=2))
    plot_cost_curve(valid_sweep, test_sweep, chosen, RESULTS / "cost_curve.png")

    print(f"Model alone: send a truck if score >= {chosen['threshold']:.2f}")
    print(f"With review: pass below {chosen['low']:.2f}, truck from {chosen['high']:.2f}, operator in between")
    print(f"Break-even cross-check, in model-score terms: {chosen['break_even_model_score']:.2f}")
    print(policies[["cost_per_1000", "review_share", "reviews_per_day", "operators_needed"]].round(2).to_string())
    return chosen


if __name__ == "__main__":
    run()
