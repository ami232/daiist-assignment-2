"""
costing.py - turns the model's scores into euros.

Used by train.py (at the end of training) and by app.py (live, when you move
the sliders). You can also run it on its own after changing the business
numbers in settings.py, without retraining:

    uv run python costing.py

THE IDEA
--------
For every test image the model gives a "rotten score" between 0 and 1
(how sure it is that the fruit is rotten). Two cut-offs turn that score
into one of three outcomes:

    score <  low            ->  PASS    (fruit goes to the shelf)
    low <= score < high     ->  REVIEW  (a worker looks at it)
    score >= high           ->  FAIL    (fruit goes in the bin)

If low == high there is no review band: the model decides alone.

We assume a worker who reviews a piece of fruit always gets it right.

WHY WE USE THE BASE RATE
------------------------
The test set has whatever mix of rotten/fresh the dataset creator picked.
Real life has a different mix (settings.BASE_RATE). So we measure two
things on the test set:
    - of the ROTTEN fruit, what share ends up pass / review / fail
    - of the FRESH fruit,  what share ends up pass / review / fail
and then re-weight them with the real-life base rate.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)  # thresholds 0.00, 0.01, ... 1.00


def review_cost_per_fruit() -> float:
    """Cost of one human check = hourly cost x time per check."""
    return S.WORKER_COST_PER_HOUR * S.SECONDS_PER_REVIEW / 3600


def outcome_shares(scores, is_rotten, low, high):
    """Share of rotten fruit and of fresh fruit that ends in pass/review/fail."""
    scores = np.asarray(scores, dtype=float)
    is_rotten = np.asarray(is_rotten, dtype=bool)

    def shares(s):
        if len(s) == 0:
            return {"pass": 0.0, "review": 0.0, "fail": 0.0}
        return {
            "pass": float(np.mean(s < low)),
            "review": float(np.mean((s >= low) & (s < high))),
            "fail": float(np.mean(s >= high)),
        }

    return shares(scores[is_rotten]), shares(scores[~is_rotten])


def evaluate(scores, is_rotten, low, high) -> dict:
    """Expected cost per 1,000 fruit and review load for one pair of cut-offs."""
    rotten, fresh = outcome_shares(scores, is_rotten, low, high)
    review = review_cost_per_fruit()

    # Average cost of one ROTTEN fruit, and of one FRESH fruit.
    cost_rotten = (rotten["fail"] * S.COST_TRUE_POSITIVE
                   + rotten["pass"] * S.COST_FALSE_NEGATIVE
                   + rotten["review"] * (review + S.COST_TRUE_POSITIVE))
    cost_fresh = (fresh["fail"] * S.COST_FALSE_POSITIVE
                  + fresh["pass"] * S.COST_TRUE_NEGATIVE
                  + fresh["review"] * (review + S.COST_TRUE_NEGATIVE))

    br = S.BASE_RATE
    cost_per_1000 = 1000 * (br * cost_rotten + (1 - br) * cost_fresh)

    review_share = br * rotten["review"] + (1 - br) * fresh["review"]
    reviews_per_day = review_share * S.FRUIT_PER_DAY
    review_hours = reviews_per_day * S.SECONDS_PER_REVIEW / 3600
    people_needed = math.ceil(review_hours / S.SHIFT_HOURS - 1e-9) if review_hours > 0 else 0

    return {
        "low": float(low),
        "high": float(high),
        "cost_per_1000": cost_per_1000,
        "review_share": review_share,
        "reviews_per_day": reviews_per_day,
        "review_hours_per_day": review_hours,
        "people_needed": people_needed,
        "rotten_caught": rotten["fail"] + rotten["review"],  # recall on rotten
        "rotten_missed": rotten["pass"],
        "fresh_binned": fresh["fail"],
    }


def sweep_single_threshold(scores, is_rotten) -> pd.DataFrame:
    """Model decides alone: try every threshold t (bin if score >= t)."""
    return pd.DataFrame([evaluate(scores, is_rotten, t, t) for t in GRID])


def best_band(scores, is_rotten) -> dict:
    """Cheapest (low, high) pair whose review load fits the workers available."""
    best = None
    for low in GRID:
        for high in GRID[GRID >= low]:
            result = evaluate(scores, is_rotten, low, high)
            if result["people_needed"] > S.REVIEWERS_AVAILABLE:
                continue
            if best is None or result["cost_per_1000"] < best["cost_per_1000"] - 1e-9:
                best = result
    return best


def alternatives(scores, is_rotten, sweep: pd.DataFrame, band: dict) -> pd.DataFrame:
    """Compare our design with the three options the brief asks for."""
    best_single = sweep.loc[sweep["cost_per_1000"].idxmin()].to_dict()
    rows = [
        ("Do nothing (everything goes to the shelf)", evaluate(scores, is_rotten, np.inf, np.inf)),
        ("A person checks every piece", evaluate(scores, is_rotten, -np.inf, np.inf)),
        ("Model decides alone (best single threshold)", best_single),
        ("Model + person checks the unsure ones", band),
    ]
    table = pd.DataFrame([{"option": name, **values} for name, values in rows])
    keep = ["option", "cost_per_1000", "review_share", "reviews_per_day", "people_needed",
            "rotten_missed", "fresh_binned"]
    return table[keep]


def run_costing(predictions: pd.DataFrame) -> dict:
    """Full cost analysis from the saved test-set scores. Writes results/."""
    scores = predictions["rotten_score"].to_numpy()
    is_rotten = predictions["is_rotten"].to_numpy().astype(bool)

    sweep = sweep_single_threshold(scores, is_rotten)
    best = sweep.loc[sweep["cost_per_1000"].idxmin()]
    band = best_band(scores, is_rotten)
    table = alternatives(scores, is_rotten, sweep, band)

    RESULTS.mkdir(exist_ok=True)
    sweep.to_csv(RESULTS / "cost_sweep.csv", index=False)
    table.to_csv(RESULTS / "alternatives.csv", index=False)
    plot_cost_curve(sweep, float(best["low"]), RESULTS / "cost_curve.png")

    summary = {
        "best_single_threshold": float(best["low"]),
        "best_single_cost_per_1000": float(best["cost_per_1000"]),
        "band_low": band["low"],
        "band_high": band["high"],
        "band_cost_per_1000": band["cost_per_1000"],
        "band_review_share": band["review_share"],
        "band_reviews_per_day": band["reviews_per_day"],
        "band_people_needed": band["people_needed"],
        "review_cost_per_fruit": review_cost_per_fruit(),
        "business_numbers_used": {
            name: getattr(S, name) for name in [
                "COST_FALSE_NEGATIVE", "COST_FALSE_POSITIVE", "COST_TRUE_POSITIVE",
                "COST_TRUE_NEGATIVE", "BASE_RATE", "FRUIT_PER_DAY", "SECONDS_PER_REVIEW",
                "WORKER_COST_PER_HOUR", "SHIFT_HOURS", "REVIEWERS_AVAILABLE"]
        },
    }
    (RESULTS / "cost_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_cost_curve(sweep: pd.DataFrame, chosen: float, path=None, low=None, high=None):
    """Cost per 1,000 fruit against threshold, with the chosen threshold marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["low"], sweep["cost_per_1000"], color="#1f6f8b", linewidth=2)
    chosen_cost = float(sweep.loc[(sweep["low"] - chosen).abs().idxmin(), "cost_per_1000"])
    ax.scatter([chosen], [chosen_cost], color="#c0392b", zorder=3, s=60,
               label=f"Cheapest threshold = {chosen:.2f} (EUR {chosen_cost:,.0f})")
    if low is not None and high is not None:
        ax.axvline(low, color="grey", linestyle="--", linewidth=1, label=f"Your pass cut-off = {low:.2f}")
        ax.axvline(high, color="black", linestyle="--", linewidth=1, label=f"Your bin cut-off = {high:.2f}")
    ax.set_xlabel("Threshold (bin the fruit if rotten score >= threshold)")
    ax.set_ylabel("Expected cost per 1,000 fruit (EUR)")
    ax.set_title("Cost against threshold (model decides alone)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    preds_file = RESULTS / "test_predictions.csv"
    if not preds_file.exists():
        raise SystemExit("results/test_predictions.csv not found. Run `uv run python main.py train` first.")
    out = run_costing(pd.read_csv(preds_file))
    print("Cost results updated in results/")
    print(f"  Best single threshold : {out['best_single_threshold']:.2f} "
          f"-> EUR {out['best_single_cost_per_1000']:,.2f} per 1,000 fruit")
    print(f"  Best review band      : pass below {out['band_low']:.2f}, bin from {out['band_high']:.2f} "
          f"-> EUR {out['band_cost_per_1000']:,.2f} per 1,000 fruit")
    print(f"  Sent to a person      : {out['band_review_share']:.1%} "
          f"= {out['band_reviews_per_day']:,.0f} per day, {out['band_people_needed']} worker(s)")
