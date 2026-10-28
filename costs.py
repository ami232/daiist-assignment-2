"""
Cost model, threshold sweep and sensitivity analysis (one fruit = one decision).

Every fruit gets a rotten score (evaluate.fruit_scores). A fruit is rejected when
its score >= t. Its expected cost per 1,000 fruit at an assumed base rate pi is

    1000 * [ pi * (1 - TPR(t)) * C_FN  +  (1 - pi) * FPR(t) * C_FP ]

where TPR / FPR come from the data and pi from the business framing, not from the
dataset's own class mix. The threshold is chosen on the validation split and the
cost reported on the test split. The same functions back the dashboard.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "artifacts" / "eval"
COSTS_DIR = ROOT / "artifacts" / "costs"

# Business assumptions (REPORT.md, "Business problem")
COSTS = {"fp": 0.20, "fn": 5.00, "tp": 0.0, "tn": 0.0}
BASE_RATE = 0.05
# Fine steps below 0.01, where the optimum sits; t = 0 would reject fruit the model never saw
THRESHOLDS = np.round(np.concatenate([np.arange(0.001, 0.01, 0.001), np.arange(0.01, 1.0, 0.01)]), 3)

BLUE, ORANGE, INK, INK_2, SURFACE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def rates(scores: pd.DataFrame, reject_at: float, review_from: float | None = None) -> dict:
    """Share of rotten / fresh fruit that is passed, reviewed and rejected.

    review_from=None means no review band: score >= reject_at is rejected, everything else passes.
    """
    review_from = reject_at if review_from is None else review_from
    out = {}
    for label, group in [("rotten", scores[scores.rotten]), ("fresh", scores[~scores.rotten])]:
        s = group.rotten_score.to_numpy()
        out[f"{label}_reject"] = float(np.mean(s >= reject_at))
        out[f"{label}_review"] = float(np.mean((s >= review_from) & (s < reject_at)))
        out[f"{label}_pass"] = float(np.mean(s < review_from))
    return out


def cost_per_1000(r: dict, base_rate: float = BASE_RATE, costs: dict = COSTS, review: dict | None = None) -> float:
    """Expected euros per 1,000 fruit. A reviewed fruit costs the review plus the reviewer's own mistakes."""
    review = review or {"cost": 0.0, "miss_rate": 0.0, "false_reject_rate": 0.0}
    rotten = (r["rotten_pass"] * costs["fn"] + r["rotten_reject"] * costs["tp"]
              + r["rotten_review"] * (review["cost"] + review["miss_rate"] * costs["fn"]
                                      + (1 - review["miss_rate"]) * costs["tp"]))
    fresh = (r["fresh_reject"] * costs["fp"] + r["fresh_pass"] * costs["tn"]
             + r["fresh_review"] * (review["cost"] + review["false_reject_rate"] * costs["fp"]
                                    + (1 - review["false_reject_rate"]) * costs["tn"]))
    return 1000 * (base_rate * rotten + (1 - base_rate) * fresh)


def reference_policies(base_rate: float = BASE_RATE, costs: dict = COSTS) -> dict:
    """Pass everything (no sorting) and reject everything, per 1,000 fruit."""
    return {
        "pass_all": 1000 * (base_rate * costs["fn"] + (1 - base_rate) * costs["tn"]),
        "reject_all": 1000 * (base_rate * costs["tp"] + (1 - base_rate) * costs["fp"]),
    }


def sweep(scores: pd.DataFrame, base_rate: float = BASE_RATE, costs: dict = COSTS) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        r = rates(scores, t)
        rows.append({"threshold": t, "tpr": r["rotten_reject"], "fpr": r["fresh_reject"],
                     "cost_per_1000": cost_per_1000(r, base_rate, costs)})
    return pd.DataFrame(rows)


def best_threshold(scores: pd.DataFrame, base_rate: float = BASE_RATE, costs: dict = COSTS) -> float:
    table = sweep(scores, base_rate, costs)
    return float(table.loc[table.cost_per_1000.idxmin(), "threshold"])


def plot_cost_curve(val: pd.DataFrame, test: pd.DataFrame, chosen: float, refs: dict, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.plot(val.threshold, val.cost_per_1000, color=BLUE, linewidth=2, label="validation (used to choose)")
    ax.plot(test.threshold, test.cost_per_1000, color=ORANGE, linewidth=2, label="test (reported)")
    for key, label in [("pass_all", "pass everything"), ("reject_all", "reject everything")]:
        ax.axhline(refs[key], color=INK_2, linestyle="--", linewidth=0.8)
        ax.text(0.99, refs[key], f"{label}: €{refs[key]:.0f}", ha="right", va="bottom", fontsize=8, color=INK_2,
                transform=ax.get_yaxis_transform())
    test_cost = float(test.loc[np.isclose(test.threshold, chosen), "cost_per_1000"].iloc[0])
    ax.axvline(chosen, color=INK, linewidth=1)
    ax.scatter([chosen], [test_cost], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"chosen t = {chosen:g}\ntest: €{test_cost:.1f} / 1,000", (chosen, test_cost),
                xytext=(12, 18), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("rotten-score threshold t (reject if score ≥ t)", color=INK_2)
    ax.set_ylabel("expected cost, € per 1,000 fruit", color=INK_2)
    ax.set_title("Expected cost against threshold", loc="left", color=INK, fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def sensitivity(val: pd.DataFrame, test: pd.DataFrame, chosen: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-choose the threshold (on val) as one assumption moves; report the test cost of that choice."""
    def row(value, base_rate, costs):
        t = best_threshold(val, base_rate, costs)
        refs = reference_policies(base_rate, costs)
        return {"value": value, "best_threshold": t,
                "model_cost": cost_per_1000(rates(test, t), base_rate, costs),
                "cost_at_chosen_t": cost_per_1000(rates(test, chosen), base_rate, costs),
                **refs}

    by_base = pd.DataFrame([row(b, b, COSTS) for b in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]])
    by_fn = pd.DataFrame([row(fn, BASE_RATE, {**COSTS, "fn": fn})
                          for fn in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]])
    return by_base, by_fn


def plot_sensitivity(table: pd.DataFrame, xlabel: str, title: str, path: Path, log_x: bool = False):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    left.plot(table.value, table.best_threshold, color=BLUE, linewidth=2, marker="o", markersize=5)
    left.set_ylabel("cost-minimising threshold (chosen on val)", color=INK_2)
    left.set_title("Where the threshold moves", loc="left", color=INK, fontsize=10)
    for col, color, label in [("model_cost", BLUE, "model, threshold re-chosen"),
                              ("pass_all", INK_2, "pass everything"), ("reject_all", ORANGE, "reject everything")]:
        right.plot(table.value, table[col], color=color, linewidth=2, label=label,
                   linestyle="--" if col == "pass_all" else "-")
    right.set_ylabel("expected cost on test, € per 1,000", color=INK_2)
    right.set_title("What it costs", loc="left", color=INK, fontsize=10)
    right.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        ax.set_xlabel(xlabel, color=INK_2)
        if log_x:
            ax.set_xscale("log")
        _style(ax)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Human review (REPORT.md, "What a person still has to check")
FRUIT_PER_DAY = 50_000
LABOUR = {"eur_per_hour": 14.0, "paid_hours": 8, "productive_hours": 7}
REVIEW = {"seconds": 4.0, "miss_rate": 0.05, "false_reject_rate": 0.05, "people": 1}
FULL_INSPECTION = {"seconds": 1.0, "miss_rate": 0.20, "false_reject_rate": 0.05}
NO_AUTO_REJECT = 1.01  # upper threshold above any score: everything over the lower one goes to a person


def eur_per_productive_second() -> float:
    return LABOUR["eur_per_hour"] * LABOUR["paid_hours"] / LABOUR["productive_hours"] / 3600


def review_terms() -> dict:
    """Per-fruit cost of a review: paid time per productive second, times the seconds it takes."""
    return {"cost": REVIEW["seconds"] * eur_per_productive_second(), "miss_rate": REVIEW["miss_rate"],
            "false_reject_rate": REVIEW["false_reject_rate"]}


def review_capacity(people: int = REVIEW["people"]) -> float:
    """Share of the day's fruit that `people` reviewers can check."""
    return people * LABOUR["productive_hours"] * 3600 / REVIEW["seconds"] / FRUIT_PER_DAY


def people_needed(share: float, seconds: float) -> int:
    return int(np.ceil(round(share * FRUIT_PER_DAY * seconds / (LABOUR["productive_hours"] * 3600), 6)))


def review_share(r: dict, base_rate: float = BASE_RATE) -> float:
    return base_rate * r["rotten_review"] + (1 - base_rate) * r["fresh_review"]


def best_review_band(scores: pd.DataFrame, base_rate: float = BASE_RATE,
                     max_share: float | None = None) -> tuple[float, float, float]:
    """(review_from, reject_at, cost per 1,000) minimising cost, with at most max_share of fruit reviewed."""
    max_share = review_capacity() if max_share is None else max_share
    grid = np.append(THRESHOLDS, NO_AUTO_REJECT)
    rotten = np.sort(scores.loc[scores.rotten, "rotten_score"].to_numpy())
    fresh = np.sort(scores.loc[~scores.rotten, "rotten_score"].to_numpy())
    ge_r = 1 - np.searchsorted(rotten, grid, side="left") / len(rotten)  # P(score >= t | rotten)
    ge_f = 1 - np.searchsorted(fresh, grid, side="left") / len(fresh)
    lo, hi = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)), indexing="ij")
    valid = hi >= lo
    r = {"rotten_reject": ge_r[hi], "rotten_review": ge_r[lo] - ge_r[hi], "rotten_pass": 1 - ge_r[lo],
         "fresh_reject": ge_f[hi], "fresh_review": ge_f[lo] - ge_f[hi], "fresh_pass": 1 - ge_f[lo]}
    cost = cost_per_1000(r, base_rate, review=review_terms())
    cost = np.where(valid & (review_share(r, base_rate) <= max_share + 1e-12), cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return float(grid[i]), float(grid[j]), float(cost[i, j])


def scenario(name: str, r: dict, people: int, seconds_per_check: float, base_rate: float = BASE_RATE,
             human: dict | None = None, paid_per_check: bool = False) -> dict:
    """Daily cost of one way of running the line. `human` = error rates of whoever checks the reviewed fruit.

    paid_per_check=False pays `people` whole shifts; True pays only the time spent checking, for a task
    that an operator already on the line absorbs.
    """
    human = human or {"miss_rate": 0.0, "false_reject_rate": 0.0}
    n_rotten, n_fresh = FRUIT_PER_DAY * base_rate, FRUIT_PER_DAY * (1 - base_rate)
    rotten_shipped = n_rotten * (r["rotten_pass"] + r["rotten_review"] * human["miss_rate"])
    fresh_lost = n_fresh * (r["fresh_reject"] + r["fresh_review"] * human["false_reject_rate"])
    fresh_shipped = n_fresh - fresh_lost
    errors = rotten_shipped * COSTS["fn"] + fresh_lost * COSTS["fp"]
    checked = FRUIT_PER_DAY * review_share(r, base_rate)
    if paid_per_check:
        labour = checked * seconds_per_check * eur_per_productive_second()
    else:
        labour = people * LABOUR["paid_hours"] * LABOUR["eur_per_hour"]
    return {"scenario": name, "people": people, "checked_per_day": round(checked),
            "checking_hours_per_day": checked * seconds_per_check / 3600,
            "share_checked": review_share(r, base_rate),
            "rotten_shipped_per_day": rotten_shipped, "fresh_lost_per_day": fresh_lost,
            "rotten_share_of_packed": rotten_shipped / (rotten_shipped + fresh_shipped),
            "error_cost_per_day": errors, "labour_cost_per_day": labour, "total_per_day": errors + labour,
            "total_per_1000": (errors + labour) / FRUIT_PER_DAY * 1000}


def scenarios(test_scores: pd.DataFrame, chosen: float, review_from: float, reject_at: float) -> pd.DataFrame:
    everything = {"rotten_pass": 0.0, "rotten_review": 1.0, "rotten_reject": 0.0,
                  "fresh_pass": 0.0, "fresh_review": 1.0, "fresh_reject": 0.0}
    nothing = {"rotten_pass": 1.0, "rotten_review": 0.0, "rotten_reject": 0.0,
               "fresh_pass": 1.0, "fresh_review": 0.0, "fresh_reject": 0.0}
    band = rates(test_scores, reject_at, review_from)
    return pd.DataFrame([
        scenario("do nothing (no sorting)", nothing, 0, 0),
        scenario("a person checks every fruit", everything,
                 people_needed(1.0, FULL_INSPECTION["seconds"]), FULL_INSPECTION["seconds"], human=FULL_INSPECTION),
        scenario(f"model alone (reject if score >= {chosen:g})", rates(test_scores, chosen), 0, 0),
        scenario(f"model + review by a line operator (review {review_from:g}-{reject_at:g}, reject >= {reject_at:g})",
                 band, people_needed(review_share(band), REVIEW["seconds"]), REVIEW["seconds"], human=REVIEW,
                 paid_per_check=True),
    ])


def analyse() -> dict:
    COSTS_DIR.mkdir(parents=True, exist_ok=True)
    val_scores = pd.read_csv(EVAL_DIR / "fruit_scores_val.csv")
    test_scores = pd.read_csv(EVAL_DIR / "fruit_scores_test.csv")

    val, test = sweep(val_scores), sweep(test_scores)
    chosen = float(val.loc[val.cost_per_1000.idxmin(), "threshold"])
    test_best = float(test.loc[test.cost_per_1000.idxmin(), "threshold"])
    refs = reference_policies()
    r = rates(test_scores, chosen)

    by_base, by_fn = sensitivity(val_scores, test_scores, chosen)
    val.to_csv(COSTS_DIR / "sweep_val.csv", index=False)
    test.to_csv(COSTS_DIR / "sweep_test.csv", index=False)
    by_base.to_csv(COSTS_DIR / "sensitivity_base_rate.csv", index=False)
    by_fn.to_csv(COSTS_DIR / "sensitivity_fn_cost.csv", index=False)
    plot_cost_curve(val, test, chosen, refs, COSTS_DIR / "cost_vs_threshold.png")
    plot_sensitivity(by_base, "assumed base rate (share of rotten fruit)", "Sensitivity to the base rate",
                     COSTS_DIR / "sensitivity_base_rate.png", log_x=True)
    plot_sensitivity(by_fn, "cost of a false negative (€)", "Sensitivity to the false-negative cost",
                     COSTS_DIR / "sensitivity_fn_cost.png", log_x=True)

    operating_point = {
        "costs_eur": COSTS, "base_rate": BASE_RATE,
        "chosen_threshold": chosen, "chosen_on": "validation",
        "test_tpr": r["rotten_reject"], "test_fpr": r["fresh_reject"],
        "test_cost_per_1000": cost_per_1000(r), "val_cost_per_1000": float(val.cost_per_1000.min()),
        "test_optimal_threshold_for_reference": test_best,
        "reference_costs_per_1000": refs,
        "bayes_threshold_on_calibrated_probability": COSTS["fp"] / (COSTS["fp"] + COSTS["fn"]),
    }

    review_from, reject_at, val_band_cost = best_review_band(val_scores)
    unconstrained = best_review_band(val_scores, max_share=1.0)
    band = rates(test_scores, reject_at, review_from)
    operating_point["review"] = {
        "fruit_per_day": FRUIT_PER_DAY, "labour": LABOUR, "review": REVIEW, "full_inspection": FULL_INSPECTION,
        "review_cost_per_fruit": review_terms()["cost"], "capacity_share_per_person": review_capacity(1),
        "review_from": review_from, "reject_at": reject_at, "chosen_on": "validation",
        "val_cost_per_1000": val_band_cost,
        "test_cost_per_1000": cost_per_1000(band, review=review_terms()),
        "test_share_reviewed": review_share(band),
        "unconstrained_band_on_val": {"review_from": unconstrained[0], "reject_at": unconstrained[1],
                                      "val_cost_per_1000": unconstrained[2]},
    }
    table = scenarios(test_scores, chosen, review_from, reject_at)
    table.to_csv(COSTS_DIR / "scenarios.csv", index=False)

    (COSTS_DIR / "operating_point.json").write_text(json.dumps(operating_point, indent=2))
    print("\n== Operating point ==\n" + json.dumps(operating_point, indent=2))
    print("\n== Scenarios (test rates, per day) ==\n" + table.round(3).to_string(index=False))
    print("\n== Sensitivity: base rate ==\n" + by_base.round(3).to_string(index=False))
    print("\n== Sensitivity: FN cost ==\n" + by_fn.round(3).to_string(index=False))
    return operating_point


if __name__ == "__main__":
    analyse()
