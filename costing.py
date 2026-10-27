"""
Cost of being wrong, threshold sweep, and the human-review band.

Imported by train.py (at the end of training) and by app.py (live, when you
move the sliders). Run it on its own to redo only the cost analysis after
changing the assumptions in config.py, without retraining:

    uv run python costing.py

The idea in one paragraph
-------------------------
Every fruit gets a score s = P(rotten) from the model. Two thresholds split
the scores into three outcomes:

    s <  t_low            -> PASS   (goes into the box)
    t_low <= s < t_high   -> REVIEW (diverted to a person, who decides)
    s >= t_high           -> FAIL   (air jet removes it)

With t_low == t_high there is no review band: the model decides alone.
We measure, separately for rotten and for fresh fruit, what share ends up in
each outcome. Then we weight those shares by the REAL base rate from
config.py (not the ~50/50 mix of the test set) and by the euro cost of each
outcome. That gives the expected cost per fruit.
"""

import json

import numpy as np
import pandas as pd

import config

THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)  # 0.00, 0.01, ..., 1.00


# ---------------------------------------------------------------------------
# Core maths (no files, no plotting: easy to test and reuse in the app)
# ---------------------------------------------------------------------------
def outcome_shares(scores, is_rotten, t_low, t_high):
    """Share of rotten and of fresh fruit that is passed, reviewed, failed."""
    scores = np.asarray(scores, dtype=float)
    is_rotten = np.asarray(is_rotten, dtype=bool)
    if t_low > t_high:
        raise ValueError("t_low must not be above t_high")

    passed = scores < t_low
    failed = scores >= t_high
    reviewed = ~passed & ~failed

    def shares(mask):
        n = max(int(mask.sum()), 1)
        return {
            "pass": float(passed[mask].sum() / n),
            "review": float(reviewed[mask].sum() / n),
            "fail": float(failed[mask].sum() / n),
        }

    return {"rotten": shares(is_rotten), "fresh": shares(~is_rotten)}


def expected_cost(shares, base_rate=None, cost_fn=None, cost_fp=None,
                  cost_tp=None, cost_tn=None, review_cost=None):
    """Expected euros per fruit at the real base rate.

    A person in the review band is assumed to decide correctly (stated as an
    assumption in REPORT.md), so a reviewed fruit costs the review labour plus
    the cost of the correct outcome.
    """
    p = config.BASE_RATE if base_rate is None else base_rate
    c_fn = config.COST_FN if cost_fn is None else cost_fn
    c_fp = config.COST_FP if cost_fp is None else cost_fp
    c_tp = config.COST_TP if cost_tp is None else cost_tp
    c_tn = config.COST_TN if cost_tn is None else cost_tn
    c_rev = config.review_cost_per_fruit() if review_cost is None else review_cost

    r, f = shares["rotten"], shares["fresh"]
    cost_rotten = r["pass"] * c_fn + r["fail"] * c_tp + r["review"] * (c_tp + c_rev)
    cost_fresh = f["pass"] * c_tn + f["fail"] * c_fp + f["review"] * (c_tn + c_rev)
    return p * cost_rotten + (1 - p) * cost_fresh


def review_share(shares, base_rate=None):
    """Share of ALL incoming fruit sent to a person, at the real base rate."""
    p = config.BASE_RATE if base_rate is None else base_rate
    return p * shares["rotten"]["review"] + (1 - p) * shares["fresh"]["review"]


def reviewers_needed(share_reviewed):
    """People needed to check that share of a day's fruit."""
    hours = config.FRUIT_PER_DAY * share_reviewed * config.REVIEW_SECONDS / 3600
    return hours / config.SHIFT_HOURS


def confusion_counts(scores, is_rotten, t_low, t_high):
    """Raw counts (not shares) for the 2x3 table shown in the dashboard."""
    scores = np.asarray(scores, dtype=float)
    is_rotten = np.asarray(is_rotten, dtype=bool)
    out = {}
    for name, mask in (("rotten", is_rotten), ("fresh", ~is_rotten)):
        s = scores[mask]
        out[name] = {
            "pass": int((s < t_low).sum()),
            "review": int(((s >= t_low) & (s < t_high)).sum()),
            "fail": int((s >= t_high).sum()),
        }
    return out


def sweep(scores, is_rotten, **cost_kwargs):
    """Single threshold (no review band): cost at every threshold."""
    rows = []
    for t in THRESHOLDS:
        sh = outcome_shares(scores, is_rotten, t, t)
        rows.append({
            "threshold": float(t),
            "recall_rotten": sh["rotten"]["fail"],      # rotten caught
            "false_alarm_rate": sh["fresh"]["fail"],    # fresh blasted off
            "cost_per_1000": 1000 * expected_cost(sh, **cost_kwargs),
        })
    return pd.DataFrame(rows)


def best_single_threshold(scores, is_rotten, **cost_kwargs):
    """Threshold with the lowest cost. If several tie (a flat bottom, common
    when the model is very confident), take the middle of the flat stretch,
    which is the choice least sensitive to small changes in the scores."""
    table = sweep(scores, is_rotten, **cost_kwargs)
    lowest = table["cost_per_1000"].min()
    tied = table[table["cost_per_1000"] <= lowest + 1e-9]
    row = tied.iloc[len(tied) // 2]
    return float(row["threshold"]), float(row["cost_per_1000"])


def best_review_band(scores, is_rotten, max_reviewers=None):
    """Cheapest (t_low, t_high) whose review load fits the people available."""
    limit = config.REVIEWERS_AVAILABLE if max_reviewers is None else max_reviewers
    best = None
    for i, t_low in enumerate(THRESHOLDS):
        for t_high in THRESHOLDS[i:]:
            sh = outcome_shares(scores, is_rotten, t_low, t_high)
            share = review_share(sh)
            people = reviewers_needed(share)
            if people > limit:
                continue
            cost = 1000 * expected_cost(sh)
            # Cheaper wins; on a tie, the band that sends fewer fruit to people
            if (best is None or cost < best["cost_per_1000"] - 1e-9
                    or (abs(cost - best["cost_per_1000"]) <= 1e-9
                        and share < best["review_share"])):
                best = {"t_low": float(t_low), "t_high": float(t_high),
                        "cost_per_1000": cost, "review_share": share,
                        "reviewers_needed": people}
    return best


def alternatives_per_1000():
    """The two strategies that don't use the model at all."""
    p = config.BASE_RATE
    do_nothing = p * config.COST_FN + (1 - p) * config.COST_TN
    check_all = config.review_cost_per_fruit() + p * config.COST_TP + (1 - p) * config.COST_TN
    return {
        "do_nothing": {"cost_per_1000": 1000 * do_nothing, "review_share": 0.0,
                       "reviewers_needed": 0.0},
        "person_checks_everything": {"cost_per_1000": 1000 * check_all,
                                     "review_share": 1.0,
                                     "reviewers_needed": reviewers_needed(1.0)},
    }


def breakeven_band(train_rotten_share):
    """Review band from break-even maths, corrected to the real base rate.

    The model learned on data that is about 50% rotten, so its score is a
    probability for a 50/50 world. Convert real-world break-even chances into
    model scores with odds: model odds = real odds / (base-rate odds /
    training odds).

      review beats passing  when real P(rotten) > review cost / cost_fn
      review beats removing when real P(fresh)  > review cost / cost_fp
    """
    c_rev = config.review_cost_per_fruit()
    shift = (config.BASE_RATE / (1 - config.BASE_RATE)) / (
        train_rotten_share / (1 - train_rotten_share))

    def to_score(q_real):
        model_odds = (q_real / (1 - q_real)) / shift
        return model_odds / (1 + model_odds)

    q_low = c_rev / (config.COST_FN - config.COST_TP)        # min real P(rotten) to review
    q_high = 1 - c_rev / (config.COST_FP - config.COST_TN)   # max real P(rotten) to review
    return {"t_low": round(to_score(q_low), 2), "t_high": round(to_score(q_high), 2),
            "real_q_low": q_low, "real_q_high": q_high, "odds_shift": shift,
            "train_rotten_share": train_rotten_share}


# ---------------------------------------------------------------------------
# The analysis run: choose on VALIDATION, report on TEST
# ---------------------------------------------------------------------------
def run_analysis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if config.ASSUMPTIONS_ARE_PLACEHOLDERS:
        print("\n" + "!" * 72)
        print("WARNING: config.py still has PLACEHOLDER business assumptions.")
        print("The cost numbers below are not yet yours. Fill them in, set")
        print("ASSUMPTIONS_ARE_PLACEHOLDERS = False, then: uv run python costing.py")
        print("!" * 72 + "\n")

    out = config.RESULTS_DIR
    scores = pd.read_csv(out / "scores.csv")
    val = scores[scores["split"] == "val"]
    test = scores[scores["split"] == "test"]

    models = {"finetuned": "p_rotten", "zero_shot": "zs_p_rotten"}

    # 1. Sweep on both splits, for both models
    sweeps = []
    for model, col in models.items():
        for split_name, df in (("val", val), ("test", test)):
            s = sweep(df[col], df["is_rotten"])
            s.insert(0, "split", split_name)
            s.insert(0, "model", model)
            sweeps.append(s)
    sweeps = pd.concat(sweeps, ignore_index=True)
    sweeps.to_csv(out / "cost_sweep.csv", index=False)

    # 2. Choose thresholds on validation only
    chosen_t, _ = best_single_threshold(val["p_rotten"], val["is_rotten"])
    zs_t, _ = best_single_threshold(val["zs_p_rotten"], val["is_rotten"])
    band = best_review_band(val["p_rotten"], val["is_rotten"])
    if band["review_share"] == 0:  # reviewing doesn't pay: same as model alone
        band.update(t_low=chosen_t, t_high=chosen_t)

    # 3. Report everything on test
    def on_test(col, t_low, t_high):
        sh = outcome_shares(test[col], test["is_rotten"], t_low, t_high)
        share = review_share(sh)
        return {"t_low": t_low, "t_high": t_high,
                "cost_per_1000": 1000 * expected_cost(sh),
                "review_share": share,
                "reviewers_needed": reviewers_needed(share),
                "shares": sh,
                "counts": confusion_counts(test[col], test["is_rotten"], t_low, t_high)}

    strategies = alternatives_per_1000()
    strategies["zero_shot_model_alone"] = on_test("zs_p_rotten", zs_t, zs_t)
    strategies["model_alone"] = on_test("p_rotten", chosen_t, chosen_t)
    strategies["model_plus_review"] = on_test("p_rotten", band["t_low"], band["t_high"])

    # Shipped band: break-even maths corrected to the real base rate. The val
    # split is stratified like train, so its rotten share stands in for train's.
    shipped = breakeven_band(float(val["is_rotten"].mean()))
    strategies["model_plus_breakeven_band"] = on_test(
        "p_rotten", shipped["t_low"], shipped["t_high"])

    # 4. Sensitivity: how does the best single threshold move if you're wrong?
    sens = []
    for p in (0.01, 0.02, 0.05, 0.10, 0.20):
        t, c = best_single_threshold(val["p_rotten"], val["is_rotten"], base_rate=p)
        sens.append({"vary": "base_rate", "value": p, "best_threshold": t,
                     "val_cost_per_1000": c})
    for mult in (0.25, 0.5, 1, 2, 4):
        t, c = best_single_threshold(val["p_rotten"], val["is_rotten"],
                                     cost_fn=config.COST_FN * mult)
        sens.append({"vary": "cost_fn_multiplier", "value": mult, "best_threshold": t,
                     "val_cost_per_1000": c})
    pd.DataFrame(sens).to_csv(out / "sensitivity.csv", index=False)

    operating_point = {
        "placeholders": config.ASSUMPTIONS_ARE_PLACEHOLDERS,
        "assumptions": {
            "base_rate": config.BASE_RATE, "cost_fn": config.COST_FN,
            "cost_fp": config.COST_FP, "cost_tp": config.COST_TP,
            "cost_tn": config.COST_TN, "review_seconds": config.REVIEW_SECONDS,
            "reviewer_cost_per_hour": config.REVIEWER_COST_PER_HOUR,
            "review_cost_per_fruit": config.review_cost_per_fruit(),
            "fruit_per_day": config.FRUIT_PER_DAY, "shift_hours": config.SHIFT_HOURS,
            "reviewers_available": config.REVIEWERS_AVAILABLE,
        },
        "chosen_on": "val",
        "reported_on": "test",
        "single_threshold": chosen_t,
        "zero_shot_threshold": zs_t,
        "review_band": {"t_low": band["t_low"], "t_high": band["t_high"]},
        "shipped_band": shipped,
        "strategies_on_test": strategies,
    }
    (out / "operating_point.json").write_text(json.dumps(operating_point, indent=2))

    # 5. Plot for the report: cost against threshold, chosen point marked
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, style in (("finetuned", "-"), ("zero_shot", ":")):
        for split_name, colour in (("val", "tab:blue"), ("test", "tab:orange")):
            d = sweeps[(sweeps.model == model) & (sweeps.split == split_name)]
            ax.plot(d.threshold, d.cost_per_1000, style, color=colour,
                    label=f"{model.replace('_', '-')} ({split_name})")
    ax.axvline(chosen_t, color="black", lw=1, ls="--",
               label=f"chosen threshold {chosen_t:.2f} (on val)")
    ax.axhline(strategies["do_nothing"]["cost_per_1000"], color="grey", lw=1,
               label="do nothing")
    ax.set_xlabel("Threshold on P(rotten)")
    ax.set_ylabel("Expected cost per 1,000 fruit (EUR)")
    ax.set_title("Cost against threshold (model decides alone)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cost_vs_threshold.png", dpi=150)
    plt.close(fig)

    # 6. Print a summary
    print(f"Chosen single threshold (on val): {chosen_t:.2f}")
    print(f"Cheapest swept band (on val): pass < {band['t_low']:.2f} <= review "
          f"< {band['t_high']:.2f} <= fail")
    print(f"Shipped band (break-even, base-rate corrected): pass < "
          f"{shipped['t_low']:.2f} <= review < {shipped['t_high']:.2f} <= fail")
    print("\nStrategies on the TEST split, per 1,000 fruit:")
    for name, s in strategies.items():
        print(f"  {name:<26} EUR {s['cost_per_1000']:>8.2f}   "
              f"review {100 * s['review_share']:5.1f}%   "
              f"people {s['reviewers_needed']:5.2f}")
    return operating_point


if __name__ == "__main__":
    run_analysis()