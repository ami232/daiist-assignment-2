"""Costing: price the outcomes, sweep thresholds, choose the operating point.

Reads costs.json and results/scores_model_{valid,test}.csv (written by train.py).
Thresholds are chosen on the valid split and reported on the test split.
Run alone with: uv run python costing.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "results"
GRID = np.append(np.round(np.arange(0.0, 1.0001, 0.01), 2), 1.01)  # 0 flags all, 1.01 flags none


def load_costs():
    c = json.loads((ROOT / "costs.json").read_text())
    c["inspection"] = c["inspector_per_hour"] * c["inspection_minutes"] / 60
    c["c_fp"] = c["inspection"]                       # good board flagged: a person checks it
    c["c_tp"] = c["inspection"] + c["repair"]         # defect flagged: check, then repair
    c["c_fn"] = sum(c["missed_defect"].values())      # defect shipped
    return c                                          # a good board passed costs nothing


def load_scores(split):
    with open(OUT / f"scores_model_{split}.csv") as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r["score"]) for r in rows]),
            np.array([int(r["truth"]) for r in rows], bool))


def rates(s, t, thr):
    flag = s >= thr
    return (flag & t).sum() / t.sum(), (flag & ~t).sum() / (~t).sum()


def single_cost(s, t, thr, c, pi):
    """Model flags -> a person checks. Expected cost per 1,000 boards at base rate pi."""
    tpr, fpr = rates(s, t, thr)
    return 1000 * (pi * (tpr * c["c_tp"] + (1 - tpr) * c["c_fn"]) + (1 - pi) * fpr * c["c_fp"])


def three_way(s, t, lo, hi, c, pi):
    """pass below lo, human review in [lo, hi), automatic fail from hi up."""
    masks = {"pass": s < lo, "review": (s >= lo) & (s < hi), "fail": s >= hi}
    d = {k: float((m & t).sum() / t.sum()) for k, m in masks.items()}    # share of defective boards
    g = {k: float((m & ~t).sum() / (~t).sum()) for k, m in masks.items()}  # share of good boards
    cost = 1000 * (pi * (d["pass"] * c["c_fn"] + d["review"] * c["c_tp"] + d["fail"] * c["repair"])
                   + (1 - pi) * (g["review"] * c["c_fp"] + g["fail"] * c["auto_fail_good_cost"]))
    share = pi * d["review"] + (1 - pi) * g["review"]
    people = c["boards_per_day"] * share * c["inspection_minutes"] / c["productive_minutes_per_person"]
    return {"cost_per_1000": float(cost), "review_share": float(share),
            "reviews_per_day": float(c["boards_per_day"] * share),
            "people_needed": float(people), "defective": d, "good": g}


def best_single(sv, tv, st, tt, c, pi):
    costs = np.array([single_cost(sv, tv, th, c, pi) for th in GRID])
    k = int(costs.argmin())
    return float(GRID[k]), float(costs[k]), float(single_cost(st, tt, GRID[k], c, pi))


def best_three_way(sv, tv, c, pi):
    best = None
    for i, lo in enumerate(GRID):
        for hi in GRID[i:]:
            r = three_way(sv, tv, lo, hi, c, pi)
            key = (r["people_needed"] > c["max_reviewers"], r["cost_per_1000"])  # feasible first
            if best is None or key < best[0]:
                best = (key, float(lo), float(hi), r)
    return best[1], best[2], best[3], not best[0][0]


def run():
    c = load_costs()
    pi = c["base_rate"]
    sv, tv = load_scores("valid")
    st, tt = load_scores("test")
    print(f"valid: {tv.sum()} defective / {(~tv).sum()} good | test: {tt.sum()} defective / {(~tt).sum()} good")

    sweep_v = [float(single_cost(sv, tv, th, c, pi)) for th in GRID]
    sweep_t = [float(single_cost(st, tt, th, c, pi)) for th in GRID]
    thr = float(GRID[int(np.argmin(sweep_v))])
    flag = st >= thr
    confusion = {"tp": int((flag & tt).sum()), "fp": int((flag & ~tt).sum()),
                 "fn": int((~flag & tt).sum()), "tn": int((~flag & ~tt).sum())}

    lo, hi, tw_valid, feasible = best_three_way(sv, tv, c, pi)
    tw_test = three_way(st, tt, lo, hi, c, pi)

    alone_thr = min(GRID, key=lambda th: three_way(sv, tv, th, th, c, pi)["cost_per_1000"])
    alone_test = three_way(st, tt, alone_thr, alone_thr, c, pi)
    per_board = c["boards_per_day"] * c["inspection_minutes"] / c["productive_minutes_per_person"]
    baselines = {
        "do_nothing": {"cost_per_1000": 1000 * pi * c["c_fn"], "people_needed": 0.0},
        "check_everything": {"cost_per_1000": 1000 * (c["inspection"] + pi * c["repair"]),
                             "people_needed": per_board},
        "model_alone": {"threshold": float(alone_thr), "cost_per_1000": alone_test["cost_per_1000"],
                        "people_needed": alone_test["people_needed"]},
        "model_plus_review": {"cost_per_1000": tw_test["cost_per_1000"],
                              "people_needed": tw_test["people_needed"]},
    }

    sens = {"missed_defect_cost": [], "base_rate": []}
    for f in (0.25, 0.5, 1, 2, 4):
        c2 = dict(c, c_fn=c["c_fn"] * f)
        t_, _, ct = best_single(sv, tv, st, tt, c2, pi)
        sens["missed_defect_cost"].append({"multiplier": f, "c_fn": c2["c_fn"], "threshold": t_, "test_cost_per_1000": ct})
    for p in (0.005, 0.01, 0.02, 0.05, 0.10):
        t_, _, ct = best_single(sv, tv, st, tt, c, p)
        sens["base_rate"].append({"base_rate": p, "threshold": t_, "test_cost_per_1000": ct})

    result = {
        "costs": {k: c[k] for k in ("inspection", "repair", "c_fp", "c_tp", "c_fn", "auto_fail_good_cost")},
        "base_rate": pi,
        "single": {"grid": GRID.tolist(), "cost_valid": sweep_v, "cost_test": sweep_t,
                   "chosen_threshold": thr, "cost_test_at_chosen": float(single_cost(st, tt, thr, c, pi)),
                   "confusion_test": confusion},
        "three_way": {"low": lo, "high": hi, "staffing_feasible": feasible,
                      "valid": tw_valid, "test": tw_test},
        "baselines": baselines,
        "sensitivity": sens,
    }
    (OUT / "cost_sweep.json").write_text(json.dumps(result, indent=2))

    print(f"\nSingle threshold (flag -> person checks), chosen on valid: {thr:.2f}")
    print(f"  test confusion: {confusion}")
    print(f"  expected cost per 1,000 boards (test): {result['single']['cost_test_at_chosen']:.0f} EUR")
    print(f"Three outcomes: pass < {lo:.2f} <= review < {hi:.2f} <= fail (staffing feasible: {feasible})")
    print(f"  test: {tw_test['cost_per_1000']:.0f} EUR/1,000 | {tw_test['review_share']*100:.1f}% to review | "
          f"{tw_test['reviews_per_day']:.0f} reviews/day | {tw_test['people_needed']:.2f} inspectors")
    print("Baselines per 1,000 boards (EUR, people needed):")
    for name, b in baselines.items():
        print(f"  {name:18s} {b['cost_per_1000']:8.0f}   {b['people_needed']:.1f} people")
    print("\nSensitivity of the chosen threshold:")
    for r in sens["missed_defect_cost"]:
        print(f"  missed-defect cost x{r['multiplier']}: threshold {r['threshold']:.2f}, cost {r['test_cost_per_1000']:.0f}")
    for r in sens["base_rate"]:
        print(f"  base rate {r['base_rate']*100:.1f}%: threshold {r['threshold']:.2f}, cost {r['test_cost_per_1000']:.0f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(GRID, sweep_v, label="valid")
        plt.plot(GRID, sweep_t, label="test")
        plt.axvline(thr, color="k", ls="--", label=f"chosen {thr:.2f}")
        plt.xlabel("confidence threshold")
        plt.ylabel("expected cost per 1,000 boards (EUR)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "cost_vs_threshold.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    run()