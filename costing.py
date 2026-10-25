"""Cost model, threshold sweep and human-review sizing for the egg inspector.

CPU only. Reads results/metrics.json, results/valid_predictions.json and
results/test_predictions.json (written earlier in train.ipynb) and writes
results/cost_sweep.json, results/scores.json and results/plots/cost_curve.png.

    uv run python costing.py

Unit of decision: one egg, seen by the camera before packing.
One-decision-per-egg rule: an egg's score is the highest "cracked" confidence
on its image (0 if the model drew no cracked box). Valid for this dataset
because images hold at most one egg (train.ipynb checks this).
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# INPUTS. Each one needs a source or a reason in REPORT.md.
# ---------------------------------------------------------------------------
CARTON_PRICE = 4.50       # EUR, price received per carton (given)
EGGS_PER_CARTON = 12      # assumption: not given, change if your cartons differ
WAGE_PER_HOUR = 14.0      # EUR per hour (given)
HANDLE_SECONDS = 40       # seconds to remove / swap a flagged egg (given)
LOOK_SECONDS = 30         # seconds to look at a flagged egg and decide: YOUR DECISION

handle_cost = HANDLE_SECONDS / 3600 * WAGE_PER_HOUR

ASSUMPTIONS = dict(
    base_rate=0.03,                               # share of eggs cracked: YOUR DECISION
    cost_fn=CARTON_PRICE,                         # missed crack: carton not sold
    cost_fp=CARTON_PRICE / EGGS_PER_CARTON + handle_cost,   # good egg lost + handling
    cost_tp=handle_cost,                          # handling a correctly caught egg
    cost_tn=0.00,
    review_seconds=LOOK_SECONDS,
    wage_per_hour=WAGE_PER_HOUR,
    human_miss_rate=0.03,                         # reviewer passes a cracked egg: YOUR DECISION
    human_false_alarm=0.01,                       # reviewer removes a good egg: YOUR DECISION
    eggs_per_day=200_000,                         # eggs through the camera: YOUR DECISION
    shift_hours=8,
    utilisation=0.75,                             # share of a shift spent reviewing
)

GRID = np.round(np.append(np.arange(0, 1.0001, 0.01), 1.01), 2)
GRID_COARSE = np.round(np.append(np.arange(0, 1.0001, 0.02), 1.01), 2)


def review_cost(A):
    return A["review_seconds"] / 3600 * A["wage_per_hour"]


def egg_scores(preds, cracked_id):
    """One decision per egg: score = highest 'cracked' confidence on the image."""
    s, y = [], []
    for p in preds:
        conf = [b[1] for b in p["pred"] if b[0] == cracked_id]
        s.append(max(conf) if conf else 0.0)
        y.append(int(any(g[0] == cracked_id for g in p["gt"])))
    return np.array(s), np.array(y)


def split_scores(s, y):
    """Sorted scores of truly cracked eggs and of everything else."""
    return np.sort(s[y == 1]), np.sort(s[y == 0])


def expected_cost(pos, neg, lo, hi, A):
    """Expected cost per 1,000 eggs for one pair of thresholds.

    score <  lo        -> pass
    lo <= score < hi   -> human review
    score >= hi        -> fail (remove the egg)
    lo == hi means no review band. Test-set rates are reweighted to base_rate.
    """
    hi = max(hi, lo)

    def fractions(x):
        below_lo = np.searchsorted(x, lo, side="left") / len(x)
        below_hi = np.searchsorted(x, hi, side="left") / len(x)
        return below_lo, below_hi - below_lo, 1 - below_hi

    p_pass, p_rev, p_fail = fractions(pos)
    n_pass, n_rev, n_fail = fractions(neg)
    c_rev, hm, hf = review_cost(A), A["human_miss_rate"], A["human_false_alarm"]

    cost_pos = (p_pass * A["cost_fn"] + p_fail * A["cost_tp"]
                + p_rev * (c_rev + hm * A["cost_fn"] + (1 - hm) * A["cost_tp"]))
    cost_neg = (n_pass * A["cost_tn"] + n_fail * A["cost_fp"]
                + n_rev * (c_rev + hf * A["cost_fp"] + (1 - hf) * A["cost_tn"]))
    pi = A["base_rate"]
    k, m = 1000 * pi, 1000 * (1 - pi)
    return {
        "cost_per_1000": k * cost_pos + m * cost_neg,
        "review_share": pi * p_rev + (1 - pi) * n_rev,
        "decisions_per_1000": {
            "pass": {"cracked": k * p_pass, "not_cracked": m * n_pass},
            "review": {"cracked": k * p_rev, "not_cracked": m * n_rev},
            "fail": {"cracked": k * p_fail, "not_cracked": m * n_fail},
        },
        "final_missed_cracked_per_1000": k * (p_pass + p_rev * hm),
        "final_good_removed_per_1000": m * (n_fail + n_rev * hf),
    }


def best_single(pos, neg, A, grid=GRID):
    costs = np.array([expected_cost(pos, neg, t, t, A)["cost_per_1000"] for t in grid])
    ties = np.where(costs <= costs.min() + 1e-9)[0]            
    return float(grid[ties[len(ties) // 2]]), costs.tolist()   


def best_band(pos, neg, A, grid=GRID):
    best = None
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            c = expected_cost(pos, neg, lo, hi, A)["cost_per_1000"]
            if best is None or c < best[0] - 1e-9:
                best = (c, float(lo), float(hi))
    return best[1], best[2]


def staffing(res, A):
    per_day = A["eggs_per_day"] * res["review_share"]
    staff = per_day * A["review_seconds"] / (A["shift_hours"] * 3600 * A["utilisation"])
    return per_day, staff


def four_designs(pos, neg, A, t_single, lo, hi):
    plan = {
        "do_nothing": (1.01, 1.01),
        "human_checks_everything": (0.0, 1.01),
        "model_alone": (t_single, t_single),
        "model_plus_review": (lo, hi),
    }
    out = {}
    for name, (a, b) in plan.items():
        r = expected_cost(pos, neg, a, b, A)
        per_day, staff = staffing(r, A)
        out[name] = {
            "lo": a, "hi": b,
            "cost_per_1000": r["cost_per_1000"],
            "cost_per_day": r["cost_per_1000"] * A["eggs_per_day"] / 1000,
            "review_share": r["review_share"],
            "reviews_per_day": per_day,
            "reviewers_needed": staff,
            "missed_cracked_per_1000": r["final_missed_cracked_per_1000"],
            "good_removed_per_1000": r["final_good_removed_per_1000"],
        }
    return out


def sensitivity(pv, nv, pt, nt, A):
    """Which design wins as the base rate and the cost of a miss change."""
    rows = []
    for pi in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.50):
        for mult in (0.25, 0.5, 1, 2, 4):
            B = {**A, "base_rate": pi, "cost_fn": A["cost_fn"] * mult}
            t, _ = best_single(pv, nv, B, GRID_COARSE)
            lo, hi = best_band(pv, nv, B, GRID_COARSE)
            d = four_designs(pt, nt, B, t, lo, hi)
            cost = {k: v["cost_per_1000"] for k, v in d.items()}
            rows.append({
                "base_rate": pi, "cost_fn": round(B["cost_fn"], 2),
                **{k: round(v, 2) for k, v in cost.items()},
                "lo": lo, "hi": hi,
                "review_share": round(d["model_plus_review"]["review_share"], 4),
                "winner": min(cost, key=cost.get),
            })
    return rows


def plot_curve(grid, cv, ct, t_single, lo, hi, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    g = grid[grid <= 1]
    ax.plot(g, cv[: len(g)], label="validation (threshold chosen here)")
    ax.plot(g, ct[: len(g)], "--", label="test (reported)")
    ax.axvline(t_single, color="k", ls=":", label=f"best single threshold = {t_single:.2f}")
    if hi > lo:
        ax.axvspan(lo, min(hi, 1), color="orange", alpha=0.2,
                   label=f"review band {lo:.2f} to {hi:.2f}")
    ax.set_xlabel("confidence threshold on the 'cracked' score")
    ax.set_ylabel("expected cost per 1,000 eggs (EUR)")
    ax.set_title("Cost against threshold (single threshold, no review)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(overrides=None):
    A = {**ASSUMPTIONS, **(overrides or {})}
    labels = json.load(open(RESULTS / "metrics.json"))["finetuned"]["confusion_matrix"]["labels"]
    cracked_id = labels.index("cracked")

    scores = {}
    for split in ("valid", "test"):
        f = RESULTS / f"{split}_predictions.json"
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run the predictions cell of train.ipynb first.")
        scores[split] = egg_scores(json.load(open(f)), cracked_id)
    pv, nv = split_scores(*scores["valid"])
    pt, nt = split_scores(*scores["test"])
    print(f"valid: {len(pv)} cracked / {len(nv)} other   test: {len(pt)} cracked / {len(nt)} other")

    t_single, cv = best_single(pv, nv, A)
    lo, hi = best_band(pv, nv, A)
    c_band = expected_cost(pv, nv, lo, hi, A)["cost_per_1000"]
    c_single = expected_cost(pv, nv, t_single, t_single, A)["cost_per_1000"]
    if c_band > c_single - 0.01:     # review does not pay: no band
        lo = hi = t_single
    ct = [expected_cost(pt, nt, t, t, A)["cost_per_1000"] for t in GRID]
    designs = four_designs(pt, nt, A, t_single, lo, hi)
    chosen = expected_cost(pt, nt, lo, hi, A)
    sens = sensitivity(pv, nv, pt, nt, A)

    out = {
        "assumptions": A,
        "grid": GRID.tolist(),
        "curve_valid": cv, "curve_test": ct,
        "chosen": {"single_threshold": t_single, "lo": lo, "hi": hi,
                   "test_result": chosen},
        "designs": designs,
        "sensitivity": sens,
        "selection": "thresholds chosen on validation, reported on test",
    }
    json.dump(out, open(RESULTS / "cost_sweep.json", "w"), indent=2)
    json.dump({sp: {"score": scores[sp][0].round(4).tolist(),
                    "cracked": scores[sp][1].tolist()} for sp in scores},
              open(RESULTS / "scores.json", "w"))
    plot_curve(GRID, cv, ct, t_single, lo, hi, RESULTS / "plots" / "cost_curve.png")

    print(f"\nBest single threshold (validation): {t_single:.2f}")
    print(f"Best review band (validation): pass < {lo:.2f} <= review < {hi:.2f} <= fail")
    print(f"\n{'design':26s}{'EUR/1000':>10s}{'review %':>10s}{'reviews/day':>13s}{'people':>8s}")
    for k, v in designs.items():
        print(f"{k:26s}{v['cost_per_1000']:10.2f}{100*v['review_share']:10.1f}"
              f"{v['reviews_per_day']:13.0f}{v['reviewers_needed']:8.1f}")
    print("\nSensitivity (winner by base rate and cost of a miss):")
    print(f"{'base':>6s}{'cost_fn':>8s}{'nothing':>9s}{'human':>8s}{'model':>8s}{'m+rev':>8s}  winner")
    for r in sens:
        print(f"{r['base_rate']:6.3f}{r['cost_fn']:8.2f}{r['do_nothing']:9.1f}"
              f"{r['human_checks_everything']:8.1f}{r['model_alone']:8.1f}"
              f"{r['model_plus_review']:8.1f}  {r['winner']}")
    return out


if __name__ == "__main__":
    run()