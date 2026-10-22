"""
Three outcomes per listing: pass, send for human review, or fail.

Two thresholds split the model's confidence score:
  score <  low   -> pass    (listing goes live)
  low <= score < high -> human review (a moderator looks at it)
  score >= high  -> fail    (the seller is asked to declare the defect or retake the photos)

The pair of thresholds is CHOSEN on the validation split and REPORTED on the
test split. It reads the scores that costing.py saved, so it needs no model.

Run on its own with:  uv run python review.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from costing import BASE_RATE, COST_FN, THRESHOLDS, read_scores

# Costs in euros. The reasoning for each is in REPORT.md.
COST_REVIEW = 0.50        # a moderator spends about 1.5 minutes on a listing
COST_WRONG_FAIL = 3.00    # a good listing is failed: seller time, support, risk of leaving
HUMAN_ACCURACY = 0.95     # how often a moderator reaches the right verdict (an assumption)

# Staffing assumptions. PLACEHOLDERS until confirmed in REPORT.md.
LISTINGS_PER_DAY = 20000
MINUTES_PER_REVIEW = 1.5
PRODUCTIVE_HOURS = 6.0    # productive hours per moderator per day


def zones(scores, low, high):
    """Share of listings that pass, go to review, and fail."""
    return (
        float(np.mean(scores < low)),
        float(np.mean((scores >= low) & (scores < high))),
        float(np.mean(scores >= high)),
    )


def three_way_cost(caught, spurious, low, high, base_rate=BASE_RATE, cost_fn=COST_FN,
                   cost_review=COST_REVIEW, cost_wrong_fail=COST_WRONG_FAIL,
                   human_acc=HUMAN_ACCURACY):
    """Expected cost per 1,000 listings, and the share of listings sent to review."""
    d_pass, d_review, d_fail = zones(caught, low, high)       # defective listings
    g_pass, g_review, g_fail = zones(spurious, low, high)     # good listings (proxy)

    # Defective: passing it is a missed defect; failing it is correct and free;
    # reviewing it costs a moderator, who misses it with probability 1 - human_acc.
    cost_defective = d_pass * cost_fn + d_review * (cost_review + (1 - human_acc) * cost_fn)
    # Good: failing it is a wrong fail; reviewing it costs a moderator, who
    # wrongly fails it with probability 1 - human_acc.
    cost_good = g_fail * cost_wrong_fail + g_review * (cost_review + (1 - human_acc) * cost_wrong_fail)

    cost = 1000 * (base_rate * cost_defective + (1 - base_rate) * cost_good)
    review_share = base_rate * d_review + (1 - base_rate) * g_review
    return cost, review_share


def outcome_table(caught, spurious, low, high, base_rate=BASE_RATE):
    """What happens to 1,000 listings: counts of pass / review / fail for each kind."""
    n_defective = 1000 * base_rate
    n_good = 1000 * (1 - base_rate)
    d = zones(caught, low, high)
    g = zones(spurious, low, high)
    return {
        "defective_listings": {"total": n_defective, "pass_(missed)": round(n_defective * d[0], 1),
                               "review": round(n_defective * d[1], 1), "fail_(caught)": round(n_defective * d[2], 1)},
        "good_listings": {"total": n_good, "pass_(correct)": round(n_good * g[0], 1),
                          "review": round(n_good * g[1], 1), "fail_(wrong)": round(n_good * g[2], 1)},
    }


def best_pair(caught, spurious, **kw):
    """Search every pair low <= high and return the cheapest (cost, low, high)."""
    best = None
    for i, low in enumerate(THRESHOLDS):
        for high in THRESHOLDS[i:]:
            cost, _ = three_way_cost(caught, spurious, low, high, **kw)
            if best is None or cost < best[0]:
                best = (cost, float(low), float(high))
    return best


def best_single(caught, spurious, **kw):
    """The model decides alone: one threshold, no review zone."""
    best = None
    for t in THRESHOLDS:
        cost, _ = three_way_cost(caught, spurious, t, t, **kw)
        if best is None or cost < best[0]:
            best = (cost, float(t))
    return best


def staffing(review_share, listings_per_day=LISTINGS_PER_DAY, minutes=MINUTES_PER_REVIEW,
             hours=PRODUCTIVE_HOURS):
    cases = review_share * listings_per_day
    hours_of_work = cases * minutes / 60
    return {"cases_per_day": round(cases), "hours_of_review_per_day": round(hours_of_work, 1),
            "moderators_needed": round(hours_of_work / hours, 1)}


def run_review(out_dir="outputs"):
    out_dir = Path(out_dir)
    v_caught, v_spur = read_scores(out_dir / "image_scores_valid.csv")
    t_caught, t_spur = read_scores(out_dir / "image_scores_test.csv")

    # Choose on validation.
    v_cost, low, high = best_pair(v_caught, v_spur)
    _, single_t = best_single(v_caught, v_spur)

    # Report on test.
    cost3, share3 = three_way_cost(t_caught, t_spur, low, high)
    cost_single, share_single = three_way_cost(t_caught, t_spur, single_t, single_t)
    cost_all, share_all = three_way_cost(t_caught, t_spur, 0.0, 1.01)   # everything to review
    cost_nothing = 1000 * BASE_RATE * COST_FN

    strategies = {
        "do nothing": {"cost_per_1000": round(cost_nothing, 2), "review_share": 0.0},
        "person checks everything": {"cost_per_1000": round(cost_all, 2), "review_share": 1.0},
        f"model decides alone (t={single_t:.2f})": {"cost_per_1000": round(cost_single, 2), "review_share": 0.0},
        f"three-way (low={low:.2f}, high={high:.2f})": {"cost_per_1000": round(cost3, 2),
                                                        "review_share": round(share3, 4)},
    }

    # What if the human or the wrong-fail assumptions are off?
    sensitivity = []
    for wrong_fail in (1.0, 3.0, 6.0):
        for human_acc in (0.90, 0.95, 0.99):
            kw = {"cost_wrong_fail": wrong_fail, "human_acc": human_acc}
            _, lo, hi = best_pair(v_caught, v_spur, **kw)
            c, s = three_way_cost(t_caught, t_spur, lo, hi, **kw)
            c_single, _ = three_way_cost(t_caught, t_spur, *(2 * [best_single(v_caught, v_spur, **kw)[1]]), **kw)
            c_all, _ = three_way_cost(t_caught, t_spur, 0.0, 1.01, **kw)
            sensitivity.append({"cost_wrong_fail": wrong_fail, "human_accuracy": human_acc,
                                "low": lo, "high": hi, "test_cost_per_1000": round(c, 2),
                                "review_share": round(s, 4), "model_alone": round(c_single, 2),
                                "check_everything": round(c_all, 2),
                                "do_nothing": round(cost_nothing, 2)})

    result = {
        "assumptions": {"cost_fn": COST_FN, "cost_review": COST_REVIEW, "cost_wrong_fail": COST_WRONG_FAIL,
                        "human_accuracy": HUMAN_ACCURACY, "base_rate": BASE_RATE,
                        "listings_per_day": LISTINGS_PER_DAY, "minutes_per_review": MINUTES_PER_REVIEW,
                        "productive_hours": PRODUCTIVE_HOURS},
        "chosen": {"low": low, "high": high, "valid_cost_per_1000": round(v_cost, 2),
                   "test_cost_per_1000": round(cost3, 2), "review_share": round(share3, 4),
                   "staffing": staffing(share3),
                   "outcomes_per_1000": outcome_table(t_caught, t_spur, low, high)},
        "strategies": strategies,
        "sensitivity": sensitivity,
    }
    (out_dir / "three_way.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(strategies)
    values = [strategies[n]["cost_per_1000"] for n in names]
    ax.barh(names, values, color=["grey", "grey", "tab:orange", "tab:blue"])
    ax.set_xlabel("expected cost per 1,000 listings (EUR), test split")
    ax.set_title("Four ways to run the check")
    for y, v in enumerate(values):
        ax.text(v, y, f" {v:.0f}", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_costs.png", dpi=150)
    plt.close(fig)

    print("\n=== THREE-WAY DECISION (thresholds chosen on validation, reported on test) ===")
    print(f"Assumptions: missed defect EUR {COST_FN}, review EUR {COST_REVIEW}, wrong fail EUR {COST_WRONG_FAIL}, "
          f"human accuracy {HUMAN_ACCURACY:.0%}, base rate {BASE_RATE:.0%}")
    print(f"Chosen: pass below {low:.2f}, review from {low:.2f} to {high:.2f}, fail from {high:.2f}")
    print("\nStrategy costs on the TEST split (EUR per 1,000 listings):")
    for n, s in strategies.items():
        print(f"  {n:40s} EUR {s['cost_per_1000']:8.2f}   sent to a person: {s['review_share']:.1%}")
    print("\nWhat happens to 1,000 listings at the chosen thresholds (test split):")
    for kind, row in result["chosen"]["outcomes_per_1000"].items():
        print(f"  {kind}: {row}")
    st = result["chosen"]["staffing"]
    print(f"\nStaffing (PLACEHOLDER: {LISTINGS_PER_DAY} listings/day, {MINUTES_PER_REVIEW} min per review, "
          f"{PRODUCTIVE_HOURS} productive hours per moderator):")
    print(f"  {st['cases_per_day']} listings/day to review, {st['hours_of_review_per_day']} hours of work, "
          f"{st['moderators_needed']} moderators")
    print("\nSensitivity (wrong-fail cost, human accuracy -> low, high, test cost | model alone | check everything | do nothing):")
    for s in sensitivity:
        print(f"  EUR {s['cost_wrong_fail']:.0f}, {s['human_accuracy']:.0%}: {s['low']:.2f}/{s['high']:.2f}  "
              f"EUR {s['test_cost_per_1000']:7.2f} ({s['review_share']:.1%} reviewed) | "
              f"alone {s['model_alone']:7.2f} | all {s['check_everything']:7.2f} | nothing {s['do_nothing']:7.2f}")
    print("\nSaved outputs/three_way.json and strategy_costs.png")
    return result


if __name__ == "__main__":
    run_review()