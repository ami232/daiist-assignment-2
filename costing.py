"""
Assignment 2: costing, threshold choice and human review.

Run on its own:   uv run python costing.py
It reads outputs/test_predictions.json (written by train.py) and writes
outputs/cost_results.json and outputs/cost_curve.png.

One "case" is one image. The positive class is Fire. A case is a real fire
if the image has at least one Fire box. The model's score for a case is the
highest confidence among its Fire boxes (0 if it found none).

Three outcomes, using two thresholds:
  score below the lower threshold  -> PASS (nothing happens)
  between the two thresholds       -> REVIEW (a person looks at the image)
  score at or above the upper one  -> FAIL (the firemen are called directly)
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
POSITIVE_CLASS = "Fire"

# Every number here is explained in REPORT.md. All money is in euros.
ASSUMPTIONS = {
    "cost_missed_fire": 1_000_000,    # a real fire that is passed (found late)
    "cost_false_callout": 1_000,      # firemen called for nothing
    "cost_review": 0.20,              # one person checks one image
    "cost_review_delay": 10_000,      # extra fire damage when a real fire waits for the person
    "base_rate": 0.001,               # 1 in 1,000 images is a real fire
    "cases_per_day": 5_000,           # images arriving per day
    "seconds_per_review": 30,         # time for one check
    "review_capacity_per_day": 1_000, # checks the team can really do per day
    "hours_per_operator_day": 8,
}

# Thresholds to try. 1.01 on the upper threshold means "never call directly".
GRID = np.round(np.arange(0.0, 1.0101, 0.01), 2)


def load_scores():
    """Return (scores of real-fire images, scores of no-fire images)."""
    data = json.loads((OUTPUT_DIR / "test_predictions.json").read_text())
    fire_id = data["names"].index(POSITIVE_CLASS)
    pos, neg = [], []
    for rec in data["images"]:
        has_fire = any(g[0] == fire_id for g in rec["gt"])
        confs = [p[1] for p in rec["pred"] if p[0] == fire_id]
        score = max(confs) if confs else 0.0
        (pos if has_fire else neg).append(score)
    return np.array(pos), np.array(neg)


def tables(scores):
    """For every threshold in GRID: share of cases below it, and at or above it."""
    below = np.array([(scores < g).mean() for g in GRID])
    above = np.array([(scores >= g).mean() for g in GRID])
    return below, above


def cost_grid(pos, neg, a):
    """Expected cost per case for every (lower, upper) pair, and the share sent
    to a person. Rows = lower threshold, columns = upper threshold."""
    pos_below, pos_above = tables(pos)
    neg_below, neg_above = tables(neg)

    pos_pass, pos_fail = pos_below[:, None], pos_above[None, :]
    neg_pass, neg_fail = neg_below[:, None], neg_above[None, :]
    pos_review = 1 - pos_pass - pos_fail
    neg_review = 1 - neg_pass - neg_fail

    p = a["base_rate"]
    cost_if_fire = (
        pos_pass * a["cost_missed_fire"]
        + pos_review * (a["cost_review"] + a["cost_review_delay"])
    )
    cost_if_no_fire = (
        neg_review * a["cost_review"] + neg_fail * a["cost_false_callout"]
    )
    cost = p * cost_if_fire + (1 - p) * cost_if_no_fire
    review_share = p * pos_review + (1 - p) * neg_review

    valid = GRID[:, None] <= GRID[None, :]  # lower threshold <= upper threshold
    return np.where(valid, cost, np.inf), review_share


def best_point(pos, neg, a, max_share=None):
    """Cheapest (lower, upper) pair, optionally limited by how many cases a
    person can check."""
    cost, share = cost_grid(pos, neg, a)
    if max_share is not None:
        cost = np.where(share <= max_share + 1e-12, cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return i, j, float(cost[i, j]), float(share[i, j])


def workload(review_share, a):
    reviews = review_share * a["cases_per_day"]
    hours = reviews * a["seconds_per_review"] / 3600
    return {
        "reviews_per_day": reviews,
        "hours_per_day": hours,
        "operators_needed": hours / a["hours_per_operator_day"],
    }


def confusion(pos, neg, t_low, t_high, a):
    p = a["base_rate"]
    out = {}
    for name, scores, expected in (
        ("fire", pos, 1000 * p),
        ("no_fire", neg, 1000 * (1 - p)),
    ):
        n = len(scores)
        passed = int((scores < t_low).sum())
        fail = int((scores >= t_high).sum())
        review = n - passed - fail
        out[name] = {
            "test_images": n,
            "pass": passed,
            "review": review,
            "fail": fail,
            "per_1000_cases": {
                "pass": expected * passed / n,
                "review": expected * review / n,
                "fail": expected * fail / n,
            },
        }
    return out


def sensitivity(pos, neg, a):
    """Change one assumption at a time and see what the best plan does."""
    rows = []
    changes = (
        ("base_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        ("cost_missed_fire", [50_000, 250_000, 1_000_000, 4_000_000]),
        ("review_capacity_per_day", [0, 250, 1_000, 2_500]),
    )
    for key, values in changes:
        for v in values:
            b = dict(a)
            b[key] = v
            max_share = b["review_capacity_per_day"] / b["cases_per_day"]
            i, j, c, share = best_point(pos, neg, b, max_share)
            rows.append({
                "changed": key,
                "value": v,
                "lower": float(GRID[i]),
                "upper": float(GRID[j]),
                "cost_per_1000": c * 1000,
                "review_share": share,
            })
    return rows


def save_plot(cost, share, i, j, max_share, capacity, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alone = np.diag(cost) * 1000
    with_review = cost[:, j] * 1000
    ok = np.isfinite(with_review)
    allowed = ok & (share[:, j] <= max_share + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(GRID, alone, color="tab:blue",
            label="Model alone (one threshold, nobody checks)")
    ax.plot(GRID[ok], with_review[ok], color="tab:orange", linestyle="--", alpha=0.6,
            label="With human review: needs more checks than the team can do")
    ax.plot(GRID[allowed], with_review[allowed], color="tab:orange",
            label=f"With human review: within {capacity:,} checks a day "
                  f"(upper threshold {GRID[j]:.2f})")

    # Cheapest point of the model-alone curve
    k = int(np.argmin(alone))
    ax.scatter([GRID[k]], [alone[k]], marker="^", color="tab:blue", s=70, zorder=5,
               label=f"Cheapest model alone: threshold {GRID[k]:.2f}")
    # Cheapest point if people had no limit
    u = np.where(ok)[0][int(np.argmin(with_review[ok]))]
    ax.scatter([GRID[u]], [with_review[u]], facecolors="none", edgecolors="black",
               s=110, linewidths=1.5, zorder=5,
               label=f"Cheapest if people had no limit: lower threshold {GRID[u]:.2f}")
    # My chosen point
    ax.scatter([GRID[i]], [cost[i, j] * 1000], color="red", s=70, zorder=6,
               label=f"My choice (cheapest within the limit): lower threshold {GRID[i]:.2f}")

    ax.set_yscale("log")
    ax.set_xlabel("Confidence threshold (lower threshold for the orange line)")
    ax.set_ylabel("Expected cost per 1,000 cases (euros, log scale)")
    ax.set_title("Cost against threshold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clean(obj):
    """Make results safe to save as JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_costing(a=None):
    a = dict(ASSUMPTIONS if a is None else a)
    p = a["base_rate"]
    pos, neg = load_scores()
    max_share = a["review_capacity_per_day"] / a["cases_per_day"]

    cost, share = cost_grid(pos, neg, a)

    # Best plan with people, limited by how many checks the team can do.
    i, j, best_cost, best_share = best_point(pos, neg, a, max_share)
    t_low, t_high = float(GRID[i]), float(GRID[j])

    # Best plan if people had no limit (to show why the limit matters).
    ui, uj, ucost, ushare = best_point(pos, neg, a, None)

    # Model alone: one threshold, nobody checks.
    diag = np.diag(cost)
    k = int(np.argmin(diag))

    alternatives = {
        "do_nothing": p * a["cost_missed_fire"] * 1000,
        "person_checks_everything": (
            p * (a["cost_review"] + a["cost_review_delay"])
            + (1 - p) * a["cost_review"]
        ) * 1000,
        "model_alone": float(diag[k]) * 1000,
        "model_plus_people": best_cost * 1000,
    }

    result = {
        "assumptions": a,
        "positive_class": POSITIVE_CLASS,
        "test_images": {"fire": len(pos), "no_fire": len(neg)},
        "chosen": {"lower_threshold": t_low, "upper_threshold": t_high,
                   "cost_per_1000": best_cost * 1000, "review_share": best_share},
        "chosen_workload": workload(best_share, a),
        "unlimited_people": {
            "lower_threshold": float(GRID[ui]), "upper_threshold": float(GRID[uj]),
            "cost_per_1000": ucost * 1000, "review_share": ushare,
            "workload": workload(ushare, a),
        },
        "model_alone": {"threshold": float(GRID[k]), "cost_per_1000": float(diag[k]) * 1000},
        "alternatives_cost_per_1000": alternatives,
        "everyone_checked_workload": workload(1.0, a),
        "confusion": confusion(pos, neg, t_low, t_high, a),
        "sweep": {
            "thresholds": GRID,
            "model_alone_per_1000": diag * 1000,
            "with_review_per_1000": cost[:, j] * 1000,
        },
        "sensitivity": sensitivity(pos, neg, a),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "cost_results.json").write_text(json.dumps(clean(result), indent=2))
    try:
        save_plot(cost, share, i, j, max_share, a["review_capacity_per_day"],
                  OUTPUT_DIR / "cost_curve.png")
    except Exception as error:  # the numbers matter more than the picture
        print("Could not draw the plot:", error)

    w = result["chosen_workload"]
    print("\n=== COST RESULTS ===")
    print(f"Test images: {len(pos)} with fire, {len(neg)} without fire (smoke only)")
    print(f"Chosen thresholds: lower {t_low:.2f}, upper {t_high:.2f}"
          + (" (upper 1.01 = never call the firemen directly)" if t_high > 1 else ""))
    print(f"Expected cost per 1,000 cases: EUR {best_cost * 1000:,.0f}")
    print(f"Sent to a person: {best_share * 100:.1f}% of cases = "
          f"{w['reviews_per_day']:.0f} per day = {w['hours_per_day']:.1f} hours of checking "
          f"= {w['operators_needed']:.2f} operators")
    print("\nWhat happens to the test images at these thresholds:")
    for name, row in result["confusion"].items():
        print(f"  {name}: {row['test_images']} images -> pass {row['pass']}, "
              f"review {row['review']}, fail {row['fail']}")
    print("\nCost per 1,000 cases, compared:")
    labels = {
        "do_nothing": "Do nothing",
        "person_checks_everything": "A person checks everything",
        "model_alone": "Model decides alone",
        "model_plus_people": "Model plus people (chosen)",
    }
    for key, label in labels.items():
        print(f"  {label}: EUR {alternatives[key]:,.0f}")
    print(f"\nIf people had no limit: lower {GRID[ui]:.2f}, upper {GRID[uj]:.2f}, "
          f"EUR {ucost * 1000:,.0f} per 1,000 cases, "
          f"{ushare * 100:.0f}% sent to a person "
          f"({result['unlimited_people']['workload']['operators_needed']:.1f} operators)")
    print("\nWhat changes if one assumption changes:")
    for row in result["sensitivity"]:
        print(f"  {row['changed']} = {row['value']:,}: lower {row['lower']:.2f}, "
              f"upper {row['upper']:.2f}, EUR {row['cost_per_1000']:,.0f} per 1,000, "
              f"{row['review_share'] * 100:.1f}% reviewed")
    return result


if __name__ == "__main__":
    run_costing()