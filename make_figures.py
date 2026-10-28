"""
This builds the figures embedded in REPORT.md from the artifacts train.ipynb already saved.
Nothing is trained or re-selected here: it only reads artifacts/ and draws.

    uv run python make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART, OUT = ROOT / "artifacts", ROOT / "figures"
OUT.mkdir(exist_ok=True)

M = json.loads((ART / "metrics.json").read_text())
R = json.loads((ART / "policy_robustness.json").read_text())
VAL, TEST = pd.read_csv(ART / "val_cases.csv"), pd.read_csv(ART / "test_cases.csv")
SEARCH = pd.read_csv(ART / "review_threshold_search_validation.csv")

COST, P = M["cost_euros"], M["base_rate_assumed"]
LOW, HIGH, SINGLE = M["low_threshold"], M["high_threshold"], M["single_threshold"]
MAX_SHARE = M["max_review_share"]
TEAL, AMBER, RED, GREY, INK = "#2a9d8f", "#e0a100", "#d6452d", "#8d99ae", "#1f2937"


def actions(df, low, high):
    s = df["unsafe_score"].to_numpy(float)
    a = np.where(s >= high, 2, np.where(s < low, 0, 1))
    return np.where(df["no_evidence"].to_numpy(bool), 1, a)


def cost_and_share(df, low, high):
    """Same maths as train.ipynb: conditional rates re-weighted to the assumed base rate."""
    truth, a = df["unsafe_gt"].to_numpy(int), actions(df, low, high)
    m = np.array([[((truth == i) & (a == j)).sum() for j in range(3)] for i in (0, 1)])
    r = m / m.sum(axis=1, keepdims=True)
    neg = r[0] @ [COST["tn"], COST["review"], COST["fp"]]
    pos = r[1] @ [COST["fn"], COST["review"] + COST["tp"], COST["tp"]]
    return 1000 * ((1 - P) * neg + P * pos), (1 - P) * r[0, 1] + P * r[1, 1]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)


# Figure 1: cost against threshold
grid = np.unique(np.round(np.geomspace(0.001, 1.0, 140), 5))
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)

# (a) model decides alone: one threshold (pass below, flag at or above)

def alone_cost(df, t):
    truth = df["unsafe_gt"].to_numpy(int)
    flag = (df["unsafe_score"].to_numpy(float) >= t).astype(int)
    fp_rate = ((truth == 0) & (flag == 1)).sum() / (truth == 0).sum()
    fn_rate = ((truth == 1) & (flag == 0)).sum() / (truth == 1).sum()
    return 1000 * ((1 - P) * fp_rate * COST["fp"] + P * (fn_rate * COST["fn"] + (1 - fn_rate) * COST["tp"]))


av = np.array([alone_cost(VAL, t) for t in grid])
at = np.array([alone_cost(TEST, t) for t in grid])
a.plot(grid, av, color=TEAL, lw=2.6, label="Validation (used to choose)")
a.plot(grid, at, color=TEAL, lw=1.6, ls=":", label="Test (held out)")
a.plot([SINGLE], [alone_cost(VAL, SINGLE)], "*", ms=16, color=INK, label=f"Minimum on validation: {SINGLE:.4f}")
a.axhline(1000 * (COST["review"] + P * COST["tp"]), color=AMBER, ls="--", lw=1.5, label="Person checks every image (€990)")
a.set_xscale("log")
a.set_ylim(0, 3000)
a.set_title("Model decides alone (pass / flag only)", fontsize=11, loc="left")
a.set_xlabel("Confidence threshold (flag if top head score ≥ threshold)")
a.set_ylabel("Expected cost per 1,000 images (€)")
a.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
style(a)

# (b) pass / check / flag: sweep the pass boundary, flag boundary fixed at the chosen value
lows = grid[grid <= HIGH]
cv = np.array([cost_and_share(VAL, t, HIGH) for t in lows])
ct = np.array([cost_and_share(TEST, t, HIGH) for t in lows])
infeasible = lows[cv[:, 1] > MAX_SHARE + 1e-12]
if len(infeasible):
    b.axvspan(grid.min(), infeasible.max(), color=RED, alpha=0.08, lw=0)
    b.text(infeasible.max() * 1.08, 2850, "shaded: needs more than\nthe 120 reviewer min/day", ha="left", va="top", fontsize=8, color=RED)
b.plot(lows, cv[:, 0], color=TEAL, lw=2.6, label="Validation (used to choose)")
b.plot(lows, ct[:, 0], color=TEAL, lw=1.6, ls=":", label="Test (held out)")
v_star = cost_and_share(VAL, LOW, HIGH)[0]
b.plot([LOW], [v_star], "*", ms=16, color=INK, label=f"Chosen pass boundary: {LOW:.5f}")
b.axhline(1000 * (COST["review"] + P * COST["tp"]), color=AMBER, ls="--", lw=1.5, label="Person checks every image (€990)")
b.set_xscale("log")
b.set_ylim(0, 3000)
b.set_title(f"Pass / person checks / flag (flag boundary fixed at {HIGH:.2f})", fontsize=11, loc="left")
b.set_xlabel("Pass boundary (pass if top head score < boundary)")
b.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
style(b)
fig.suptitle(f"Cost against threshold at an assumed {P:.0%} unsafe images; missed case €{COST['fn']:.0f}, false alarm €{COST['fp']:.0f}", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "cost_vs_threshold.png")
plt.close(fig)

# Figure 2: what the reviewer's miss rate does to each way of working
sweep = pd.DataFrame(R["reviewer_miss_rate_sweep"])
x = sweep["miss_rate"] * 100
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
ax.plot(x, sweep["review_all"], color=AMBER, lw=2.4, marker="o", ms=4, label="Person checks every image")
ax.plot(x, sweep["hybrid"], color=TEAL, lw=2.8, marker="o", ms=4, label="Model + person (submitted)")
ax.plot(x, sweep["model_alone"], color=GREY, lw=2.2, ls="--", label="Model decides alone")
ax.set_ylim(0, 7000)
ax.annotate("Do nothing: €30,000 (off the scale)", xy=(0.2, 6800), fontsize=9, color=RED, va="top")
ax.set_xlabel("Share of unsafe images a human reviewer misses (%)")
ax.set_ylabel("Expected cost per 1,000 images (€)")
ax.set_title("The reviewer's miss rate decides whether the model earns its place", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.62))
style(ax)
fig.tight_layout()
fig.savefig(OUT / "reviewer_miss_rate.png")
plt.close(fig)

print("Wrote", *(p.name for p in sorted(OUT.glob("*.png"))))
