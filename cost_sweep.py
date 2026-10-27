"""Cost sweep and human-review band. Run: uv run python cost_sweep.py

A frame's score is the highest 'pothole' confidence in it.
  score >= hi        -> flag it, send a repair crew
  lo <= score < hi   -> a person reviews it (assumed to always be right)
  score < lo         -> pass
TPR/FPR come from the test split and are re-weighted by an assumed base rate.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# assumptions
COST_FP = 150.0              # EUR: crew sent to a frame with no pothole
COST_FN = 1000.0             # EUR: pothole left unrepaired
BASE_RATES = [0.005, 0.01, 0.05, 0.42]
CENTRAL = 0.01
FRAMES_PER_DAY = 43_200      # 2 vehicles x 12 h x 1,800 frames/h (1 frame / 2 s)
REVIEWS_PER_PERSON_PER_DAY = 400
REVIEWER_EUR_PER_HOUR = 25.0
REVIEWER_HOURS_PER_DAY = 8.0
N_REVIEWERS = 5              # people available
POSITIVE = "pothole"

REVIEW_COST = REVIEWER_EUR_PER_HOUR * REVIEWER_HOURS_PER_DAY / REVIEWS_PER_PERSON_PER_DAY
PER_PERSON = REVIEWS_PER_PERSON_PER_DAY
SHARE_MAX = N_REVIEWERS * PER_PERSON / FRAMES_PER_DAY   # max share of frames a team can review

metrics = json.loads((ART / "metrics.json").read_text())
pid = list(metrics["per_class"]).index(POSITIVE)
preds = json.loads((ART / "predictions.json").read_text())

scores, labels = [], []
for rec in preds.values():
    conf = [p[1] for p in rec["pred"] if p[0] == pid]
    scores.append(max(conf, default=0.0))
    labels.append(any(g[0] == pid for g in rec["gt"]))
scores, labels = np.array(scores), np.array(labels)
n_pos, n_neg = int(labels.sum()), int((~labels).sum())

thr = np.round(np.linspace(0, 1, 201), 3)
tpr = np.array([(scores[labels] >= t).sum() / n_pos for t in thr])
fpr = np.array([(scores[~labels] >= t).sum() / n_neg for t in thr])
n = len(thr)
I, J = np.meshgrid(range(n), range(n), indexing="ij")  # I = lo index, J = hi index
valid = J >= I


def band(pi):
    """Cost per 1,000 frames and review share for every (lo, hi) pair."""
    share = pi * (tpr[I] - tpr[J]) + (1 - pi) * (fpr[I] - fpr[J])
    cost = 1000 * (pi * (1 - tpr[I]) * COST_FN
                   + (1 - pi) * fpr[J] * COST_FP
                   + share * REVIEW_COST)
    return cost, share


def best(cost, mask):
    c = np.where(mask, cost, np.inf)
    k = np.unravel_index(np.argmin(c), c.shape)
    return int(k[0]), int(k[1])


def people_needed(share):
    return math.ceil(share * FRAMES_PER_DAY / PER_PERSON)


result = {
    "positive_class": POSITIVE, "cost_fp": COST_FP, "cost_fn": COST_FN,
    "review_cost_per_frame": REVIEW_COST, "frames_per_day": FRAMES_PER_DAY,
    "reviews_per_person_per_day": REVIEWS_PER_PERSON_PER_DAY, "n_reviewers": N_REVIEWERS,    "frames_per_person_per_day": PER_PERSON,
    "central_base_rate": CENTRAL,
    "test_frames": {"positive": n_pos, "negative": n_neg},
    "thresholds": thr.tolist(), "tpr": tpr.tolist(), "fpr": fpr.tolist(),
    "by_base_rate": {},
}
print(f"Review cost {REVIEW_COST:.4f} EUR/frame; {N_REVIEWERS} people can review "
      f"{N_REVIEWERS * PER_PERSON:,.0f} frames/day = {SHARE_MAX:.1%} of {FRAMES_PER_DAY:,}\n")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, pi in zip(axes.ravel(), BASE_RATES):
    cost, share = band(pi)
    i1, j1 = best(cost, valid & (I == J))                  # model alone
    i2, j2 = best(cost, valid & (share <= SHARE_MAX))      # band, staffed
    i3, j3 = best(cost, valid)                             # band, unlimited staff
    nothing = 1000 * pi * COST_FN
    review_all = 1000 * REVIEW_COST
    row = {
        "do_nothing": nothing, "review_everything": review_all,
        "people_for_review_everything": people_needed(1.0),
        "model_alone": {"threshold": float(thr[i1]), "cost": float(cost[i1, j1])},
        "hybrid_staffed": {
            "lo": float(thr[i2]), "hi": float(thr[j2]), "cost": float(cost[i2, j2]),
            "review_share": float(share[i2, j2]),
            "reviews_per_day": float(share[i2, j2] * FRAMES_PER_DAY),
            "people": people_needed(share[i2, j2]),
            "tpr_lo": float(tpr[i2]), "fpr_hi": float(fpr[j2])},
        "hybrid_unlimited": {
            "lo": float(thr[i3]), "hi": float(thr[j3]), "cost": float(cost[i3, j3]),
            "review_share": float(share[i3, j3]),
            "reviews_per_day": float(share[i3, j3] * FRAMES_PER_DAY),
            "people": people_needed(share[i3, j3])},
    }
    result["by_base_rate"][str(pi)] = row
    h, u = row["hybrid_staffed"], row["hybrid_unlimited"]
    print(f"base rate {pi:.1%}  (EUR per 1,000 frames)")
    print(f"  do nothing          {nothing:>10,.0f}")
    print(f"  review everything   {review_all:>10,.0f}   needs {people_needed(1.0)} people")
    print(f"  model alone         {cost[i1, j1]:>10,.0f}   t={thr[i1]:.3f}")
    print(f"  hybrid, {N_REVIEWERS} people    {h['cost']:>10,.0f}   lo={h['lo']:.3f} hi={h['hi']:.3f}  "
          f"review {h['review_share']:.1%} = {h['reviews_per_day']:,.0f}/day")
    print(f"  hybrid, unlimited   {u['cost']:>10,.0f}   lo={u['lo']:.3f} hi={u['hi']:.3f}  "
          f"review {u['review_share']:.1%} = {u['reviews_per_day']:,.0f}/day "
          f"-> {u['people']} people\n")
    single = cost[np.arange(n), np.arange(n)]
    ax.plot(thr, single, label="model alone")
    ax.axhline(nothing, ls="--", color="gray", label="do nothing")
    ax.axvline(thr[i2], color="green", ls=":", label=f"lo={thr[i2]:.2f}")
    ax.axvline(thr[j2], color="red", ls=":", label=f"hi={thr[j2]:.2f}")
    ax.set_title(f"base rate {pi:.1%}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("EUR per 1,000 frames")
    ax.legend(fontsize=7)

c = result["by_base_rate"][str(CENTRAL)]["hybrid_staffed"]
result["chosen"] = {"lo": c["lo"], "hi": c["hi"], "base_rate": CENTRAL}
fig.tight_layout()
fig.savefig(ART / "cost_vs_threshold.png", dpi=130)
(ART / "cost_sweep.json").write_text(json.dumps(result))
print("saved artifacts/cost_sweep.json and cost_vs_threshold.png")