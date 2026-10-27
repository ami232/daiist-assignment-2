"""Confusion matrix from artifacts/predictions.json. Run: uv run python confusion.py

Rows are the predicted class, columns the true class (Ultralytics convention).
A prediction is kept if its confidence >= CONF; it matches a label if IoU >= IOU_MIN.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
ART = Path(__file__).resolve().parent / "artifacts"
CONF, IOU_MIN = 0.25, 0.45

names = list(json.loads((ART / "metrics.json").read_text())["per_class"])
bg = len(names)
labels = names + ["background"]
preds = json.loads((ART / "predictions.json").read_text())


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


M = np.zeros((bg + 1, bg + 1), dtype=int)
for rec in preds.values():
    P = [p for p in rec["pred"] if p[1] >= CONF]   # [cls, score, x1, y1, x2, y2]
    G = rec["gt"]                                  # [cls, x1, y1, x2, y2]
    pairs = sorted(((iou(g[1:5], p[2:6]), gi, pi)
                    for gi, g in enumerate(G) for pi, p in enumerate(P)), reverse=True)
    used_g, used_p = set(), set()
    for v, gi, pi in pairs:
        if v < IOU_MIN:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        M[int(P[pi][0]), int(G[gi][0])] += 1
    for gi, g in enumerate(G):
        if gi not in used_g:
            M[bg, int(g[0])] += 1                  # missed: predicted background
    for pi, p in enumerate(P):
        if pi not in used_p:
            M[int(p[0]), bg] += 1                  # false alarm on background

norm = M / np.maximum(M.sum(axis=0, keepdims=True), 1)
idx = ["pred " + s for s in labels]
cols = ["true " + s for s in labels]
print(f"Counts (confidence >= {CONF}, IoU >= {IOU_MIN})")
print(pd.DataFrame(M, index=idx, columns=cols))
print("\nShare of each TRUE class (columns sum to 1)")
print(pd.DataFrame(norm.round(2), index=idx, columns=cols))

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(bg + 1), labels, rotation=30)
ax.set_yticks(range(bg + 1), labels)
ax.set_xlabel("true class")
ax.set_ylabel("predicted class")
for r in range(bg + 1):
    for c in range(bg + 1):
        ax.text(c, r, f"{norm[r, c]:.2f}", ha="center", va="center",
                color="white" if norm[r, c] > 0.5 else "black")
fig.tight_layout()
fig.savefig(ART / "confusion_matrix_own.png", dpi=130)
print("\nsaved artifacts/confusion_matrix_own.png")