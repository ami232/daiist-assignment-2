"""
Assignment 2: object-level evaluation on the TEST split.

Run on its own:   uv run python evaluation.py
It reads outputs/test_predictions.json (written by train.py) and, if it
exists, the lower threshold chosen in outputs/cost_results.json. It writes
outputs/evaluation.json and outputs/confusion_matrix.png.

A prediction counts as a hit when its box overlaps a real box by at least
50% (IoU 0.5). Rows of the confusion matrix are what was really there,
columns are what the model said. "Background" means nothing: a prediction
in the Background column is a missed object, a real object in the Background
row is a false alarm.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
IOU_MATCH = 0.5
DEFAULT_CONF = 0.25
SIZE_BUCKETS = [
    (0.0, 0.01, "under 1% of the image"),
    (0.01, 0.05, "1% to 5%"),
    (0.05, 0.20, "5% to 20%"),
    (0.20, 1.01, "over 20%"),
]


def iou(a, b):
    """a and b are [x1, y1, x2, y2]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe(gts):
    """Remove repeated labels (same class, almost the same box)."""
    kept = []
    for g in gts:
        if not any(k[0] == g[0] and iou(k[1:], g[1:]) > 0.95 for k in kept):
            kept.append(g)
    return kept


def chosen_conf():
    f = OUTPUT_DIR / "cost_results.json"
    if f.exists():
        try:
            return float(json.loads(f.read_text())["chosen"]["lower_threshold"])
        except Exception:
            pass
    return DEFAULT_CONF


def bucket_of(area):
    for b, (low, high, _) in enumerate(SIZE_BUCKETS):
        if low <= area < high:
            return b
    return len(SIZE_BUCKETS) - 1


def evaluate(records, n_classes, conf):
    bg = n_classes  # index of "Background"
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    total = np.zeros((n_classes, len(SIZE_BUCKETS)), dtype=int)
    found = np.zeros((n_classes, len(SIZE_BUCKETS)), dtype=int)

    for rec in records:
        gts = dedupe(rec["gt"])
        preds = sorted((p for p in rec["pred"] if p[1] >= conf), key=lambda p: -p[1])
        matched, correct = set(), set()
        for p in preds:
            best_j, best_iou = None, IOU_MATCH
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                v = iou(p[2:], g[1:])
                if v >= best_iou:
                    best_j, best_iou = j, v
            if best_j is None:
                m[bg][p[0]] += 1  # the model saw something where there was nothing
            else:
                matched.add(best_j)
                m[gts[best_j][0]][p[0]] += 1
                if gts[best_j][0] == p[0]:
                    correct.add(best_j)
        for j, g in enumerate(gts):
            if j not in matched:
                m[g[0]][bg] += 1  # a real object the model missed
            area = max(0.0, g[3] - g[1]) * max(0.0, g[4] - g[2])
            b = bucket_of(area)
            total[g[0]][b] += 1
            if j in correct:
                found[g[0]][b] += 1
    return m, total, found


def save_plot(m, labels, conf, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("What the model said")
    ax.set_ylabel("What was really there")
    ax.set_title(f"Confusion matrix on the test images\n(confidence at least {conf:.2f})")
    top = m.max() if m.max() > 0 else 1
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            ax.text(c, r, str(m[r][c]), ha="center", va="center",
                    color="white" if m[r][c] > top / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(conf=None):
    data = json.loads((OUTPUT_DIR / "test_predictions.json").read_text())
    names = data["names"]
    conf = chosen_conf() if conf is None else conf
    m, total, found = evaluate(data["images"], len(names), conf)
    labels = names + ["Background"]

    per_class = {}
    for c, name in enumerate(names):
        tp = int(m[c][c])
        said = int(m[:, c].sum())
        real = int(m[c, :].sum())
        per_class[name] = {
            "objects": real,
            "hits": tp,
            "precision": tp / said if said else 0.0,
            "recall": tp / real if real else 0.0,
            "recall_by_size": {
                SIZE_BUCKETS[b][2]: {
                    "objects": int(total[c][b]),
                    "found": int(found[c][b]),
                    "recall": float(found[c][b] / total[c][b]) if total[c][b] else None,
                }
                for b in range(len(SIZE_BUCKETS))
            },
        }

    result = {
        "confidence_used": conf,
        "iou_match": IOU_MATCH,
        "labels": labels,
        "matrix_rows_actual_columns_predicted": m.tolist(),
        "per_class": per_class,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(result, indent=2))
    try:
        save_plot(m, labels, conf, OUTPUT_DIR / "confusion_matrix.png")
    except Exception as error:
        print("Could not draw the confusion matrix:", error)

    print(f"\n=== EVALUATION (confidence at least {conf:.2f}, overlap at least {IOU_MATCH}) ===")
    print("Rows = what was really there, columns = what the model said")
    print(" " * 12 + "".join(f"{l:>12}" for l in labels))
    for r, l in enumerate(labels):
        print(f"{l:>12}" + "".join(f"{m[r][c]:>12}" for c in range(len(labels))))
    for name, row in per_class.items():
        print(f"\n{name}: {row['objects']} real objects, {row['hits']} found correctly, "
              f"precision {row['precision']:.3f}, recall {row['recall']:.3f}")
        for size, s in row["recall_by_size"].items():
            if s["recall"] is None:
                continue
            print(f"   size {size}: {s['found']} of {s['objects']} found ({s['recall']:.0%})")
    return result


if __name__ == "__main__":
    run_evaluation()