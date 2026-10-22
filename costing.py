"""
Costing: turn the model's confidence scores into euros.

For every image we keep two scores:
  score_caught    highest confidence of a box that overlaps a real defect
  score_spurious  highest confidence of a box that overlaps nothing real

A defective listing is caught at threshold t if score_caught >= t.
A good listing is falsely flagged at threshold t if score_spurious >= t
(a proxy: our images all contain defects, so stray boxes stand in for the
false alarms a good garment would cause).

The threshold is CHOSEN on the validation split and REPORTED on the test split,
so the test number is not tuned on the images it describes.

Run on its own with:  uv run python costing.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# Cost matrix in euros (the reasoning is in REPORT.md).
COST_FN = 13.0   # a defect is missed: return 4 + support 4 + goodwill 5
COST_FP = 0.50   # a good listing is flagged: about 1.5 minutes of moderator time
COST_TP = 0.0    # a defect is correctly flagged: no extra cost
COST_TN = 0.0    # a good listing is correctly passed: no extra cost
BASE_RATE = 0.05  # assumed share of listings that really have an undeclared defect
IOU_MATCH = 0.5   # a box counts as overlapping a defect at this IoU (same as AP50)
THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 2)

# Pink and purple colour scale for the confusion matrix.
PINK_PURPLE = LinearSegmentedColormap.from_list(
    "pinkpurple", ["#FFF5FB", "#F4B6D8", "#E0218A", "#7B2CBF", "#3B0F66"]
)


def read_label_rows(label_file, width, height):
    """Read YOLO labels: boxes as pixel corners (x1, y1, x2, y2) and their class numbers."""
    boxes, classes = [], []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            xc, yc, w, h = (float(v) for v in parts[1:5])
            boxes.append([(xc - w / 2) * width, (yc - h / 2) * height,
                          (xc + w / 2) * width, (yc + h / 2) * height])
            classes.append(int(parts[0]))
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), classes


def read_labels(label_file, width, height):
    """Same as read_label_rows, but only the boxes."""
    return read_label_rows(label_file, width, height)[0]


def collect_scores(model, folder, split, device, imgsz):
    """Predict on every image of a split and record the two scores described above."""
    folder = Path(folder)
    images = sorted(p for p in (folder / split / "images").iterdir() if p.is_file())
    rows = []
    for path in images:
        # One image per call. A list of paths is run as one big batch, and the
        # non-max-suppression time limit then silently drops most predictions.
        result = model.predict(
            source=str(path),
            conf=0.01,
            imgsz=imgsz,
            device=device,
            max_det=100,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        gt = read_labels(folder / split / "labels" / f"{path.stem}.txt", width, height)
        pred = result.boxes.xyxy.cpu()
        conf = result.boxes.conf.cpu()
        caught = 0.0
        spurious = 0.0
        if len(pred):
            if len(gt):
                matched = (box_iou(gt, pred) >= IOU_MATCH).any(dim=0)
            else:
                matched = torch.zeros(len(pred), dtype=torch.bool)
            if matched.any():
                caught = float(conf[matched].max())
            if (~matched).any():
                spurious = float(conf[~matched].max())
        rows.append({
            "image": path.name,
            "n_defects": len(gt),
            "score_caught": round(caught, 4),
            "score_spurious": round(spurious, 4),
        })
    return rows


def class_confusion(model, folder, split, threshold, device, imgsz, class_names):
    """Confusion matrix at one threshold, box by box.

    matrix[predicted][true], with 'background' as the last row and column.
    A box that overlaps a real defect (IoU >= 0.5) is matched one to one.
    A real defect with no box is a miss (predicted background).
    A box that matches nothing is a stray box (true background).
    """
    n = len(class_names)
    matrix = np.zeros((n + 1, n + 1), dtype=int)
    folder = Path(folder)
    images = sorted(p for p in (folder / split / "images").iterdir() if p.is_file())
    for path in images:
        result = model.predict(source=str(path), conf=threshold, imgsz=imgsz,
                               device=device, max_det=100, verbose=False)[0]
        height, width = result.orig_shape
        gt, gt_cls = read_label_rows(folder / split / "labels" / f"{path.stem}.txt", width, height)
        pred = result.boxes.xyxy.cpu()
        pred_cls = result.boxes.cls.cpu().long().tolist()
        used_gt, used_pred = set(), set()
        if len(gt) and len(pred):
            iou = box_iou(gt, pred)
            pairs = (iou >= IOU_MATCH).nonzero().tolist()
            pairs.sort(key=lambda gp: -float(iou[gp[0], gp[1]]))
            for g, p in pairs:
                if g in used_gt or p in used_pred:
                    continue
                used_gt.add(g)
                used_pred.add(p)
                matrix[pred_cls[p], gt_cls[g]] += 1
        for g in range(len(gt)):
            if g not in used_gt:
                matrix[n, gt_cls[g]] += 1
        for p in range(len(pred)):
            if p not in used_pred:
                matrix[pred_cls[p], n] += 1
    return matrix


def confusion_figure(matrix, labels, title):
    """Draw a confusion matrix in pink and purple (rows = predicted, columns = true)."""
    matrix = np.asarray(matrix)
    fig = Figure(figsize=(6.5, 5.5))
    ax = fig.subplots()
    im = ax.imshow(matrix, cmap=PINK_PURPLE, vmin=0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("True (what was really there)")
    ax.set_ylabel("Predicted (what the model said)")
    ax.set_title(title)
    top = matrix.max() if matrix.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == len(labels) - 1 and j == len(labels) - 1:
                continue
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if matrix[i, j] > top * 0.5 else "#2B0A3D")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def read_scores(path):
    """Read a saved scores file and return (score_caught, score_spurious) as arrays."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    caught = np.array([float(r["score_caught"]) for r in rows])
    spurious = np.array([float(r["score_spurious"]) for r in rows])
    return caught, spurious


def cost_at(threshold, caught, spurious, base_rate=BASE_RATE, cost_fn=COST_FN, cost_fp=COST_FP):
    """Expected cost per 1,000 listings at one threshold."""
    recall = float(np.mean(caught >= threshold))      # share of defects caught
    fpr = float(np.mean(spurious >= threshold))       # share of good listings flagged
    cost_defective = recall * COST_TP + (1 - recall) * cost_fn
    cost_good = fpr * cost_fp + (1 - fpr) * COST_TN
    cost = 1000 * (base_rate * cost_defective + (1 - base_rate) * cost_good)
    return cost, recall, fpr


def make_sweep(caught, spurious):
    sweep = []
    for t in THRESHOLDS:
        cost, recall, fpr = cost_at(t, caught, spurious)
        sweep.append({"threshold": float(t), "cost_per_1000": round(cost, 2),
                      "recall": round(recall, 4), "false_alarm_rate": round(fpr, 4)})
    return sweep


def save_scores(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_costing(weights, folder, device, imgsz, out_dir):
    """Choose the threshold on validation, report on test, check other assumptions, save."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    model = YOLO(str(weights))

    valid_rows = collect_scores(model, folder, "valid", device, imgsz)
    test_rows = collect_scores(model, folder, "test", device, imgsz)
    save_scores(valid_rows, out_dir / "image_scores_valid.csv")
    save_scores(test_rows, out_dir / "image_scores_test.csv")

    v_caught = np.array([r["score_caught"] for r in valid_rows])
    v_spur = np.array([r["score_spurious"] for r in valid_rows])
    t_caught = np.array([r["score_caught"] for r in test_rows])
    t_spur = np.array([r["score_spurious"] for r in test_rows])

    sweep_valid = make_sweep(v_caught, v_spur)
    sweep_test = make_sweep(t_caught, t_spur)

    # The threshold is chosen on validation only.
    best_valid = min(sweep_valid, key=lambda s: s["cost_per_1000"])
    chosen = best_valid["threshold"]
    test_cost, test_recall, test_fpr = cost_at(chosen, t_caught, t_spur)
    best_test = min(sweep_test, key=lambda s: s["cost_per_1000"])  # for comparison only
    do_nothing = 1000 * BASE_RATE * COST_FN

    # Confusion matrix on the test split at the chosen threshold.
    names = [model.names[i] for i in sorted(model.names)]
    labels = names + ["background"]
    matrix = class_confusion(model, folder, "test", chosen, device, imgsz, names)
    (out_dir / "confusion_matrix.json").write_text(json.dumps(
        {"split": "test", "threshold": chosen, "labels": labels, "matrix": matrix.tolist()}, indent=2))
    confusion_figure(matrix, labels, f"Test split, threshold {chosen:.2f}").savefig(
        out_dir / "test_confusion_matrix_at_threshold.png", dpi=150)

    sensitivity = []
    for base in (0.02, 0.05, 0.10):
        for cost_fn in (5.0, 13.0, 30.0):
            options = [(cost_at(t, v_caught, v_spur, base, cost_fn)[0], float(t)) for t in THRESHOLDS]
            _, t = min(options)
            cost_on_test = cost_at(t, t_caught, t_spur, base, cost_fn)[0]
            sensitivity.append({"base_rate": base, "cost_fn": cost_fn, "threshold_chosen_on_valid": t,
                                "test_cost_per_1000": round(cost_on_test, 2),
                                "do_nothing_per_1000": round(1000 * base * cost_fn, 2)})

    result = {
        "assumptions": {"cost_fn": COST_FN, "cost_fp": COST_FP, "cost_tp": COST_TP,
                        "cost_tn": COST_TN, "base_rate": BASE_RATE, "iou_match": IOU_MATCH},
        "chosen": {
            "threshold": chosen,
            "valid_cost_per_1000": best_valid["cost_per_1000"],
            "test_cost_per_1000": round(test_cost, 2),
            "test_recall": round(test_recall, 4),
            "test_false_alarm_rate": round(test_fpr, 4),
        },
        "best_on_test_for_comparison": best_test,
        "do_nothing_cost_per_1000": do_nothing,
        "sweep_valid": sweep_valid,
        "sweep_test": sweep_test,
        "sensitivity": sensitivity,
    }
    (out_dir / "cost_sweep.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(THRESHOLDS, [s["cost_per_1000"] for s in sweep_valid], color="#7B2CBF",
            label="validation (used to choose)")
    ax.plot(THRESHOLDS, [s["cost_per_1000"] for s in sweep_test], color="#E0218A",
            label="test (reported)")
    ax.axhline(do_nothing, color="grey", linestyle="--", label=f"do nothing (EUR {do_nothing:.0f})")
    ax.axvline(chosen, color="#3B0F66", linestyle=":", label=f"chosen threshold {chosen:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("expected cost per 1,000 listings (EUR)")
    ax.set_title("Cost against threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cost_curve.png", dpi=150)
    plt.close(fig)

    print("\n=== COST SWEEP (threshold chosen on validation, reported on test) ===")
    print(f"Assumptions: missed defect EUR {COST_FN}, false alarm EUR {COST_FP}, base rate {BASE_RATE:.0%}")
    print(f"Doing nothing: EUR {do_nothing:.1f} per 1,000 listings")
    print(f"Chosen threshold (validation): {chosen:.2f}  (validation cost EUR {best_valid['cost_per_1000']:.2f})")
    print(f"On the TEST split at that threshold: EUR {test_cost:.2f} per 1,000, "
          f"catches {test_recall:.1%} of defects, flags {test_fpr:.1%} of good listings")
    print(f"For comparison, the best threshold if we had tuned on test: {best_test['threshold']:.2f} "
          f"-> EUR {best_test['cost_per_1000']:.2f}")
    print("\nTest cost at other thresholds:")
    for t in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        s = next(x for x in sweep_test if abs(x["threshold"] - t) < 1e-9)
        print(f"  t={t:.2f}: EUR {s['cost_per_1000']:7.2f}  recall {s['recall']:.1%}  false alarms {s['false_alarm_rate']:.1%}")
    print("\nWhat if the assumptions are wrong?  (base rate, missed-defect cost -> threshold from validation, test cost, do nothing)")
    for s in sensitivity:
        print(f"  {s['base_rate']:.0%}, EUR {s['cost_fn']:>4.0f}: threshold {s['threshold_chosen_on_valid']:.2f}, "
              f"EUR {s['test_cost_per_1000']:7.2f}, do nothing EUR {s['do_nothing_per_1000']:7.2f}")
    print(f"\nConfusion matrix on test at threshold {chosen:.2f} (rows = predicted, columns = true):")
    print("  labels:", labels)
    for label, row in zip(labels, matrix.tolist()):
        print(f"  {label:15s} {row}")
    print("\nSaved outputs/image_scores_valid.csv, image_scores_test.csv, cost_sweep.json, cost_curve.png, "
          "confusion_matrix.json, test_confusion_matrix_at_threshold.png")
    return result


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    run_costing("weights/best.pt", "data/fabric-defect", device, 512, "outputs")