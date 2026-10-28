"""
Assignment 2: training pipeline.

Downloads the fire/smoke/lights dataset from Roboflow, finetunes yolo11n on it,
and evaluates it on the test split against two baselines:

- the pretrained COCO yolo11n used zero-shot (COCO has no fire, smoke or light
  class, so there is nothing to map its predictions onto), and
- trivial frame-level baselines: predict nothing / always escalate.

Outputs:
- weights/fire_smoke_yolo11n.pt   the finetuned model the app serves
- artifacts/metrics.json          per-class and frame-level results
- artifacts/test_scores.csv       per-image scores, used for the cost sweep
- artifacts/cost_sweep.csv/.png   expected cost per 1,000 cases by threshold

Ultralytics picks the device itself: the GPU if there is one, else the CPU.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from roboflow import Roboflow
from ultralytics import YOLO

import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"
FINAL_WEIGHTS = WEIGHTS_DIR / "fire_smoke_yolo11n.pt"

WORKSPACE = "pangarap"
PROJECT = "3classes-nmslq"
VERSION = 3

CLASSES = ["Fire", "Lights", "Smoke"]
POSITIVE_IDS = {0, 2}  # Fire or Smoke means escalate; Lights is a negative

BASE_MODEL = "yolo11n.pt"
EPOCHS = 40
PATIENCE = 10
IMGSZ = 640
BATCH = 16
DEFAULT_CONF = 0.25  # ultralytics default, used only for the comparison table
SEED = 0


def get_data(dest="datasets"):
    rf = Roboflow(api_key=load_api_key())
    ver = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    ds = ver.download("yolov11", location=f"{dest}/{PROJECT}-v{VERSION}")
    return Path(ds.location) / "data.yaml"


def finetune(data_yaml):
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        project=str(ROOT / "runs"),
        name="finetune",
        exist_ok=True,
        plots=True,
    )
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, FINAL_WEIGHTS)
    return FINAL_WEIGHTS


def training_history(weights):
    """Epochs actually run and the best one, from the history saved in the checkpoint."""
    history = pd.DataFrame(torch.load(weights, map_location="cpu", weights_only=False)["train_results"])
    history.columns = history.columns.str.strip()
    best = history.loc[history["metrics/mAP50-95(B)"].idxmax()]
    return {
        "epochs_run": int(history["epoch"].max()),
        "best_epoch": int(best["epoch"]),
        "best_val_map50": float(best["metrics/mAP50(B)"]),
        "best_val_map50_95": float(best["metrics/mAP50-95(B)"]),
        "train_minutes": round(float(history["time"].max()) / 60, 1),
    }


def evaluate_detection(weights, data_yaml):
    """Box-level metrics on the test split: overall and per class."""
    metrics = YOLO(weights).val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
        project=str(ROOT / "runs"), name="test_eval",
        exist_ok=True, plots=True,
    )
    box = metrics.box
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    for plot in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        if (Path(metrics.save_dir) / plot).exists():
            shutil.copy(Path(metrics.save_dir) / plot, ARTIFACTS_DIR / plot)
    matrix = metrics.confusion_matrix.matrix
    labels = CLASSES + ["background"]
    confusions = sorted(
        ({"true": labels[t], "predicted": labels[p], "count": int(matrix[p, t])}
         for p in range(len(labels)) for t in range(len(labels))
         if p != t and matrix[p, t] > 0),
        key=lambda c: -c["count"],
    )
    per_class = {
        CLASSES[c]: {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
        }
        for i, c in enumerate(box.ap_class_index)
    }
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        # rows = predicted, cols = true; last row/col is background
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": labels,
        "top_confusions": confusions,
    }


def ground_truth(label_path):
    ids = set()
    if label_path.exists():
        ids = {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}
    return {f"gt_{name.lower()}": int(i in ids) for i, name in enumerate(CLASSES)} | {
        "gt_positive": int(bool(ids & POSITIVE_IDS)),
    }


def score_test_images(weights, test_dir):
    """Highest confidence per class for every test image, at a near-zero threshold."""
    rows = []
    model = YOLO(weights)
    for r in model.predict(source=str(test_dir / "images"), conf=0.001, imgsz=IMGSZ,
                           stream=True, verbose=False):
        path = Path(r.path)
        scores = {f"score_{name.lower()}": 0.0 for name in CLASSES}
        for c, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            key = f"score_{CLASSES[int(c)].lower()}"
            scores[key] = max(scores[key], conf)
        rows.append({"image": path.name}
                    | ground_truth(test_dir / "labels" / f"{path.stem}.txt")
                    | scores
                    | {"score_positive": max(scores["score_fire"], scores["score_smoke"])})
    return pd.DataFrame(rows)


def coco_zero_shot(test_dir, gt):
    """What the off-the-shelf COCO model sees on positive vs negative frames."""
    seen = {1: Counter(), 0: Counter()}
    any_detection = {}
    model = YOLO(BASE_MODEL)
    for r in model.predict(source=str(test_dir / "images"), conf=DEFAULT_CONF, imgsz=IMGSZ,
                           stream=True, verbose=False):
        name = Path(r.path).name
        label = int(gt.loc[name])
        names = [model.names[int(c)] for c in r.boxes.cls.tolist()]
        seen[label].update(set(names))
        any_detection[name] = int(len(names) > 0)
    pred = pd.Series(any_detection).reindex(gt.index)
    return {
        "note": "COCO has no Fire, Smoke or Lights class, so box-level AP is 0 by construction.",
        "map50": 0.0,
        "top_classes_on_fire_smoke_frames": seen[1].most_common(5),
        "top_classes_on_negative_frames": seen[0].most_common(5),
        "frame_level_any_detection": frame_metrics(gt, pred),
        # escalate on any detection, reweighted to the assumed base rate
        "cost_per_1000": float(costing.cost_per_1000(pred, gt, 0.5, 0.5)["cost"]),
    }


def frame_metrics(truth, pred):
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / len(truth),
    }


def cost_analysis(scores):
    """Sweep the threshold, pick the cheapest, and check how fragile that pick is."""
    s, y = scores["score_positive"], scores["gt_positive"]
    sweep = costing.sweep(s, y)
    best = costing.best_threshold(sweep)
    sweep.to_csv(ARTIFACTS_DIR / "cost_sweep.csv", index=False)
    plot_sweep(sweep, best, ARTIFACTS_DIR / "cost_sweep.png")

    band = costing.best_band(costing.sweep_band(s, y))

    sensitivity = []
    for base_rate in (0.001, 0.01, 0.05, float(y.mean())):
        for fn_cost in (10_000, 50_000, 200_000):
            costs = costing.COSTS | {"fn": fn_cost}
            b = costing.best_threshold(costing.sweep(s, y, base_rate, costs))
            # the band chosen at the default assumptions, re-priced under these ones
            fixed = costing.cost_per_1000(s, y, band["low"], band["high"], base_rate, costs)
            sensitivity.append({"base_rate": base_rate, "fn_cost": fn_cost,
                                "best_threshold": float(b["threshold"]),
                                "cost_per_1000": float(b["cost"]),
                                "band_cost_per_1000": float(fixed["cost"])})

    reference = {
        "always_escalate": costing.cost_per_1000(s, y, 0.0, 0.0),
        "never_escalate": costing.cost_per_1000(s, y, 1.01, 1.01),
        "default_conf_0.25": costing.cost_per_1000(s, y, DEFAULT_CONF, DEFAULT_CONF),
        "review_everything": costing.cost_per_1000(s, y, 0.0, 1.01),
    }
    return {
        "costs": costing.COSTS,
        "base_rate": costing.BASE_RATE,
        "review_cost": costing.REVIEW_COST,
        "max_review_share": costing.MAX_REVIEW_SHARE,
        "test_negatives": int((y == 0).sum()),
        "best_threshold": float(best["threshold"]),
        "best": {k: float(v) for k, v in best.items()},
        "band": {k: float(v) for k, v in band.items()},
        "reference_points": reference,
        "sensitivity": sensitivity,
    }


def plot_sweep(sweep, best, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["cost"], color="#2a6fdb", linewidth=2)
    ax.axvline(best["threshold"], color="#555555", linestyle="--", linewidth=1)
    ax.annotate(f"chosen: {best['threshold']:.2f}\n€{best['cost']:,.0f} per 1,000 cases",
                xy=(best["threshold"], best["cost"]), xytext=(10, 45), bbox={"facecolor": "white", "edgecolor": "none"},
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("Confidence threshold (Fire or Smoke)")
    ax.set_ylabel("Expected cost per 1,000 cases")
    ax.yaxis.set_major_formatter(lambda v, _: f"€{v:,.0f}")
    ax.set_title(f"Cost against threshold, base rate {costing.BASE_RATE:.0%}", loc="left")
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    data_yaml = get_data()
    print(f"Dataset ready: {data_yaml}")
    test_dir = data_yaml.parent / "test"

    weights = finetune(data_yaml)
    detection = evaluate_detection(weights, data_yaml)

    scores = score_test_images(weights, test_dir)
    gt = scores.set_index("image")["gt_positive"]
    finetuned_pred = (scores.set_index("image")["score_positive"] >= DEFAULT_CONF).astype(int)

    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "patience": PATIENCE,
                     "imgsz": IMGSZ, "batch": BATCH, "seed": SEED},
        "test_positive_rate": float(gt.mean()),
        "finetuned": detection | {
            "frame_level_at_default_conf": frame_metrics(gt, finetuned_pred),
        },
        "coco_zero_shot": coco_zero_shot(test_dir, gt),
        "predict_nothing": {"map50": 0.0, "frame_level": frame_metrics(gt, gt * 0)},
        "always_escalate": {"frame_level": frame_metrics(gt, gt * 0 + 1)},
    }

    results["operating_point"] = cost_analysis(scores)
    best_pred = (scores.set_index("image")["score_positive"]
                 >= results["operating_point"]["best_threshold"]).astype(int)
    results["finetuned"]["frame_level_at_best_threshold"] = frame_metrics(gt, best_pred)
    results["training"] |= training_history(weights)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    scores.to_csv(ARTIFACTS_DIR / "test_scores.csv", index=False)
    print(json.dumps({k: v for k, v in results.items() if k != "finetuned"}, indent=2))
    print(f"Finetuned test mAP50: {detection['map50']:.3f}  mAP50-95: {detection['map50_95']:.3f}")
    print(f"Saved weights to {weights} and results to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
