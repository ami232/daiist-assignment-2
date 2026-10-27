"""
Assignment 2: training pipeline.

Downloads the shelf dataset, describes it, finetunes YOLO11n, evaluates it
against the COCO zero-shot and predict-nothing baselines, scores every shelf
slot in val and test, sweeps thresholds on val, and writes
artifacts/results.json so app.py never trains or re-scores.

TRAIN_EPOCHS and TRAIN_FRACTION environment variables shorten a run for a
quick end-to-end check. TRAIN_MODEL, TRAIN_IMGSZ and TRAIN_BATCH override the
model settings for experiments. Leave them all unset for the real run.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from matplotlib.figure import Figure
from roboflow import Roboflow
from ultralytics import YOLO

import config
import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

WORKSPACE = "final-project-object-detection-for-instore-inventory-management"
PROJECT = "empty-spaces-in-a-supermarket-hanger-1upsp"
VERSION = 26

BASE_MODEL = os.environ.get("TRAIN_MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 50))
FRACTION = float(os.environ.get("TRAIN_FRACTION", 1.0))
# Gaps cover ~0.3% of a 1536x2048 photo; at 640 px they shrink to ~35 px.
IMGSZ = int(os.environ.get("TRAIN_IMGSZ", 1024))
BATCH = int(os.environ.get("TRAIN_BATCH", 8))
SEED = 0
MAX_DET = 1000     # photos hold ~150 slots, so the default cap of 300 is too tight
MIN_CONF = 0.001   # lowest confidence kept when scoring slots
IOU_MATCH = 0.5    # a prediction counts for a slot at this overlap, as in AP50
SPLITS = {"train": "train", "val": "valid", "test": "test"}
THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)


def download() -> Path:
    location = DATA_DIR / f"shelves-v{VERSION}"
    if (location / "data.yaml").exists():
        return location
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    return Path(version.download("yolov11", location=str(location)).location)


def write_data_yaml(root: Path) -> tuple[Path, list[str]]:
    """Rewrite Roboflow's data.yaml with absolute paths, which Ultralytics resolves reliably."""
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    for key, folder in SPLITS.items():
        if not (root / folder / "images").is_dir():
            raise SystemExit(f"Dataset has no {folder}/ split. Pick a version that has one.")
        cfg[key] = str(root / folder / "images")
    cfg.pop("path", None)
    names = cfg["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if config.POSITIVE_CLASS not in names:
        raise SystemExit(f"{config.POSITIVE_CLASS!r} is not a class here: {names}")
    path = root / "data_abs.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, names


def describe(root: Path, names: list[str]) -> dict:
    """Images per split, instances per class, and median box area as % of the image."""
    stats = {}
    for key, folder in SPLITS.items():
        instances = {n: 0 for n in names}
        images_with = {n: 0 for n in names}
        areas = {n: [] for n in names}
        for label_file in (root / folder / "labels").glob("*.txt"):
            seen = set()
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                name = names[int(parts[0])]
                instances[name] += 1
                seen.add(name)
                areas[name].append(float(parts[3]) * float(parts[4]))
            for name in seen:
                images_with[name] += 1
        stats[key] = {
            "images": sum(1 for _ in (root / folder / "images").iterdir()),
            "instances": instances,
            "images_with_class": images_with,
            "median_box_area_pct": {n: round(100 * float(np.median(a)), 2) if a else None
                                    for n, a in areas.items()},
        }
    return stats


def read_slots(label_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Labelled slots in one image: class ids and normalised xyxy boxes."""
    rows = [r.split() for r in label_file.read_text().splitlines() if len(r.split()) == 5]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    data = np.array(rows, dtype=float)
    xc, yc, w, h = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    boxes = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
    return data[:, 0].astype(int), boxes


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def score_slots(slots: np.ndarray, pred: np.ndarray, conf: np.ndarray):
    """Per-slot score and phantom confidences for one image.

    A slot's score is the highest positive-class confidence among predictions
    overlapping it at IoU >= IOU_MATCH (0 if none). A positive prediction that
    overlaps no labelled slot is a phantom, kept with its confidence.
    """
    if len(slots) and len(pred):
        overlap = iou(slots, pred) >= IOU_MATCH
    else:
        overlap = np.zeros((len(slots), len(pred)), bool)
    scores = np.where(overlap, conf[None, :], 0).max(axis=1, initial=0)
    return scores, conf[~overlap.any(axis=0)]


def slot_cases(model: YOLO, root: Path, folder: str, names: list[str],
               positive_ids: set[int]) -> dict:
    """Score every labelled slot in a split, and collect phantom detections."""
    positive = names.index(config.POSITIVE_CLASS)
    scores, labels, phantoms = [], [], []
    for r in model.predict(source=str(root / folder / "images"), imgsz=IMGSZ, conf=MIN_CONF,
                           max_det=MAX_DET, stream=True, verbose=False):
        classes, slots = read_slots(root / folder / "labels" / f"{Path(r.path).stem}.txt")
        keep = np.isin(r.boxes.cls.cpu().numpy().astype(int), list(positive_ids))
        slot_scores, image_phantoms = score_slots(slots, r.boxes.xyxyn.cpu().numpy()[keep],
                                                  r.boxes.conf.cpu().numpy()[keep])
        scores += slot_scores.round(4).tolist()
        labels += (classes == positive).astype(int).tolist()
        phantoms += image_phantoms.round(4).tolist()
    return {"scores": scores, "labels": labels, "phantoms": phantoms}


def box_metrics(model: YOLO, data_yaml: Path, names: list[str]) -> dict:
    m = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, max_det=MAX_DET, plots=True,
                  project=str(ROOT / "runs"), name="test", exist_ok=True, verbose=False)
    per_class = [{"class": names[int(c)],
                  "precision": round(float(m.box.p[i]), 4),
                  "recall": round(float(m.box.r[i]), 4),
                  "AP50": round(float(m.box.ap50[i]), 4),
                  "AP50-95": round(float(m.box.ap[i]), 4)}
                 for i, c in enumerate(m.box.ap_class_index)]
    plot = Path(m.save_dir) / "confusion_matrix.png"
    if plot.exists():
        shutil.copy(plot, ARTIFACTS / "box_confusion_matrix.png")
    return {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
            "per_class": per_class,
            "confusion_matrix": m.confusion_matrix.matrix.astype(int).tolist(),
            "confusion_labels": names + ["background"]}


def save_figures(results: dict) -> None:
    """Cost-against-threshold curve and a readable two-class confusion matrix for REPORT.md."""
    thresholds = np.array(results["thresholds"])
    op = results["operating_point"]
    test = results["cases"]["test"]
    args = (np.array(test["finetuned"]), np.array(test["labels"]))
    phantoms = np.array(test["finetuned_phantoms"])
    nothing = costing.cost_per_1000(*args, np.inf, np.inf)
    band = costing.cost_per_1000(*args, op["t_low"], op["t_high"], phantoms)

    fig = Figure(figsize=(9, 5.5))
    ax = fig.add_subplot()
    ax.plot(thresholds, results["sweep"]["test"], lw=2, label="Model decides alone (test)")
    ax.plot(thresholds, results["sweep"]["val"], ls=":", lw=2, label="Model decides alone (validation)")
    ax.axhline(nothing, color="grey", ls="--", label=f"Do nothing: €{nothing:.2f}")
    ax.axvspan(op["t_low"], op["t_high"], color="orange", alpha=0.15,
               label=f"Review band {op['t_low']:.2f}–{op['t_high']:.2f}")
    ax.axhline(band, color="darkorange", ls="-.", label=f"Model + review band: €{band:.2f} (test)")
    t = op["single_threshold"]
    single = results["sweep"]["test"][int(np.argmin(np.abs(thresholds - t)))]
    ax.axvline(t, color="black", ls="--", lw=1)
    ax.plot(t, single, "ko")
    ax.annotate(f"single threshold {t:.2f}\n(chosen on validation): €{single:.2f}", (t, single),
                xytext=(10, 25), textcoords="offset points", arrowprops={"arrowstyle": "-"})
    # Below ~0.05 nearly every slot is flagged and the cost climbs past €700, which would
    # flatten the part of the curve that matters, so clip the axis.
    ax.set(xlabel="Confidence threshold for 'missing'", ylabel="Expected cost, € per 1,000 slots",
           title=f"Cost against threshold (base rate {config.BASE_RATE:.0%}; "
                 "curves continue above the top edge)",
           xlim=(0, 1), ylim=(0, nothing * 1.6))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(ARTIFACTS / "cost_curve.png", dpi=150, bbox_inches="tight")

    # Ultralytics' matrix has empty rows for the junk classes and a background column inflated
    # by its 0.001 confidence cut-off, so keep the two real classes and show shares per true class.
    box = results["box_metrics"]["finetuned"]
    labels = box["confusion_labels"]
    keep = [labels.index(n) for n in ("missing", "product", "background")]
    counts = np.array(box["confusion_matrix"])[np.ix_(keep, keep[:2])]  # rows predicted, columns true
    shares = counts / counts.sum(axis=0, keepdims=True)
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot()
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for (i, j), n in np.ndenumerate(counts):
        ax.text(j, i, f"{n:,}\n{shares[i, j]:.1%}", ha="center", va="center",
                color="white" if shares[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], xticklabels=["missing", "product"],
           yticks=[0, 1, 2], yticklabels=["missing", "product", "no detection"],
           xlabel="True class", ylabel="Predicted class",
           title="Box confusion matrix, test split\n(share of each true class; IoU ≥ 0.45, conf ≥ 0.001)")
    fig.savefig(ARTIFACTS / "confusion_matrix_clean.png", dpi=150, bbox_inches="tight")


def main() -> None:
    costing.check_config()
    ARTIFACTS.mkdir(exist_ok=True)
    WEIGHTS.parent.mkdir(exist_ok=True)

    root = download()
    data_yaml, names = write_data_yaml(root)
    stats = describe(root, names)
    print(json.dumps(stats, indent=2))

    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, fraction=FRACTION, imgsz=IMGSZ,
                batch=BATCH, seed=SEED, max_det=MAX_DET, project=str(ROOT / "runs"),
                name="finetune", exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)
    for curve in ("results.csv", "results.png"):
        if (Path(model.trainer.save_dir) / curve).exists():
            shutil.copy(Path(model.trainer.save_dir) / curve, ARTIFACTS / f"training_{curve}")
    finetuned = YOLO(str(WEIGHTS))

    coco = YOLO(BASE_MODEL)
    positive_ids = {names.index(config.POSITIVE_CLASS)}
    coco_ids = {i for i, n in coco.names.items() if n == config.POSITIVE_CLASS}  # empty: COCO has no such class

    cases = {}
    for split in ("val", "test"):
        fine = slot_cases(finetuned, root, SPLITS[split], names, positive_ids)
        zero = slot_cases(coco, root, SPLITS[split], names, coco_ids)
        cases[split] = {"labels": fine["labels"],
                        "finetuned": fine["scores"], "finetuned_phantoms": fine["phantoms"],
                        "coco_zero_shot": zero["scores"], "coco_zero_shot_phantoms": zero["phantoms"],
                        "trivial": [0.0] * len(fine["labels"]), "trivial_phantoms": []}

    # Choose thresholds on validation, report them on test.
    val, test = cases["val"], cases["test"]
    val_args = (np.array(val["finetuned"]), np.array(val["labels"]))
    test_args = (np.array(test["finetuned"]), np.array(test["labels"]))
    val_ph, test_ph = np.array(val["finetuned_phantoms"]), np.array(test["finetuned_phantoms"])
    sweep_val = costing.sweep(*val_args, THRESHOLDS, val_ph)
    t_single = float(THRESHOLDS[int(np.argmin(sweep_val))])
    t_low, t_high = costing.best_band(*val_args, THRESHOLDS, val_ph, config.MAX_REVIEWERS)

    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "train_settings": {"base_model": BASE_MODEL, "epochs": EPOCHS, "fraction": FRACTION,
                           "imgsz": IMGSZ, "batch": BATCH, "seed": SEED},
        "names": names,
        "positive_class": config.POSITIVE_CLASS,
        "stats": stats,
        "box_metrics": {
            "finetuned": box_metrics(finetuned, data_yaml, names),
            "coco_zero_shot": "COCO has no empty-slot class, so AP for it is 0 by construction",
            "trivial": "predicts nothing, so AP is 0 for every class",
        },
        "cases": cases,
        "slot_level_baselines": {
            name: {**costing.binary_metrics(np.array(test[name]), test_args[1], t_single,
                                            np.array(test[f"{name}_phantoms"])),
                   "cost_per_1000": round(costing.cost_per_1000(
                       np.array(test[name]), test_args[1], t_single, t_single,
                       np.array(test[f"{name}_phantoms"])), 2)}
            for name in ("finetuned", "coco_zero_shot", "trivial")
        },
        "thresholds": THRESHOLDS.tolist(),
        "sweep": {"val": sweep_val, "test": costing.sweep(*test_args, THRESHOLDS, test_ph)},
        "operating_point": {"single_threshold": t_single, "t_low": t_low, "t_high": t_high},
        "alternatives_test": costing.alternatives(*test_args, t_single, t_low, t_high, test_ph),
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
    save_figures(results)
    print(json.dumps({k: results[k] for k in ("operating_point", "alternatives_test")}, indent=2))


if __name__ == "__main__":
    main()
