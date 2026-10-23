"""
Assignment 2: training pipeline for hard-hat checks at a construction-site entrance.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download Hard Hat Workers v5 (joseph-nelson/hard-hat-workers) from Roboflow.
2. Finetune yolo11n from its COCO-pretrained checkpoint and copy the best
   weights into weights/.
3. Evaluate on the test split at two levels:
   - detection level (mAP50, mAP50-95, per-class P/R/AP50, confusion matrix),
     the usual YOLO numbers;
   - case level, where one case is one person walking through the gate. Each
     ground-truth head/helmet box gets a "no helmet" score, which is what the
     gate decision is made on.
   Both levels are computed for the finetuned model and for the two baselines:
   the COCO model zero-shot, and a trivial "predict nothing" model (nobody is
   ever stopped, i.e. doing nothing).
4. Sweep the confidence threshold, compute the expected cost per 1,000 cases at
   the base rate we assume for production (not the test split's own mix), and
   pick the threshold with the lowest cost.
5. Add a human-review band (stop / guard reviews / let in), pick the cheapest
   pair of thresholds that fits the guard's capacity at peak, and price it
   against doing nothing, the guard checking everyone, and the model alone.
6. Save everything app.py needs into outputs/, so the app never trains.

The cost assumptions and formulas live in costing.py, shared with app.py.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: this runs as a plain script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"

# --- Dataset ------------------------------------------------------------------
WORKSPACE, PROJECT, VERSION = "joseph-nelson", "hard-hat-workers", 5
CLASS_NAMES = ["head", "helmet"]  # class 0 "head" = no helmet = our positive class
POSITIVE_CLASS = 0

# --- Training settings ----------------------------------------------------------
BASE_MODEL = "yolo11n.pt"
EPOCHS = 30
PATIENCE = 10  # stop early if val mAP has not improved for this many epochs
IMGSZ = 512    # images are ~500 px wide, so 640 would only upscale them
BATCH = 16
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
WEIGHTS_FILE = WEIGHTS_DIR / "hardhat_yolo11n.pt"
CM_CONF = 0.25  # confidence at which the detection confusion matrix is read


# =============================================================================
# 1. Data
# =============================================================================
def download_dataset() -> Path:
    """Download the dataset once and return a data.yaml with absolute paths.

    Roboflow's own data.yaml uses paths relative to a folder that Ultralytics
    does not resolve the same way, so we write our own next to it.
    """
    location = DATASETS_DIR / f"{PROJECT}-{VERSION}"
    if not (location / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(location)
        )
    data_yaml = location / "data_abs.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(location),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(CLASS_NAMES)),
    }))
    return data_yaml


# =============================================================================
# 2. Finetuning
# =============================================================================
def finetune(data_yaml: Path) -> Path:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ,
        batch=BATCH, device=DEVICE, seed=SEED, deterministic=True,
        project=str(RUNS_DIR), name="hardhat", exist_ok=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_FILE)
    return WEIGHTS_FILE


# =============================================================================
# 3a. Detection-level evaluation
# =============================================================================
def detection_metrics(model: YOLO, data_yaml: Path, name: str, classes=None) -> dict:
    """mAP and per-class numbers on the test split, as Ultralytics computes them.

    For the COCO model we keep only its class 0 ("person"), which therefore
    gets scored as our class 0 ("head"). COCO has no helmet class at all.
    """
    r = model.val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, device=DEVICE,
        classes=classes, project=str(RUNS_DIR), name=f"test_{name}", exist_ok=True,
    )
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        # Ultralytics reports P and R at the confidence that maximises F1.
        per_class[cls] = {
            "precision": float(r.box.p[i]) if i < len(r.box.p) else 0.0,
            "recall": float(r.box.r[i]) if i < len(r.box.r) else 0.0,
            "ap50": float(r.box.ap50[i]) if i < len(r.box.ap50) else 0.0,
            "ap50_95": float(r.box.ap[i]) if i < len(r.box.ap) else 0.0,
        }
    # mAP needs every box down to conf 0.001, but a confusion matrix at that
    # confidence is mostly noise, so it gets its own pass at a readable 0.25.
    cm = model.val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, device=DEVICE, conf=CM_CONF,
        classes=classes, project=str(RUNS_DIR), name=f"test_{name}_cm", exist_ok=True,
    )
    matrix = cm.confusion_matrix.matrix  # rows = predicted, cols = true; last = background
    return {
        "map50": float(r.box.map50),
        "map50_95": float(r.box.map),
        "per_class": per_class,
        "confusion_matrix": {
            "conf": CM_CONF,
            "labels": CLASS_NAMES + ["background"],
            "rows_predicted_cols_true": matrix.astype(int).tolist(),
        },
        "plots_dir": str(cm.save_dir),
    }


# =============================================================================
# 3b. Case-level evaluation: one case = one person at the gate
# =============================================================================
def yolo_labels(label_file: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth classes and xyxy pixel boxes from a YOLO label file."""
    rows = np.array([l.split() for l in label_file.read_text().splitlines() if l.strip()], float)
    cls = rows[:, 0].astype(int)
    xc, yc, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    return cls, np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def overlap(pred: np.ndarray, gt: np.ndarray, kind: str) -> np.ndarray:
    """Pairwise overlap, shape (n_pred, n_gt).

    "iou": intersection over union, the usual detection match.
    "ioa": intersection over the ground-truth box's area, i.e. how much of the
           person's head a prediction covers. Used for the COCO model, whose
           person boxes cover the whole body and so never reach IoU 0.5 with a
           head box.
    """
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    if kind == "ioa":
        return inter / area(gt)[None, :]
    return inter / (area(pred)[:, None] + area(gt)[None, :] - inter)


def case_scores(model: YOLO, test_dir: Path, score_class: int, kind: str, classes=None) -> pd.DataFrame:
    """One row per ground-truth person: its true class and a "no helmet" score.

    At the gate there is one person in the frame, so any "head" box the model
    draws anywhere counts against that person, including boxes on nobody.
    The test images hold several people, so each predicted box is attributed
    to the person it overlaps most, or to the nearest one if it overlaps
    nobody. A person's score is the highest confidence among the boxes
    attributed to them; with none, the score is 0 and they walk in.
    """
    rows = []
    for result in model.predict(
        source=str(test_dir / "images"), imgsz=IMGSZ, device=DEVICE, conf=0.001,
        classes=classes, stream=True, verbose=False,
    ):
        image = Path(result.path)
        height, width = result.orig_shape
        gt_cls, gt_box = yolo_labels(test_dir / "labels" / f"{image.stem}.txt", width, height)
        boxes = result.boxes
        keep = boxes.cls.cpu().numpy().astype(int) == score_class
        pred_box = boxes.xyxy.cpu().numpy()[keep]
        pred_conf = boxes.conf.cpu().numpy()[keep]
        score = np.zeros(len(gt_cls))
        if len(pred_box):
            ov = overlap(pred_box, gt_box, kind)
            centre = lambda b: (b[:, :2] + b[:, 2:]) / 2
            distance = np.linalg.norm(centre(pred_box)[:, None] - centre(gt_box)[None], axis=2)
            owner = np.where(ov.max(axis=1) > 0, ov.argmax(axis=1), distance.argmin(axis=1))
            np.maximum.at(score, owner, pred_conf)
        rows += [
            {"image": image.name, "true_class": CLASS_NAMES[c], "positive": int(c == POSITIVE_CLASS), "score": float(s)}
            for c, s in zip(gt_cls, score)
        ]
    return pd.DataFrame(rows)


# =============================================================================
# 4. Costing, threshold sweep and human review (formulas in costing.py)
# =============================================================================
def sensitivity(df: pd.DataFrame) -> list[dict]:
    """How the best single threshold moves if our weakest assumptions are wrong."""
    scores, positive = df["score"].to_numpy(), df["positive"].to_numpy()
    scenarios = [("baseline", costing.BASE_RATE, costing.P_INSPECTION, costing.P_ACCIDENT)]
    for rate in (0.02, 0.10):
        scenarios.append((f"base rate {rate:.0%}", rate, costing.P_INSPECTION, costing.P_ACCIDENT))
    for factor in (0.5, 2):
        scenarios.append((f"P(inspection) x{factor}", costing.BASE_RATE, costing.P_INSPECTION * factor, costing.P_ACCIDENT))
        scenarios.append((f"P(accident) x{factor}", costing.BASE_RATE, costing.P_INSPECTION, costing.P_ACCIDENT * factor))
    out = []
    for name, rate, p_insp, p_acc in scenarios:
        costs = {**costing.COSTS, "fn": costing.fn_cost(p_insp, p_acc)}
        best = costing.best_threshold(costing.sweep(scores, positive, costs, rate))
        out.append({"scenario": name, "base_rate": rate, "fn_cost": costs["fn"], **best})
    return out


def plot_cost_curve(curves: dict[str, pd.DataFrame], chosen: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in curves.items():
        ax.plot(curve["threshold"], curve["cost_per_1000"], label=name)
    ax.scatter([chosen["threshold"]], [chosen["cost_per_1000"]], color="black", zorder=3,
               label=f"chosen t={chosen['threshold']:.2f} ({chosen['cost_per_1000']:.0f} EUR)")
    ax.set_xlabel("confidence threshold for 'no helmet'")
    ax.set_ylabel(f"EUR per 1,000 entrances (base rate {costing.BASE_RATE:.0%})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =============================================================================
# 5. Pipeline
# =============================================================================
def evaluate(weights: Path, data_yaml: Path) -> dict:
    """Everything after training: metrics, cost sweep, review band, outputs/."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    test_dir = data_yaml.parent / "test"
    finetuned, coco = YOLO(str(weights)), YOLO(BASE_MODEL)
    detection = {
        "finetuned": detection_metrics(finetuned, data_yaml, "finetuned"),
        "coco_zero_shot": detection_metrics(coco, data_yaml, "coco", classes=[0]),
        "predict_nothing": {"map50": 0.0, "map50_95": 0.0, "per_class": {
            c: {"precision": 0.0, "recall": 0.0, "ap50": 0.0, "ap50_95": 0.0} for c in CLASS_NAMES}},
    }
    shutil.copy(Path(detection["finetuned"]["plots_dir"]) / "confusion_matrix.png",
                OUTPUTS_DIR / "confusion_matrix.png")

    cases = case_scores(finetuned, test_dir, score_class=POSITIVE_CLASS, kind="iou")
    coco_cases = case_scores(coco, test_dir, score_class=0, kind="ioa", classes=[0])
    cases["score_coco"] = coco_cases["score"].to_numpy()  # same images, same order of boxes
    cases.to_csv(OUTPUTS_DIR / "case_scores.csv", index=False)

    positive = cases["positive"].to_numpy()
    curves = {
        "finetuned": costing.sweep(cases["score"].to_numpy(), positive),
        "coco_zero_shot": costing.sweep(cases["score_coco"].to_numpy(), positive),
        # Predict nothing: nobody ever has a score, so nobody is stopped.
        "predict_nothing": costing.sweep(np.full(len(cases), -1.0), positive),
    }
    for name, curve in curves.items():
        curve.to_csv(OUTPUTS_DIR / f"cost_sweep_{name}.csv", index=False)
    chosen = costing.best_threshold(curves["finetuned"])
    plot_cost_curve(curves, chosen, OUTPUTS_DIR / "cost_curve.png")

    band = costing.best_review_band(cases["score"].to_numpy(), positive)
    summary = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "classes": CLASS_NAMES,
                    "positive_class": CLASS_NAMES[POSITIVE_CLASS]},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "patience": PATIENCE, "imgsz": IMGSZ,
                     "batch": BATCH, "seed": SEED, "device": str(DEVICE)},
        "assumptions": {
            "costs": costing.COSTS, "base_rate": costing.BASE_RATE, "cases_per_day": costing.CASES_PER_DAY,
            "p_inspection": costing.P_INSPECTION, "fine": costing.FINE,
            "p_accident": costing.P_ACCIDENT, "accident_cost": costing.ACCIDENT_COST,
            "review_seconds": costing.REVIEW_SECONDS, "review_cost": costing.REVIEW_COST,
            "peak_arrivals": costing.PEAK_ARRIVALS, "peak_window_min": costing.PEAK_WINDOW_MIN,
            "max_review_share": costing.MAX_REVIEW_SHARE,
        },
        "test_cases": {"people": len(cases), "without_helmet": int(positive.sum())},
        "detection": detection,
        "case_level_best": {name: costing.best_threshold(curve) for name, curve in curves.items()},
        "operating_point": chosen,
        "sensitivity": sensitivity(cases),
        "review_band": band,
        "alternatives": costing.alternatives(cases["score"].to_numpy(), positive, band, chosen),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    data_yaml = download_dataset()
    weights = finetune(data_yaml)
    summary = evaluate(weights, data_yaml)
    print(json.dumps({k: summary[k] for k in ["operating_point", "review_band", "alternatives"]}, indent=2))


if __name__ == "__main__":
    main()
