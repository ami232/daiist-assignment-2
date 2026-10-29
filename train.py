"""
Assignment 2: training pipeline.

Run via `uv run python main.py train`. Start to finish, with no manual steps:

1. Download the chicken-droppings dataset (Roboflow Universe, v1, YOLO format).
2. Finetune yolo11s from the COCO-pretrained checkpoint, with the classification
   term of the detection loss replaced by a BCE + focal mix.
3. Evaluate the best checkpoint on the untouched test split.
4. Save the weights to weights/ and the metrics plus raw test predictions to
   outputs/, so the app never has to run anything heavy at startup.
"""

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"

# Dataset (Roboflow Universe). Native train/valid/test split is kept as published.
RF_WORKSPACE = "thesis-pr4oh"
RF_PROJECT = "chicken-droppings-analysis"
RF_VERSION = 1

# Training settings
BASE_MODEL = "yolo11s.pt"
IMGSZ = 640
EPOCHS = 30
PATIENCE = 5
BATCH = 16
OPTIMIZER = "AdamW"
LR0 = 0.001
SEED = 0

# Loss: box and dfl gains are the Ultralytics defaults; the cls term is
# CLS_GAIN * (W_BCE * BCE + W_FOCAL * Focal), element-wise per anchor and class.
BOX_GAIN = 7.5
DFL_GAIN = 1.5
CLS_GAIN = 0.5
W_BCE = 0.7
W_FOCAL = 0.3
FOCAL_GAMMA = 1.5
FOCAL_ALPHA = 0.25

# Predictions saved for the threshold sweep are kept down to this confidence.
PRED_CONF_FLOOR = 0.001

# Decision and costing. One image is one dropping sample. Positive = diseased
# (any of Coccidiosis, NCD, Salmonella); the image score is the highest
# confidence among disease-class detections. Outcomes: pass (< t_low), human
# review (between), fail (>= t_high). A reviewer is assumed to decide correctly.
# Assumed per-sample costs and operating figures (justified in REPORT.md):
DISEASE_CLASSES = ["Coccidiosis-like", "NCD-like", "Salmonella-like"]
BASE_RATE = 0.05          # assumed share of floor samples that are diseased
COST_FN = 2000.0          # euro: diseased sample passed, infection spreads in the house
COST_FP = 80.0            # euro: healthy sample failed, confirmatory PCR + vet visit
COST_TP = 80.0            # euro: diseased sample failed, same PCR + vet visit
COST_TN = 0.0             # euro: healthy sample passed, no action
REVIEW_MINUTES = 3.0      # assumed time for one person to check one sample
REVIEW_COST_PER_HOUR = 40.0
CASES_PER_DAY = 500       # assumed samples photographed per day
WORK_MINUTES_PER_DAY = 450
THRESHOLD_GRID = [round(i * 0.01, 2) for i in range(0, 101)]


class BCEFocalLoss(nn.Module):
    """Weighted sum of BCE and focal loss on logits, returned unreduced.

    Drop-in replacement for v8DetectionLoss.bce, which is
    BCEWithLogitsLoss(reduction="none"): same inputs (logits and the
    assigner's soft targets in [0, 1]) and same (batch, anchors, classes)
    output, so the existing normalisation by target_scores_sum still applies.
    """

    def __init__(self, w_bce: float, w_focal: float, gamma: float, alpha: float):
        super().__init__()
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = pred.sigmoid()
        p_t = target * prob + (1 - target) * (1 - prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal = bce * (1.0 - p_t) ** self.gamma * alpha_t
        return self.w_bce * bce + self.w_focal * focal


def use_bce_focal_loss(trainer) -> None:
    """Callback: give the model being trained the BCE + focal criterion.

    Attached at train start rather than by subclassing the model, so the saved
    checkpoint is a stock DetectionModel that loads anywhere. Ultralytics
    strips the criterion before saving. Validation runs on the EMA copy, which
    builds its own default criterion, so the reported val cls_loss is plain BCE.
    """
    model = getattr(trainer.model, "module", trainer.model)
    criterion = v8DetectionLoss(model)
    criterion.bce = BCEFocalLoss(W_BCE, W_FOCAL, FOCAL_GAMMA, FOCAL_ALPHA)
    model.criterion = criterion


def download_dataset() -> Path:
    """Download the dataset once and return a data.yaml with absolute paths."""
    location = DATASETS_DIR / f"{RF_PROJECT}-v{RF_VERSION}"
    if not (location / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION)
        version.download("yolov11", location=str(location))

    # Roboflow writes split paths relative to the yaml ("../train/images"),
    # which Ultralytics resolves against its own datasets dir. Pin them.
    data = yaml.safe_load((location / "data.yaml").read_text())
    data["path"] = str(location)
    data["train"], data["val"], data["test"] = "train/images", "valid/images", "test/images"
    resolved = location / "data_resolved.yaml"
    resolved.write_text(yaml.safe_dump(data, sort_keys=False))
    return resolved


def train(data_yaml: Path) -> Path:
    model = YOLO(BASE_MODEL)
    model.add_callback("on_train_start", use_bce_focal_loss)
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        optimizer=OPTIMIZER,
        lr0=LR0,
        seed=SEED,
        deterministic=True,
        box=BOX_GAIN,
        cls=CLS_GAIN,
        dfl=DFL_GAIN,
        project=str(RUNS_DIR),
        name="finetune",
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_DIR / "best.pt")
    return WEIGHTS_DIR / "best.pt"


def evaluate(weights: Path, data_yaml: Path) -> None:
    """Per-class test metrics and raw test predictions, written to outputs/."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    model = YOLO(str(weights))
    names = model.names

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(RUNS_DIR),
        name="test_eval",
        exist_ok=True,
    )
    box = metrics.box
    per_class = {
        names[c]: {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
        }
        for i, c in enumerate(box.ap_class_index)
    }
    summary = {
        "weights": weights.name,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        "notes": "precision/recall are at Ultralytics' max-F1 confidence. The image-level "
                 "confusion matrix is in image_confusion.json, built at a stated threshold.",
    }
    (OUTPUTS_DIR / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    # Raw predictions on every test image, with ground truth alongside, so the
    # cost sweep and the app can re-threshold without running the model again.
    test_dir = Path(yaml.safe_load(data_yaml.read_text())["path"]) / "test"
    records = []
    for image in sorted((test_dir / "images").iterdir()):
        label_file = test_dir / "labels" / f"{image.stem}.txt"
        truth = [
            [int(row.split()[0])] + [float(v) for v in row.split()[1:5]]
            for row in label_file.read_text().splitlines() if row.strip()
        ]
        result = model.predict(str(image), imgsz=IMGSZ, conf=PRED_CONF_FLOOR, verbose=False)[0]
        preds = [
            [int(c), float(s)] + [float(v) for v in xywhn]
            for c, s, xywhn in zip(result.boxes.cls, result.boxes.conf, result.boxes.xywhn)
        ]
        records.append({"image": image.name, "truth": truth, "preds": preds})
    (OUTPUTS_DIR / "test_predictions.json").write_text(json.dumps({
        "names": names,
        "format": "truth: [cls, x, y, w, h]; preds: [cls, conf, x, y, w, h]; xywh normalised",
        "conf_floor": PRED_CONF_FLOOR,
        "images": records,
    }))


def image_level(predictions: dict) -> list[dict]:
    """One row per test image: true label, disease score, top predicted class."""
    names = {int(k): v for k, v in predictions["names"].items()}
    disease_ids = {i for i, n in names.items() if n in DISEASE_CLASSES}
    rows = []
    for rec in predictions["images"]:
        true_classes = [int(t[0]) for t in rec["truth"]]
        true_class = max(set(true_classes), key=true_classes.count)
        disease_conf = [p[1] for p in rec["preds"] if p[0] in disease_ids]
        top = max(rec["preds"], key=lambda p: p[1]) if rec["preds"] else None
        rows.append({
            "image": rec["image"],
            "true_class": names[true_class],
            "positive": any(c in disease_ids for c in true_classes),
            "score": max(disease_conf, default=0.0),
            "top_class": names[top[0]] if top else None,
            "top_conf": top[1] if top else 0.0,
        })
    return rows


def expected_cost(rows: list[dict], t_low: float, t_high: float, costs: dict | None = None) -> dict:
    """Expected cost per case at the assumed base rate, not the test mix.

    Rates are measured separately among positive and negative test images,
    then reweighted by BASE_RATE, so the dataset's 68% disease share does not
    leak into the cost.
    """
    c = {"fn": COST_FN, "fp": COST_FP, "tp": COST_TP, "tn": COST_TN,
         "review": REVIEW_MINUTES / 60 * REVIEW_COST_PER_HOUR, "base_rate": BASE_RATE}
    c.update(costs or {})
    pos = [r["score"] for r in rows if r["positive"]]
    neg = [r["score"] for r in rows if not r["positive"]]

    def shares(scores):
        n = max(len(scores), 1)
        fail = sum(s >= t_high for s in scores) / n
        passed = sum(s < t_low for s in scores) / n
        return passed, 1 - fail - passed, fail

    p_pass, p_rev, p_fail = shares(pos)
    n_pass, n_rev, n_fail = shares(neg)
    br = c["base_rate"]
    # A reviewed case ends correctly: positives get the TP action, negatives TN.
    cost_pos = p_pass * c["fn"] + p_fail * c["tp"] + p_rev * (c["review"] + c["tp"])
    cost_neg = n_fail * c["fp"] + n_pass * c["tn"] + n_rev * (c["review"] + c["tn"])
    review_share = br * p_rev + (1 - br) * n_rev
    return {
        "t_low": t_low, "t_high": t_high,
        "cost_per_case": br * cost_pos + (1 - br) * cost_neg,
        "review_share": review_share,
        "recall": p_fail + p_rev, "fpr": n_fail,
        "counts": {
            "pos_pass": sum(s < t_low for s in pos), "pos_review": round(p_rev * len(pos)),
            "pos_fail": sum(s >= t_high for s in pos),
            "neg_pass": sum(s < t_low for s in neg), "neg_review": round(n_rev * len(neg)),
            "neg_fail": sum(s >= t_high for s in neg),
        },
    }


def class_confusion(rows: list[dict], threshold: float, names: list[str]) -> dict:
    """4-class image-level confusion matrix; below threshold counts as 'none'."""
    labels = names + ["none"]
    matrix = [[0] * len(labels) for _ in names]
    for r in rows:
        pred = r["top_class"] if r["top_conf"] >= threshold and r["top_class"] else "none"
        matrix[names.index(r["true_class"])][labels.index(pred)] += 1
    return {"threshold": threshold, "rows_true": names, "cols_pred": labels, "matrix": matrix}


def baselines(data_yaml: Path, rows: list[dict]) -> dict:
    """COCO zero-shot and trivial baselines on the same test images."""
    test_dir = Path(yaml.safe_load(data_yaml.read_text())["path"]) / "test" / "images"
    coco = YOLO(BASE_MODEL)
    coco_hits = 0
    for image in sorted(test_dir.iterdir()):
        result = coco.predict(str(image), imgsz=IMGSZ, conf=0.25, verbose=False)[0]
        coco_hits += len(result.boxes) > 0
    n = len(rows)
    n_pos = sum(r["positive"] for r in rows)
    majority = max({r["true_class"] for r in rows}, key=lambda k: sum(r["true_class"] == k for r in rows))
    return {
        "coco_zero_shot": {
            "note": "COCO has no dropping or disease class, so no detection can match a label: "
                    "mAP50 = 0, disease recall = 0. Equivalent to passing every sample.",
            "images_with_any_coco_detection_at_0.25": coco_hits,
            "map50": 0.0, "disease_recall": 0.0,
            "cost_per_case": expected_cost(rows, 1.01, 1.01)["cost_per_case"],
        },
        "predict_nothing": {
            "map50": 0.0, "disease_recall": 0.0,
            "cost_per_case": expected_cost(rows, 1.01, 1.01)["cost_per_case"],
        },
        "majority_class": {
            "class": majority,
            "accuracy": sum(r["true_class"] == majority for r in rows) / n,
            "disease_recall": 1.0 if majority in DISEASE_CLASSES else 0.0,
            "cost_per_case": expected_cost(rows, 0.0, 0.0)["cost_per_case"]
            if majority in DISEASE_CLASSES else expected_cost(rows, 1.01, 1.01)["cost_per_case"],
        },
        "test_images": n, "test_positive_images": n_pos,
    }


def costing(data_yaml: Path) -> None:
    """Threshold sweep, chosen operating points and baselines, to outputs/."""
    predictions = json.loads((OUTPUTS_DIR / "test_predictions.json").read_text())
    rows = image_level(predictions)
    names = [predictions["names"][str(i)] for i in range(len(predictions["names"]))]

    single = [expected_cost(rows, t, t) for t in THRESHOLD_GRID]
    best_single = min(single, key=lambda r: r["cost_per_case"])
    pairs = [expected_cost(rows, lo, hi) for lo in THRESHOLD_GRID for hi in THRESHOLD_GRID if lo <= hi]
    best_pair = min(pairs, key=lambda r: r["cost_per_case"])

    review = REVIEW_MINUTES / 60 * REVIEW_COST_PER_HOUR
    per_day = best_pair["review_share"] * CASES_PER_DAY
    alternatives = {
        "do_nothing": expected_cost(rows, 1.01, 1.01)["cost_per_case"],
        "human_checks_all": BASE_RATE * (review + COST_TP) + (1 - BASE_RATE) * (review + COST_TN),
        "model_alone": best_single["cost_per_case"],
        "model_plus_review": best_pair["cost_per_case"],
    }
    (OUTPUTS_DIR / "cost_sweep.json").write_text(json.dumps({
        "assumptions": {
            "base_rate": BASE_RATE, "cost_fn": COST_FN, "cost_fp": COST_FP, "cost_tp": COST_TP,
            "cost_tn": COST_TN, "review_minutes": REVIEW_MINUTES,
            "review_cost_per_hour": REVIEW_COST_PER_HOUR, "cases_per_day": CASES_PER_DAY,
            "work_minutes_per_day": WORK_MINUTES_PER_DAY, "disease_classes": DISEASE_CLASSES,
        },
        "single_threshold_sweep": single,
        "chosen_single": best_single,
        "chosen_pair": best_pair,
        "review_load": {
            "cases_per_day": per_day,
            "minutes_per_day": per_day * REVIEW_MINUTES,
            "people_needed": per_day * REVIEW_MINUTES / WORK_MINUTES_PER_DAY,
        },
        "alternatives_cost_per_case": alternatives,
    }, indent=2))
    (OUTPUTS_DIR / "image_scores.json").write_text(json.dumps(rows))
    (OUTPUTS_DIR / "image_confusion.json").write_text(
        json.dumps(class_confusion(rows, best_single["t_high"], names), indent=2))
    (OUTPUTS_DIR / "baselines.json").write_text(json.dumps(baselines(data_yaml, rows), indent=2))


if __name__ == "__main__":
    data_yaml = download_dataset()
    weights = train(data_yaml)
    evaluate(weights, data_yaml)
    costing(data_yaml)
