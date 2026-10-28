"""
Assignment 2: training pipeline.

    uv run python main.py train          # the full pipeline
    uv run python train.py --eval-only   # re-run evaluation and the cost sweep
                                         # on the committed weights, no training
    QUICK=1 uv run python train.py       # smoke test: 1 epoch on 5% of train,
                                         # written to runs/quick, not committed

Steps:

1. Download joseph-nelson/hard-hat-workers v5 from Roboflow Universe.
2. Finetune yolo11n from the pretrained COCO checkpoint, copy the best epoch to
   weights/best.pt.
3. Evaluate on the test split: overall and per-class detection metrics, and
   the detection confusion matrix.
4. Score every test image for the gate decision (highest `head` confidence),
   for the finetuned model and the COCO zero-shot baseline.
5. Compare against the baselines, sweep the threshold with costs.py, and save
   everything app.py needs into artifacts/.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import costs
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent

WORKSPACE, PROJECT, VERSION = "joseph-nelson", "hard-hat-workers", 5
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-{VERSION}"
POSITIVE_CLASS = "head"

# Sized to train on a laptop CPU (Apple M2) in a few hours. 416 is close to the
# images' native width of about 500 px, so little detail is lost.
BASE_MODEL = "yolo11n.pt"
EPOCHS = 10
PATIENCE = 15
IMGSZ = 416
BATCH = 16
SEED = 0
CONFUSION_CONF = 0.25

QUICK = os.environ.get("QUICK") == "1"
if QUICK:
    WEIGHTS = ROOT / "runs" / "quick" / "best.pt"
    ARTIFACTS = ROOT / "runs" / "quick" / "artifacts"
else:
    WEIGHTS = ROOT / "weights" / "best.pt"
    ARTIFACTS = ROOT / "artifacts"


def download_dataset() -> Path:
    """Download the dataset once and point data.yaml at absolute paths.

    Roboflow writes paths like ../train/images, which Ultralytics resolves
    against the wrong folder, so the yaml is rewritten with an explicit root.
    """
    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR)
        )
    config = yaml.safe_load(data_yaml.read_text())
    config.update(
        path=str(DATA_DIR), train="train/images", val="valid/images", test="test/images"
    )
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return data_yaml


def train(data_yaml: Path) -> dict:
    """Finetune from the pretrained checkpoint and copy the best weights."""
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=1 if QUICK else EPOCHS,
        fraction=0.05 if QUICK else 1.0,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="quick" if QUICK else "finetune",
        exist_ok=True,
    )
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)
    return {
        "base_model": BASE_MODEL,
        "epochs_max": 1 if QUICK else EPOCHS,
        "epochs_run": model.trainer.epoch + 1,
        "patience": PATIENCE,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "device": str(model.trainer.device),
        "quick": QUICK,
    }


def evaluate_detection(data_yaml: Path) -> dict:
    """Overall and per-class detection metrics on the test split."""
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(ROOT / "runs"),
        name="test",
        exist_ok=True,
    )
    names = [model.names[i] for i in sorted(model.names)]
    box = metrics.box
    instances = getattr(metrics, "nt_per_class", None)
    per_class = {}
    for i, name in enumerate(names):
        per_class[name] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
            "instances": int(instances[i]) if instances is not None else None,
        }
    # AP needs every box, but a confusion matrix built from boxes down to
    # conf 0.001 mostly counts guesses no threshold would keep. Rebuild it at
    # the usual 0.25 instead.
    at_conf = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        conf=CONFUSION_CONF,
        project=str(ROOT / "runs"),
        name="test_conf",
        exist_ok=True,
    )
    plots = {
        "BoxPR_curve.png": metrics.save_dir,
        "confusion_matrix.png": at_conf.save_dir,
        "confusion_matrix_normalized.png": at_conf.save_dir,
    }
    for plot, folder in plots.items():
        source = Path(folder) / plot
        if source.exists():
            shutil.copy(source, ARTIFACTS / plot)
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": names + ["background"],
            "orientation": "rows are predicted class, columns are true class",
            "confidence": CONFUSION_CONF,
            "matrix": at_conf.confusion_matrix.matrix.astype(int).tolist(),
        },
    }


def image_has_positive(image: Path, class_id: int) -> bool:
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    if not label.exists():
        return False
    return any(
        line.split()[0] == str(class_id)
        for line in label.read_text().splitlines()
        if line.strip()
    )


def score_images(weights: str, images: list[Path], class_name: str) -> np.ndarray:
    """Each image's highest confidence for one class, 0 if none was detected.

    Images go in chunks of BATCH: Ultralytics treats a list source as a single
    batch, so passing all test images at once runs out of memory.
    """
    model = YOLO(weights)
    class_id = {v: k for k, v in model.names.items()}[class_name]
    scores = []
    for start in range(0, len(images), BATCH):
        chunk = [str(p) for p in images[start:start + BATCH]]
        for result in model.predict(source=chunk, imgsz=IMGSZ, conf=0.001, verbose=False):
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            hits = confs[classes == class_id]
            scores.append(float(hits.max()) if hits.size else 0.0)
    return np.array(scores)


def decision_summary(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Image-level numbers for one rule: fail if score >= threshold."""
    shares = costs.outcome_shares(scores, labels, threshold, threshold)
    predicted = scores >= threshold
    flagged = predicted.sum()
    return {
        "threshold": threshold,
        "accuracy_test": float((predicted == labels).mean()),
        "recall": float(shares["pos_fail"]),
        "false_positive_rate": float(shares["neg_fail"]),
        "precision_test": float((predicted & labels).sum() / flagged) if flagged else None,
        "cost_per_1000": 1000 * costs.cost_per_case(shares),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="skip training and evaluate the committed weights",
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data_yaml = download_dataset()
    data_config = yaml.safe_load(data_yaml.read_text())
    positive_id = data_config["names"].index(POSITIVE_CLASS)

    metrics_path = ARTIFACTS / "metrics.json"
    if args.eval_only:
        if not WEIGHTS.exists():
            sys.exit(f"--eval-only needs {WEIGHTS.relative_to(ROOT)}. Train first.")
        previous = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        training = previous.get("training", {})
    else:
        training = train(data_yaml)

    detection = evaluate_detection(data_yaml)

    images = sorted((DATA_DIR / "test" / "images").glob("*.*"))
    labels = np.array([image_has_positive(p, positive_id) for p in images])
    finetuned_scores = score_images(str(WEIGHTS), images, POSITIVE_CLASS)
    coco_scores = score_images(BASE_MODEL, images, "person")
    pd.DataFrame({
        "image": [p.name for p in images],
        "has_no_hard_hat": labels,
        "score": finetuned_scores,
        "coco_person_score": coco_scores,
    }).to_csv(ARTIFACTS / "test_scores.csv", index=False)

    sweep = costs.sweep(finetuned_scores, labels)
    sweep.to_csv(ARTIFACTS / "cost_sweep.csv", index=False)
    best = sweep.loc[sweep["cost_per_1000"].idxmin()]
    chosen = float(best["threshold"])

    image_level = {
        "finetuned_at_chosen_threshold": decision_summary(finetuned_scores, labels, chosen),
        "finetuned_at_default_0.25": decision_summary(finetuned_scores, labels, 0.25),
        "coco_zero_shot_fail_if_person": decision_summary(coco_scores, labels, 0.25),
        "trivial_everyone_passes": decision_summary(np.zeros(len(labels)), labels, 1.01),
    }

    metrics = {
        "dataset": {
            "workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
            "classes": data_config["names"], "positive_class": POSITIVE_CLASS,
            "test_images": len(images), "test_positive_images": int(labels.sum()),
        },
        "training": training,
        "detection": {
            "finetuned": detection,
            "coco_zero_shot": {
                "map50": 0.0, "map50_95": 0.0,
                "note": "COCO has no head or helmet class, so it cannot detect either.",
            },
            "trivial_predict_nothing": {"map50": 0.0, "map50_95": 0.0},
        },
        "image_level": image_level,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    low, high, band_cost = costs.best_band(finetuned_scores, labels)
    band_shares = costs.outcome_shares(finetuned_scores, labels, low, high)
    review_share = costs.review_share(band_shares)

    assumptions = {
        name: getattr(costs, name)
        for name in [
            "BASE_RATE", "COST_FP", "COST_FN", "COST_TP", "COST_TN", "REVIEW_COST",
            "REVIEWER_ACCURACY", "GUARD_COST_PER_CASE", "CASES_PER_DAY",
        ]
    }
    (ARTIFACTS / "operating_point.json").write_text(json.dumps({
        "threshold": chosen,
        "cost_per_1000": float(best["cost_per_1000"]),
        "recall": float(best["recall"]),
        "false_positive_rate": float(best["false_positive_rate"]),
        "review_band": {
            "low": low,
            "high": high,
            "cost_per_1000": 1000 * band_cost,
            "review_share": review_share,
            "reviews_per_day": review_share * costs.CASES_PER_DAY,
            "shares": {k: float(v) for k, v in band_shares.items()},
        },
        "assumptions": assumptions,
    }, indent=2))

    print("\n" + "=" * 72)
    print(f"Weights: {WEIGHTS.relative_to(ROOT)}   Artifacts: {ARTIFACTS.relative_to(ROOT)}")
    print(f"Test mAP50 {detection['map50']:.3f}   mAP50-95 {detection['map50_95']:.3f}")
    for name, row in detection["per_class"].items():
        print(f"  {name:7s} P {row['precision']:.3f}  R {row['recall']:.3f}  AP50 {row['ap50']:.3f}")
    print(f"Cheapest threshold {chosen:.2f}: {best['cost_per_1000']:.0f} EUR per 1,000 gate passes")
    print(
        f"Cheapest review band: pass below {low:.2f}, fail at or above {high:.2f}: "
        f"{1000 * band_cost:.0f} EUR per 1,000, {review_share:.1%} to review"
    )
    print(pd.DataFrame(image_level).T.round(3).to_string())
    print("=" * 72)


if __name__ == "__main__":
    main()
