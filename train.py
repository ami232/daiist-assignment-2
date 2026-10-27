"""
Assignment 2: training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file. main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via uv run python main.py train, this file must, start to finish, with no
manual steps in between:

- Get your Roboflow API key via from api_key import load_api_key, so that it
  works whether the key is in your .env file or in the environment, and
  download your dataset version from Roboflow Universe.
- Finetune yolo11n/yolo11s (or the -cls variant) on that dataset, from the
  pretrained checkpoint.
- Evaluate on the test split against the two baselines named in the brief: the
  pretrained COCO model zero-shot, and a trivial predict-nothing / majority
  class baseline.
- Sweep the confidence threshold, compute the expected cost at each point from
  the cost matrix you defined in REPORT.md, and save the sweep.
- Save everything app.py needs to build its dashboard without retraining:
  the metrics, the per-class numbers, the cost sweep, and the chosen operating
  point.

This is your real training pipeline. It takes as long as it takes, and it runs
wherever training is practical for you: your own machine, a GPU you have access
to, or a hosted notebook. Nothing re-runs it to check your submission.

The weights it produces are what you commit to weights/ and what your app
serves. Say in REPORT.md where you trained and what settings you used.

Save the trained model into weights/. That is the file you commit and the file
app.py loads.

How you structure the code beyond that is your call to make and be able to
defend: file layout, function boundaries, naming.
"""


"""Assignment 2: YOLO11 training, evaluation, and cost optimization."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = ROOT / "roboflow_download"
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"

CLASS_NAMES = ["Damaged road", "Potholes"]
CLASS_COUNT = len(CLASS_NAMES)

# Illustrative costs, in euros per image-level decision.
COSTS = {
    "false_negative": 100.0,
    "false_positive": 20.0,
    "human_review": 10.0,
}

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = os.getenv("DEVICE", "0" if torch.cuda.is_available() else "cpu")
EPOCHS = int(os.getenv("EPOCHS", "30"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16" if torch.cuda.is_available() else "4"))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def image_files(directory):
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGES
    )


def read_ground_truth(label_file):
    """Return the target class IDs present in a YOLO label file."""
    classes = set()
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                classes.add(int(line.split()[0]))
    return classes


def prepare_dataset():
    """Download Roboflow Version 2 and retain the two supported classes."""
    load_api_key()

    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)

    rf = Roboflow(api_key=load_api_key())
    project = rf.workspace("moh-fzwtt").project("pothole-detection-hg5ye")
    dataset = project.version(2).download(
        "yolov11", location=str(DOWNLOAD_DIR)
    )

    dataset_dir = Path(dataset.location).resolve()
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Could not find {data_yaml}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    # Remove class 2 annotations from every split. The original dataset
    # contains only four such annotations, all in the training split.
    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

        for label_file in labels_dir.glob("*.txt"):
            kept = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if line.strip() and int(line.split()[0]) in (0, 1):
                    kept.append(line)
            label_file.write_text(
                "\n".join(kept) + ("\n" if kept else ""),
                encoding="utf-8",
            )

    # Set paths explicitly so training works regardless of where the
    # downloaded directory is located.
    config["path"] = str(dataset_dir)
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"
    config["nc"] = CLASS_COUNT
    config["names"] = CLASS_NAMES

    data_yaml.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        if not image_files(split_dir / "images"):
            raise RuntimeError(f"No images found in {split_dir / 'images'}")

    print(f"Dataset prepared at: {dataset_dir}")
    print(f"Dataset YAML: {data_yaml}")
    return dataset_dir, data_yaml


def collect_predictions(model, images_dir, labels_dir, split_name):
    """Collect image-level scores and labels for threshold analysis.

    An image is positive if it contains at least one retained object.
    Its score is the highest confidence among detections of either target
    class. This image-level analysis is separate from YOLO's object-level
    precision, recall, and AP metrics.
    """
    images = image_files(images_dir)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    model_names = model.names
    if isinstance(model_names, list):
        model_names = dict(enumerate(model_names))

    def normalize(value):
        return " ".join(str(value).lower().replace("_", " ").split())

    # Only exact target-class names are mapped. COCO has neither of these
    # classes, so its zero-shot target-hazard scores will be zero.
    name_to_id = {normalize(name): idx for idx, name in enumerate(CLASS_NAMES)}
    mapped = {
        int(class_id): name_to_id[normalize(name)]
        for class_id, name in model_names.items()
        if normalize(name) in name_to_id
    }

    records = []
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=IMAGE_SIZE,
        conf=0.001,
        device=DEVICE,
        stream=True,
        verbose=False,
    )

    for result in results:
        image_path = Path(result.path)
        gt_ids = read_ground_truth(labels_dir / f"{image_path.stem}.txt")
        class_scores = [0.0] * CLASS_COUNT
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                original_id = int(box.cls.item())
                if original_id not in mapped:
                    continue

                target_id = mapped[original_id]
                confidence = float(box.conf.item())
                class_scores[target_id] = max(
                    class_scores[target_id], confidence
                )
                detections.append({
                    "class_id": target_id,
                    "class_name": CLASS_NAMES[target_id],
                    "confidence": confidence,
                    "xyxy": [float(x) for x in box.xyxy[0].tolist()],
                })

        records.append({
            "image": image_path.name,
            "truth": int(bool(gt_ids.intersection({0, 1}))),
            "truth_classes": sorted(gt_ids.intersection({0, 1})),
            "class_scores": class_scores,
            "score": max(class_scores),
            "detections": detections,
        })

    save_json(
        ARTIFACTS_DIR / f"{split_name}_predictions.json",
        records,
    )
    return records


def evaluate_policy(records, low, high):
    """Evaluate an image-level policy with a human-review band.

    score < low: no automatic flag
    low <= score < high: human review
    score >= high: automatic hazard flag

    The confusion matrix excludes reviewed cases. Review cost is counted
    separately, and reviewed cases are assumed to be resolved correctly.
    """
    tn = fp = fn = tp = reviewed = 0
    total_cost = 0.0

    for record in records:
        truth = bool(record["truth"])
        score = float(record["score"])

        if low < high and low <= score < high:
            reviewed += 1
            total_cost += COSTS["human_review"]
        else:
            predicted = score >= high
            if truth and predicted:
                tp += 1
            elif truth and not predicted:
                fn += 1
                total_cost += COSTS["false_negative"]
            elif not truth and predicted:
                fp += 1
                total_cost += COSTS["false_positive"]
            else:
                tn += 1

    n = len(records)
    return {
        "low_threshold": float(low),
        "high_threshold": float(high),
        "images": n,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "reviewed": reviewed,
        "review_share": reviewed / n if n else 0.0,
        "cost_total_eur": total_cost,
        "cost_per_1000_images_eur": total_cost * 1000 / n if n else 0.0,
        "confusion_matrix_auto_only": [[tn, fp], [fn, tp]],
    }


def optimize_thresholds(records):
    """Select a cost-minimizing review band using validation predictions."""
    thresholds = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    sweep = []

    for threshold in thresholds:
        result = evaluate_policy(records, threshold, threshold)
        sweep.append({
            "threshold": threshold,
            "cost_per_1000_images_eur":
                result["cost_per_1000_images_eur"],
            "false_positive": result["false_positive"],
            "false_negative": result["false_negative"],
        })

    single_best = min(
        sweep,
        key=lambda row: row["cost_per_1000_images_eur"],
    )

    bands = []
    for low in thresholds:
        for high in thresholds:
            if low < high:
                result = evaluate_policy(records, low, high)
                bands.append(result)

    best_band = min(
        bands,
        key=lambda row: (
            row["cost_per_1000_images_eur"],
            row["review_share"],
        ),
    )

    save_json(ARTIFACTS_DIR / "cost_sweep.json", sweep)
    save_json(ARTIFACTS_DIR / "review_band_sweep.json", bands)
    save_json(ARTIFACTS_DIR / "operating_point.json", {
        "single_threshold_best": single_best,
        "review_band_best": best_band,
        "selection_split": "validation",
        "costs_eur": COSTS,
    })

    return single_best, best_band


def extract_per_class_metrics(metrics):
    """Extract class-level precision, recall, AP50 and AP50-95."""
    box = metrics.box
    class_indices = np.asarray(
        getattr(box, "ap_class_index", []), dtype=int
    )
    output = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        positions = np.where(class_indices == class_id)[0]
        if not len(positions):
            output[class_name] = {
                "precision": None,
                "recall": None,
                "ap50": None,
                "map50_95": None,
            }
            continue

        pos = int(positions[0])

        def value(attribute):
            arr = np.asarray(getattr(box, attribute, []))
            return float(arr[pos]) if arr.ndim and len(arr) > pos else None

        output[class_name] = {
            "precision": value("p"),
            "recall": value("r"),
            "ap50": value("ap50"),
            "map50_95": value("ap"),
        }

    return output


def binary_baseline_metrics(records, threshold=0.5):
    """Image-level metrics for a score-based prediction baseline."""
    return evaluate_policy(records, threshold, threshold)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir, data_yaml = prepare_dataset()

    print(f"\nTraining YOLO11n on {DEVICE} for {EPOCHS} epochs.")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=10,
        workers=2,
        device=DEVICE,
        seed=42,
        pretrained=True,
        project=str(ROOT / "runs" / "detect"),
        name="road_damage_yolo11n",
        exist_ok=True,
        plots=True,
    )

    best_path = Path(model.trainer.best)
    if not best_path.exists():
        raise FileNotFoundError(f"Training did not produce {best_path}")

    final_weights = WEIGHTS_DIR / "road_damage_yolo11n.pt"
    shutil.copy2(best_path, final_weights)
    print(f"Saved model weights to {final_weights}")

    trained_model = YOLO(str(final_weights))

    print("\nEvaluating fine-tuned model on the test split.")
    test_metrics = trained_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name="road_damage_test",
        exist_ok=True,
    )

    box = test_metrics.box
    object_metrics = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": extract_per_class_metrics(test_metrics),
    }

    try:
        object_metrics["confusion_matrix"] = (
            test_metrics.confusion_matrix.matrix.tolist()
        )
    except (AttributeError, TypeError):
        object_metrics["confusion_matrix"] = None

    valid_records = collect_predictions(
        trained_model,
        dataset_dir / "valid" / "images",
        dataset_dir / "valid" / "labels",
        "validation",
    )
    test_records = collect_predictions(
        trained_model,
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "labels",
        "test",
    )

    single_best, best_band = optimize_thresholds(valid_records)
    test_policy = evaluate_policy(
        test_records,
        best_band["low_threshold"],
        best_band["high_threshold"],
    )

    print("\nEvaluating pretrained COCO YOLO11n zero-shot baseline.")
    coco_model = YOLO("yolo11n.pt")
    coco_records = collect_predictions(
        coco_model,
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "labels",
        "coco_zero_shot_test",
    )

    # This model has no "Damaged road" or "Potholes" class in COCO.
    # Therefore no COCO class is counted as a target-hazard detection.
    coco_baseline = binary_baseline_metrics(coco_records)
    trivial_baseline = binary_baseline_metrics([
        {**record, "score": 0.0} for record in test_records
    ])

    summary = {
        "classes": CLASS_NAMES,
        "dataset": {
            "source": "Roboflow Universe",
            "workspace": "moh-fzwtt",
            "project": "pothole-detection-hg5ye",
            "version": 2,
            "directory": str(dataset_dir),
            "train_images": len(image_files(dataset_dir / "train" / "images")),
            "validation_images": len(image_files(dataset_dir / "valid" / "images")),
            "test_images": len(image_files(dataset_dir / "test" / "images")),
        },
        "training": {
            "model": "YOLO11n",
            "device": str(DEVICE),
            "epochs_requested": EPOCHS,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "weights": str(final_weights.relative_to(ROOT)),
        },
        "costs_eur": COSTS,
        "object_detection_test_metrics": object_metrics,
        "image_level_test_metrics": {
            "fine_tuned_at_validation_selected_band": test_policy,
            "pretrained_coco_zero_shot": coco_baseline,
            "predict_nothing": trivial_baseline,
        },
        "validation_selected_operating_points": {
            "single_threshold": single_best,
            "review_band": best_band,
        },
        "caveat": (
            "Image-level costs treat an image as the decision unit. "
            "Reviewed images incur review cost and are assumed to be "
            "resolved correctly. The confusion matrix excludes reviewed "
            "images. COCO has neither target class, so its target-class "
            "zero-shot baseline makes no mapped target detections."
        ),
    }

    save_json(ARTIFACTS_DIR / "metrics.json", summary)

    print("\nTraining and evaluation complete.")
    print(f"Model weights: {final_weights}")
    print(f"Metrics: {ARTIFACTS_DIR / 'metrics.json'}")
    print(f"Cost sweep: {ARTIFACTS_DIR / 'cost_sweep.json'}")
    print(f"Chosen review band: {best_band['low_threshold']:.2f}–"
          f"{best_band['high_threshold']:.2f}")
    print(f"Test cost per 1,000 images: "
          f"€{test_policy['cost_per_1000_images_eur']:.2f}")


if __name__ == "__main__":
    main()
