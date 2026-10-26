import json
import os
import shutil
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from dataset import (
    PROJECT,
    RESULTS_DIR,
    VERSION,
    WORKSPACE,
    class_names,
    describe_dataset,
    download_dataset,
    write_local_yaml,
)
from costing import run_costing
from duplicates import DUPLICATE_MAX_DISTANCE, SWEEP_THRESHOLDS, find_duplicates
from evaluate import evaluate_predictions, load_ground_truth, predict

ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = ROOT / "pretrained"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"

BASE_MODEL = "yolo11n"
EPOCHS = int(os.environ.get("EPOCHS", "100"))
IMGSZ = 512
BATCH = 16
PATIENCE = 10
SEED = 0
REPORT_CONF = 0.25
COCO_IMGSZ = 640
COCO_PERSON = 0
ZERO_SHOT_TARGET = "swimmer"


def pretrained_path() -> str:
    PRETRAINED_DIR.mkdir(exist_ok=True)
    return str(PRETRAINED_DIR / f"{BASE_MODEL}.pt")


def finetune(data_yaml: Path) -> Path:
    model = YOLO(pretrained_path())
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        project=str(RUNS_DIR),
        name="finetune",
        exist_ok=True,
    )
    WEIGHTS_DIR.mkdir(exist_ok=True)
    target = WEIGHTS_DIR / "best.pt"
    shutil.copy(Path(model.trainer.best), target)
    for stray in WEIGHTS_DIR.glob("*.pt"):
        if stray.name != target.name:
            stray.unlink()
    return target


def keep_distant(truth: dict, distances: dict, threshold: int) -> dict:
    return {path: item for path, item in truth.items() if distances[path] > threshold}


def evaluate_all(dataset_dir: Path, names: list[str], weights: Path, distances: dict) -> dict:
    truth = load_ground_truth(dataset_dir, "test")
    images = list(truth)

    finetuned_predictions = predict(YOLO(str(weights)), images, IMGSZ)
    zero_shot_predictions = predict(
        YOLO(pretrained_path()),
        images,
        COCO_IMGSZ,
        classes=[COCO_PERSON],
        class_map={COCO_PERSON: names.index(ZERO_SHOT_TARGET)},
    )
    predictions = {
        "finetuned": finetuned_predictions,
        "coco_zero_shot": zero_shot_predictions,
        "predict_nothing": {},
    }

    full = {
        name: evaluate_predictions(preds, truth, names, REPORT_CONF)
        for name, preds in predictions.items()
    }

    default_subset = keep_distant(truth, distances, DUPLICATE_MAX_DISTANCE)
    deduplicated = {
        name: evaluate_predictions(preds, default_subset, names, REPORT_CONF)
        for name, preds in predictions.items()
    }

    by_threshold = {}
    for threshold in SWEEP_THRESHOLDS:
        subset = keep_distant(truth, distances, threshold)
        result = evaluate_predictions(finetuned_predictions, subset, names, REPORT_CONF)
        result["test_images"] = len(subset)
        by_threshold[str(threshold)] = result

    return {
        "full": full,
        "deduplicated": deduplicated,
        "finetuned_by_duplicate_threshold": by_threshold,
        "test_images_full": len(truth),
        "test_images_deduplicated": len(default_subset),
    }


def show_models(title: str, results: dict) -> None:
    print(f"\n=== {title} ===")
    for model_name, result in results.items():
        print(f"\n{model_name}: mAP50 {result['map50']}  mAP50-95 {result['map50_95']}  (conf {result['conf_threshold']})")
        for class_name, row in result["per_class"].items():
            print(
                f"  {class_name:10s} n={row['instances']:4d}  AP50 {row['ap50']}  "
                f"P {row['precision']}  R {row['recall']}  TP {row['tp']}  FP {row['fp']}  FN {row['fn']}"
            )


def show(results: dict) -> None:
    show_models(f"full test split ({results['test_images_full']} images)", results["full"])
    show_models(
        f"deduplicated test split, distance > {DUPLICATE_MAX_DISTANCE} ({results['test_images_deduplicated']} images)",
        results["deduplicated"],
    )
    print("\n=== finetuned model as stricter or looser duplicates are removed ===")
    for threshold, result in results["finetuned_by_duplicate_threshold"].items():
        cells = "  ".join(
            f"{name}: n={row['instances']} AP50 {row['ap50']} R {row['recall']}"
            for name, row in result["per_class"].items()
        )
        print(f"distance > {threshold}: {result['test_images']} images, mAP50 {result['map50']}  |  {cells}")
    print("\nfinetuned confusion matrix, full test split (rows true, columns predicted)")
    labels = results["full"]["finetuned"]["confusion_labels"]
    print("            " + "  ".join(f"{label[:9]:>9s}" for label in labels))
    for label, row in zip(labels, results["full"]["finetuned"]["confusion_matrix"]):
        print(f"{label[:10]:>10s}  " + "  ".join(f"{value:9d}" for value in row))


def main() -> None:
    dataset_dir = download_dataset()
    describe_dataset(dataset_dir)
    distances = find_duplicates(dataset_dir)
    names = class_names(dataset_dir)
    data_yaml = write_local_yaml(dataset_dir)

    weights = finetune(data_yaml)
    results = evaluate_all(dataset_dir, names, weights, distances)

    payload = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "training": {
            "base_model": BASE_MODEL,
            "epochs_requested": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "patience": PATIENCE,
            "seed": SEED,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        "zero_shot_mapping": {"coco_class": "person", "dataset_class": ZERO_SHOT_TARGET},
        "duplicate_default_distance": DUPLICATE_MAX_DISTANCE,
        "split": "test",
        "models": results["full"],
        "models_deduplicated": results["deduplicated"],
        "finetuned_by_duplicate_threshold": results["finetuned_by_duplicate_threshold"],
        "test_images_full": results["test_images_full"],
        "test_images_deduplicated": results["test_images_deduplicated"],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    show(results)
    run_costing(dataset_dir, names, weights, IMGSZ)


if __name__ == "__main__":
    main()
