
from pathlib import Path
from collections import Counter
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key
from decision import BASE_RATE, choose_triage, outcome, sweep

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "ppe-v1"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"

WORKSPACE = "ppe-detection-e2ly1"
PROJECT = "ppe-detection-huwcd"
VERSION = 1

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def save_json(name, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def download_dataset():
    if not (DATA / "data.yaml").exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)

        print("Downloading Roboflow dataset...", flush=True)

        rf = Roboflow(api_key=load_api_key())

        rf.workspace(WORKSPACE).project(PROJECT).version(
            VERSION
        ).download("yolov11", location=str(DATA))

    file = DATA / "data.yaml"

    if not file.exists():
        raise FileNotFoundError("Dataset download failed")

    cfg = yaml.safe_load(file.read_text())

    names = cfg["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    else:
        names = {int(k): v for k, v in names.items()}

    cfg["names"] = names

    for split, folder in [
        ("train", "train"),
        ("val", "valid"),
        ("test", "test")
    ]:
        images = DATA / folder / "images"

        if not images.exists():
            raise FileNotFoundError(str(images))

        cfg[split] = str(images.resolve())

    cfg["path"] = str(DATA.resolve())

    config = DATA / "training.yaml"
    config.write_text(yaml.safe_dump(cfg))

    return config, cfg


def image_files(folder):
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]
    )


def annotations(image):
    label = (
        image.parent.parent
        / "labels"
        / (image.stem + ".txt")
    )

    if not label.exists():
        return []

    rows = []

    for line in label.read_text().splitlines():
        parts = line.split()

        if len(parts) >= 5:
            rows.append((
                int(parts[0]),
                *map(float, parts[1:5])
            ))

    return rows


def dataset_statistics(cfg, nohat, names):
    counts = Counter()
    sizes = {}
    positive_images = {}
    box_sizes = []

    for split in ["train", "val", "test"]:
        images = image_files(cfg[split])

        sizes[split] = len(images)
        positive_images[split] = 0

        for image in images:
            labels = annotations(image)

            if any(a[0] == nohat for a in labels):
                positive_images[split] += 1

            for cls, x, y, w, h in labels:
                counts[cls] += 1

                if cls == nohat:
                    box_sizes.append((w, h, w * h))

    stats = {
        "split_images": sizes,
        "positive_images": positive_images,
        "class_instances": {
            name: counts[i]
            for i, name in names.items()
        },
        "no_hardhat_median_box_size": (
            np.median(box_sizes, axis=0).tolist()
            if box_sizes else None
        )
    }

    save_json("dataset_stats.json", stats)

    print(
        "DATASET STATISTICS:",
        json.dumps(stats, indent=2),
        flush=True
    )

    return stats


def prediction_rows(model, folder, nohat):
    rows = []

    predictions = model.predict(
        source=str(folder),
        stream=True,
        conf=0.001,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )

    for prediction in predictions:
        image = Path(prediction.path)
        score = 0.0

        if prediction.boxes is not None:
            boxes = prediction.boxes

            if len(boxes):
                classes = (
                    boxes.cls.cpu().numpy().astype(int)
                )
                confidences = (
                    boxes.conf.cpu().numpy()
                )

                relevant = confidences[
                    classes == nohat
                ]

                if len(relevant):
                    score = float(relevant.max())

        positive = any(
            a[0] == nohat
            for a in annotations(image)
        )

        rows.append({
            "image": image.name,
            "score": score,
            "positive": int(positive)
        })

    return rows


def evaluation_metrics(metrics, names):
    box = metrics.box
    per_class = {}

    for j, cls_id in enumerate(box.ap_class_index):
        name = names[int(cls_id)]

        per_class[name] = {
            "precision": float(box.p[j]),
            "recall": float(box.r[j]),
            "AP50": float(box.ap50[j]),
            "mAP50_95": float(box.ap[j])
        }

    return {
        "AP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": per_class
    }


def main():
    print("Starting Assignment 2 training...", flush=True)

    config, cfg = download_dataset()
    names = cfg["names"]

    matches = [
        i for i, name in names.items()
        if "".join(
            c.lower() for c in name
            if c.isalnum()
        ) == "nohardhat"
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Cannot identify NO-Hardhat class: {names}"
        )

    nohat = matches[0]

    stats = dataset_statistics(
        cfg, nohat, names
    )

    epochs = int(
        os.environ.get("TRAIN_EPOCHS", "15")
    )

    print(
        f"Training YOLO11n for {epochs} epochs",
        flush=True
    )

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(config),
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=DEVICE,
        workers=2,
        seed=42,
        project=str(ROOT / "runs"),
        name="helmet",
        exist_ok=True
    )

    WEIGHTS.mkdir(exist_ok=True)

    best = WEIGHTS / "helmet_best.pt"
    shutil.copy2(model.trainer.best, best)

    print("Saved model:", best, flush=True)

    trained = YOLO(str(best))

    metrics = trained.val(
        data=str(config),
        split="test",
        imgsz=640,
        device=DEVICE,
        plots=True,
        project=str(ROOT / "runs"),
        name="test_evaluation",
        exist_ok=True
    )

    detection = evaluation_metrics(
        metrics, names
    )

    save_json(
        "per_class_metrics.json",
        detection
    )

    validation = prediction_rows(
        trained, cfg["val"], nohat
    )

    test = prediction_rows(
        trained, cfg["test"], nohat
    )

    save_json(
        "validation_predictions.json",
        validation
    )

    save_json(
        "test_predictions.json",
        test
    )

    scores = np.array([
        row["score"] for row in validation
    ])

    truth = np.array([
        row["positive"] for row in validation
    ], dtype=bool)

    curve, best_single = sweep(
        scores, truth, base_rate=BASE_RATE
    )

    policy = choose_triage(
        scores, truth, base_rate=BASE_RATE
    )

    save_json("cost_curve.json", curve)
    save_json("policy.json", policy)

    plt.figure(figsize=(8, 4))

    plt.plot(
        [r["threshold"] for r in curve],
        [r["eur_per_1000"] for r in curve]
    )

    plt.axvline(
        best_single["threshold"],
        color="red",
        linestyle="--",
        label="Selected threshold"
    )

    plt.xlabel("Confidence threshold")
    plt.ylabel("Expected cost per 1,000 images (€)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(RESULTS / "cost_curve.png")
    plt.close()

    test_scores = np.array([
        row["score"] for row in test
    ])

    test_truth = np.array([
        row["positive"] for row in test
    ], dtype=bool)

    test_policy = outcome(
        test_scores,
        test_truth,
        policy["lower"],
        policy["upper"],
        BASE_RATE
    )

    # COCO has no NO-Hardhat class.
    # Its zero-shot helmet violation recall is zero.
    # Run it on the same test images for comparison.
    coco = YOLO("yolo11n.pt")
    coco_person_boxes = 0

    for prediction in coco.predict(
        source=cfg["test"],
        stream=True,
        imgsz=640,
        conf=0.25,
        device=DEVICE,
        verbose=False
    ):
        if prediction.boxes is not None:
            coco_person_boxes += int(
                (
                    prediction.boxes.cls.cpu().numpy()
                    == 0
                ).sum()
            )

    analysis = {
        "dataset": stats,
        "classes": names,
        "positive_class": names[nohat],
        "training": {
            "epochs": epochs,
            "image_size": 640,
            "device": str(DEVICE)
        },
        "finetuned_test": detection,
        "coco_zero_shot": {
            "no_hardhat_recall": 0.0,
            "person_detections_test": coco_person_boxes
        },
        "trivial_predict_nothing": {
            "no_hardhat_recall": 0.0
        },
        "best_single_threshold_validation": best_single,
        "review_policy_validation": policy,
        "review_policy_test": test_policy
    }

    save_json("analysis.json", analysis)

    print("TRAINING AND EVALUATION COMPLETE!")
    print("Model: weights/helmet_best.pt")
    print("Results: results/analysis.json")
    print("Cost curve: results/cost_curve.png")
    print("Confusion matrix: runs/test_evaluation/")


if __name__ == "__main__":
    main()
