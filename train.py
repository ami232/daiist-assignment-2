import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key


RAW = Path("data/railway_v1")
CLEAN = Path("data/railway_v1_clean")
WEIGHTS = Path("weights")
RESULTS = Path("results")

CRITICAL = {
    "fishplate_boltmissing",
    "track_boltmissing",
    "track_crack",
}

# Illustrative assumptions, not official Adif costs.
COST = {"TP": 150, "FP": 150, "FN": 3000, "TN": 0}
THRESHOLDS = [0.001, 0.01] + [
    round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)
]


def source_id(path):
    return path.stem.split(".rf.")[0]


def image_files(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def get_dataset():
    if (RAW / "data.yaml").exists():
        print("Original dataset already available.")
        return

    if RAW.exists() and not any(RAW.iterdir()):
        RAW.rmdir()

    rf = Roboflow(api_key=load_api_key())
    project = rf.workspace("kerusakan-rel-kereta").project(
        "railway-track-fault-detection-fgbu5-eys2z"
    )

    project.version(1).download(
        "yolov11",
        location=str(RAW),
    )

    if not (RAW / "data.yaml").exists():
        raise RuntimeError("Roboflow download is incomplete.")


def prepare_clean_dataset():
    splits = ("train", "valid", "test")

    original = {
        s: image_files(RAW / s / "images")
        for s in splits
    }

    ids = {
        s: {source_id(p) for p in original[s]}
        for s in splits
    }

    # Keep test intact. Remove overlaps from validation
    # and training to avoid sharing source IDs across splits.
    keep = {
        "test": original["test"],
        "valid": [
            p for p in original["valid"]
            if source_id(p) not in ids["test"]
        ],
        "train": [
            p for p in original["train"]
            if source_id(p) not in ids["test"]
            and source_id(p) not in ids["valid"]
        ],
    }

    clean_ids = {
        s: {source_id(p) for p in keep[s]}
        for s in splits
    }

    assert not (clean_ids["train"] & clean_ids["valid"])
    assert not (clean_ids["train"] & clean_ids["test"])
    assert not (clean_ids["valid"] & clean_ids["test"])

    counts = {s: len(keep[s]) for s in splits}

    if not (CLEAN / "data.yaml").exists():
        CLEAN.mkdir(parents=True, exist_ok=True)

        for split in splits:
            (CLEAN / split / "images").mkdir(parents=True, exist_ok=True)
            (CLEAN / split / "labels").mkdir(parents=True, exist_ok=True)

            for img in keep[split]:
                label = RAW / split / "labels" / (img.stem + ".txt")

                if not label.exists():
                    raise FileNotFoundError(label)

                shutil.copy2(
                    img, CLEAN / split / "images" / img.name
                )
                shutil.copy2(
                    label, CLEAN / split / "labels" / label.name
                )

        shutil.copy2(RAW / "data.yaml", CLEAN / "data.yaml")

    for split in splits:
        actual = len(image_files(CLEAN / split / "images"))
        if actual != counts[split]:
            raise RuntimeError(
                f"Incorrect clean split {split}: {actual} vs {counts[split]}"
            )

    print("Clean dataset:", counts)
    return CLEAN / "data.yaml", counts


def ground_truth(split, critical_ids):
    records = []

    for img in image_files(CLEAN / split / "images"):
        label = CLEAN / split / "labels" / (img.stem + ".txt")
        classes = set()

        for line in label.read_text().splitlines():
            if line.strip():
                classes.add(int(line.split()[0]))

        records.append({
            "path": str(img),
            "positive": bool(classes & critical_ids),
        })

    return records


def collect_predictions(model, records, critical_ids):
    output = []

    for item in records:
        result = model.predict(
            source=item["path"],
            conf=0.001,
            imgsz=416,
            device="cpu",
            verbose=False,
        )[0]

        scores = [
            float(conf)
            for cls, conf in zip(
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
            )
            if int(cls) in critical_ids
        ]

        output.append({
            "positive": item["positive"],
            "score": max(scores, default=0.0),
        })

    return output


def confusion_at_threshold(records, threshold):
    tp = fp = fn = tn = 0

    for r in records:
        predicted = r["score"] >= threshold
        actual = r["positive"]

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def evaluate_cost(records, threshold):
    cm = confusion_at_threshold(records, threshold)
    total = sum(cm[k] * COST[k] for k in COST)

    return {
        "threshold": threshold,
        "confusion": cm,
        "total_cost_eur": total,
        "cost_per_1000_eur": round(
            1000 * total / len(records), 2
        ),
    }


def metrics_from_confusion(cm):
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]

    return {
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "accuracy": (
            (tp + cm["TN"]) / sum(cm.values())
        ),
    }


def save_json(name, data):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved:", path)


def main():
    WEIGHTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    get_dataset()
    config, split_counts = prepare_clean_dataset()
    config = config.resolve()

    dataset_info = yaml.safe_load(config.read_text())
    names = dataset_info["names"]
    critical_ids = {
        i for i, name in enumerate(names)
        if name in CRITICAL
    }

    if len(critical_ids) != 3:
        raise RuntimeError("Unexpected critical class names.")


    # 1. Train only if weights do not already exist.
    best_path = WEIGHTS / "best.pt"

    if best_path.exists():
        print("Existing trained model found. Skipping training.")
    else:
        model = YOLO("yolo11n.pt")

        model.train(
            data=str(config),
            epochs=30,
            imgsz=416,
            batch=4,
            device="cpu",
            workers=0,
            seed=42,
            project="runs",
            name="railway_final",
            exist_ok=True,
            plots=True,
        )

        shutil.copy2(model.trainer.best, best_path)
        print("Trained weights saved:", best_path)

    trained = YOLO(str(best_path))

    # 2. Object detection metrics on the held-out test.
    test_metrics = trained.val(
        data=str(config),
        split="test",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project="runs",
        name="railway_test_evaluation",
        exist_ok=True,
    )

    per_class = {}
    for idx, cls_id in enumerate(test_metrics.box.ap_class_index):
        per_class[names[int(cls_id)]] = {
            "precision": float(test_metrics.box.p[idx]),
            "recall": float(test_metrics.box.r[idx]),
            "AP50": float(test_metrics.box.ap50[idx]),
        }

    detection_summary = {
        "mAP50": float(test_metrics.box.map50),
        "mAP50_95": float(test_metrics.box.map),
        "per_class": per_class,
    }
    save_json("detection_metrics.json", detection_summary)

    # 3. Image-level ground truth and predictions.
    valid_gt = ground_truth("valid", critical_ids)
    test_gt = ground_truth("test", critical_ids)

    valid_predictions = collect_predictions(
        trained, valid_gt, critical_ids
    )
    test_predictions = collect_predictions(
        trained, test_gt, critical_ids
    )

    save_json("validation_predictions.json", valid_predictions)
    save_json("test_predictions.json", test_predictions)

    # 4. Select threshold on VALIDATION only.
    sweep = [
        evaluate_cost(valid_predictions, threshold)
        for threshold in THRESHOLDS
    ]
    chosen = min(
        sweep,
        key=lambda row: (
            row["total_cost_eur"],
            row["threshold"],
        ),
    )
    threshold = chosen["threshold"]

    save_json("cost_sweep.json", sweep)

    # 5. Evaluate selected threshold on TEST.
    final = evaluate_cost(test_predictions, threshold)
    final["metrics"] = metrics_from_confusion(final["confusion"])
    final["base_rate_test"] = (
        sum(x["positive"] for x in test_predictions)
        / len(test_predictions)
    )

    save_json("operating_point.json", {
        "selected_on": "validation",
        "threshold": threshold,
        "validation": chosen,
        "test": final,
        "cost_assumptions_eur": COST,
    })

    # 6. Trivial baseline: predict no critical defects.
    none_records = [
        {"positive": x["positive"], "score": 0.0}
        for x in test_gt
    ]
    trivial_cm = confusion_at_threshold(none_records, 0.5)

    # 7. COCO zero-shot baseline.
    # No exact class-name correspondence between COCO's
    # 80 categories and our seven railway classes.
    # We run inference, but do not invent a class mapping.
    coco = YOLO("yolo11n.pt")
    coco_names = {
        str(name).lower()
        for name in coco.names.values()
    }
    direct_matches = sorted(
        set(name.lower() for name in names) & coco_names
    )

    for item in test_gt:
        coco.predict(
            source=item["path"],
            imgsz=416,
            device="cpu",
            verbose=False,
        )

    # Without a valid mapping, no predicted COCO category
    # can count as a correctly named railway defect.
    coco_cm = confusion_at_threshold(none_records, 0.5)

    save_json("baselines.json", {
        "finetuned": {
            "image_confusion": final["confusion"],
            "image_metrics": final["metrics"],
            "detection_mAP50": detection_summary["mAP50"],
        },
        "predict_nothing": {
            "image_confusion": trivial_cm,
            "image_metrics": metrics_from_confusion(trivial_cm),
        },
        "coco_zero_shot": {
            "image_confusion": coco_cm,
            "image_metrics": metrics_from_confusion(coco_cm),
            "direct_class_matches": direct_matches,
            "limitation": (
                "COCO has no equivalent railway defect labels; "
                "zero-shot predictions cannot be meaningfully "
                "mapped to the dataset's critical classes."
            ),
        },
    })

    save_json("dataset_summary.json", {
        "version": 1,
        "clean_split_images": split_counts,
        "classes": names,
        "critical_classes": sorted(CRITICAL),
        "training": {
            "model": "yolo11n.pt",
            "epochs": 30,
            "image_size": 416,
            "batch": 4,
            "device": "cpu",
        },
        "limitations": [
            "Small test set",
            "Source-ID filtering does not rule out all visual duplicates",
            "Operational representativeness not established",
            "No direct COCO class mapping",
        ],
    })

    print("\nPIPELINE COMPLETED")
    print("Best weights:", best_path)
    print("Chosen threshold:", threshold)
    print("Test results:", final)


if __name__ == "__main__":
    main()
    from operating_policy import main as run_policy
    run_policy()