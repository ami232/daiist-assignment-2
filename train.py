"""
Assignment 2: training pipeline.

    uv run python main.py train

Start to finish, no manual steps:

1. Download pill-j8vgy v63 from Roboflow Universe.
2. Split its 600 valid images, with a fixed seed, into a tuning half (used for
   early stopping and for choosing thresholds) and a test half (touched once,
   for every number reported).
3. Finetune yolo11s from the COCO checkpoint and save weights/best.pt.
4. Evaluate on the test half: per-class box metrics for the finetuned model,
   and blister-level decisions for the finetuned model, the COCO model
   zero-shot, and two trivial baselines.
5. Sweep the threshold on the tuning half, pick the cheapest operating point,
   and price it on the test half.
6. Write everything app.py needs into artifacts/.

For a quick smoke run, set EPOCHS and FRACTION (share of training images
used) in the environment, e.g. EPOCHS=1 FRACTION=0.03.
"""

import json
import os
import random
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO

import config
import costing
from api_key import load_api_key


def device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download():
    location = config.DATA_DIR / f"{config.PROJECT}-{config.VERSION}"
    if not (location / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(config.WORKSPACE).project(config.PROJECT).version(config.VERSION)
        version.download("yolov8", location=str(location))
    return location


def labels_of(label_path):
    if not label_path.exists():
        return set()
    return {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}


def as_box(line):
    """Some pockets are labelled as polygons. Turn those rows into their bounding box.

    Ultralytics drops any label file that mixes polygon and box rows, which
    would silently remove 86 images, so every row is written as a box.
    """
    values = line.split()
    if len(values) <= 5:
        return " ".join(values)
    xs, ys = [float(v) for v in values[1::2]], [float(v) for v in values[2::2]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return f"{values[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def copy_images(images, source_labels, target):
    (target / "images").mkdir(parents=True)
    (target / "labels").mkdir(parents=True)
    for image in images:
        # YOLO shrinks every image to IMGSZ before it sees it, so doing it once
        # here changes nothing for the model but saves decoding 4K JPEGs every epoch.
        picture = Image.open(image).convert("RGB")
        picture.thumbnail((config.IMGSZ, config.IMGSZ))
        picture.save(target / "images" / image.name, quality=95)
        label = source_labels / f"{image.stem}.txt"
        if label.exists():
            rows = [as_box(line) for line in label.read_text().splitlines() if line.strip()]
            (target / "labels" / label.name).write_text("\n".join(rows) + "\n")


def make_split(source):
    """Copy train/ as is, split valid/ into tune/ and test/ halves, write a data yaml."""
    split_dir = config.DATA_DIR / "split"
    images = sorted((source / "valid" / "images").iterdir())
    random.Random(config.SPLIT_SEED).shuffle(images)
    halves = {"tune": images[: len(images) // 2], "test": images[len(images) // 2 :]}

    shutil.rmtree(split_dir, ignore_errors=True)
    copy_images(sorted((source / "train" / "images").iterdir()), source / "train" / "labels", split_dir / "train")
    for name, files in halves.items():
        copy_images(files, source / "valid" / "labels", split_dir / name)

    yaml_path = split_dir / "data.yaml"
    yaml_path.write_text(
        f"train: {(split_dir / 'train' / 'images').resolve()}\n"
        f"val: {(split_dir / 'tune' / 'images').resolve()}\n"
        f"test: {(split_dir / 'test' / 'images').resolve()}\n"
        f"nc: {len(config.CLASS_NAMES)}\n"
        f"names: {config.CLASS_NAMES}\n"
    )
    return yaml_path, split_dir


def finetune(yaml_path):
    epochs = int(os.environ.get("EPOCHS", config.EPOCHS))
    dev = device()
    model = YOLO(config.BASE_MODEL)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=config.IMGSZ,
        batch=16 if dev == 0 else 8,
        patience=config.PATIENCE,
        seed=0,
        device=dev,
        project=str(config.ROOT / "runs"),
        name="pill",
        exist_ok=True,
        fraction=float(os.environ.get("FRACTION", 1.0)),
        cache="ram",
        plots=True,
    )
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, config.WEIGHTS)
    return {"epochs_requested": epochs, "device": str(dev), "batch": 16 if dev == 0 else 8,
            "imgsz": config.IMGSZ, "base_model": config.BASE_MODEL}


def box_metrics(yaml_path):
    """Per-class precision, recall, AP50 and mAP50-95 on the test half."""
    metrics = YOLO(config.WEIGHTS).val(
        data=str(yaml_path), split="test", imgsz=config.IMGSZ, device=device(),
        project=str(config.ROOT / "runs"), name="test", exist_ok=True, plots=True,
    )
    box = metrics.box
    per_class = []
    for i, cls in enumerate(box.ap_class_index):
        per_class.append({
            "class": config.CLASS_NAMES[int(cls)],
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "map50_95": float(box.ap[i]),
        })
    shutil.copy(metrics.save_dir / "BoxPR_curve.png", config.ARTIFACTS / "test_BoxPR_curve.png")

    # AP needs every low-confidence box, but a confusion matrix at conf 0.001 is
    # buried in background boxes. Count it again at a confidence a person would use.
    at_025 = YOLO(config.WEIGHTS).val(
        data=str(yaml_path), split="test", imgsz=config.IMGSZ, device=device(), conf=0.25,
        project=str(config.ROOT / "runs"), name="test_conf025", exist_ok=True, plots=True,
    )
    for png in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        shutil.copy(at_025.save_dir / png, config.ARTIFACTS / f"test_{png}")
    matrix = at_025.confusion_matrix.matrix.tolist()
    return {
        "overall": {"precision": float(box.mp), "recall": float(box.mr),
                    "ap50": float(box.map50), "map50_95": float(box.map)},
        "per_class": per_class,
        "box_confusion_matrix": {"labels": config.CLASS_NAMES + ["background"], "conf": 0.25, "matrix": matrix},
    }


def blister_table(split_dir):
    """One row per image: ground truth, which side is shown, and model scores."""
    finetuned = YOLO(config.WEIGHTS)
    coco = YOLO(config.BASE_MODEL)
    rows, coco_seen = [], {}
    for split in ["tune", "test"]:
        for image in sorted((split_dir / split / "images").iterdir()):
            truth = labels_of(split_dir / split / "labels" / f"{image.stem}.txt")
            result = finetuned.predict(str(image), conf=0.001, imgsz=config.IMGSZ, device=device(), verbose=False)[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            empty = np.isin(classes, config.DEFECT_CLASSES)

            # The COCO model has no pocket classes, so it can never flag an
            # empty pocket. Record what it does see, to show that in the report.
            coco_result = coco.predict(str(image), conf=0.25, device=device(), verbose=False)[0]
            for c in coco_result.boxes.cls.cpu().numpy().astype(int):
                name = coco.names[c]
                coco_seen[name] = coco_seen.get(name, 0) + 1

            rows.append({
                "split": split,
                "image": image.name,
                "defective": bool(truth & set(config.DEFECT_CLASSES)),
                "side": "back" if truth & set(config.BACK_CLASSES) else "front",
                "score": float(confs[empty].max()) if empty.any() else 0.0,
                "coco_score": 0.0,
            })
    return pd.DataFrame(rows), coco_seen


def blister_results(test, t_low, t_high, t_single):
    """Blister-level comparison on the test half, at the single-threshold point."""
    def row(name, scores, threshold):
        counts = costing.outcome_counts(scores, test.defective, threshold, threshold)
        tp, fn = counts.at["defective", "reject"], counts.at["defective", "pass"]
        fp, tn = counts.at["good", "reject"], counts.at["good", "pass"]
        low, high = costing.wilson(tp, tp + fn)
        return {
            "model": name, "threshold": threshold,
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "recall": tp / max(tp + fn, 1), "recall_ci95": [low, high],
            "precision": tp / max(tp + fp, 1) if tp + fp else 0.0,
            "accuracy": (tp + tn) / len(test),
            "cost_per_1000": costing.cost_per_1000(counts),
        }

    ones = np.ones(len(test))
    table = [
        row("Finetuned yolo11s", test.score.values, t_single),
        row("COCO yolo11s zero-shot", test.coco_score.values, t_single),
        row("Trivial: predict nothing (pass all)", 0 * ones, 0.5),
        row("Trivial: test majority (reject all)", ones, 0.5),
    ]

    # Recall by side, since back-side defects are rare.
    by_side = {}
    for side, part in test.groupby("side"):
        counts = costing.outcome_counts(part.score, part.defective, t_low, t_high)
        caught = counts.at["defective", "reject"] + counts.at["defective", "review"]
        n = counts.loc["defective"].sum()
        by_side[side] = {
            "images": len(part), "defective": int(n), "caught_by_band": int(caught),
            "recall_band": caught / n if n else None, "recall_ci95": costing.wilson(caught, n),
            "single_threshold_recall": float(
                costing.rates(costing.outcome_counts(part.score, part.defective, t_single, t_single)).at["defective", "reject"]
            ),
        }
    return table, by_side


def save_examples(test, split_dir):
    """A few test images so the dashboard has something to click on."""
    out = config.ARTIFACTS / "examples"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    picks = pd.concat([test[test.defective].head(2), test[~test.defective].head(2)])
    for row in picks.itertuples():
        image = Image.open(split_dir / "test" / "images" / row.image)
        tag = "defective" if row.defective else "good"
        image.save(out / f"{tag}_{row.side}_{row.Index}.jpg", quality=85)


def plot_sweep(sweep, t_single):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, style in [("tune", "-"), ("test", "--")]:
        part = sweep[sweep.split == split]
        ax.plot(part.threshold, part.cost_per_1000, style, label=f"{split} half")
    best = sweep[(sweep.split == "tune") & (sweep.threshold == t_single)].cost_per_1000.iloc[0]
    ax.scatter([t_single], [best], color="red", zorder=3, label=f"chosen t = {t_single:.2f}")
    ax.set_xlabel("Confidence threshold on blister score")
    ax.set_ylabel("Expected cost per 1,000 blisters (EUR)")
    ax.set_yscale("log")
    ax.set_title("Model decides alone: cost against threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.ARTIFACTS / "cost_vs_threshold.png", dpi=150)
    plt.close(fig)


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    source = download()
    yaml_path, split_dir = make_split(source)
    training = finetune(yaml_path)

    boxes = box_metrics(yaml_path)
    blisters, coco_seen = blister_table(split_dir)
    blisters.to_csv(config.ARTIFACTS / "blister_scores.csv", index=False)
    tune, test = blisters[blisters.split == "tune"], blisters[blisters.split == "test"]

    # Thresholds are chosen on the tuning half only, then priced on the test half.
    sweep = pd.concat([
        costing.sweep_single(part.score.values, part.defective.values).assign(split=name)
        for name, part in [("tune", tune), ("test", test)]
    ])
    sweep.to_csv(config.ARTIFACTS / "cost_sweep.csv", index=False)
    tune_sweep = sweep[sweep.split == "tune"]
    t_single = float(tune_sweep.loc[tune_sweep.cost_per_1000.idxmin(), "threshold"])
    t_low, t_high = costing.best_band(tune.score.values, tune.defective.values)
    if config.CHOSEN_T_LOW is not None:
        t_low, t_high = config.CHOSEN_T_LOW, config.CHOSEN_T_HIGH
    plot_sweep(sweep, t_single)
    save_examples(test, split_dir)

    table, by_side = blister_results(test, t_low, t_high, t_single)
    alternatives = costing.alternatives(test.score.values, test.defective.values, t_low, t_high, t_single)
    sensitivity = costing.sensitivity(tune, test)
    sensitivity.to_csv(config.ARTIFACTS / "sensitivity.csv", index=False)

    results = {
        "dataset": {"workspace": config.WORKSPACE, "project": config.PROJECT, "version": config.VERSION,
                    "split_seed": config.SPLIT_SEED,
                    "images": {name: len(part) for name, part in blisters.groupby("split")},
                    "defective": {name: int(part.defective.sum()) for name, part in blisters.groupby("split")}},
        "training": training,
        "box_metrics_test": boxes,
        "coco_zero_shot_detections_test_and_tune": dict(sorted(coco_seen.items(), key=lambda kv: -kv[1])),
        "blister_results_test": table,
        "recall_by_side_test": by_side,
        "operating_point": {"t_single": t_single, "t_low": t_low, "t_high": t_high,
                            "chosen_on": "tune half", "priced_on": "test half"},
        "alternatives_test": alternatives,
        "sensitivity_test": sensitivity.to_dict("records"),
        "assumptions": {k: getattr(config, k) for k in [
            "FN_COMPONENTS", "COST_FN", "COST_FP", "COST_REVIEW", "COST_TP", "COST_TN", "BASE_RATE",
            "LINE_RATE_PER_MIN", "HOURS_PER_DAY", "SHIFT_HOURS", "PRODUCTIVE_HOURS_PER_SHIFT", "REVIEW_SECONDS"]},
    }
    (config.ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["operating_point"], indent=2))
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()
