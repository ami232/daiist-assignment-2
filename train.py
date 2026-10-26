"""
Assignment 2: training pipeline. Run with

    uv run python main.py train

Start to finish, with no manual steps, this:
  1. downloads the dataset version named in config.py from Roboflow Universe,
  2. describes it (images and instances per class per split, box sizes),
  3. finetunes the YOLO checkpoint in config.py and saves weights/best.pt,
  4. evaluates it on the test split, per class, with its confusion matrix,
  5. scores every test image for both the finetuned model and the COCO
     model used zero-shot, plus a predict-nothing baseline,
  6. sweeps the threshold, prices every option, and saves it all to results/.

app.py builds the dashboard from weights/ and results/ alone. Commit both.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

import config as cfg
import decision

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / f"{cfg.PROJECT}-{cfg.VERSION}"
RUNS_DIR = ROOT / "runs"
WEIGHTS = ROOT / "weights" / "best.pt"
RESULTS = ROOT / "results"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}  # ultralytics name -> folder


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def download_dataset() -> Path:
    """Download the dataset once; later runs reuse the local copy."""
    if not (DATA_DIR / "data.yaml").exists():
        from roboflow import Roboflow

        from api_key import load_api_key

        print(f"Downloading {cfg.WORKSPACE}/{cfg.PROJECT}/{cfg.VERSION} ...")
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(cfg.WORKSPACE).project(cfg.PROJECT).version(cfg.VERSION)
        version.download("yolov11", location=str(DATA_DIR), overwrite=True)

    # Roboflow writes relative paths that ultralytics resolves inconsistently,
    # so rewrite data.yaml with an absolute root.
    data = yaml.safe_load((DATA_DIR / "data.yaml").read_text())
    if not (DATA_DIR / "test" / "images").is_dir():
        raise SystemExit("This dataset version has no test split. Pick a version that has one.")
    names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])
    if cfg.CLASS_NAMES:
        if sorted(cfg.CLASS_NAMES) != list(range(len(names))):
            raise SystemExit(
                f"config.CLASS_NAMES has ids {sorted(cfg.CLASS_NAMES)} but the dataset has "
                f"{len(names)} classes {names}. Give one name per class id."
            )
        renamed = [cfg.CLASS_NAMES[i] for i in range(len(names))]
        print(f"Renaming classes {names} -> {renamed} (from config.CLASS_NAMES)")
        names = renamed
    clean = {"path": str(DATA_DIR.resolve()), "names": names, "nc": len(names)}
    clean.update({split: f"{folder}/images" for split, folder in SPLITS.items()})
    yaml_path = DATA_DIR / "data_abs.yaml"
    yaml_path.write_text(yaml.safe_dump(clean, sort_keys=False))
    return yaml_path


def class_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def positive_ids(names: list[str]) -> list[int]:
    ids = [i for i, n in enumerate(names) if any(k in n.lower() for k in cfg.POSITIVE_CLASS_KEYWORDS)]
    if not ids or len(ids) == len(names):
        raise SystemExit(
            f"POSITIVE_CLASS_KEYWORDS {cfg.POSITIVE_CLASS_KEYWORDS} matched {ids} of {names}. "
            "Edit config.py so that some, but not all, classes count as positive."
        )
    return ids


def images_in(split_folder: str) -> list[Path]:
    folder = DATA_DIR / split_folder / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_labels(image: Path) -> list[tuple[int, float]]:
    """(class id, box area as a share of the image) for every object in one image."""
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    if not label.exists():
        return []
    objects = []
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, coords = int(float(parts[0])), [float(v) for v in parts[1:]]
        if len(coords) == 4:  # box: cx cy w h
            area = coords[2] * coords[3]
        else:  # polygon: x1 y1 x2 y2 ... -> its bounding box
            xs, ys = coords[0::2], coords[1::2]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        objects.append((cls, area))
    return objects


def describe_dataset(names: list[str], pos_ids: list[int]) -> dict:
    stats = {"classes": names, "positive_classes": [names[i] for i in pos_ids], "splits": {}}
    for split, folder in SPLITS.items():
        images = images_in(folder)
        per_image = [read_labels(p) for p in images]
        instances = {n: 0 for n in names}
        areas = {n: [] for n in names}
        for objects in per_image:
            for cls, area in objects:
                instances[names[cls]] += 1
                areas[names[cls]].append(area)
        bad_images = sum(any(c in pos_ids for c, _ in objs) for objs in per_image)
        stats["splits"][split] = {
            "images": len(images),
            "images_with_no_labels": sum(len(o) == 0 for o in per_image),
            "images_with_a_bad_weld_or_defect": bad_images,
            "images_with_only_good_welds_or_nothing": len(images) - bad_images,
            "instances_per_class": instances,
            "median_box_area_share": {
                n: round(float(np.median(a)), 4) if a else None for n, a in areas.items()
            },
            "p10_box_area_share": {
                n: round(float(np.percentile(a, 10)), 4) if a else None for n, a in areas.items()
            },
            "p90_box_area_share": {
                n: round(float(np.percentile(a, 90)), 4) if a else None for n, a in areas.items()
            },
        }
    return stats


def print_dataset(stats: dict) -> None:
    print("\nDataset composition")
    rows = []
    for split, s in stats["splits"].items():
        row = {"split": split, "images": s["images"], "bad images": s["images_with_a_bad_weld_or_defect"],
               "good images": s["images_with_only_good_welds_or_nothing"]}
        row.update(s["instances_per_class"])
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    test = stats["splits"]["test"]
    if min(test["images_with_a_bad_weld_or_defect"], test["images_with_only_good_welds_or_nothing"]) < 15:
        print(
            "\nWARNING: the test split has fewer than 15 images on one side of the good/bad "
            "decision. Any cost figure built on it is fragile. Say so in REPORT.md."
        )


# ----------------------------------------------------------------------------
# 2. Training
# ----------------------------------------------------------------------------
def pick_device():
    if os.environ.get("TRAIN_DEVICE"):
        return os.environ["TRAIN_DEVICE"]
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(data_yaml: Path, device) -> Path:
    model = YOLO(cfg.BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        imgsz=cfg.IMG_SIZE,
        batch=cfg.BATCH,
        seed=cfg.SEED,
        deterministic=True,
        device=device,
        project=str(RUNS_DIR),
        name="finetune",
        exist_ok=True,
        plots=True,
    )
    best = RUNS_DIR / "finetune" / "weights" / "best.pt"
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS)
    print(f"\nSaved {WEIGHTS.relative_to(ROOT)}")
    return WEIGHTS


# ----------------------------------------------------------------------------
# 3. Evaluation
# ----------------------------------------------------------------------------
def evaluate_detection(weights: Path, data_yaml: Path, names: list[str], device) -> dict:
    """Overall and per-class detection metrics on the test split, plus the confusion matrix."""
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), split="test", imgsz=cfg.IMG_SIZE, batch=cfg.BATCH, device=device,
        project=str(RUNS_DIR), name="test_eval", exist_ok=True, plots=True, verbose=False,
    )
    box = metrics.box
    per_class = []
    for k, cls in enumerate(metrics.ap_class_index):
        per_class.append({
            "class": names[int(cls)],
            "precision": round(float(box.p[k]), 4),
            "recall": round(float(box.r[k]), 4),
            "AP50": round(float(box.ap50[k]), 4),
            "AP50-95": round(float(box.ap[k]), 4),
        })

    # mAP needs every box down to conf 0.001, but a confusion matrix at that
    # level counts thousands of junk boxes. Build it in a second pass at a
    # working confidence instead.
    at_conf = YOLO(str(weights)).val(
        data=str(data_yaml), split="test", imgsz=cfg.IMG_SIZE, batch=cfg.BATCH, device=device,
        conf=cfg.CONFUSION_MATRIX_CONF, project=str(RUNS_DIR), name="test_confusion",
        exist_ok=True, plots=True, verbose=False,
    )
    for png in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        if (Path(at_conf.save_dir) / png).exists():
            shutil.copy2(Path(at_conf.save_dir) / png, RESULTS / png)
    matrix = at_conf.confusion_matrix.matrix  # rows = predicted, columns = true, last = background
    return {
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": names + ["background"],
            "confidence": cfg.CONFUSION_MATRIX_CONF,
            "rows_are": "predicted",
            "columns_are": "true",
            "matrix": np.asarray(matrix).astype(int).tolist(),
        },
    }


def image_scores(weights: str, images: list[Path], class_ids: list[int] | None, device) -> np.ndarray:
    """Highest confidence for any of class_ids in each image (any class if None); 0 if none."""
    model = YOLO(weights)
    scores = []
    for result in model.predict(
        source=[str(p) for p in images], conf=0.001, imgsz=cfg.IMG_SIZE,
        device=device, stream=True, verbose=False,
    ):
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        keep = np.ones_like(cls, bool) if class_ids is None else np.isin(cls, class_ids)
        scores.append(float(conf[keep].max()) if keep.any() else 0.0)
    return np.array(scores)


def decision_summary(name: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    t = decision.best_threshold(scores, labels)
    (p_pass, _, p_fail), (_, _, n_fail) = decision.outcome_rates(scores, labels, t, t)
    return {
        "model": name,
        "ROC AUC (bad vs good image)": round(decision.roc_auc(scores, labels), 4),
        "best threshold": t,
        "cost per 1,000 at best threshold (€)": round(1000 * decision.cost_per_case(scores, labels, t, t), 2),
        "recall on bad welds": round(p_fail, 4),
        "false-alarm rate on good welds": round(n_fail, 4),
    }


def sensitivity(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """How the best model-alone threshold moves if the base rate or the cost of a miss is wrong."""
    rows = []
    for base_rate in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
            costs = cfg.costs() | {"fn": cfg.COST_FN * multiplier}
            t = decision.best_threshold(scores, labels, base_rate, costs)
            rows.append({
                "base_rate": base_rate,
                "cost_fn": cfg.COST_FN * multiplier,
                "best_threshold": t,
                "cost_per_1000": round(1000 * decision.cost_per_case(scores, labels, t, t, base_rate, costs), 2),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def main() -> None:
    started = time.time()
    RESULTS.mkdir(exist_ok=True)
    device = pick_device()
    print(f"Training device: {device}")

    data_yaml = download_dataset()
    names = class_names(data_yaml)
    pos_ids = positive_ids(names)
    print(f"Classes: {names}. Treated as positive (needs action): {[names[i] for i in pos_ids]}")

    stats = describe_dataset(names, pos_ids)
    print_dataset(stats)
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    weights = train(data_yaml, device)
    detection = evaluate_detection(weights, data_yaml, names, device)

    # Image-level decision on the test split.
    test_images = images_in("test")
    labels = np.array([int(any(c in pos_ids for c, _ in read_labels(p))) for p in test_images])
    finetuned = image_scores(str(weights), test_images, pos_ids, device)
    coco = image_scores(cfg.COCO_BASELINE, test_images, None, device)  # COCO has no weld class: any box counts
    nothing = np.zeros(len(test_images))  # predict nothing: every weld passes
    pd.DataFrame({
        "image": [p.name for p in test_images], "is_bad": labels,
        "score_finetuned": finetuned, "score_coco_zero_shot": coco,
    }).to_csv(RESULTS / "test_scores.csv", index=False)

    baselines = [
        {"model": f"Finetuned {cfg.BASE_MODEL}", **detection["overall"]},
        {"model": "COCO zero-shot", "precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0,
         "note": "COCO has no weld or defect class, so it cannot score on these classes"},
        {"model": "Predict nothing", "precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0},
    ]
    image_level = [
        decision_summary(f"Finetuned {cfg.BASE_MODEL}", finetuned, labels),
        decision_summary("COCO zero-shot (any box = defect)", coco, labels),
        {
            "model": "Predict nothing (pass all)",
            "ROC AUC (bad vs good image)": 0.5,
            "best threshold": None,
            "cost per 1,000 at best threshold (€)": round(1000 * decision.cost_per_case(nothing, labels, 1.01, 1.01), 2),
            "recall on bad welds": 0.0,
            "false-alarm rate on good welds": 0.0,
        },
    ]

    # Costing and the operating point.
    decision.sweep(finetuned, labels).to_csv(RESULTS / "cost_sweep.csv", index=False)
    sensitivity(finetuned, labels).to_csv(RESULTS / "sensitivity.csv", index=False)
    point = decision.operating_point(finetuned, labels)
    alternatives = decision.alternatives(finetuned, labels, point["threshold"], point["band"])

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"workspace": cfg.WORKSPACE, "project": cfg.PROJECT, "version": cfg.VERSION},
        "training": {
            "base_model": cfg.BASE_MODEL, "epochs_max": cfg.EPOCHS, "patience": cfg.PATIENCE,
            "imgsz": cfg.IMG_SIZE, "batch": cfg.BATCH, "seed": cfg.SEED, "device": str(device),
            "minutes": round((time.time() - started) / 60, 1),
        },
        "test_images": int(len(labels)), "test_bad_images": int(labels.sum()),
        "detection": detection, "baselines_detection": baselines, "baselines_image_level": image_level,
        "operating_point": point, "alternatives": alternatives.to_dict(orient="records"),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\nPer-class results on the test split")
    print(pd.DataFrame(detection["per_class"]).to_string(index=False))
    print("\nBaselines (image-level decision)")
    print(pd.DataFrame(image_level).to_string(index=False))
    print(f"\nOperating point ({point['chosen_by']}): threshold {point['threshold']:.2f}, band {point['band']}")
    print(alternatives.to_string(index=False))
    print(f"\nDone in {summary['training']['minutes']} min. Commit weights/ and results/.")


if __name__ == "__main__":
    main()
