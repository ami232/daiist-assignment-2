from __future__ import annotations

import json
import torch
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
DATASET_DIR = DATASETS_DIR / "valorant-object-detection2-v7"
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"

WORKSPACE = "kwan-li-jqief"
PROJECT = "valorant-object-detection2"
VERSION = 7
EXPORT_FORMAT = "yolov11"
MODEL_NAME = "yolo11n.pt"
EPOCHS = int(os.environ.get("A2_EPOCHS", "50"))
IMGSZ = int(os.environ.get("A2_IMGSZ", "640"))
BATCH = int(os.environ.get("A2_BATCH", "16"))
SEED = 42
IOU_MATCH = 0.50

# Proposed business assumptions. Keep these only if you personally agree and
# can defend them in the written comprehension check.
ANALYST_EUR_PER_HOUR = 48.0
REVIEW_SECONDS = 15.0
FALSE_POSITIVE_SECONDS = 15.0
FALSE_NEGATIVE_RECOVERY_SECONDS = 120.0
CASES_PER_DAY = 2000
REVIEW_COST_EUR = ANALYST_EUR_PER_HOUR / 3600 * REVIEW_SECONDS
FALSE_POSITIVE_COST_EUR = ANALYST_EUR_PER_HOUR / 3600 * FALSE_POSITIVE_SECONDS
FALSE_NEGATIVE_COST_EUR = ANALYST_EUR_PER_HOUR / 3600 * FALSE_NEGATIVE_RECOVERY_SECONDS


def ensure_dirs() -> None:
    for path in (DATASETS_DIR, WEIGHTS_DIR, ARTIFACTS_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    api_key = load_api_key()
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset = version.download(EXPORT_FORMAT, location=str(DATASET_DIR), overwrite=True)
    location = Path(getattr(dataset, "location", DATASET_DIR))
    yaml_candidates = list(location.rglob("data.yaml"))
    if not yaml_candidates:
        raise FileNotFoundError(f"No data.yaml found under {location}")
    return yaml_candidates[0]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_names_from_yaml(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def resolve_split_dir(data_yaml: Path, split_value: str | list[str]) -> Path:
    if isinstance(split_value, list):
        split_value = split_value[0]

    p = Path(split_value)

    if p.is_absolute():
        return p

    # Normal relative path
    candidate = (data_yaml.parent / p).resolve()
    if candidate.exists():
        return candidate

    # Roboflow exports can use ../train/images even though the folders
    # are actually inside the dataset root.
    cleaned_parts = [part for part in p.parts if part not in ("..", ".")]
    candidate = data_yaml.parent.joinpath(*cleaned_parts).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve split path '{split_value}' from {data_yaml}"
    )


def label_dir_for_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def collect_dataset_stats(data_yaml: Path) -> dict[str, Any]:
    data = load_yaml(data_yaml)
    names = class_names_from_yaml(data)
    stats: dict[str, Any] = {
        "workspace": WORKSPACE,
        "project": PROJECT,
        "version": VERSION,
        "classes": names,
        "splits": {},
        "instances_per_class": {name: 0 for name in names.values()},
        "box_area_ratio": {},
    }
    areas_by_class: dict[str, list[float]] = {name: [] for name in names.values()}

    for split_key in ("train", "val", "test"):
        value = data.get(split_key)
        if not value:
            continue
        images_dir = resolve_split_dir(data_yaml, value)
        labels_dir = label_dir_for_images(images_dir)
        image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        split_counts = {name: 0 for name in names.values()}
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                bits = line.strip().split()
                if len(bits) < 5:
                    continue
                class_id = int(float(bits[0]))
                w, h = float(bits[3]), float(bits[4])
                class_name = names.get(class_id, str(class_id))
                split_counts[class_name] = split_counts.get(class_name, 0) + 1
                stats["instances_per_class"][class_name] = stats["instances_per_class"].get(class_name, 0) + 1
                areas_by_class.setdefault(class_name, []).append(w * h)
        stats["splits"][split_key] = {"images": len(image_files), "instances_per_class": split_counts}

    for class_name, values in areas_by_class.items():
        if values:
            arr = np.asarray(values, dtype=float)
            stats["box_area_ratio"][class_name] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
        else:
            stats["box_area_ratio"][class_name] = {"median": None, "mean": None, "p25": None, "p75": None}

    (ARTIFACTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def train_model(data_yaml: Path) -> Path:
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0 if torch.cuda.is_available() else "cpu",
        seed=SEED,
        project=str(RUNS_DIR),
        name="valorant_assignment2",
        exist_ok=True,
        plots=True,
    )
    best = Path(model.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"Training completed but best weights were not found at {best}")
    destination = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best, destination)
    return destination


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ultralytics_metrics(model_path: Path, data_yaml: Path) -> dict[str, Any]:
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, plots=True, project=str(RUNS_DIR), name="test_evaluation", exist_ok=True)
    names = {int(k): str(v) for k, v in dict(model.names).items()}
    box = metrics.box
    p = np.asarray(getattr(box, "p", []), dtype=float)
    r = np.asarray(getattr(box, "r", []), dtype=float)
    ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    maps = np.asarray(getattr(box, "maps", []), dtype=float)
    class_ids = np.asarray(getattr(box, "ap_class_index", np.arange(len(p))), dtype=int)
    per_class = []
    for i, cid in enumerate(class_ids.tolist()):
        per_class.append({
            "class_id": int(cid),
            "class_name": names.get(int(cid), str(cid)),
            "precision": safe_float(p[i]) if i < len(p) else None,
            "recall": safe_float(r[i]) if i < len(r) else None,
            "ap50": safe_float(ap50[i]) if i < len(ap50) else None,
            "map50_95": safe_float(maps[i]) if i < len(maps) else None,
        })
    output = {
        "model": "finetuned_yolo11n",
        "mean_precision": safe_float(getattr(box, "mp", None)),
        "mean_recall": safe_float(getattr(box, "mr", None)),
        "map50": safe_float(getattr(box, "map50", None)),
        "map50_95": safe_float(getattr(box, "map", None)),
        "per_class": per_class,
    }
    (ARTIFACTS_DIR / "finetuned_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(per_class).to_csv(ARTIFACTS_DIR / "per_class_metrics.csv", index=False)
    return output


def xywhn_to_xyxy(row: list[float], width: int, height: int) -> np.ndarray:
    _, xc, yc, w, h = row
    return np.array([(xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height], dtype=float)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(data_yaml: Path, split: str = "test"):
    data = load_yaml(data_yaml)
    names = class_names_from_yaml(data)
    images_dir = resolve_split_dir(data_yaml, data[split])
    labels_dir = label_dir_for_images(images_dir)
    image_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    gt = {}
    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rows = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                bits = line.strip().split()
                if len(bits) < 5:
                    continue
                vals = [float(x) for x in bits[:5]]
                cid = int(vals[0])
                rows.append({"class_id": cid, "class_name": names.get(cid, str(cid)), "box": xywhn_to_xyxy(vals, width, height)})
        gt[image_path] = rows
    return image_files, gt, names


def predict_all(model: YOLO, image_files: list[Path], conf: float = 0.001):
    predictions = {}
    results = model.predict(source=[str(p) for p in image_files], conf=conf, imgsz=IMGSZ, verbose=False, stream=False)
    for image_path, result in zip(image_files, results):
        rows = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, score, cid in zip(boxes, scores, classes):
                rows.append({"class_id": int(cid), "score": float(score), "box": np.asarray(box, dtype=float)})
        predictions[image_path] = rows
    return predictions


def class_detection_counts(gt_rows, pred_rows, class_id: int, threshold: float):
    gt_boxes = [r["box"] for r in gt_rows if r["class_id"] == class_id]
    preds = sorted([r for r in pred_rows if r["class_id"] == class_id and r["score"] >= threshold], key=lambda r: r["score"], reverse=True)
    matched_gt = set()
    tp = fp = 0
    for pred in preds:
        best_iou, best_idx = 0.0, None
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            score = iou_xyxy(pred["box"], gt_box)
            if score > best_iou:
                best_iou, best_idx = score, idx
        if best_idx is not None and best_iou >= IOU_MATCH:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def image_level_scores(image_files, gt, preds, positive_class_id: int) -> pd.DataFrame:
    rows = []
    for image_path in image_files:
        actual = int(any(r["class_id"] == positive_class_id for r in gt.get(image_path, [])))
        scores = [r["score"] for r in preds.get(image_path, []) if r["class_id"] == positive_class_id]
        rows.append({"image": image_path.name, "actual": actual, "score": max(scores) if scores else 0.0})
    return pd.DataFrame(rows)


def confusion_from_scores(df: pd.DataFrame, threshold: float) -> dict[str, int]:
    pred = (df["score"].to_numpy() >= threshold).astype(int)
    actual = df["actual"].to_numpy().astype(int)
    return {
        "tp": int(np.sum((pred == 1) & (actual == 1))),
        "fp": int(np.sum((pred == 1) & (actual == 0))),
        "tn": int(np.sum((pred == 0) & (actual == 0))),
        "fn": int(np.sum((pred == 0) & (actual == 1))),
    }


def model_only_cost_sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        cm = confusion_from_scores(df, float(threshold))
        total_cost = cm["fp"] * FALSE_POSITIVE_COST_EUR + cm["fn"] * FALSE_NEGATIVE_COST_EUR
        rows.append({"threshold": float(threshold), **cm, "total_cost_eur": float(total_cost), "cost_per_1000_eur": float(total_cost / max(len(df), 1) * 1000)})
    return pd.DataFrame(rows)


def assisted_metrics(df: pd.DataFrame, low: float, high: float) -> dict[str, Any]:
    actual = df["actual"].to_numpy().astype(int)
    score = df["score"].to_numpy()
    auto_negative = score < low
    auto_positive = score >= high
    review = (~auto_negative) & (~auto_positive)
    fn = int(np.sum(auto_negative & (actual == 1)))
    fp = int(np.sum(auto_positive & (actual == 0)))
    tp_auto = int(np.sum(auto_positive & (actual == 1)))
    tn_auto = int(np.sum(auto_negative & (actual == 0)))
    review_pos = int(np.sum(review & (actual == 1)))
    review_neg = int(np.sum(review & (actual == 0)))
    tp, tn = tp_auto + review_pos, tn_auto + review_neg
    review_count = int(np.sum(review))
    total_cost = fp * FALSE_POSITIVE_COST_EUR + fn * FALSE_NEGATIVE_COST_EUR + review_count * REVIEW_COST_EUR
    n = max(len(df), 1)
    review_share = review_count / n
    daily_reviews = review_share * CASES_PER_DAY
    daily_review_hours = daily_reviews * REVIEW_SECONDS / 3600
    return {
        "low_threshold": float(low), "high_threshold": float(high),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "review_count": review_count, "review_share": float(review_share),
        "total_cost_eur": float(total_cost), "cost_per_1000_eur": float(total_cost / n * 1000),
        "daily_reviews": float(daily_reviews), "daily_review_hours": float(daily_review_hours),
        "reviewer_fte_8h": float(daily_review_hours / 8.0),
    }


def optimize_review_band(df: pd.DataFrame) -> dict[str, Any]:
    candidates = []
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for low in grid:
        for high in grid:
            if low < high:
                candidates.append(assisted_metrics(df, float(low), float(high)))
    candidates.sort(key=lambda x: (x["total_cost_eur"], x["review_share"]))
    return candidates[0]


def per_class_counts_at_threshold(gt, preds, names, threshold: float = 0.50):
    rows = []
    for cid, name in names.items():
        tp = fp = fn = 0
        for image_path, gt_rows in gt.items():
            a, b, c = class_detection_counts(gt_rows, preds.get(image_path, []), cid, threshold)
            tp += a; fp += b; fn += c
        rows.append({
            "class_id": cid, "class_name": name, "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        })
    return rows


def coco_zero_shot_baseline(image_files, gt, names):
    coco = YOLO(MODEL_NAME)
    raw = predict_all(coco, image_files, conf=0.001)
    person_class = next((cid for cid, name in coco.names.items() if str(name) == "person"), 0)
    body_class = next((cid for cid, name in names.items() if name.lower() == "enemybody"), min(names))
    mapped = {}
    for image_path, rows in raw.items():
        mapped[image_path] = [{"class_id": body_class, "score": r["score"], "box": r["box"]} for r in rows if r["class_id"] == int(person_class)]
    per_class = per_class_counts_at_threshold(gt, mapped, names, threshold=0.50)
    summary = {"model": "pretrained_coco_yolo11n_zero_shot", "mapping": "COCO person -> enemyBody; no COCO equivalent for enemyHead", "per_class_at_0_50": per_class}
    (ARTIFACTS_DIR / "coco_zero_shot.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def trivial_baseline(image_files, gt, names):
    per_class = []
    for cid, name in names.items():
        positives = sum(1 for image_path in image_files if any(r["class_id"] == cid for r in gt.get(image_path, [])))
        per_class.append({"class_id": cid, "class_name": name, "strategy": "predict nothing", "precision": 0.0, "recall": 0.0, "positive_images": positives})
    output = {"model": "trivial_predict_nothing", "per_class": per_class}
    (ARTIFACTS_DIR / "trivial_baseline.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    ensure_dirs()
    print("1/8 Downloading exact Roboflow dataset...")
    data_yaml = download_dataset()
    print("2/8 Describing dataset...")
    stats = collect_dataset_stats(data_yaml)
    print(json.dumps(stats["splits"], indent=2))
    print("3/8 Finetuning YOLO11n...")
    best_weights = train_model(data_yaml)
    print(f"Saved best weights to {best_weights}")
    print("4/8 Running held-out test evaluation...")
    fine_metrics = ultralytics_metrics(best_weights, data_yaml)
    print("5/8 Building threshold-level predictions...")
    image_files, gt, names = load_ground_truth(data_yaml, "test")
    preds = predict_all(YOLO(str(best_weights)), image_files, conf=0.001)
    positive_class_id = next((cid for cid, name in names.items() if name.lower() == "enemybody"), min(names))
    positive_class_name = names[positive_class_id]
    score_df = image_level_scores(image_files, gt, preds, positive_class_id)
    score_df.to_csv(ARTIFACTS_DIR / "test_image_scores.csv", index=False)
    print("6/8 Comparing baselines...")
    coco_baseline = coco_zero_shot_baseline(image_files, gt, names)
    trivial = trivial_baseline(image_files, gt, names)
    print("7/8 Sweeping confidence thresholds and optimizing human review...")
    sweep = model_only_cost_sweep(score_df)
    sweep.to_csv(ARTIFACTS_DIR / "cost_sweep.csv", index=False)
    best_single_row = sweep.loc[sweep["total_cost_eur"].idxmin()].to_dict()
    best_review = optimize_review_band(score_df)
    class_counts = per_class_counts_at_threshold(gt, preds, names, threshold=0.50)
    pd.DataFrame(class_counts).to_csv(ARTIFACTS_DIR / "per_class_counts_at_0_50.csv", index=False)
    base_rate = float(score_df["actual"].mean()) if len(score_df) else 0.0
    print("8/8 Saving dashboard/report artifacts...")
    operating = {
        "business_framing": {
            "decision": "Flag Valorant VOD frames that contain a visible enemyBody so a coach/analyst can focus review on relevant combat situations.",
            "positive_class": positive_class_name,
            "base_rate_proxy": base_rate,
            "base_rate_note": "Observed enemyBody image prevalence in the held-out test set; used as a proxy because no production base-rate dataset is available.",
        },
        "cost_assumptions": {
            "analyst_eur_per_hour": ANALYST_EUR_PER_HOUR,
            "false_positive_cost_eur": FALSE_POSITIVE_COST_EUR,
            "false_positive_logic": f"{FALSE_POSITIVE_SECONDS:g} seconds of unnecessary analyst review at €{ANALYST_EUR_PER_HOUR:.0f}/hour.",
            "false_negative_cost_eur": FALSE_NEGATIVE_COST_EUR,
            "false_negative_logic": f"{FALSE_NEGATIVE_RECOVERY_SECONDS:g} seconds of manual VOD recovery/search at €{ANALYST_EUR_PER_HOUR:.0f}/hour.",
            "human_review_cost_eur": REVIEW_COST_EUR,
            "human_review_logic": f"{REVIEW_SECONDS:g} seconds of analyst review at €{ANALYST_EUR_PER_HOUR:.0f}/hour.",
            "cases_per_day": CASES_PER_DAY,
            "IMPORTANT": "These are proposed assumptions. Keep them only if you personally agree and can defend them in the written comprehension check.",
        },
        "best_model_only_threshold": best_single_row,
        "best_assisted_review_band": best_review,
        "finetuned_metrics": fine_metrics,
        "zero_shot_summary": coco_baseline,
        "trivial_baseline": trivial,
    }
    (ARTIFACTS_DIR / "operating_point.json").write_text(json.dumps(operating, indent=2), encoding="utf-8")
    print("\nTraining pipeline complete.")
    print(f"Positive class: {positive_class_name}")
    print(f"Observed test base rate: {base_rate:.3f}")
    print(f"Best model-only threshold: {best_single_row['threshold']:.2f}")
    print(f"Best assisted band: {best_review['low_threshold']:.2f} - {best_review['high_threshold']:.2f}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Weights: {best_weights}")


if __name__ == "__main__":
    main()
