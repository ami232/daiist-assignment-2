"""
Assignment 2: training pipeline for out-of-stock (empty shelf gap) detection.

    uv run python main.py train

1. Download tossoengineering/retail-shelf-detection-peimi v1 from Roboflow.
2. Describe it: instances per class, split sizes, box size relative to image.
3. Finetune yolo11n from the COCO checkpoint.
4. Evaluate the finetuned model, COCO yolo11n zero-shot and a predict-nothing
   baseline on the test split with one evaluator, so the numbers compare.
5. Score every photo (max "empty" confidence), sweep the threshold, pick the
   single threshold and the review band on the validation split, report them
   on the test split.
6. Write everything app.py needs to artifacts/ and the weights to weights/.

EPOCHS / IMGSZ / DEVICE environment variables override the defaults below.
Trained on a laptop CPU: ~3 min per epoch, ~1.5 h for the default 30 epochs.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key
from costs import (
    BASE_RATE, COSTS, OPERATIONS, POSITIVE_CLASS, alternatives, expected_cost,
    outcome_rates, reviewers_needed,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "retail-shelf"
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

WORKSPACE, PROJECT, VERSION = "tossoengineering", "retail-shelf-detection-peimi", 1
BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
DEVICE = os.environ.get("DEVICE", "cpu")
SEED = 0

# COCO classes a zero-shot model could plausibly fire on for goods on a shelf.
# COCO has nothing that means "empty space", so zero-shot can never find gaps.
COCO_AS_PRODUCT = {"bottle", "cup", "bowl", "vase", "book", "wine glass", "banana",
                   "apple", "orange", "sandwich", "broccoli", "carrot", "cake",
                   "donut", "pizza", "hot dog", "toothbrush", "scissors", "teddy bear"}

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
REPORT_CONF = 0.25  # confidence at which per-class precision/recall are quoted
SPLITS = {"train": "train", "val": "valid", "test": "test"}


# --------------------------------------------------------------------------- data

def download_dataset() -> Path:
    if not (DATA_DIR / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download("yolov11", location=str(DATA_DIR), overwrite=True)
    with open(DATA_DIR / "data.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = meta["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    # Roboflow's relative paths depend on the working directory; pin them.
    local = {"path": str(DATA_DIR), "names": dict(enumerate(names)),
             **{split: f"{folder}/images" for split, folder in SPLITS.items()}}
    local_yaml = DATA_DIR / "data_local.yaml"
    with open(local_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(local, f)
    return local_yaml


def class_names(data_yaml: Path) -> list[str]:
    with open(data_yaml, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    return [names[k] for k in sorted(names)]


def split_images(split: str) -> list[Path]:
    folder = DATA_DIR / SPLITS[split] / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def read_labels(image: Path) -> np.ndarray:
    """Ground truth as rows of (class, x1, y1, x2, y2), normalised to [0, 1]."""
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    rows = []
    if label.exists():
        for line in label.read_text().split("\n"):
            values = [float(v) for v in line.split()]
            if len(values) == 5:
                c, cx, cy, w, h = values
                rows.append([c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            elif len(values) > 5:  # polygon: use its bounding box
                xs, ys = values[1::2], values[2::2]
                rows.append([values[0], min(xs), min(ys), max(xs), max(ys)])
    return np.array(rows).reshape(-1, 5)


def describe_dataset(names: list[str]) -> dict:
    out = {"splits": {}, "per_class": {}}
    records = []
    for split in SPLITS:
        images = split_images(split)
        with_empty = 0
        for image in images:
            gt = read_labels(image)
            with_empty += int((gt[:, 0] == names.index(POSITIVE_CLASS)).any())
            for c, x1, y1, x2, y2 in gt:
                records.append({"split": split, "class": names[int(c)],
                                "area": (x2 - x1) * (y2 - y1)})
        out["splits"][split] = {"images": len(images), "images_with_gap": with_empty}
    df = pd.DataFrame(records)
    for name in names:
        sub = df[df["class"] == name]
        out["per_class"][name] = {
            **{f"{s}_instances": int((sub["split"] == s).sum()) for s in SPLITS},
            "median_area_pct": round(100 * float(sub["area"].median()), 2) if len(sub) else None,
            "p10_area_pct": round(100 * float(sub["area"].quantile(0.1)), 2) if len(sub) else None,
            "p90_area_pct": round(100 * float(sub["area"].quantile(0.9)), 2) if len(sub) else None,
        }
    return out


# ----------------------------------------------------------------------- training

def finetune(data_yaml: Path) -> Path:
    model = YOLO(BASE_MODEL)
    result = model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, device=DEVICE,
                         batch=16, patience=15, seed=SEED, deterministic=True, workers=0,
                         project=str(ROOT / "runs"), name="finetune", exist_ok=True,
                         plots=True, verbose=False)
    best = Path(result.save_dir) / "weights" / "best.pt"
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    return Path(result.save_dir)


# --------------------------------------------------------------------- evaluation

def predict(model: YOLO, images: list[Path], class_map: dict[int, int]) -> list[np.ndarray]:
    """Per image, rows of (class, conf, x1, y1, x2, y2) in our class ids."""
    preds = []
    for result in model.predict([str(p) for p in images], conf=0.001, iou=0.7, imgsz=IMGSZ,
                                device=DEVICE, verbose=False, stream=True):
        boxes = result.boxes
        rows = []
        for c, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxyn.tolist()):
            if int(c) in class_map:
                rows.append([class_map[int(c)], conf, *xyxy])
        preds.append(np.array(rows).reshape(-1, 6))
    return preds


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def match(preds: list[np.ndarray], gts: list[np.ndarray], cls: int, thr: float):
    """Greedy COCO-style matching. Returns (confidences, is_tp, n_gt)."""
    confs, tps, n_gt = [], [], 0
    for pred, gt in zip(preds, gts):
        p = pred[pred[:, 0] == cls]
        g = gt[gt[:, 0] == cls][:, 1:]
        n_gt += len(g)
        used = np.zeros(len(g), dtype=bool)
        for row in p[np.argsort(-p[:, 1])]:
            hit = False
            if len(g):
                ious = iou(row[2:], g)
                ious[used] = -1
                j = int(np.argmax(ious))
                if ious[j] >= thr:
                    used[j] = hit = True
            confs.append(row[1]); tps.append(hit)
    return np.array(confs), np.array(tps, dtype=bool), n_gt


def average_precision(confs, tps, n_gt) -> float:
    """101-point interpolated AP, as COCO computes it."""
    if n_gt == 0 or len(confs) == 0:
        return 0.0
    order = np.argsort(-confs)
    tp = np.cumsum(tps[order]); fp = np.cumsum(~tps[order])
    recall = tp / n_gt
    precision = np.maximum.accumulate((tp / (tp + fp))[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def evaluate(preds: list[np.ndarray], gts: list[np.ndarray], names: list[str]) -> dict:
    per_class = {}
    for cls, name in enumerate(names):
        aps = [average_precision(*match(preds, gts, cls, t)) for t in IOU_THRESHOLDS]
        confs, tps, n_gt = match(preds, gts, cls, 0.5)
        keep = confs >= REPORT_CONF
        tp = int(tps[keep].sum()); fp = int((~tps[keep]).sum())
        per_class[name] = {
            "instances": n_gt, "ap50": aps[0], "map50_95": float(np.mean(aps)),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / n_gt if n_gt else 0.0, "tp": tp, "fp": fp, "fn": n_gt - tp,
        }
    return {
        "per_class": per_class,
        "ap50": float(np.mean([v["ap50"] for v in per_class.values()])),
        "map50_95": float(np.mean([v["map50_95"] for v in per_class.values()])),
    }


def image_scores(preds: list[np.ndarray], gts: list[np.ndarray], images: list[Path],
                 pos: int) -> pd.DataFrame:
    return pd.DataFrame({
        "image": [p.name for p in images],
        "has_gap": [int((g[:, 0] == pos).any()) for g in gts],
        "score": [float(p[p[:, 0] == pos][:, 1].max()) if (p[:, 0] == pos).any() else 0.0
                  for p in preds],
    })


def image_level(df: pd.DataFrame, threshold: float) -> dict:
    r = outcome_rates(df["has_gap"], df["score"], threshold, threshold)
    pos, neg = int(df["has_gap"].sum()), int((1 - df["has_gap"]).sum())
    tp, fp, tn = r["tp"] * pos, r["fp"] * neg, r["tn"] * neg
    return {"threshold": threshold, "photos": pos + neg, "photos_with_gap": pos,
            "recall": r["tp"], "false_alarm_rate": r["fp"],
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "accuracy": (tp + tn) / max(pos + neg, 1), **expected_cost(r)}


# ------------------------------------------------------------------ cost sweeps

GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def single_threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in GRID:
        r = outcome_rates(df["has_gap"], df["score"], t, t)
        rows.append({"threshold": t, **r, **expected_cost(r)})
    return pd.DataFrame(rows)


def band_sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for low in GRID[::2]:
        for high in GRID[::2]:
            if high >= low:
                r = outcome_rates(df["has_gap"], df["score"], low, high)
                rows.append({"low": low, "high": high, **expected_cost(r)})
    return pd.DataFrame(rows)


def cheapest_threshold(sweep: pd.DataFrame) -> float:
    """Middle of the range of thresholds that tie for the lowest cost.

    With a small validation split many thresholds tie; the middle one keeps the
    largest margin to the photos on either side of it.
    """
    tied = sweep.loc[np.isclose(sweep["cost_per_1000"], sweep["cost_per_1000"].min()), "threshold"]
    return float(np.round((tied.min() + tied.max()) / 2, 2))


def cheapest_band(bands: pd.DataFrame) -> pd.Series:
    """Widest review band among those that tie for the lowest cost.

    Photos scoring between the lowest-scoring gap and the highest-scoring full
    fridge seen in validation are the ones the model has no evidence about, so
    those are the ones a person should look at.
    """
    tied = bands[np.isclose(bands["cost_per_1000"], bands["cost_per_1000"].min())]
    return tied.assign(width=tied["high"] - tied["low"]).sort_values(
        ["width", "low"], ascending=[False, True], kind="stable").iloc[0]


def bootstrap_cost(df: pd.DataFrame, low: float, high: float, n: int = 2000) -> dict:
    """5th-95th percentile of cost per 1,000 when the test photos are resampled.

    The test split is small, so this says how much of the headline cost is noise.
    """
    rng = np.random.default_rng(SEED)
    y, s = df["has_gap"].to_numpy(), df["score"].to_numpy()
    costs = []
    for _ in range(n):
        i = rng.integers(0, len(df), len(df))
        if 0 < y[i].sum() < len(i):  # need both classes to have rates
            costs.append(expected_cost(outcome_rates(y[i], s[i], low, high))["cost_per_1000"])
    return {"p5": float(np.percentile(costs, 5)), "p95": float(np.percentile(costs, 95))}


# --------------------------------------------------------------------------- main

def main() -> None:
    data_yaml = download_dataset()
    run_dir = finetune(data_yaml)
    evaluate_and_cost(data_yaml, run_dir)


def evaluate_and_cost(data_yaml: Path, run_dir: Path = ROOT / "runs" / "finetune") -> None:
    """Everything after training. Re-runnable on its own when only costs.py changes:

        uv run python -c "import train; train.evaluate_and_cost(train.download_dataset())"
    """
    ARTIFACTS.mkdir(exist_ok=True)
    names = class_names(data_yaml)
    pos = names.index(POSITIVE_CLASS)
    dataset = describe_dataset(names)
    print(json.dumps(dataset, indent=2))
    finetuned = YOLO(str(WEIGHTS))

    # Ultralytics' own object-level confusion matrix on the test split.
    val = finetuned.val(data=str(data_yaml), split="test", imgsz=IMGSZ, device=DEVICE,
                        conf=REPORT_CONF, project=str(ROOT / "runs"), name="test",
                        exist_ok=True, plots=True, verbose=False)
    matrix = val.confusion_matrix.matrix.astype(int).tolist()
    for png in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        if (Path(val.save_dir) / png).exists():
            shutil.copy(Path(val.save_dir) / png, ARTIFACTS / png)
    if (run_dir / "results.png").exists():
        shutil.copy(run_dir / "results.png", ARTIFACTS / "training_curves.png")

    coco = YOLO(BASE_MODEL)
    coco_map = {i: names.index("product") for i, n in coco.names.items() if n in COCO_AS_PRODUCT}
    ft_map = {i: names.index(n) for i, n in finetuned.names.items() if n in names}

    results, scores = {}, {}
    for split in ("val", "test"):
        images = split_images(split)
        gts = [read_labels(p) for p in images]
        preds = {
            "finetuned yolo11n": predict(finetuned, images, ft_map),
            "COCO yolo11n zero-shot": predict(coco, images, coco_map),
            "predict nothing": [np.zeros((0, 6)) for _ in images],
        }
        if split == "test":
            results = {model: evaluate(p, gts, names) for model, p in preds.items()}
        scores[split] = {model: image_scores(p, gts, images, pos) for model, p in preds.items()}

    ft_val, ft_test = scores["val"]["finetuned yolo11n"], scores["test"]["finetuned yolo11n"]
    pd.concat([ft_val.assign(split="val"), ft_test.assign(split="test")]) \
        .to_csv(ARTIFACTS / "image_scores.csv", index=False)

    # Choose on validation, report on test: picking the threshold on the same
    # photos you quote the cost on flatters the cost.
    sweep_val, sweep_test = single_threshold_sweep(ft_val), single_threshold_sweep(ft_test)
    sweep = sweep_test.merge(sweep_val[["threshold", "cost_per_1000"]], on="threshold",
                             suffixes=("", "_val"))
    sweep.to_csv(ARTIFACTS / "cost_sweep.csv", index=False)
    chosen_t = cheapest_threshold(sweep_val)

    bands = band_sweep(ft_val)
    bands.to_csv(ARTIFACTS / "band_sweep_val.csv", index=False)
    band = cheapest_band(bands)
    low, high = float(band["low"]), float(band["high"])
    band_test = expected_cost(outcome_rates(ft_test["has_gap"], ft_test["score"], low, high))

    image_baselines = {
        model: image_level(df, chosen_t if model == "finetuned yolo11n" else 0.5)
        for model, df in scores["test"].items()
    }
    alone = image_level(ft_test, chosen_t)

    summary = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "classes": names, **dataset},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ,
                     "device": DEVICE, "seed": SEED},
        "positive_class": POSITIVE_CLASS,
        "object_level_test": results,
        "object_confusion_matrix_test": {"labels": names + ["background"], "matrix": matrix,
                                          "note": "rows = predicted, columns = true"},
        "image_level_test": image_baselines,
        "costs": COSTS, "base_rate": BASE_RATE, "operations": OPERATIONS,
        "operating_point": {
            "single_threshold": chosen_t,
            "single_threshold_test": alone,
            "single_threshold_test_90ci": bootstrap_cost(ft_test, chosen_t, chosen_t),
            "band": {"low": low, "high": high},
            "band_val": {k: float(band[k]) for k in ("cost_per_1000", "review_share")},
            "band_test": band_test,
            "band_test_90ci": bootstrap_cost(ft_test, low, high),
            "band_staffing": reviewers_needed(band_test["review_share"]),
        },
        "alternatives_per_1000": {
            **alternatives(),
            "model decides alone": alone["cost_per_1000"],
            "model + human review band": band_test["cost_per_1000"],
        },
    }
    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: summary[k] for k in ("object_level_test", "operating_point",
                                               "alternatives_per_1000")}, indent=2, default=float))


if __name__ == "__main__":
    main()
