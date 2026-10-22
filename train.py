"""
Assignment 2 - training + evaluation pipeline.
Run with: uv run python main.py train

Produces, with no manual steps:
  weights/best.pt          the finetuned model the app loads
  results/metrics.json     finetuned vs COCO zero-shot vs trivial baselines
  results/per_class.csv    per-class precision / recall / AP50
  results/confusion_matrix.csv
  results/test_scores.csv  one row per test image: y_true, score  <- feeds the app
  results/cost_curve.png   cost against threshold, with the chosen point marked
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"

# --- Dataset identity. Must match REPORT.md exactly. ---
RF_WORKSPACE = "intelligent-fruit-quality-grading-system"
RF_PROJECT = "fruit-quality-inspection-system"
RF_VERSION = 6

# --- Which classes mean "defective". Everything else is "good". ---
DEFECT_KEYWORDS = ("bad",)

# --- Training settings. Recorded in REPORT.md. ---
BASE_MODEL = "yolo11n.pt"
EPOCHS = 40
IMGSZ = 640
BATCH = 16
SEED = 0

# --- Cost model. FILL THESE IN and defend them in REPORT.md. ---
COST_FN = 10.0
COST_FP = 0.25
COST_TP = 0.0
COST_TN = 0.0
COST_REVIEW = 0.05
BASE_RATE = 0.05


def device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def download_dataset() -> Path:
    """Pull the exact pinned version from Roboflow Universe."""
    from roboflow import Roboflow

    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / f"{RF_PROJECT}-{RF_VERSION}"
    if not (target / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION)
        version.download("yolov11", location=str(target))

    yaml_path = target / "data.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    cfg["path"] = str(target.resolve())
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        cfg[split] = f"{folder}/images"
    yaml_path.write_text(yaml.safe_dump(cfg))
    return yaml_path


def describe_dataset(yaml_path: Path) -> dict:
    """Instance counts per class, split sizes, object size vs image."""
    cfg = yaml.safe_load(yaml_path.read_text())
    names = cfg["names"]
    root = Path(cfg["path"])
    summary = {"classes": names, "splits": {}, "instances": {}, "box_area_pct": {}}

    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        labels = sorted((root / folder / "labels").glob("*.txt"))
        counts = {n: 0 for n in names}
        areas = []
        for f in labels:
            for line in f.read_text().split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                counts[names[int(parts[0])]] += 1
                areas.append(float(parts[3]) * float(parts[4]) * 100)
        summary["splits"][split] = len(labels)
        summary["instances"][split] = counts
        summary["box_area_pct"][split] = {
            "median": round(float(np.median(areas)), 2) if areas else None,
            "p10": round(float(np.percentile(areas, 10)), 2) if areas else None,
            "p90": round(float(np.percentile(areas, 90)), 2) if areas else None,
        }
    return summary


def image_scores(model: YOLO, yaml_path: Path, defect_ids: set) -> pd.DataFrame:
    """Per-image binary view of the business decision.

    y_true = 1 if the image contains at least one defective piece.
    score  = highest confidence given to any defect class.
    """
    cfg = yaml.safe_load(yaml_path.read_text())
    root = Path(cfg["path"]) / "test"
    rows = []
    for img in sorted((root / "images").glob("*")):
        label = root / "labels" / f"{img.stem}.txt"
        ids = set()
        if label.exists():
            ids = {int(l.split()[0]) for l in label.read_text().split("\n") if l.split()}
        y_true = int(bool(ids & defect_ids))

        pred = model.predict(str(img), conf=0.01, verbose=False)[0]
        confs = [float(c) for c, k in zip(pred.boxes.conf, pred.boxes.cls)
                 if int(k) in defect_ids]
        rows.append({"image": img.name, "y_true": y_true,
                     "score": max(confs) if confs else 0.0})
    return pd.DataFrame(rows)


def counts_at(df: pd.DataFrame, thr: float) -> dict:
    pred = (df["score"] >= thr).astype(int)
    y = df["y_true"]
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def expected_cost(df: pd.DataFrame, thr: float) -> float:
    """Cost per 1,000 pieces, reweighted from test prevalence to BASE_RATE."""
    c = counts_at(df, thr)
    tpr = c["tp"] / max(c["tp"] + c["fn"], 1)
    fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
    p = BASE_RATE
    per_case = (p * (tpr * COST_TP + (1 - tpr) * COST_FN)
                + (1 - p) * (fpr * COST_FP + (1 - fpr) * COST_TN))
    return 1000 * per_case


def main() -> None:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    yaml_path = download_dataset()
    cfg = yaml.safe_load(yaml_path.read_text())
    names = cfg["names"]
    defect_ids = {i for i, n in enumerate(names)
                  if any(k in n.lower() for k in DEFECT_KEYWORDS)}
    print(f"Defect classes: {[names[i] for i in sorted(defect_ids)]}")

    description = describe_dataset(yaml_path)
    (RESULTS_DIR / "dataset_description.json").write_text(json.dumps(description, indent=2))

    # 1. Finetune
    model = YOLO(BASE_MODEL)
    model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                seed=SEED, device=device(), project=str(ROOT / "runs"),
                name="finetune", exist_ok=True)
    best = ROOT / "runs" / "finetune" / "weights" / "best.pt"
    shutil.copy(best, WEIGHTS_DIR / "best.pt")

    tuned = YOLO(WEIGHTS_DIR / "best.pt")
    m = tuned.val(data=str(yaml_path), split="test", device=device())

    per_class = pd.DataFrame({
        "class": [names[i] for i in m.box.ap_class_index],
        "precision": m.box.p,
        "recall": m.box.r,
        "ap50": m.box.ap50,
        "ap50_95": m.box.maps[m.box.ap_class_index],
    })
    per_class.to_csv(RESULTS_DIR / "per_class.csv", index=False)
    pd.DataFrame(m.confusion_matrix.matrix).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    # 2. Baseline A: pretrained COCO zero-shot on the same test split.
    #    COCO knows apple/banana/orange but has no notion of "bad",
    #    so its recall on every defect class is zero by construction.
    coco = YOLO(BASE_MODEL)
    coco_m = coco.val(data=str(yaml_path), split="test", device=device())

    # 3. Baseline B: trivial. Predict nothing, or flag everything.
    scores = image_scores(tuned, yaml_path, defect_ids)
    scores.to_csv(RESULTS_DIR / "test_scores.csv", index=False)
    n, pos = len(scores), int(scores["y_true"].sum())

    # 4. Threshold sweep
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    sweep = pd.DataFrame([{"threshold": float(t), **counts_at(scores, t),
                           "cost_per_1000": expected_cost(scores, t)} for t in grid])
    sweep.to_csv(RESULTS_DIR / "cost_sweep.csv", index=False)
    chosen = float(sweep.loc[sweep["cost_per_1000"].idxmin(), "threshold"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["cost_per_1000"])
    ax.axvline(chosen, linestyle="--", color="red")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("expected cost per 1,000 pieces (EUR)")
    ax.set_title(f"minimum at threshold {chosen}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cost_curve.png", dpi=150)

    metrics = {
        "dataset": {"workspace": RF_WORKSPACE, "project": RF_PROJECT, "version": RF_VERSION},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ,
                     "batch": BATCH, "seed": SEED, "device": device()},
        "costs": {"fn": COST_FN, "fp": COST_FP, "tp": COST_TP, "tn": COST_TN,
                  "review": COST_REVIEW, "base_rate": BASE_RATE},
        "finetuned": {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)},
        "coco_zero_shot": {"mAP50": float(coco_m.box.map50), "mAP50_95": float(coco_m.box.map)},
        "trivial_predict_nothing": {"accuracy": (n - pos) / n, "recall_on_defect": 0.0},
        "trivial_flag_everything": {"accuracy": pos / n, "recall_on_defect": 1.0},
        "test_prevalence": pos / n,
        "chosen_threshold": chosen,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
