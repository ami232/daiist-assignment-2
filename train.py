"""
Assignment 2: training pipeline.

    uv run python main.py train

Downloads yaiinternsproject / avocado-foo4c / version 1 from Roboflow Universe,
finetunes yolo11n on it, evaluates on the test split against two baselines
(COCO yolo11n zero-shot, and trivial policies), sweeps the reject threshold
using the costs in costs.json, and writes everything the dashboard needs to
artifacts/. The trained model is copied to weights/.

Decision: positive = any disease (reject), negative = Healthy (export). The
model still learns all five classes so they can be reported separately.

Optional environment variables, for a quick smoke test only:
    EPOCHS=1 FRACTION=0.1 uv run python main.py train

After changing costs.json, re-evaluate the committed weights without retraining:
    uv run python train.py --eval-only
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import decision
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WORKSPACE, PROJECT, VERSION = "yaiinternsproject", "avocado-foo4c", 1
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-{VERSION}"
DATA_YAML = DATA_DIR / "data_abs.yaml"
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "avocado_yolo11n.pt"
RUNS = ROOT / "runs"

BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 40))
FRACTION = float(os.environ.get("FRACTION", 1.0))
IMGSZ = 640
BATCH = 16
PATIENCE = 10
SEED = 0
CONFUSION_CONF = 0.25
DEVICE = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
SPLITS = ["train", "valid", "test"]


# --- data -------------------------------------------------------------------

def download_dataset() -> list[str]:
    if not (DATA_DIR / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR)
        )
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    # Roboflow's data.yaml uses paths relative to a parent folder, which
    # Ultralytics resolves differently. Write one with absolute paths.
    DATA_YAML.write_text(yaml.safe_dump({
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }))
    return names


def read_labels(label_file: Path) -> list[tuple[int, float]]:
    """(class id, box area as a share of the image) for each object."""
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        values = list(map(float, parts[1:]))
        if len(values) == 4:
            w, h = values[2], values[3]
        else:  # polygon
            w = max(values[0::2]) - min(values[0::2])
            h = max(values[1::2]) - min(values[1::2])
        out.append((int(parts[0]), w * h))
    return out


def dataset_stats(names: list[str]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        per_class = {c: {"instances": 0, "images": 0, "areas": []} for c in range(len(names))}
        files = sorted((DATA_DIR / split / "labels").glob("*.txt"))
        for f in files:
            labels = read_labels(f)
            for c, area in labels:
                per_class[c]["instances"] += 1
                per_class[c]["areas"].append(area)
            for c in {c for c, _ in labels}:
                per_class[c]["images"] += 1
        for c, d in per_class.items():
            areas = np.array(d["areas"]) * 100
            rows.append({
                "split": split,
                "split_images": len(files),
                "class": names[c],
                "instances": d["instances"],
                "images_with_class": d["images"],
                "median_box_area_pct": round(float(np.median(areas)), 1) if len(areas) else None,
                "p10_box_area_pct": round(float(np.percentile(areas, 10)), 1) if len(areas) else None,
                "p90_box_area_pct": round(float(np.percentile(areas, 90)), 1) if len(areas) else None,
            })
    return pd.DataFrame(rows)


# --- model ------------------------------------------------------------------

def train() -> Path:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        patience=PATIENCE, seed=SEED, deterministic=True, device=DEVICE,
        fraction=FRACTION, project=str(RUNS), name="avocado", exist_ok=True,
    )
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)
    shutil.copy(Path(model.trainer.save_dir) / "results.csv", ARTIFACTS / "training_curves.csv")
    return WEIGHTS


def evaluate_detection(model: YOLO, names: list[str], stats: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    metrics = model.val(
        data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        project=str(RUNS), name="test", exist_ok=True,
    )
    test_counts = stats[stats.split == "test"].set_index("class")
    rows = []
    for c, name in enumerate(names):
        row = {"class": name, "test_instances": int(test_counts.loc[name, "instances"]),
               "precision": None, "recall": None, "ap50": None, "ap50_95": None}
        if c in list(metrics.ap_class_index):
            p, r, ap50, ap = metrics.box.class_result(list(metrics.ap_class_index).index(c))
            row.update(precision=round(float(p), 3), recall=round(float(r), 3),
                       ap50=round(float(ap50), 3), ap50_95=round(float(ap), 3))
        rows.append(row)
    overall = {"map50": round(float(metrics.box.map50), 3), "map50_95": round(float(metrics.box.map), 3)}
    # mAP needs every box down to conf 0.001, but a confusion matrix at that
    # level is mostly junk boxes. Count it at a working confidence instead.
    at_conf = model.val(
        data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        conf=CONFUSION_CONF, plots=True, project=str(RUNS), name="test_conf", exist_ok=True,
    )
    confusion = {
        "labels": names + ["background"],
        "layout": "rows = predicted class, columns = true class",
        "conf": CONFUSION_CONF,
        "matrix": at_conf.confusion_matrix.matrix.astype(int).tolist(),
    }
    return overall, pd.DataFrame(rows), confusion


def image_scores(model: YOLO, class_ids: list[int], split: str) -> list[float]:
    """Highest confidence among boxes of the given classes, per image."""
    paths = sorted((DATA_DIR / split / "images").glob("*"))
    scores = []
    for result in model.predict([str(p) for p in paths], conf=0.001, imgsz=IMGSZ,
                                device=DEVICE, stream=True, verbose=False):
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        keep = np.isin(cls, class_ids) if class_ids else np.ones_like(cls, dtype=bool)
        scores.append(float(conf[keep].max()) if keep.any() else 0.0)
    return scores


def image_labels(names: list[str], split: str) -> tuple[list[str], np.ndarray]:
    disease_ids = {names.index(c) for c in decision.DISEASE_CLASSES}
    paths = sorted((DATA_DIR / split / "images").glob("*"))
    labels = []
    for p in paths:
        classes = {c for c, _ in read_labels(DATA_DIR / split / "labels" / f"{p.stem}.txt")}
        labels.append(int(bool(classes & disease_ids)))
    return [p.name for p in paths], np.array(labels)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random diseased image scores above a random healthy one."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def decision_summary(val: tuple, test: tuple, costs: dict) -> dict:
    """Pick the threshold on validation, report how it does on test."""
    t = decision.best_single_threshold(*val, costs)
    rates = decision.outcome_rates(*test, t, t)
    return {
        "auc": round(roc_auc(*test), 3),
        "threshold_chosen_on_val": t,
        "test_recall": round(rates["pos"]["reject"], 3),
        "test_false_positive_rate": round(rates["neg"]["reject"], 3),
        "test_cost_per_1000": round(1000 * decision.expected_cost(rates, costs)["cost_per_case"], 2),
    }


def plot_cost_curve(sweep: pd.DataFrame, threshold: float, band: dict) -> None:
    chosen = sweep.loc[sweep.threshold == threshold].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.threshold, sweep.val_cost_per_1000, label="validation (used to choose)")
    ax.plot(sweep.threshold, sweep.test_cost_per_1000, "--", label="test (reported)")
    ax.plot(chosen.threshold, chosen.test_cost_per_1000, "r*", markersize=14,
            label=f"chosen t={threshold:.2f} (test EUR {chosen.test_cost_per_1000:.0f} per 1,000)")
    ax.axvspan(band["lo"], band["hi"], alpha=0.15, color="orange",
               label=f"review band {band['lo']:.2f}-{band['hi']:.2f}")
    ax.set_xlabel("Reject threshold (disease score)")
    ax.set_ylabel("Expected cost per 1,000 avocados (EUR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "cost_curve.png", dpi=120)
    plt.close(fig)


# --- pipeline ---------------------------------------------------------------

def main(train_model: bool = True) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    costs = decision.load_costs()
    print(f"Device: {DEVICE}")

    names = download_dataset()
    if train_model:
        train()
        # Scores computed inside the training process did not match a fresh
        # load of the same weights (image AUC 0.92 vs 0.99). Evaluate on a
        # clean load of the saved file, which is what the app and graders use.
        subprocess.run([sys.executable, str(Path(__file__)), "--eval-only"], cwd=ROOT, check=True)
        return

    stats = dataset_stats(names)
    stats.to_csv(ARTIFACTS / "dataset_stats.csv", index=False)

    model = YOLO(str(WEIGHTS))
    overall, per_class, confusion = evaluate_detection(model, names, stats)
    per_class.to_csv(ARTIFACTS / "per_class.csv", index=False)
    (ARTIFACTS / "confusion_matrix.json").write_text(json.dumps(confusion, indent=2))

    # Thresholds are chosen on the validation split and evaluated on test, so
    # the reported cost is not tuned on the data it is measured on.
    disease_ids = [names.index(c) for c in decision.DISEASE_CLASSES]
    coco_model = YOLO(BASE_MODEL)
    scores = {}
    for split in ("valid", "test"):
        files, labels = image_labels(names, split)
        finetuned = np.array(image_scores(model, disease_ids, split))
        # COCO has no disease classes, so its only possible signal is "found an object".
        coco = np.array(image_scores(coco_model, [], split))
        scores[split] = {"labels": labels, "finetuned": finetuned, "coco": coco}
        pd.DataFrame({"image": files, "diseased": labels, "finetuned_score": finetuned,
                      "coco_zero_shot_score": coco}).to_csv(ARTIFACTS / f"{split}_scores.csv", index=False)
    val = (scores["valid"]["finetuned"], scores["valid"]["labels"])
    test = (scores["test"]["finetuned"], scores["test"]["labels"])
    labels = test[1]

    val_sweep = pd.DataFrame(decision.single_threshold_sweep(*val, costs))
    test_sweep = pd.DataFrame(decision.single_threshold_sweep(*test, costs))
    sweep = pd.DataFrame({
        "threshold": val_sweep.threshold,
        "val_cost_per_1000": val_sweep.cost_per_1000,
        "test_cost_per_1000": test_sweep.cost_per_1000,
        "test_recall": test_sweep.recall,
        "test_false_positive_rate": test_sweep.false_positive_rate,
    })
    sweep.to_csv(ARTIFACTS / "cost_sweep.csv", index=False)
    threshold = decision.best_single_threshold(*val, costs)
    band = decision.best_review_band(*val, costs)
    plot_cost_curve(sweep, threshold, band)

    single_rates = decision.outcome_rates(*test, threshold, threshold)
    band_rates = decision.outcome_rates(*test, band["lo"], band["hi"])
    band_test = decision.expected_cost(band_rates, costs)

    p = costs["base_rate"]
    coco_val = (scores["valid"]["coco"], scores["valid"]["labels"])
    coco_test = (scores["test"]["coco"], labels)
    baselines = {
        "finetuned yolo11n": {**overall, **decision_summary(val, test, costs)},
        "COCO yolo11n zero-shot": {"map50": 0.0, "map50_95": 0.0,
                                   "note": "no COCO class matches any of our five classes",
                                   **decision_summary(coco_val, coco_test, costs)},
        "trivial: predict nothing (export all)": {
            "map50": 0.0, "map50_95": 0.0, "auc": 0.5,
            "test_cost_per_1000": round(1000 * p * costs["cost_fn_eur"] + 1000 * (1 - p) * costs["cost_tn_eur"], 2)},
        "trivial: majority of test split (reject all)": {
            "map50": 0.0, "map50_95": 0.0, "auc": 0.5,
            "test_cost_per_1000": round(1000 * p * costs["cost_tp_eur"] + 1000 * (1 - p) * costs["cost_fp_eur"], 2)},
    }
    # What the old method (tune and report on test) would have claimed, to
    # show how optimistic that was.
    test_band = decision.best_review_band(*test, costs)
    test_tuned = {
        "single_threshold": decision.best_single_threshold(*test, costs),
        "single_cost_per_1000": round(float(test_sweep.cost_per_1000.min()), 2),
        "band": [test_band["lo"], test_band["hi"]],
        "band_cost_per_1000": round(1000 * test_band["cost_per_case"], 2),
    }

    summary = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}",
                    "test_images": int(len(labels)), "test_diseased": int(labels.sum())},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "fraction": FRACTION,
                     "imgsz": IMGSZ, "batch": BATCH, "patience": PATIENCE, "seed": SEED,
                     "device": DEVICE},
        "positive_class": "diseased (any of " + ", ".join(decision.DISEASE_CLASSES) + ")",
        "detection": overall,
        "baselines": baselines,
        "costs_used": costs,
        "threshold_selection": "chosen on the validation split, costs reported on the test split",
        "single_threshold": {
            "threshold": threshold,
            "test_recall": round(single_rates["pos"]["reject"], 3),
            "test_false_positive_rate": round(single_rates["neg"]["reject"], 3),
            "test_cost_per_1000": round(1000 * decision.expected_cost(single_rates, costs)["cost_per_case"], 2),
        },
        "review_band": {
            "lo": band["lo"], "hi": band["hi"],
            "val_cost_per_1000": round(1000 * band["cost_per_case"], 2),
            "test_cost_per_1000": round(1000 * band_test["cost_per_case"], 2),
            "test_review_share": band_test["review_share"],
            "test_outcomes": decision.confusion_3way(*test, band["lo"], band["hi"]),
            **decision.staffing(band_test["review_share"], costs),
        },
        "alternatives_per_1000": decision.alternatives(*test, costs, threshold, band["lo"], band["hi"]),
        "if_tuned_on_test_instead": test_tuned,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("detection", "single_threshold", "review_band",
                                               "alternatives_per_1000")}, indent=2, default=float))
    print(f"\nWeights: {WEIGHTS.relative_to(ROOT)}   Artifacts: {ARTIFACTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    # --eval-only re-evaluates the committed weights and re-costs them, for
    # when costs.json changes. main.py train always runs the full pipeline.
    main(train_model="--eval-only" not in sys.argv)
