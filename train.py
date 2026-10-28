"""
Assignment 2 training pipeline: download, finetune, evaluate, export scores.

Run with: uv run python main.py train
Optional env vars: EPOCHS (default 30), BATCH (default 16),
SKIP_TRAIN=1 (re-evaluate the weights already in weights/best.pt).
"""
import csv
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

ROOT = Path(__file__).parent
WORKSPACE, PROJECT, VERSION = "university-of-strathclyde-m436x", "pcb-defect-94yfu", 1
DATA_DIR = ROOT / "data" / f"{PROJECT}-v{VERSION}"
WEIGHTS_DIR = ROOT / "weights"
OUT_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

BASE_MODEL = "yolo11n.pt"  # COCO-pretrained; also used as the zero-shot baseline
EPOCHS = int(os.environ.get("EPOCHS", 30))
BATCH = int(os.environ.get("BATCH", 16))
IMGSZ = 640
PATIENCE = 10
SEED = 0
SCORE_CONF = 0.001   # keep almost every box so the threshold sweep sees all scores
BASELINE_CONF = 0.25  # a typical default threshold, used for the quick baseline table


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_dataset():
    if not (DATA_DIR / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR)
        )
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    cfg = {"path": str(DATA_DIR), "train": "train/images", "val": "valid/images",
           "test": "test/images", "names": names}
    data_yaml = DATA_DIR / "data_abs.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg))
    return data_yaml, names


def count_labels(split):
    counts, n_empty = {}, 0
    files = list((DATA_DIR / split / "labels").glob("*.txt"))
    for f in files:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        n_empty += not lines
        for l in lines:
            c = int(l.split()[0])
            counts[c] = counts.get(c, 0) + 1
    return counts, len(files), n_empty


def train(data_yaml, device):
    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                device=device, seed=SEED, patience=PATIENCE, plots=True,
                project=str(RUNS_DIR), name="pcb", exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS_DIR / "best.pt")
    return WEIGHTS_DIR / "best.pt"


def evaluate_detection(weights, data_yaml, names, device):
    """Per-class and overall detection metrics on the test split."""
    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                               device=device, plots=True, project=str(RUNS_DIR),
                               name="eval_test", exist_ok=True)
    counts, _, _ = count_labels("test")
    per_class = []
    for i, c in enumerate(m.box.ap_class_index):
        per_class.append({
            "class": names[int(c)], "test_instances": counts.get(int(c), 0),
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i]),
        })
    cm_png = Path(m.save_dir) / "confusion_matrix.png"
    if cm_png.exists():
        OUT_DIR.mkdir(exist_ok=True)
        shutil.copy(cm_png, OUT_DIR / "confusion_matrix.png")
    return {
        "overall": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                    "ap50": float(m.box.map50), "map50_95": float(m.box.map)},
        "per_class": per_class,
        "confusion_matrix": m.confusion_matrix.matrix.tolist(),
        "confusion_matrix_labels": list(names) + ["background"],
    }


def image_scores(model_path, split, names, device):
    """One row per image: true label (any defect?) and the model's highest box confidence."""
    model = YOLO(str(model_path))
    lbl_dir = DATA_DIR / split / "labels"
    rows = []
    for r in model.predict(source=str(DATA_DIR / split / "images"), conf=SCORE_CONF,
                           iou=0.6, imgsz=IMGSZ, device=device, stream=True, verbose=False):
        lbl = lbl_dir / (Path(r.path).stem + ".txt")
        truth = int(lbl.exists() and any(l.strip() for l in lbl.read_text().splitlines()))
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            score, top = float(r.boxes.conf[k]), r.names[int(r.boxes.cls[k])]
        else:
            score, top = 0.0, ""
        rows.append({"image": Path(r.path).name, "truth": truth, "score": score, "top_class": top})
    return rows


def save_scores(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "truth", "score", "top_class"])
        w.writeheader()
        w.writerows(rows)


def auroc(rows):
    s = np.array([r["score"] for r in rows])
    t = np.array([r["truth"] for r in rows], bool)
    pos, neg = s[t], s[~t]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def confusion_at(rows, thr):
    s = np.array([r["score"] for r in rows])
    t = np.array([r["truth"] for r in rows], bool)
    flag = s >= thr
    return {"tp": int((flag & t).sum()), "fp": int((flag & ~t).sum()),
            "fn": int((~flag & t).sum()), "tn": int((~flag & ~t).sum())}


def main():
    device = pick_device()
    print(f"Device: {device}")
    data_yaml, names = download_dataset()

    if os.environ.get("SKIP_TRAIN") == "1" and (WEIGHTS_DIR / "best.pt").exists():
        weights = WEIGHTS_DIR / "best.pt"
    else:
        weights = train(data_yaml, device)

    OUT_DIR.mkdir(exist_ok=True)
    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "classes": list(names)},
        "settings": {"base_model": BASE_MODEL, "epochs": EPOCHS, "batch": BATCH,
                     "imgsz": IMGSZ, "patience": PATIENCE, "seed": SEED, "device": str(device)},
        "splits": {},
        "detection_test": evaluate_detection(weights, data_yaml, names, device),
        "image_level": {},
    }
    for split in ("train", "valid", "test"):
        counts, n_img, n_empty = count_labels(split)
        results["splits"][split] = {"images": n_img, "images_without_defect": n_empty,
                                    "instances_per_class": {names[c]: n for c, n in sorted(counts.items())}}

    # Image-level scores: the fine-tuned model and the COCO zero-shot baseline.
    for tag, path in (("model", weights), ("coco_zero_shot", BASE_MODEL)):
        results["image_level"][tag] = {}
        for split in ("valid", "test"):
            rows = image_scores(path, split, names, device)
            save_scores(rows, OUT_DIR / f"scores_{tag}_{split}.csv")
            results["image_level"][tag][split] = {
                "auroc": auroc(rows), f"confusion_at_{BASELINE_CONF}": confusion_at(rows, BASELINE_CONF)}

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved weights to {weights} and results to {OUT_DIR}/")

    if (ROOT / "costs.json").exists():  # added in the costing step
        from costing import run
        run()


if __name__ == "__main__":
    main()