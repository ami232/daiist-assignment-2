"""Training pipeline: uv run python main.py train

1. download the public Roboflow dataset (needs ROBOFLOW_API_KEY in the environment)
2. describe it (instances per class, split sizes, object size)
3. finetune YOLO11, copy best weights to weights/best.pt
4. evaluate on the TEST split: per-class metrics + confusion matrix
5. score every test image (max confidence of a positive-class box), for the
   finetuned model and the two baselines -> outputs/test_scores.csv

Run it on Colab (see SETUP.md), then download weights/ and outputs/ and commit them.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import config as C
import costing

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------------ data
def download_dataset() -> Path:
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:  # try a .env file (python-dotenv, if installed)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("ROBOFLOW_API_KEY")
        except ImportError:
            pass
    if not key:  # try the repo's own api_key.py, whatever the variable is called
        try:
            import api_key as ak
            key = next((getattr(ak, n) for n in ("ROBOFLOW_API_KEY", "API_KEY", "api_key", "key")
                        if isinstance(getattr(ak, n, None), str) and getattr(ak, n)), None)
        except Exception:
            pass
    if not key:
        raise SystemExit("Roboflow key not found. Set ROBOFLOW_API_KEY (see SETUP.md).")
    rf = Roboflow(api_key=key)
    version = rf.workspace(C.ROBOFLOW_WORKSPACE).project(C.ROBOFLOW_PROJECT).version(C.ROBOFLOW_VERSION)
    ds = version.download("yolov11", location=str(C.DATA_DIR))
    return Path(ds.location)


def _split_dir(root: Path, name: str) -> Path | None:
    for cand in ([name] if name != "val" else ["valid", "val"]):
        if (root / cand / "images").exists():
            return root / cand
    return None


def describe_dataset(root: Path, names: dict) -> dict:
    """Instances per class, split sizes and object size relative to the image."""
    stats = {"splits": {}, "instances": {}, "box_area_fraction": {}}
    for split in ("train", "val", "test"):
        d = _split_dir(root, split)
        if d is None:
            continue
        imgs = [p for p in (d / "images").iterdir() if p.suffix.lower() in IMG_EXT]
        counts = {n: 0 for n in names.values()}
        imgs_with_pos = 0
        areas = []
        for lab in (d / "labels").glob("*.txt"):
            has_pos = False
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                counts[names[cls]] += 1
                has_pos |= names[cls] in C.POSITIVE_CLASSES
                w, h = float(parts[3]), float(parts[4])  # normalised
                areas.append(w * h)
            imgs_with_pos += has_pos
        stats["splits"][split] = {"images": len(imgs), "images_with_positive": imgs_with_pos}
        stats["instances"][split] = counts
        if areas:
            stats["box_area_fraction"][split] = {
                "median": float(np.median(areas)),
                "p10": float(np.percentile(areas, 10)),
                "p90": float(np.percentile(areas, 90)),
            }
    return stats


# -------------------------------------------------------------- scoring
def image_labels(test_dir: Path, names: dict) -> dict[str, int]:
    """Ground truth per image: 1 if it contains any positive-class box."""
    pos_ids = {i for i, n in names.items() if n in C.POSITIVE_CLASSES}
    out = {}
    for img in sorted(p for p in (test_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT):
        lab = test_dir / "labels" / (img.stem + ".txt")
        ids = {int(float(l.split()[0])) for l in lab.read_text().splitlines() if l.strip()} if lab.exists() else set()
        out[img.name] = int(bool(ids & pos_ids))
    return out


def score_images(model, test_dir: Path, files: list[str]) -> list[float]:
    pos_ids = {i for i, n in model.names.items() if n in C.POSITIVE_CLASSES}
    scores = []
    for f in files:
        r = model.predict(str(test_dir / "images" / f), conf=0.001, verbose=False)[0]
        s = 0.0
        if r.boxes is not None and len(r.boxes):
            for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(cls) in pos_ids:
                    s = max(s, float(conf))
        scores.append(s)
    return scores


# ----------------------------------------------------------------- main
def main():
    from ultralytics import YOLO

    C.OUT.mkdir(exist_ok=True)
    C.WEIGHTS.parent.mkdir(exist_ok=True)

    root = download_dataset()
    data_yaml = root / "data.yaml"
    raw_names = yaml.safe_load(data_yaml.read_text())["names"]
    names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else {int(k): v for k, v in raw_names.items()}
    missing = [p for p in C.POSITIVE_CLASSES if p not in names.values()]
    if missing:
        raise SystemExit(f"POSITIVE_CLASSES {missing} not in dataset classes {list(names.values())}")

    ds_stats = describe_dataset(root, names)
    (C.OUT / "dataset_stats.json").write_text(json.dumps(ds_stats, indent=2))

    # ---- finetune
    model = YOLO(C.MODEL_NAME)
    model.train(data=str(data_yaml), epochs=C.EPOCHS, imgsz=C.IMGSZ, batch=C.BATCH,
                seed=C.SEED, project=str(C.ROOT / "runs"), name="train", exist_ok=True)
    best = C.ROOT / "runs" / "train" / "weights" / "best.pt"
    shutil.copy(best, C.WEIGHTS)
    model = YOLO(str(C.WEIGHTS))

    # ---- per-class evaluation on the TEST split
    m = model.val(data=str(data_yaml), split="test", conf=0.001, plots=True,
                  project=str(C.ROOT / "runs"), name="test_eval", exist_ok=True)
    per_class = []
    for k, ci in enumerate(m.box.ap_class_index):
        per_class.append({
            "class": names[int(ci)],
            "precision": float(m.box.p[k]),   # at the max-F1 confidence
            "recall": float(m.box.r[k]),
            "ap50": float(m.box.ap50[k]),
            "map50_95": float(m.box.ap[k]),
        })
    overall = {"map50": float(m.box.map50), "map50_95": float(m.box.map)}
    for fname in ("confusion_matrix_normalized.png", "confusion_matrix.png"):
        src = C.ROOT / "runs" / "test_eval" / fname
        if src.exists():
            shutil.copy(src, C.OUT / fname)

    # ---- image-level scores: finetuned model + baselines
    test_dir = _split_dir(root, "test")
    if test_dir is None:
        raise SystemExit("Dataset has no test split; pick one that does.")
    labels = image_labels(test_dir, names)
    files = list(labels)
    y = [labels[f] for f in files]

    s_model = score_images(model, test_dir, files)

    # Baseline A: pretrained COCO model, zero-shot. COCO has no "diseased" class, so it
    # cannot express the decision at all: it predicts "nothing" for the positive class.
    # That is itself a finding -- report it, and note what COCO *does* detect on your images.
    coco = YOLO("yolo11n.pt")
    s_coco = [0.0] * len(files)
    _ = coco  # loaded so the baseline is reproducible/documented; scores stay 0 by construction

    # Baseline B: trivial -- majority class. Positive is the minority, so it predicts nothing.
    majority_is_positive = np.mean(y) > 0.5
    s_trivial = [1.0 if majority_is_positive else 0.0] * len(files)

    pd.DataFrame({"image": files, "label": y, "score": s_model,
                  "score_coco_zero_shot": s_coco, "score_trivial": s_trivial}
                 ).to_csv(C.OUT / "test_scores.csv", index=False)

    # Validation split too: the test split alone (a couple of dozen images) is too small to
    # choose a threshold from. The dashboard can pool valid + test for the cost sweep.
    valid_dir = _split_dir(root, "val")
    if valid_dir is not None:
        vlabels = image_labels(valid_dir, names)
        vfiles = list(vlabels)
        pd.DataFrame({"image": vfiles, "label": [vlabels[f] for f in vfiles],
                      "score": score_images(model, valid_dir, vfiles)}
                     ).to_csv(C.OUT / "valid_scores.csv", index=False)

    baselines = {
        "finetuned": costing.image_metrics(y, s_model, 0.5),
        "coco_zero_shot": costing.image_metrics(y, s_coco, 0.5),
        "trivial_majority": costing.image_metrics(y, s_trivial, 0.5),
    }
    metrics = {
        "config": {"model": C.MODEL_NAME, "epochs": C.EPOCHS, "imgsz": C.IMGSZ,
                   "batch": C.BATCH, "seed": C.SEED,
                   "dataset": f"{C.ROBOFLOW_WORKSPACE}/{C.ROBOFLOW_PROJECT}/v{C.ROBOFLOW_VERSION}"},
        "overall_detection": overall,
        "per_class": per_class,
        "image_level_at_0.5": baselines,
        "positive_classes": C.POSITIVE_CLASSES,
    }
    (C.OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Done. Commit weights/best.pt and outputs/.")


if __name__ == "__main__":
    main()
