"""
Assignment 2: training pipeline (run with `uv run python main.py train`).

1. Download the Roboflow dataset (project-irggm / -w4yds / v1) if it is not on disk.
2. Describe it and drop from train/valid every copy of a test image (dataset_stats.py).
3. Finetune yolo11s and yolo11n from their COCO checkpoints on the cleaned split.
4. Keep the model with the higher rotten-class recall on the validation split as
   weights/best.pt (the other one is kept for comparison).
5. Evaluate both against the COCO zero-shot and predict-nothing baselines on the
   test split (evaluate.py), then sweep the cost threshold (costs.py).

Environment variables:
  RUNS_DIR   where Ultralytics writes runs (default ./runs). Point it at Google
             Drive on Colab: an interrupted run is resumed from its last.pt.
  SMOKE=1    tiny CPU run (1 epoch, 3% of train, yolo11n only) to test the
             pipeline. Writes its weights to runs/smoke/, never to weights/,
             and stops before evaluation so it never overwrites artifacts/.
"""

import json
import os
import platform
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from api_key import load_api_key
from costs import analyse
from dataset_stats import describe
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "fruit-v1"
ARTIFACTS = ROOT / "artifacts"
SMOKE = os.environ.get("SMOKE") == "1"
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs")) / ("smoke" if SMOKE else "")
WEIGHTS = RUNS_DIR / "weights" if SMOKE else ROOT / "weights"

ROBOFLOW = {"workspace": "project-irggm", "project": "-w4yds", "version": 1, "format": "yolov11"}
MODELS = ["yolo11n"] if SMOKE else ["yolo11s", "yolo11n"]
TRAIN_ARGS = {
    "epochs": 1 if SMOKE else 50,
    "patience": 10,
    "imgsz": 320 if SMOKE else 640,
    "batch": 8 if SMOKE else 32,
    "fraction": 0.03 if SMOKE else 1.0,
    "seed": 0,
    "deterministic": True,
    "workers": 0 if SMOKE else 4,
    "plots": not SMOKE,
}


def ensure_dataset() -> Path:
    if not (DATASET / "test" / "images").is_dir():
        from roboflow import Roboflow

        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(ROBOFLOW["workspace"]).project(ROBOFLOW["project"]).version(ROBOFLOW["version"])
        version.download(ROBOFLOW["format"], location=str(DATASET), overwrite=False)

    # Roboflow writes "../train/images"-style paths; make them relative to the dataset folder.
    config_path = DATASET / "data.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update({"train": "train/images", "val": "valid/images", "test": "test/images"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def write_clean_split(config_path: Path) -> Path:
    """Data config whose train/val are image lists without the copies of test images."""
    excluded = set(pd.read_csv(ARTIFACTS / "dataset" / "excluded_train_val.csv").itertuples(index=False, name=None))
    config = yaml.safe_load(config_path.read_text())
    clean = {"path": str(DATASET), "test": "test/images", "names": config["names"], "nc": config["nc"]}
    for split, key in [("train", "train"), ("valid", "val")]:
        kept = [str(p) for p in sorted((DATASET / split / "images").iterdir()) if (split, p.name) not in excluded]
        (DATASET / f"clean_{split}.txt").write_text("\n".join(kept) + "\n")
        clean[key] = f"clean_{split}.txt"
        print(f"{split}: {len(kept)} images after removing copies of test images")
    clean_path = DATASET / "data_clean.yaml"
    clean_path.write_text(yaml.safe_dump(clean, sort_keys=False))
    return clean_path


def train(model_name: str, data: Path) -> Path:
    """Train one model, resuming an interrupted run. Returns the path of its best.pt."""
    run_dir = RUNS_DIR / model_name
    done_marker = run_dir / "done.json"
    last = run_dir / "weights" / "last.pt"

    if done_marker.exists():
        print(f"{model_name}: finished run found in {run_dir}, not retraining")
    else:
        started = time.time()
        if last.exists():
            print(f"{model_name}: resuming interrupted run from {last}")
            YOLO(str(last)).train(resume=True)
        else:
            YOLO(f"{model_name}.pt").train(data=str(data), project=str(RUNS_DIR), name=model_name,
                                            exist_ok=True, **TRAIN_ARGS)
        done_marker.write_text(json.dumps({"train_seconds_last_session": round(time.time() - started)}))
    return run_dir / "weights" / "best.pt"


def rotten_recall_on_val(weights: Path, data: Path, names: list[str]) -> dict:
    metrics = YOLO(str(weights)).val(data=str(data), split="val", imgsz=TRAIN_ARGS["imgsz"],
                                     batch=TRAIN_ARGS["batch"], plots=False, verbose=False)
    box = metrics.box
    per_class = {names[c]: {"precision": float(box.p[i]), "recall": float(box.r[i]),
                            "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i])}
                 for i, c in enumerate(box.ap_class_index)}
    rotten = [v["recall"] for k, v in per_class.items() if k.startswith("rotten")]
    return {"rotten_recall_mean": sum(rotten) / len(rotten), "map50": float(box.map50),
            "map50_95": float(box.map), "per_class": per_class}


def main():
    config_path = ensure_dataset()
    describe(DATASET)
    data = write_clean_split(config_path)
    names = yaml.safe_load(data.read_text())["names"]

    results = {}
    for model_name in MODELS:
        best = train(model_name, data)
        results[model_name] = {"weights": best, **rotten_recall_on_val(best, data, names)}
        print(f"{model_name}: val rotten recall {results[model_name]['rotten_recall_mean']:.3f}, "
              f"mAP50 {results[model_name]['map50']:.3f}")

    chosen = max(results, key=lambda m: results[m]["rotten_recall_mean"])
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    shutil.copy(results[chosen]["weights"], WEIGHTS / "best.pt")
    for model_name in results:
        if model_name != chosen:
            shutil.copy(results[model_name]["weights"], WEIGHTS / f"{model_name}_comparison.pt")

    summary = {
        "chosen_model": chosen,
        "selection_rule": "highest mean recall over the 3 rotten classes on the cleaned validation split",
        "dataset": ROBOFLOW,
        "train_args": TRAIN_ARGS,
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "val": {m: {k: v for k, v in r.items() if k != "weights"} for m, r in results.items()},
    }
    out = ARTIFACTS / ("training_smoke.json" if SMOKE else "training.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nChosen: {chosen} -> {WEIGHTS / 'best.pt'}\nSummary: {out}")
    if SMOKE:
        return

    finetuned = {chosen: WEIGHTS / "best.pt"}
    finetuned.update({m: WEIGHTS / f"{m}_comparison.pt" for m in results if m != chosen})
    evaluate(finetuned, coco_weights=f"{chosen}.pt", dataset=DATASET, data_config=data,
             imgsz=TRAIN_ARGS["imgsz"])
    analyse()


if __name__ == "__main__":
    main()
