
"""Finetune YOLO11 on Roboflow fire-smoke-lwjte v1 and save weights/best.pt.

Run via:  uv run python main.py train
Settings can be overridden with env vars (handy for a quick smoke test):
    MODEL=yolo11n.pt EPOCHS=1 FRACTION=0.05
"""
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

WORKSPACE = "object-detection-yxxz2"
PROJECT = "fire-smoke-lwjte"
VERSION = 1

MODEL = os.environ.get("MODEL", "yolo11s.pt")        # or yolo11n.pt
EPOCHS = int(os.environ.get("EPOCHS", 50))
FRACTION = float(os.environ.get("FRACTION", 1.0))     # share of train set used
IMGSZ = 640
BATCH = int(os.environ.get("BATCH", 16))
SEED = 0

DATA_DIR = (Path("datasets") / f"{PROJECT}-v{VERSION}").resolve()
WEIGHTS_DIR = Path("weights")


def download() -> Path:
    if not DATA_DIR.exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR)
        )
    # Roboflow's data.yaml uses relative paths that break on some setups,
    # so write our own with absolute ones.
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    cfg = {
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }
    out = DATA_DIR / "data_local.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def main() -> None:
    data_yaml = download()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training {MODEL} on {device} for {EPOCHS} epochs")

    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        fraction=FRACTION,
        device=device,
        project=str(Path("runs").resolve()),
        name="train",
        exist_ok=True,
    )

    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS_DIR / "best.pt")
    (WEIGHTS_DIR / "train_settings.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "epochs": EPOCHS,
                "fraction": FRACTION,
                "imgsz": IMGSZ,
                "batch": BATCH,
                "seed": SEED,
                "device": torch.cuda.get_device_name(0) if device == 0 else "cpu",
                "dataset": f"{WORKSPACE}/{PROJECT}/v{VERSION}",
            },
            indent=2,
        )
    )
    print("Saved weights/best.pt")


if __name__ == "__main__":
    main()