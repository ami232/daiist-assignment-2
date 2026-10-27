"""Dataset download helper, shared by stats.py and train.py."""
from pathlib import Path

import yaml
from roboflow import Roboflow

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WORKSPACE = "firas"
PROJECT = "road-damage-detection-cu846"
VERSION = 2
DATASET_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
DATA_YAML = DATASET_DIR / "data_local.yaml"


def get_dataset() -> Path:
    """Download the dataset if needed. Return a YAML path Ultralytics can use."""
    if DATA_YAML.exists():
        return DATA_YAML
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download("yolov11", location=str(DATASET_DIR))
    original = yaml.safe_load((DATASET_DIR / "data.yaml").read_text())
    local = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": original["names"],
    }
    DATA_YAML.write_text(yaml.safe_dump(local))
    return DATA_YAML


if __name__ == "__main__":
    print(get_dataset())