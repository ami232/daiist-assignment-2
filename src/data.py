"""
Dataset access for Assignment 2.

One dataset, pinned to one Roboflow Universe version, downloaded once into
data/raw/ in YOLOv11 format. Everything that needs the data (training,
evaluation, the dataset description) calls `get_dataset()` and reads the
returned data.yaml. `data/` is git-ignored, so the download happens on first
use on each machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKSPACE = "roboflow-ngkro"
PROJECT = "package-detection-e1ssd"
VERSION = 6
DATASET_ID = f"{WORKSPACE}/{PROJECT}/{VERSION}"
MODEL_FORMAT = "yolov11"

RAW_DIR = ROOT / "data" / "raw"
DATA_YAML = RAW_DIR / "data.yaml"

SPLITS = ("train", "valid", "test")
CLASS_NAMES = ["Box", "Box_broken", "Open_package", "Package"]


def is_downloaded(raw_dir: Path = RAW_DIR) -> bool:
    """True when data.yaml and every split's images folder are present."""
    if not (raw_dir / "data.yaml").is_file():
        return False
    return all((raw_dir / split / "images").is_dir() for split in SPLITS)


def normalise_data_yaml(data_yaml: Path) -> Path:
    """Make the Roboflow data.yaml usable from any working directory.

    Roboflow writes `train: ../train/images`, which Ultralytics resolves
    relative to its datasets directory, not to the yaml. Set an absolute
    `path` and split folders relative to it instead. Idempotent.
    """
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent.resolve()
    wanted = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    if all(cfg.get(k) == v for k, v in wanted.items()):
        return data_yaml
    cfg.update(wanted)
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return data_yaml


def ensure_dataset(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download the pinned dataset version if needed and return data.yaml.

    Skips the download when a complete export is already in `raw_dir`. The
    returned data.yaml always has absolute paths (see normalise_data_yaml).
    """
    if is_downloaded(raw_dir) and not force:
        return normalise_data_yaml(raw_dir / "data.yaml")

    from roboflow import Roboflow

    from api_key import load_api_key

    raw_dir.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    dataset = version.download(
        model_format=MODEL_FORMAT, location=str(raw_dir), overwrite=True
    )
    location = Path(dataset.location)
    data_yaml = location / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"download finished but {data_yaml} is missing")
    return normalise_data_yaml(data_yaml)


get_dataset = ensure_dataset  # older name, kept for scripts/describe_dataset.py


def get_version_metadata() -> dict:
    """Roboflow's own record of this version: preprocessing, augmentation,
    split sizes and whatever else the API returns. Needs the API key."""
    from roboflow import Roboflow

    from api_key import load_api_key

    rf = Roboflow(api_key=load_api_key())
    project = rf.workspace(WORKSPACE).project(PROJECT)
    for entry in project.get_version_information():
        if str(entry.get("id", "")).rsplit("/", 1)[-1] == str(VERSION):
            return dict(entry)
    raise RuntimeError(f"version {VERSION} not found in {WORKSPACE}/{PROJECT}")


if __name__ == "__main__":
    print(ensure_dataset())

# Image-level decision classes come from src/config.py (owner-supplied). An
# image is "compromised" (positive) when it holds at least one box of a
# positive class, "intact" (negative) when every box is a negative class, and
# "unlabelled" when it has no boxes.
from src.config import NEGATIVE_CLASSES, POSITIVE_CLASSES  # noqa: E402,F401
