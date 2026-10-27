"""
Every decision in this project, in one place.

train.py, costing.py and app.py all read from here, so changing a number here
changes it everywhere. Lines marked YOUR DECISION are the ones the written
comprehension check can ask you to defend.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset: Roboflow Universe, used as published (not forked)
# https://universe.roboflow.com/fruit-fresh-rotten/fresh-rotten-6yh5b
# ---------------------------------------------------------------------------
RF_WORKSPACE = "fruit-fresh-rotten"
RF_PROJECT = "fresh-rotten-6yh5b"
RF_VERSION = 1

KEEP_CLASSES = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]

POSITIVE_CLASSES = ["rottenapples", "rottenbanana", "rottenoranges"]

# ---------------------------------------------------------------------------
# Splits. The Roboflow version is 80% train / 20% valid / 0% test.
#   Roboflow "valid" -> our TEST set (never seen during training).
#   A slice of Roboflow "train" -> our VALIDATION set (YOLO picks its best
#   epoch on it, and we choose thresholds on it).
# ---------------------------------------------------------------------------
VAL_FRACTION_OF_TRAIN = 0.15
SEED = 42  # fixed, so the split is identical every run

# ---------------------------------------------------------------------------
# Training settings (the EPOCHS / IMGSZ overrides exist only for quick tests)
# ---------------------------------------------------------------------------
BASE_MODEL = "yolo11n-cls.pt"  # ImageNet-pretrained classifier, smallest size
EPOCHS = int(os.environ.get("EPOCHS", 20))
IMGSZ = int(os.environ.get("IMGSZ", 224))  # YOLO's default for classification
BATCH = 32

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_RAW = ROOT / "datasets" / f"{RF_PROJECT}-v{RF_VERSION}"  # as downloaded
DATASET_PREPARED = ROOT / "datasets" / "prepared"  # our train/val/test folders
RUNS_DIR = ROOT / "runs"  # YOLO's scratch output, not committed
WEIGHTS_PATH = ROOT / "weights" / "best.pt"  # committed, loaded by the app
RESULTS_DIR = ROOT / "results"  # committed, read by the app and the report

# ---------------------------------------------------------------------------
# Business assumptions: YOUR DECISION, every one of them.
#
# PLACEHOLDERS for now. They don't affect training, only the cost analysis.
# Once you have your figures, put them in, set the flag to False, and rerun
# just the analysis (no retraining) with:   uv run python costing.py
# ---------------------------------------------------------------------------
ASSUMPTIONS_ARE_PLACEHOLDERS = False

# One average fruit is about 180 g.
BASE_RATE = 0.05  # 5% of incoming fruit rotten; sensitivity tests 1-20%

# Euros per fruit, relative to a perfect sort
COST_FN = 1.00  # rotten fruit boxed: credit 0.16 + rot spreading 0.25 + claims 0.60
COST_FP = 0.15  # good fruit to juice: (0.90 - 0.15 EUR/kg) x 0.18 kg + handling
COST_TP = 0.00  # rotten fruit correctly removed
COST_TN = 0.00  # good fruit correctly boxed

# Human review station: fruit in the review band is diverted to a person
REVIEW_SECONDS = 4.0  # pick up, turn once, decide
REVIEWER_COST_PER_HOUR = 13.0  # minimum wage 17,094/yr + ~30% social security, / 1,800 h
FRUIT_PER_DAY = 100_000  # small 2-lane line: ~2,500 kg/h / 0.18 kg x 8 h
SHIFT_HOURS = 8.0
REVIEWERS_AVAILABLE = 1


def review_cost_per_fruit() -> float:
    """Labour cost of one human check, in euros."""
    return REVIEW_SECONDS / 3600 * REVIEWER_COST_PER_HOUR
