"""
Every assumption the pipeline and the dashboard share, in one place.

train.py uses these to build the cost sweep; app.py uses the same numbers as
the defaults for its sliders. Change a figure here and both follow. Each figure
is argued for in REPORT.md ("Cost matrix and operating point").
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data ---------------------------------------------------------------------
# Public Roboflow Universe dataset, used as published (not forked).
# https://universe.roboflow.com/work-6qkmv/solder-2-hcmiq/dataset/37
RF_WORKSPACE = "work-6qkmv"
RF_PROJECT = "solder-2-hcmiq"
RF_VERSION = 37
RF_FORMAT = "yolov11"

# The positive class of the business decision is "this inspection window
# contains at least one defective joint". Every class except Good is a defect.
GOOD_CLASSES = {"good"}  # compared case-insensitively

# --- Model --------------------------------------------------------------------
BASE_MODEL = "yolo11s.pt"  # finetuned from COCO-pretrained weights
COCO_MODEL = "yolo11s.pt"  # the same checkpoint, used zero-shot as a baseline
IMGSZ = 640
EPOCHS = 80                # overridable with the EPOCHS environment variable
PATIENCE = 20
SEED = 0
VAL_FRACTION = 0.10        # carved out of Roboflow's train split, by source image

# --- Paths --------------------------------------------------------------------
DATA_DIR = ROOT / "datasets"
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

# --- Business assumptions (euros, per inspection window) ----------------------
# One "case" is one camera window from the inspection station: one image,
# typically two or three joints.
COST_FN = 72.0      # defective joint passed: expected downstream cost of an escape
COST_FP = 1.50      # good window failed: rework technician touches a good joint
COST_TP = 1.50      # defective window failed: the same rework, now worth doing
COST_TN = 0.0       # good window passed: nothing happens
COST_REVIEW = 0.12  # a person looks at one window on screen: 12 s at 35 EUR/h

# A human reviewer is good but not perfect.
HUMAN_SENSITIVITY = 0.95  # share of defective windows a reviewer catches
HUMAN_FALSE_FAIL = 0.02   # share of good windows a reviewer wrongly fails

# How often the positive class really occurs on the line. The test split is far
# richer in defects than production, so costs are re-weighted to this rate.
BASE_RATE = 0.002  # 0.2% of windows contain a defective joint

# Volume and staffing.
BOARDS_PER_DAY = 2000      # two shifts
WINDOWS_PER_BOARD = 40     # inspection windows the camera takes per board
REVIEW_SECONDS = 12        # time to judge one window on screen
PRODUCTIVE_HOURS_PER_REVIEWER = 6.5  # per shift, after breaks and changeovers
