"""
settings.py - every number YOU decide lives in this one file.

train.py, costing.py and app.py all read from here.

If you change the DATASET or TRAINING settings, re-run the training:
    uv run python main.py train

If you only change the BUSINESS numbers (section 3), you do NOT need to
retrain. Just re-run the cost part, which takes a few seconds:
    uv run python costing.py

IMPORTANT: the business numbers below are PLACEHOLDERS so the code runs.
Choosing them is the graded part of the assignment. Replace each one with
your own figure and explain where it comes from in REPORT.md.
"""

# ---------------------------------------------------------------------------
# 1. Dataset on Roboflow Universe
#    Check the exact version number on the dataset's page before training.
# ---------------------------------------------------------------------------
WORKSPACE = "fruit-onxba"
PROJECT = "rotten-fresh-fruitclassification"
VERSION = 1

# Any class whose name contains this word is a POSITIVE case:
# the thing the business acts on (here: rotten fruit goes in the bin).
POSITIVE_WORD = "rotten"

# ---------------------------------------------------------------------------
# 2. Training settings
# ---------------------------------------------------------------------------
BASE_MODEL = "yolo11n-cls.pt"  # small pretrained classification model
EPOCHS = 15                    # passes over the training images
IMAGE_SIZE = 224               # images are resized to 224x224 pixels
BATCH = 32                     # images processed together in one step
SEED = 0                       # fixed so the run is repeatable

# ---------------------------------------------------------------------------
# 3. Business numbers  (PLACEHOLDERS - replace with your own)
#    All costs are in EUR for ONE piece of fruit.
# ---------------------------------------------------------------------------

# What each kind of result costs.
COST_FALSE_NEGATIVE = 3.00   # PLACEHOLDER: rotten fruit missed, reaches the shelf
COST_FALSE_POSITIVE = 0.30   # PLACEHOLDER: good fruit wrongly thrown away
COST_TRUE_POSITIVE = 0.00    # PLACEHOLDER: rotten fruit correctly thrown away
COST_TRUE_NEGATIVE = 0.00    # PLACEHOLDER: good fruit correctly sent to the shelf

# How often fruit is ACTUALLY rotten in real life (0.10 = 10%).
# This is NOT the share in the dataset, which the dataset creator chose.
BASE_RATE = 0.10             # PLACEHOLDER

# The human review step.
FRUIT_PER_DAY = 20_000       # PLACEHOLDER: pieces of fruit checked per day
SECONDS_PER_REVIEW = 3       # PLACEHOLDER: time a worker needs to check one piece
WORKER_COST_PER_HOUR = 20.0  # PLACEHOLDER: wage + overheads, EUR per hour
SHIFT_HOURS = 8              # PLACEHOLDER: hours one worker works per day
REVIEWERS_AVAILABLE = 1      # PLACEHOLDER: workers you can put on reviewing
