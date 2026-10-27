"""Single source of truth for every number the pipeline and dashboard use.

Everything marked TODO / PLACEHOLDER is YOUR decision. You will be asked to
defend each of them in the comprehension check, so write the justification in
REPORT.md next to the value you choose.
"""
from pathlib import Path

# ----------------------------------------------------------------- dataset
# TODO: exact Roboflow Universe workspace / project / version (public, NOT forked)
ROBOFLOW_WORKSPACE = "pbl-ez66k"
ROBOFLOW_PROJECT = "rotten-and-fresh-fruits"
ROBOFLOW_VERSION = 1

# Class name(s) that mean "bad" -- must match data.yaml names exactly.
# These are the POSITIVE class: the thing we are trying to catch.
# (Dataset classes: fresh-apple, fresh-banana, rotten-apple, rotten-banana)
POSITIVE_CLASSES = ["rotten-apple", "rotten-banana"]

# ---------------------------------------------------------------- training
MODEL_NAME = "yolo11s.pt"      # yolo11n.pt or yolo11s.pt
EPOCHS = 50
IMGSZ = 640
BATCH = 16
SEED = 0

# ------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
OUT = ROOT / "outputs"
DATA_DIR = ROOT / "datasets" / "data"

# --------------------------------------------------------------- cost model
# A "case" = one image = one unit that gets a pass / review / fail decision
# (e.g. one fruit, or one crate photo). Decide and defend what a case is.
#
# PLACEHOLDER VALUES. Replace each with a figure you can justify
# ("I assumed X because Y, and the answer changes if X drops below Z").
PLACEHOLDER_COSTS = False      # set False once you've replaced everything below

COSTS = {                      # euros per case (one case = one box of fruit)
    "fn": 34.0,   # diseased box ships: refund 20 + return shipping 6 + complaint handling 8
    "fp": 3.0,    # good box rejected: manual re-sort, 10 min at 18 EUR/h
    "tp": 3.0,    # diseased box rejected: same manual re-sort
    "tn": 0.0,    # good box passes: nothing
}

# Assumed share of boxes with at least one rotten fruit IN PRODUCTION (not the test-set share).
BASE_RATE = 0.05

# ------------------------------------------------------------ human review
REVIEW = {
    "seconds": 15,         # a person opens and checks one box
    "accuracy": 0.98,      # how often the human gets it right
    "hourly_cost": 18.0,   # EUR / hour, fully loaded
}
REVIEW["cost_per_case"] = REVIEW["seconds"] / 3600 * REVIEW["hourly_cost"]

CASES_PER_DAY = 20_000     # boxes through the line per day
SHIFT_HOURS = 7.0          # productive hours per reviewer per day
REVIEWERS_AVAILABLE = 2    # people you could realistically staff

# ----------------------------------------------------- chosen thresholds
# Fill in AFTER you've run the sweep and defended the choice in REPORT.md.
# score < LO -> pass ; LO <= score < HI -> human review ; score >= HI -> fail
CHOSEN_LO = 0.05
CHOSEN_HI = 0.30
CHOSEN_SINGLE = 0.28        # middle of the minimum-cost plateau; bootstrap median also 0.28    # best single threshold (no review band), for the curve marker
