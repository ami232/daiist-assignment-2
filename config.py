"""
Every assumption the pipeline and the dashboard rely on, in one place.

train.py and app.py both import from here, so the costs used to pick the
threshold are exactly the costs the dashboard shows. Each figure is justified
in REPORT.md; change it here and rerun train.py to see how the answer moves.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data ---------------------------------------------------------------------
WORKSPACE = "zhaosi-nicholas-z2xrh"
PROJECT = "pill-j8vgy"
VERSION = 63
DATA_DIR = ROOT / "data"
SPLIT_SEED = 42  # fixes which valid images become the tuning half vs the test half

# Class indices as exported by Roboflow (alphabetical).
CLASS_NAMES = ["nopill_back", "nopill_front", "pill_back", "pill_front"]
DEFECT_CLASSES = [0, 1]  # a blister is defective if any pocket is empty
BACK_CLASSES = [0, 2]    # any back-side label means the photo shows the foil side

# --- Model --------------------------------------------------------------------
BASE_MODEL = "yolo11s.pt"
EPOCHS = 50
IMGSZ = 640
PATIENCE = 20
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS = WEIGHTS_DIR / "best.pt"
ARTIFACTS = ROOT / "artifacts"

# --- Costs, in euros per blister ----------------------------------------------
# Positive class = defective blister (at least one empty pocket).
# Brazilian figures converted at roughly 6 BRL = 1 EUR.

# A false negative is priced as an expected value: what each consequence costs,
# times how likely one escaped blister is to cause it.
FN_COMPONENTS = {
    # name: (probability per escaped blister, cost in EUR if it happens)
    "complaint investigation": (0.10, 1_500),      # QA investigation, batch record review, CAPA
    "batch recall": (0.01, 1_000_000),             # retrieval from pharmacies, destruction, lost batch
    "ANVISA sanction": (0.005, 250_000),           # top of the Lei 6.437/1977 range, ~R$1.5M
    "patient harm and liability": (0.001, 500_000),
}
COST_FN = sum(p * cost for p, cost in FN_COMPONENTS.values())  # = 11,900

COST_FP = 0.50      # production cost of a branded blister plus pharmaceutical waste disposal
COST_REVIEW = 0.015  # 10 s of an operator at ~R$30/h loaded (~5 EUR/h)
COST_TP = 0.0       # rejecting a defective blister: it was lost anyway
COST_TN = 0.0       # passing a good blister

# Share of blisters on a real line that are defective. The test split is ~61%
# defective, so every cost below reweights the test rates to this base rate.
BASE_RATE = 0.001

# Reviewers are assumed to always decide correctly.

# Sensitivity analysis: thresholds are re-chosen for every combination.
SENSITIVITY_COST_FN = [500, 2_000, 5_000, COST_FN, 25_000, 50_000]
SENSITIVITY_BASE_RATE = [0.0001, 0.001, 0.01]

# --- Line and staffing --------------------------------------------------------
LINE_RATE_PER_MIN = 300
HOURS_PER_DAY = 16
SHIFT_HOURS = 8
PRODUCTIVE_HOURS_PER_SHIFT = 7
REVIEW_SECONDS = 10

# Thresholds on the blister score (highest confidence of any empty-pocket box).
# Score below T_LOW -> pass, at or above T_HIGH -> reject, in between -> review.
# None means "use the minimum-cost pair found on the tuning half".
CHOSEN_T_LOW = None
CHOSEN_T_HIGH = None
