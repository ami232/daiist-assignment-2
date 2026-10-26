"""
Every assumption the pipeline and the dashboard depend on, in one place.

train.py and app.py both import this file. The app recomputes all costs from
the saved test-set scores every time it starts, so after changing a cost figure
you only need to restart the app. Re-run training if you change the dataset or
training settings, or to refresh the cost files saved in results/.

BUSINESS CASE (fictional): Chispa Componentes S.L., a Tier-1 automotive
supplier in Getafe, robot-welds steel rear subframes for a European carmaker's
SUV. A subframe holds the rear suspension, so a failed weld is a safety issue.
Today inspectors check every critical seam by hand; the plant wants a camera +
model system that passes, fails, or routes each weld photo to a human.
The reasoning behind every figure below is written up in REPORT.md.
"""

# ----------------------------------------------------------------------------
# Dataset on Roboflow Universe
# ----------------------------------------------------------------------------
WORKSPACE = "defect-detection-fxtd5"
PROJECT = "weld-defect-lze1x"
VERSION = 1

# ----------------------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"  # finetuned model starts from this COCO checkpoint
COCO_BASELINE = "yolo11n.pt"  # same checkpoint, used zero-shot as a baseline
EPOCHS = 60
PATIENCE = 20  # stop early if validation mAP has not improved for this many epochs
IMG_SIZE = 640
BATCH = 16
SEED = 0
CONFUSION_MATRIX_CONF = 0.25  # boxes below this confidence are ignored in the detection confusion matrix

# ----------------------------------------------------------------------------
# Class names
# ----------------------------------------------------------------------------
# Version 1 of this dataset exports its classes as "0", "1" and "2", with no
# names. Checked against the labelled photos: 0 = Bad Weld, 1 = Good Weld,
# 2 = Defect (Defect boxes are the smallest, about 1% of the image).
CLASS_NAMES = {0: "Bad Weld", 1: "Good Weld", 2: "Defect"}

# ----------------------------------------------------------------------------
# The decision
# ----------------------------------------------------------------------------
# One case = one inspection photo of a weld. The case is "positive" (needs
# action) if it contains at least one Bad Weld or Defect box. Everything else
# is treated as a good weld.
POSITIVE_CLASS_KEYWORDS = ("bad", "defect")

# ----------------------------------------------------------------------------
# Costs in euros per case
# ----------------------------------------------------------------------------
# FN: expected cost of an escaped bad weld. 80% are caught at the end-of-line
# check (scrap the finished subframe, ~250) + 19% reach the carmaker's plant
# (chargeback, containment sorting, 8D report, ~3,000) + 1% reach a car on the
# road (share of warranty/recall costs, ~100,000) = ~1,770, rounded up for the
# risk of losing the contract.
COST_FN = 2000.0  # bad weld passed (missed)

# FP: part diverted to the rework station (~5 handling), where a welder
# re-checks the seam and usually re-welds it as a precaution (~20).
COST_FP = 25.0  # good weld failed (needless rework / re-check)

# TP: grind out the defect, re-weld and re-inspect (~30), plus ~10% of bad
# welds cannot be repaired and the ~200 subframe is scrapped (~20).
COST_TP = 50.0  # bad weld correctly failed (the rework itself)

COST_TN = 0.0  # good weld correctly passed: nothing happens

# Production base rate: share of welds that are actually bad on the real line.
# This is NOT the share in the test set; the cost maths reweights to this.
# 1% = ~80 bad welds a day, plausible for a line still tuning its weld
# parameters after adding a new part variant.
BASE_RATE = 0.01

# ----------------------------------------------------------------------------
# Human review
# ----------------------------------------------------------------------------
# 2 robot cells x 25 subframes/hour x 16 hours (two shifts) x 10 critical seams
DAILY_VOLUME = 8000  # welds photographed per day

# Locate the seam, inspect under a lamp, gauge any undercut, scan, log result.
REVIEW_SECONDS = 60  # time for an inspector to check one case

# Certified weld inspector: ~30k gross + ~30% employer social security +
# ~15k training, certification and supervision = ~54k over ~1,700 hours.
INSPECTOR_HOURLY_COST = 32.0  # fully loaded cost per hour, EUR

# 8-hour shift minus breaks, handovers, paperwork and gauge calibration.
INSPECTOR_HOURS_PER_DAY = 6.5  # productive checking hours per person per day

# Repetitive visual work: fatigue and subtle porosity mean even a certified
# inspector passes some bad welds.
HUMAN_MISS_RATE = 0.10  # share of bad welds an inspector still passes

# Safety-critical part, so inspectors fail a seam when in doubt.
HUMAN_FALSE_ALARM_RATE = 0.02  # share of good welds an inspector fails

COST_REVIEW = REVIEW_SECONDS / 3600 * INSPECTOR_HOURLY_COST  # labour per check, EUR

# ----------------------------------------------------------------------------
# Operating point
# ----------------------------------------------------------------------------
# Leave as None to use the cost-minimising values from the sweep. Set them to
# override, for example if the cost-optimal band needs more inspectors than
# you have. Scores are the model's highest defect confidence in the image.
CHOSEN_THRESHOLD = None  # model-alone threshold, e.g. 0.35
CHOSEN_BAND = None  # (low, high): below low = pass, at/above high = fail, between = human


def costs() -> dict[str, float]:
    return {
        "fn": COST_FN,
        "fp": COST_FP,
        "tp": COST_TP,
        "tn": COST_TN,
        "review": COST_REVIEW,
        "human_miss": HUMAN_MISS_RATE,
        "human_false_alarm": HUMAN_FALSE_ALARM_RATE,
    }


def as_table() -> list[tuple[str, str]]:
    """The assumptions, as rows for the dashboard."""
    return [
        ("Dataset", f"{WORKSPACE} / {PROJECT} / v{VERSION}"),
        ("Model", f"{BASE_MODEL}, {EPOCHS} epochs max, imgsz {IMG_SIZE}, batch {BATCH}"),
        ("Class names", ", ".join(f"{k} = {v}" for k, v in (CLASS_NAMES or {}).items()) or "as exported"),
        ("Positive class keywords", ", ".join(POSITIVE_CLASS_KEYWORDS)),
        ("Cost of a missed bad weld (FN)", f"€{COST_FN:,.2f}"),
        ("Cost of failing a good weld (FP)", f"€{COST_FP:,.2f}"),
        ("Cost of failing a bad weld (TP)", f"€{COST_TP:,.2f}"),
        ("Cost of passing a good weld (TN)", f"€{COST_TN:,.2f}"),
        ("Production base rate of bad welds", f"{BASE_RATE:.1%}"),
        ("Cases per day", f"{DAILY_VOLUME:,}"),
        ("Seconds per human check", f"{REVIEW_SECONDS}"),
        ("Inspector cost per hour", f"€{INSPECTOR_HOURLY_COST:,.2f}"),
        ("Productive checking hours per inspector per day", f"{INSPECTOR_HOURS_PER_DAY}"),
        ("Cost per human check", f"€{COST_REVIEW:,.2f}"),
        ("Inspector miss rate", f"{HUMAN_MISS_RATE:.1%}"),
        ("Inspector false-alarm rate", f"{HUMAN_FALSE_ALARM_RATE:.1%}"),
    ]
