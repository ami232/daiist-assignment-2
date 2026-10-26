"""
Every decision this project depends on, in one place.

train.py, costing.py and app.py all read these values, so each number lives in
exactly one spot. The business figures in sections 4 and 5 are your own
judgement: each one needs a reason and a source in REPORT.md, plus a sentence on
what would change if it were wrong.

While PLACEHOLDERS is True the dashboard shows a warning banner. Set it to False
once every TODO below holds a figure you can defend.
"""

# 1. The decision -------------------------------------------------------------
# One or two sentences: what is decided, by whom, from which image.
DECISION = ("At a Canary Islands banana packing house, a camera photographs every hand of bananas on the "
            "conveyor. The model decides whether the hand is packed (pass), shown to a line operator who "
            "checks the photo (review), or taken off the line (fail): reprocessed if possible, otherwise discarded.")

# What a positive case is called on screen, e.g. "defect" or "violation".
POSITIVE_NAME = "rotten hand"

# A case is one image. It is POSITIVE when it contains at least one object of
# any of these classes. Use the dataset's class names exactly as published.
POSITIVE_CLASSES: list[str] = ["rotten"]   # your call: add "overripe" if that also triggers the action
# Classes in this dataset: freshripe, freshunripe, overripe, ripe, rotten, unripe

# 2. The dataset: Roboflow Universe, used as published, never forked ----------
# Read them off the URL: universe.roboflow.com/<workspace>/<project>/dataset/<version>
ROBOFLOW_WORKSPACE = "fruit-ripening"
ROBOFLOW_PROJECT = "fruit-ripening-process"
ROBOFLOW_VERSION = 1   # "raw": 5,264 train / 1,525 valid / 757 test, no augmentation

# 3. Training ------------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"   # pretrained on COCO; "yolo11s.pt" is bigger and slower
EPOCHS = 50
IMAGE_SIZE = 640
BATCH = 16
PATIENCE = 15               # stop early after this many epochs without improvement
SEED = 0
DEVICE = "auto"             # "auto" = NVIDIA GPU if present, else CPU. "mps" on an Apple Silicon Mac.
WHERE_TRAINED = "Google Colab, NVIDIA T4"

# COCO zero-shot baseline. A COCO class whose name equals one of your classes is
# matched automatically. To let a COCO class stand in for one of yours, add it as
# {"your_class": ["coco_class"]} and defend that choice in REPORT.md.
COCO_STAND_INS: dict[str, list[str]] = {}

# 4. The cost of being wrong, in euros per case --------------------------------
# A case is one hand of bananas (~2 kg) valued at the 0.75 EUR/kg production cost.
COST_FALSE_POSITIVE = 1.50  # a good hand is thrown away: 2 kg x 0.75 EUR/kg
COST_FALSE_NEGATIVE = 18.0  # a rotten hand is packed: lose the 12 kg box plus the claim, ~2 boxes x 12 kg x 0.75
COST_TRUE_POSITIVE = 0.0    # a rotten hand is thrown away: it was already worthless
COST_TRUE_NEGATIVE = 0.0
BASE_RATE = 0.01            # 1% of hands arrive rotten at the packing house

# 5. Human review --------------------------------------------------------------
CASES_PER_DAY = 25_000                # 50,000 kg a day / 2 kg per hand
SECONDS_PER_REVIEW = 5               # provisional: time yourself reviewing photos
REVIEWER_COST_PER_HOUR = 12.4          # 2026 minimum wage, 17,094 EUR / 1,808 h, plus employer social security
PRODUCTIVE_HOURS_PER_REVIEWER = 7.0    # an 8-hour shift minus breaks
REVIEWERS_AVAILABLE = 2.0              # two line operators

PLACEHOLDERS = False
