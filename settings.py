"""
Every assumption the pipeline and the dashboard share, in one place.

train.py and app.py both import from here, so the numbers in the dashboard
and the numbers in REPORT.md come from the same figures. Change a cost here
and restart the app: it re-costs the saved test predictions without
retraining anything.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"

# --- Data: Roboflow Universe, used as published (not forked) -----------------
WORKSPACE = "roboflow-universe-projects"
PROJECT = "basketball-players-fy4c2"
# v21 "raw-images-scoreboardClassesOnly-withNullImages": the 170 original frames,
# no offline augmentation, 114/32/24 split, frames without a scoreboard kept.
VERSION = 21
EXPORT_FORMAT = "yolov11"

# The class the decision is about. A Team Points box is what gets read and
# published as the score.
POSITIVE_CLASS = "Team Points"

# --- Training ---------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"  # yolo11s at 960 px swapped an 8 GB M1 to a halt (20 s per batch)
TRAIN_ARGS = dict(
    epochs=100,
    patience=30,
    imgsz=960,      # scoreboard digits are small; 640 shrinks them to a few pixels
    batch=4,        # keeps MPS memory well under the 8 GB the M1 shares with the OS
    fliplr=0.0,     # a mirrored scoreboard is not a scoreboard: never flip text
    seed=0,
    deterministic=True,
    plots=True,
)

# --- Matching ---------------------------------------------------------------
IOU_MATCH = 0.5       # a prediction counts as correct at IoU >= 0.5 with a same-class box
MIN_CONF = 0.001      # predictions are saved down to here so any threshold can be costed later

# --- Label audit --------------------------------------------------------------
# Frames published as "no scoreboard" that plainly show this game's scoreboard
# (checked by eye: 70-81, 42-50, 108-108). A Team Points box the model finds on
# them is counted as a false positive by the published labels although it is
# right. Costs are reported both ways; thresholds are chosen on the audited view.
# The data itself is not changed: these frames are only left out of the audited view.
MISLABELLED_NULL_FRAMES = {
    "val": [
        "youtube-158_jpg.rf.f8a72f7fc50ac50b931c671c608f8029.jpg",
        "youtube-44_jpg.rf.d7ca27d95b5ed6b49f6d8f8b7531fc30.jpg",
    ],
    "test": [
        "youtube-230_jpg.rf.59e3f2de6a94d36154e93c91d98fb8ef.jpg",
    ],
    # Same problem in train (106-106, 108-108). Reported, not acted on: the model
    # was taught that these scoreboards are background.
    "train": [
        "youtube-209_jpg.rf.49bdc6fcd76852caa2b266b676aa6fa9.jpg",
        "youtube-232_jpg.rf.67cef7c1c422efb7a2182cabc59b2ab0.jpg",
    ],
}

# Test frames copied to artifacts/examples for the dashboard: a clean read, a
# promo graphic with other games' numbers, an ESPN ticker with no game
# scoreboard, and one of the mislabelled frames.
EXAMPLE_FRAMES = [
    "youtube-48_jpg.rf.686f58e064a91f3889cea78a847fb2a4.jpg",
    "youtube-13_jpg.rf.6622569a192c63bca32587765cccbef6.jpg",
    "youtube-9_jpg.rf.0f8a04bbbe99c28abce8b8d25c08d2bf.jpg",
    "youtube-230_jpg.rf.59e3f2de6a94d36154e93c91d98fb8ef.jpg",
]

# --- Volume ---------------------------------------------------------------------
GAMES_PER_DAY = 40
FRAME_INTERVAL_S = 2          # one frame read every 2 s of broadcast
BROADCAST_MINUTES = 130       # a basketball broadcast, tip-off to final buzzer
PEAK_CONCURRENT_GAMES = 12    # evening windows: this many games run at once
REVIEW_SECONDS = 3            # one look at a cropped scoreboard
OPERATOR_HOURLY_EUR = 30.0    # loaded cost of an operator or a scout
SCORE_CHANGES_PER_TEAM = 40   # scoring plays per team per game (~100 points)

FRAMES_PER_GAME = BROADCAST_MINUTES * 60 // FRAME_INTERVAL_S
FRAMES_PER_DAY = GAMES_PER_DAY * FRAMES_PER_GAME
# Chance that a team's score changed since the previous frame. A correct reading
# only reaches an operator then, because the pipeline only asks about readings
# that would change the published score.
P_SCORE_CHANGE = SCORE_CHANGES_PER_TEAM / FRAMES_PER_GAME

# --- Cost of being wrong (EUR), per Team Points reading ------------------------
# See REPORT.md, "Cost matrix and operating point", for where each comes from.
COST_FP = 150.0       # a wrong number published to bookmakers as the score
DELAYED_UPDATE_EUR = 20.0  # a real score change that has to be typed in by hand, late
# A missed reading only matters if the score had just changed: the feed holds the
# last score, and the update has to be keyed in by hand.
COST_FN = P_SCORE_CHANGE * DELAYED_UPDATE_EUR
# 3 s of operator time per frame they look at, plus the 3 s a review delays publication.
COST_REVIEW = REVIEW_SECONDS / 3600 * OPERATOR_HOURLY_EUR * 2
COST_TP = 0.0         # a correct automatic reading costs nothing extra

# How often a sampled broadcast frame actually shows the scoreboard.
BASE_RATE = 0.80

# Doing nothing: today, one scout per game keys in every change.
SCOUT_EUR_PER_FRAME = FRAME_INTERVAL_S / 3600 * OPERATOR_HOURLY_EUR
