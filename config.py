"""
Business assumptions for the shelf-gap decision. train.py and app.py both read
this file, so every number here is a decision defended in REPORT.md.

Setting: a ~1,500 m2 supermarket in Madrid. Staff photograph ~150 shelf
sections twice a day. One case is one shelf slot: the decision is whether
someone should restock that spot. Confident gaps go straight to restockers'
handhelds; uncertain ones go to a shift supervisor who checks a cropped photo.
"""

POSITIVE_CLASS = "missing"  # the empty-slot class in the dataset's data.yaml

# Euros per slot.
COST_FN = 1.30      # 0.6 units/h (fast-seller skew, ~2.7x store avg) x 6 h to next manual round x 40% lost x EUR 0.90 margin
COST_FP = 0.40      # restocker at EUR 16/h loaded x 1.5 min wasted walk and check
COST_TP = 0.30      # only the extra ~1 min trip outside the round at EUR 16/h; the round would restock anyway
COST_TN = 0.0
COST_REVIEW = 0.02  # supervisor at EUR 20/h x 4 s per crop

BASE_RATE = 0.04    # test split shows 3.6%; assumes a slightly messier store

IMAGES_PER_DAY = 300       # 150 sections x 2 rounds
SLOTS_PER_IMAGE = 150      # dataset average
SECONDS_PER_REVIEW = 4
REVIEWER_SECONDS_PER_DAY = 7.5 * 3600
MAX_REVIEWERS = 1 / 7.5    # one supervisor-hour a day, in full 7.5 h days of review; None = no cap
