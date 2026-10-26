"""Cost model, in euros per case (one case = one image).
Every number is my own assumption and is justified in REPORT.md."""

C_FP = 2.5       # supervisor walks over to check a worker who wears a helmet
C_FN = 300       # a missed bare head (expected cost of the exposure)
C_TP = 2.5       # supervisor walks over and corrects a real violation
C_TN = 0         # correctly passing a helmet costs nothing
C_REVIEW = 0.25  # a person looks at one image
BASE_RATE = 0.05 # share of images with a bare head on a real site


def require_filled():
    missing = [k for k, v in globals().items() if k.isupper() and v is None]
    if missing:
        raise SystemExit(f"Fill in costs.py first, missing: {', '.join(missing)}")