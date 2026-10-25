"""
Assignment 2: training pipeline (skin lesion triage, YOLO11n classification).

Run with:  uv run python main.py train

Steps:
  1. Download the Roboflow dataset (version 2) into datasets/
  2. Make a second copy of the images with the dermoscope frame painted over
  3. Finetune yolo11n-cls twice: on the original images and on the frame-free ones
  4. Predict class probabilities on validation and test with both models
  5. Choose the two thresholds (pass/review/fail) for each model on VALIDATION,
     at a realistic base rate and within the dermatologists' capacity
  6. Keep the model with the lower validation cost; evaluate it on TEST against
     two baselines (zero-shot pretrained model, predict-nothing), per class
  7. Save the weights to weights/ and everything app.py needs to artifacts/
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key
from preprocess import remove_frame

WORKSPACE, PROJECT, VERSION = "riteshs-workspace-qkstg", "skin-lesion-classification-sh4sk", 2
DATA = Path("datasets/skin-lesions")
DATA_NOFRAME = Path("datasets/skin-lesions-noframe")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)
Path("weights").mkdir(exist_ok=True)

POSITIVE = ["AK", "BCC", "MEL", "SCC"]  # malignant or pre-malignant -> needs a dermatologist in person

# ---------------------------------------------------------------------------
# Business assumptions (justified in REPORT.md, "Cost matrix and operating point")
# ---------------------------------------------------------------------------
APPOINTMENT = 100   # € one in-person dermatologist visit
REVIEW = 4          # € 2 minutes of a dermatologist at ~€120/hour looking at the photo
REVIEW_MISS = 0.10  # share of cancers the dermatologist misses when judging only from a photo
MISS_COST = {       # € extra cost of finding this cancer months later instead of now
    "MEL": 20_000,  # thin melanoma removed early vs spread melanoma (surgery, scans, immunotherapy)
    "SCC": 3_000,   # bigger surgery, small risk of spread
    "BCC": 1_000,   # grows slowly, bigger excision later
    "AK": 300,      # pre-cancer: cheap to treat now, small chance it becomes SCC
}
BASE_RATE = 0.10    # share of photos in a real inbox that are malignant/pre-malignant
DAILY_PHOTOS = 500  # photos per day the service receives
REVIEWS_PER_DERM_PER_DAY = 120  # 4 hours/day of photo review at 2 minutes each
DERMATOLOGISTS = 2  # dermatologists the service has available for photo review
CAPACITY_MARGIN = 0.90  # plan to use only 90% of their capacity, leaving room for busy days

device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Download dataset
# ---------------------------------------------------------------------------
if not (DATA / "train").exists():
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("folder", location=str(DATA))

# ---------------------------------------------------------------------------
# 2. Frame-free copy: the black dermoscope frame is far more common in the
#    cancer classes, so the model can use it as a shortcut (see REPORT.md)
# ---------------------------------------------------------------------------
for f in DATA.glob("*/*/*"):
    out = DATA_NOFRAME / f.relative_to(DATA)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        remove_frame(Image.open(f)).save(out)


# ---------------------------------------------------------------------------
# 3. Finetune yolo11n-cls (same settings for both versions of the data)
# ---------------------------------------------------------------------------
def finetune(data_dir, name):
    model = YOLO("yolo11n-cls.pt")  # ImageNet-pretrained checkpoint
    model.train(
        data=str(data_dir.resolve()),
        epochs=40,
        patience=10,       # stop if validation accuracy hasn't improved for 10 epochs
        imgsz=224,         # the dataset is already 224x224
        batch=64,
        flipud=0.5,        # a lesion photo has no "up": vertical flips are as valid as horizontal ones
        device=device,
        seed=0,
        project="runs",
        name=name,
        exist_ok=True,
    )
    weights = Path(f"weights/skin_{name}.pt")
    shutil.copy(Path(model.trainer.best), weights)
    shutil.copy(Path(model.trainer.save_dir) / "results.csv", ART / f"training_curves_{name}.csv")
    return weights


VARIANTS = {"original": DATA, "noframe": DATA_NOFRAME}
weights = {name: finetune(d, name) for name, d in VARIANTS.items()}


# ---------------------------------------------------------------------------
# 4. Probabilities on validation and test
# ---------------------------------------------------------------------------
def has_frame(path):
    """True if the top-left corner of the ORIGINAL image is black (dermoscope frame)."""
    return bool(np.asarray(Image.open(path).convert("L"))[:10, :10].mean() < 20)


def predict_split(m, data_dir, split):
    files = sorted((data_dir / split).glob("*/*"))
    rows = []
    for i in range(0, len(files), 64):
        for f, r in zip(files[i:i + 64], m.predict([str(f) for f in files[i:i + 64]], imgsz=224, verbose=False)):
            probs = r.probs.data.cpu().numpy()
            rows.append({"file": str(f), "true": f.parent.name,
                         "frame": has_frame(DATA / f.relative_to(data_dir)),
                         **{m.names[k]: float(probs[k]) for k in m.names}})
    df = pd.DataFrame(rows)
    df["pred"] = df[list(m.names.values())].idxmax(axis=1)
    df["score"] = df[POSITIVE].sum(axis=1)         # P(malignant or pre-malignant)
    df["positive"] = df["true"].isin(POSITIVE).astype(int)
    return df


preds = {}
for name, d in VARIANTS.items():
    m = YOLO(str(weights[name]))
    preds[name] = {"valid": predict_split(m, d, "valid"), "test": predict_split(m, d, "test")}
CLASSES = list(YOLO(str(weights["original"])).names.values())


# ---------------------------------------------------------------------------
# 5. Costs. Every case gets one of three outcomes:
#    score <  low          -> pass   (sent home)
#    low <= score < high   -> review (dermatologist looks at the photo)
#    score >= high         -> fail   (in-person appointment)
# The test/valid sets are 55-57% malignant; a real inbox is ~10%. Each case is
# weighted so positives count for BASE_RATE of the total, then costs are per 1,000 photos.
# ---------------------------------------------------------------------------
def case_weights(df):
    n_pos, n_neg = df["positive"].sum(), (1 - df["positive"]).sum()
    return np.where(df["positive"] == 1, BASE_RATE / n_pos, (1 - BASE_RATE) / n_neg)


def outcome_costs(df):
    """€ for each case under each outcome."""
    miss = df["true"].map(MISS_COST).fillna(0).values
    pos = df["positive"].values
    # Review: the dermatologist catches 90% of cancers (-> appointment) and misses 10% (-> sent home)
    review = REVIEW + pos * ((1 - REVIEW_MISS) * APPOINTMENT + REVIEW_MISS * miss)
    return {"pass": miss, "review": review, "fail": np.full(len(df), APPOINTMENT)}


def evaluate(df, low, high):
    w, c, s = case_weights(df), outcome_costs(df), df["score"].values
    passed, failed = s < low, s >= high
    review = ~passed & ~failed
    cost = np.where(passed, c["pass"], np.where(failed, c["fail"], c["review"]))
    pos = df["positive"].values == 1
    return {
        "low": round(low, 3), "high": round(high, 3),
        "cost_per_1000": round(1000 * float(np.sum(w * cost)), 1),
        "pass_share": round(float(np.sum(w * passed)), 4),
        "review_share": round(float(np.sum(w * review)), 4),
        "fail_share": round(float(np.sum(w * failed)), 4),
        "cancers_sent_home_by_model": round(float(np.sum(w * (passed & pos)) / BASE_RATE), 4),
        "melanomas_sent_home_by_model": int(np.sum(passed & (df["true"].values == "MEL"))),
    }


# Finer steps at low scores, where the model's confident "benign" cases sit
grid = np.unique(np.round(np.r_[np.arange(0, 0.1, 0.005), np.arange(0.1, 1.0001, 0.01)], 3))


def capacity(n_derms):
    """Largest share of photos n dermatologists can review, keeping a margin for busy days."""
    return n_derms * REVIEWS_PER_DERM_PER_DAY * CAPACITY_MARGIN / DAILY_PHOTOS


def choose_thresholds(val, n_derms):
    """Cheapest (low, high) pair on validation that fits in the review capacity."""
    pairs = pd.DataFrame([evaluate(val, lo, hi) for lo in grid for hi in grid if hi >= lo])
    ok = pairs[pairs["review_share"] <= capacity(n_derms) + 1e-9]
    best = ok.loc[ok["cost_per_1000"].idxmin()]
    return float(best["low"]), float(best["high"]), float(best["cost_per_1000"])


def auc(df):
    """Probability that a random malignant case scores higher than a random benign one."""
    r = df["score"].rank()
    n1, n0 = df["positive"].sum(), (1 - df["positive"]).sum()
    return (r[df["positive"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def frame_effect(df):
    return {f"{'with' if fr else 'without'} frame": {
        "images": int(len(g)),
        "accuracy": round((g["pred"] == g["true"]).mean(), 3),
        "benign_flagged_malignant": round((g.loc[g["positive"] == 0, "score"] >= 0.5).mean(), 3),
        "malignant_recall": round((g.loc[g["positive"] == 1, "score"] >= 0.5).mean(), 3),
    } for fr, g in df.groupby("frame")}


# ---------------------------------------------------------------------------
# 6. Compare the two models and keep the one with the lower VALIDATION cost
# ---------------------------------------------------------------------------
comparison, chosen = [], {}
for name in VARIANTS:
    v, t = preds[name]["valid"], preds[name]["test"]
    low, high, val_cost = choose_thresholds(v, DERMATOLOGISTS)
    chosen[name] = (low, high)
    fe = frame_effect(t)
    comparison.append({
        "model": name,
        "val_accuracy": round((v["pred"] == v["true"]).mean(), 3),
        "val_cost_per_1000": val_cost,
        "test_accuracy": round((t["pred"] == t["true"]).mean(), 3),
        "test_auc_malignant": round(auc(t), 3),
        "test_melanoma_recall": round(((t["true"] == "MEL") & (t["pred"] == "MEL")).sum() / (t["true"] == "MEL").sum(), 3),
        "benign_flagged_with_frame": fe["with frame"]["benign_flagged_malignant"],
        "benign_flagged_without_frame": fe["without frame"]["benign_flagged_malignant"],
        "low": low, "high": high,
        "test_cost_per_1000": evaluate(t, low, high)["cost_per_1000"],
    })
comparison = pd.DataFrame(comparison)
print("\nModel comparison:\n", comparison.to_string(index=False))

BEST = comparison.loc[comparison["val_cost_per_1000"].idxmin(), "model"]
LOW, HIGH = chosen[BEST]
val, test = preds[BEST]["valid"], preds[BEST]["test"]
MODEL_FILE = str(weights[BEST])
print(f"\nKeeping the '{BEST}' model ({MODEL_FILE}): pass < {LOW}, appointment >= {HIGH}")

# Per class
per_class = []
for c in CLASSES:
    tp = ((test["pred"] == c) & (test["true"] == c)).sum()
    p, n = (test["pred"] == c).sum(), (test["true"] == c).sum()
    per_class.append({"class": c, "malignant": c in POSITIVE, "test_images": int(n),
                      "precision": round(tp / p, 3) if p else 0.0, "recall": round(tp / n, 3)})
per_class = pd.DataFrame(per_class)
confusion = pd.crosstab(test["true"], test["pred"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)

# Zero-shot baseline: the ImageNet-pretrained yolo11n-cls has 1,000 everyday
# classes and none of them is a skin lesion, so it cannot output any of ours.
zs = YOLO("yolo11n-cls.pt")
zs_labels = [zs.names[int(r.probs.top1)] for r in
             zs.predict(sorted(str(f) for f in (DATA / "test").glob("*/*")), imgsz=224, verbose=False)]
zs_top = pd.Series(zs_labels).value_counts().head(5)

# Trivial baseline: always say the most common class in TRAINING
majority = max(CLASSES, key=lambda c: len(list((DATA / "train" / c).glob("*"))))
baselines = {
    f"finetuned yolo11n-cls ({BEST})": {"accuracy": round((test["pred"] == test["true"]).mean(), 3),
                                        "auc_malignant": round(auc(test), 3)},
    f"trivial: always '{majority}' (refers nobody)": {"accuracy": round((test["true"] == majority).mean(), 3),
                                                      "auc_malignant": 0.5},
    "zero-shot ImageNet yolo11n-cls": {"accuracy": 0.0, "auc_malignant": None},
}
print("\nBaselines:", baselines)
print("Zero-shot model's most common labels on lesions:\n", zs_top)

# Single threshold (model alone) and the alternatives
single_val = pd.DataFrame([evaluate(val, t, t) for t in grid])
single_test = pd.DataFrame([evaluate(test, t, t) for t in grid])
t_alone = float(single_val.loc[single_val["cost_per_1000"].idxmin(), "low"])
ours = evaluate(test, LOW, HIGH)
alternatives = {
    "do nothing (send everyone home)": evaluate(test, 1.01, 1.01),
    "dermatologist checks every photo": evaluate(test, 0.0, 1.01),
    f"model alone (threshold {t_alone})": evaluate(test, t_alone, t_alone),
    f"model + review band ({LOW}-{HIGH})": ours,
}

# What each extra dermatologist is worth (thresholds re-chosen on validation for each)
staffing_options = []
for n in [1, 2, 3, 4, 5]:
    lo, hi, _ = choose_thresholds(val, n)
    staffing_options.append({"dermatologists": n, **evaluate(test, lo, hi)})
staffing_options = pd.DataFrame(staffing_options)

reviews_per_day = ours["review_share"] * DAILY_PHOTOS
staffing = {
    "daily_photos": DAILY_PHOTOS,
    "reviews_per_day": round(reviews_per_day, 1),
    "minutes_per_review": 2,
    "reviews_per_dermatologist_per_day": REVIEWS_PER_DERM_PER_DAY,
    "dermatologists_available": DERMATOLOGISTS,
    "planned_capacity_share": round(capacity(DERMATOLOGISTS), 3),
    "dermatologists_needed": round(reviews_per_day / REVIEWS_PER_DERM_PER_DAY, 2),
    "everyone_reviewed_dermatologists": round(DAILY_PHOTOS / REVIEWS_PER_DERM_PER_DAY, 2),
}
print("\nAlternatives (test):\n", pd.DataFrame(alternatives).T.to_string())
print("\nStaffing options (test):\n", staffing_options.to_string(index=False))
print("Staffing:", staffing)

# ---------------------------------------------------------------------------
# 7. Save for the app
# ---------------------------------------------------------------------------
val.to_csv(ART / "predictions_valid.csv", index=False)
test.to_csv(ART / "predictions_test.csv", index=False)
per_class.to_csv(ART / "per_class_test.csv", index=False)
confusion.to_csv(ART / "confusion_test.csv")
comparison.to_csv(ART / "model_comparison.csv", index=False)
single_test.to_csv(ART / "cost_sweep_test.csv", index=False)
single_val.to_csv(ART / "cost_sweep_valid.csv", index=False)
pd.DataFrame(alternatives).T.rename_axis("strategy").to_csv(ART / "alternatives_test.csv")
staffing_options.to_csv(ART / "staffing_options_test.csv", index=False)
json.dump({
    "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
    "model_file": MODEL_FILE, "remove_frame": BEST == "noframe",
    "classes": CLASSES, "positive_classes": POSITIVE,
    "low": LOW, "high": HIGH, "model_alone_threshold": t_alone,
    "appointment": APPOINTMENT, "review": REVIEW, "review_miss": REVIEW_MISS, "miss_cost": MISS_COST,
    "base_rate": BASE_RATE, "daily_photos": DAILY_PHOTOS,
    "reviews_per_derm_per_day": REVIEWS_PER_DERM_PER_DAY, "dermatologists": DERMATOLOGISTS,
    "capacity_margin": CAPACITY_MARGIN,
    "baselines": baselines, "zero_shot_top_labels": zs_top.to_dict(),
    "frame_check": frame_effect(test), "staffing": staffing,
    "training": {"model": "yolo11n-cls", "epochs_max": 40, "patience": 10, "imgsz": 224,
                 "batch": 64, "device": device, "flipud": 0.5},
}, open(ART / "config.json", "w"), indent=2, default=float)
print(f"\nSaved artifacts to {ART}/")
