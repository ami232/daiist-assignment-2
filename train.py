"""
Assignment 2 - training pipeline.

Run with:   uv run python main.py train

What it does, start to finish, with no manual steps:
  1. Downloads the dataset from Roboflow Universe (key read via api_key.py).
  2. Describes the dataset (images per class and per split, image sizes).
  3. Finetunes yolo11n-cls from the pretrained checkpoint.
  4. Evaluates on the TEST split: the finetuned model, the pretrained model
     used zero-shot, and a trivial majority-class baseline.
  5. Sweeps the confidence threshold, prices every point with the cost matrix
     in costing.py, and picks the thresholds on the VALIDATION split.
  6. Saves everything the dashboard needs into outputs/ and the model into
     weights/best.pt.
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO

import costing
from api_key import load_api_key

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Roboflow Universe dataset (public, used as it is, not forked)
WORKSPACE = "fruit-classification-v0e3w"
PROJECT = "fresh-and-rotten-zj6kl"
VERSION = 6

DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
WEIGHTS_DIR = ROOT / "weights"
OUT_DIR = ROOT / "outputs"

# Finetuning settings
BASE_MODEL = "yolo11n-cls.pt"  # smallest YOLO11 classifier, pretrained on ImageNet
EPOCHS = 10
IMG_SIZE = 224
BATCH = 32
SEED = 0
# Use the Mac's GPU (Apple "mps") if there is one, otherwise the CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)  # 0.00, 0.01, ..., 1.00


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def download_dataset():
    """Download the dataset in folder format: <split>/<class>/<image>.jpg"""
    if not DATA_DIR.exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download("folder", location=str(DATA_DIR))
    # Ultralytics expects the validation folder to be called "val"
    if (DATA_DIR / "valid").exists() and not (DATA_DIR / "val").exists():
        (DATA_DIR / "valid").rename(DATA_DIR / "val")


def list_images(split):
    """Return a table with one row per image: path, class name."""
    rows = []
    for class_dir in sorted(p for p in (DATA_DIR / split).iterdir() if p.is_dir()):
        for img in sorted(class_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                rows.append({"path": str(img), "true_class": class_dir.name})
    return pd.DataFrame(rows)


def is_rotten_class(name):
    return "rotten" in name.lower()


def describe_dataset():
    """Images per class and split, and typical image size."""
    counts = {}
    sizes = []
    for split in ["train", "val", "test"]:
        df = list_images(split)
        counts[split] = df["true_class"].value_counts().sort_index().to_dict()
        for path in df["path"]:
            with Image.open(path) as im:
                sizes.append(im.size)
    sizes = np.array(sizes)
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    table.loc["TOTAL"] = table.sum()
    table.to_csv(OUT_DIR / "dataset_counts.csv")
    print("\nImages per class and split:\n", table)
    return {
        "counts": counts,
        "image_width_median": int(np.median(sizes[:, 0])),
        "image_height_median": int(np.median(sizes[:, 1])),
        "note": "Classification: one fruit fills most of each image, so object "
                "size relative to the image is close to 100%.",
    }


# --------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------
def finetune():
    """Finetune yolo11n-cls on our data and copy the best weights to weights/."""
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        seed=SEED,
        device=DEVICE,
        project=str(ROOT / "runs"),
        name="finetune",
        exist_ok=True,
        plots=False,
    )
    best = ROOT / "runs" / "finetune" / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_DIR / "best.pt")
    # Keep the epoch-by-epoch training log for the report
    shutil.copy(ROOT / "runs" / "finetune" / "results.csv", OUT_DIR / "training_log.csv")
    return YOLO(str(WEIGHTS_DIR / "best.pt"))


def predict_probs(model, paths):
    """Class probabilities for every image (one row per image)."""
    probs = []
    for result in model.predict(paths, imgsz=IMG_SIZE, stream=True, verbose=False):
        probs.append(result.probs.data.cpu().numpy())
    return np.array(probs)


def score_finetuned(model, images, class_names):
    """Predicted class and rotten score = sum of P(all rotten classes)."""
    probs = predict_probs(model, list(images["path"]))
    rotten_cols = [i for i, n in enumerate(class_names) if is_rotten_class(n)]
    out = images.copy()
    out["pred_class"] = [class_names[i] for i in probs.argmax(axis=1)]
    out["score"] = probs[:, rotten_cols].sum(axis=1)
    return out


def score_zero_shot(images, class_names):
    """Pretrained ImageNet classifier, no training on our data.

    ImageNet knows 'Granny_Smith' (apple), 'banana' and 'orange', but has no idea
    of rotten. So: predicted class = fresh + whichever fruit it finds most likely,
    and rotten score = 1 - P(it looks like one of those three fruits).
    """
    model = YOLO(BASE_MODEL)
    imagenet = model.names
    fruit_ids = {
        "apple": [k for k, v in imagenet.items() if v == "Granny_Smith"][0],
        "banana": [k for k, v in imagenet.items() if v == "banana"][0],
        "orange": [k for k, v in imagenet.items() if v == "orange"][0],
    }
    fresh_class = {
        fruit: next(c for c in class_names if "fresh" in c.lower() and fruit[:5] in c.lower())
        for fruit in fruit_ids
    }
    probs = predict_probs(model, list(images["path"]))
    fruit_probs = probs[:, list(fruit_ids.values())]
    best_fruit = np.array(list(fruit_ids))[fruit_probs.argmax(axis=1)]
    out = images.copy()
    out["pred_class"] = [fresh_class[f] for f in best_fruit]
    out["score"] = 1 - fruit_probs.sum(axis=1)
    return out


def score_majority(images, train_images):
    """Trivial baseline: always predict the most common training class."""
    majority = train_images["true_class"].value_counts().idxmax()
    out = images.copy()
    out["pred_class"] = majority
    out["score"] = 1.0 if is_rotten_class(majority) else 0.0
    return out


# --------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------
def per_class_report(scored, class_names):
    """Confusion matrix (rows = true, columns = predicted) and precision/recall."""
    cm = pd.crosstab(scored["true_class"], scored["pred_class"]).reindex(
        index=class_names, columns=class_names, fill_value=0
    )
    rows = []
    for c in class_names:
        tp = cm.loc[c, c]
        precision = tp / cm[c].sum() if cm[c].sum() else 0.0
        recall = tp / cm.loc[c].sum() if cm.loc[c].sum() else 0.0
        rows.append({"class": c, "support": int(cm.loc[c].sum()),
                     "precision": round(precision, 3), "recall": round(recall, 3)})
    return cm, pd.DataFrame(rows)


def binary_metrics(scored, t):
    """Rotten-vs-fresh precision and recall at one threshold (positive = rotten)."""
    y = scored["true_class"].map(is_rotten_class).values
    pred = scored["score"].values >= t
    tp, fp = np.sum(pred & y), np.sum(pred & ~y)
    fn, tn = np.sum(~pred & y), np.sum(~pred & ~y)
    return {
        "threshold": float(t), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision_rotten": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall_rotten": round(tp / (tp + fn), 3) if tp + fn else 0.0,
    }


def original_name(path):
    """Name of the original photo an image was made from.

    Roboflow adds '.rf.<hash>' and the source data adds prefixes such as
    'rotated_by_15_' or 'saltandpepper_' to augmented copies of the same photo.
    """
    name = Path(path).name.split(".rf.")[0]
    return re.sub(r"^(rotated_by_\d+_|translation_|saltandpepper_|vertical_flip_)+", "", name)


def leakage_check(test_scored, t):
    """How many test images are copies of a photo that is also in train?

    Those copies make the test score look better than it really is, so we also
    report the model on the 'clean' test images whose photo never appears in train.
    """
    train_originals = set(list_images("train")["path"].map(original_name))
    leaked = test_scored["path"].map(original_name).isin(train_originals)
    clean = test_scored[~leaked]
    return {
        "test_images": len(test_scored),
        "test_images_copied_from_train": int(leaked.sum()),
        "clean_test_images": len(clean),
        "clean_test_per_class": clean["true_class"].value_counts().to_dict(),
        "clean_test_accuracy": round(float((clean["true_class"] == clean["pred_class"]).mean()), 3),
        "clean_test_binary_at_chosen_threshold": binary_metrics(clean, t),
    }


def cost_sweep(scored):
    """Expected cost per piece for every single threshold (no human review)."""
    y = scored["true_class"].map(is_rotten_class).values
    s = scored["score"].values
    return np.array([costing.expected_cost(s, y, t, t) for t in THRESHOLDS])


def best_band(scored, max_review_share=None):
    """Search every (t_low, t_high) pair and keep the cheapest one.

    If max_review_share is given, pairs that send more cases to people than the
    reviewers can handle are not allowed.

    Tie-break: when two pairs cost exactly the same, keep the WIDER review band.
    A band that costs nothing on the validation images is free insurance for
    production images that land between the two thresholds.
    """
    y = scored["true_class"].map(is_rotten_class).values
    s = scored["score"].values
    best = None
    for i, lo in enumerate(THRESHOLDS):
        for hi in THRESHOLDS[i:]:
            share = costing.review_share(s, y, lo, hi)
            if max_review_share is not None and share > max_review_share:
                continue
            cost = costing.expected_cost(s, y, lo, hi)
            cheaper = best is None or cost < best["cost"] - 1e-12
            same_but_wider = (best is not None and abs(cost - best["cost"]) <= 1e-12
                              and hi - lo > best["t_high"] - best["t_low"])
            if cheaper or same_but_wider:
                best = {"t_low": float(lo), "t_high": float(hi), "cost": cost, "review_share": share}
    return best


def policy_on_test(test, lo, hi):
    """Cost per 1,000 pieces, review share and staff for a (t_low, t_high) on test."""
    y = test["true_class"].map(is_rotten_class).values
    s = test["score"].values
    share = costing.review_share(s, y, lo, hi)
    return {
        "t_low": lo, "t_high": hi,
        "cost_per_1000": round(1000 * costing.expected_cost(s, y, lo, hi), 2),
        "review_share": round(share, 4),
        "reviews_per_day": round(share * costing.DAILY_VOLUME),
        "staff_needed": round(costing.staff_needed(share), 2),
    }


def plot_cost_curve(costs_val, costs_test, t_chosen, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(THRESHOLDS, 1000 * costs_val, label="validation (used to choose)")
    ax.plot(THRESHOLDS, 1000 * costs_test, label="test", linestyle="--")
    ax.axvline(t_chosen, color="red", linestyle=":", label=f"chosen t = {t_chosen:.2f}")
    ax.set_xlabel("Rotten threshold (score >= t -> discard)")
    ax.set_ylabel("Expected cost per 1,000 pieces (EUR)")
    ax.set_title("Cost against threshold (model decides alone)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("1/6 Downloading dataset ...")
    download_dataset()
    train_images, val_images, test_images = (list_images(s) for s in ["train", "val", "test"])
    class_names = sorted(train_images["true_class"].unique())

    print("2/6 Describing dataset ...")
    dataset_info = describe_dataset()

    print("3/6 Finetuning", BASE_MODEL, "...")
    model = finetune()
    assert list(model.names.values()) == class_names, "class order mismatch"

    print("4/6 Scoring test (and validation) images with the three models ...")
    scored = {
        "finetuned": score_finetuned(model, test_images, class_names),
        "zero_shot_pretrained": score_zero_shot(test_images, class_names),
        "majority_class": score_majority(test_images, train_images),
    }
    val_scored = score_finetuned(model, val_images, class_names)

    print("5/6 Evaluating and costing ...")
    model_table = []
    for name, df in scored.items():
        cm, per_class = per_class_report(df, class_names)
        cm.to_csv(OUT_DIR / f"confusion_matrix_{name}.csv")
        per_class.to_csv(OUT_DIR / f"per_class_{name}.csv", index=False)
        costs = cost_sweep(df)
        model_table.append({
            "model": name,
            "accuracy_top1": round(float((df["true_class"] == df["pred_class"]).mean()), 3),
            "min_cost_per_1000_test": round(1000 * float(costs.min()), 2),
            "best_threshold_on_test": float(THRESHOLDS[costs.argmin()]),
        })
    pd.DataFrame(model_table).to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(pd.DataFrame(model_table))

    # Single threshold: chosen on VALIDATION, reported on TEST
    costs_val = cost_sweep(val_scored)
    costs_test = cost_sweep(scored["finetuned"])
    t_chosen = float(THRESHOLDS[costs_val.argmin()])
    pd.DataFrame({"threshold": THRESHOLDS, "cost_per_piece_val": costs_val,
                  "cost_per_piece_test": costs_test}).to_csv(OUT_DIR / "cost_sweep.csv", index=False)
    plot_cost_curve(costs_val, costs_test, t_chosen, OUT_DIR / "cost_vs_threshold.png")

    # Three outcomes: review band chosen on VALIDATION, within reviewer capacity
    capacity_share = costing.REVIEW_CAPACITY_PER_DAY / costing.DAILY_VOLUME
    band_free = best_band(val_scored)
    band = best_band(val_scored, max_review_share=capacity_share)

    test = scored["finetuned"]
    alternatives = {
        "do_nothing_pass_all": policy_on_test(test, 1.01, 1.01),
        "person_checks_everything": policy_on_test(test, 0.0, 1.01),
        "model_alone": policy_on_test(test, t_chosen, t_chosen),
        "model_plus_review_band": policy_on_test(test, band["t_low"], band["t_high"]),
        "model_plus_review_unconstrained": policy_on_test(test, band_free["t_low"], band_free["t_high"]),
    }
    print(json.dumps(alternatives, indent=2))

    print("6/6 Saving outputs for the dashboard ...")
    test.to_csv(OUT_DIR / "test_scores.csv", index=False)
    val_scored.to_csv(OUT_DIR / "val_scores.csv", index=False)
    summary = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, **dataset_info},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMG_SIZE,
                     "batch": BATCH, "seed": SEED, "device": DEVICE},
        "positive_class": "rotten (any rotten* class)",
        "class_names": class_names,
        "costs": {"FN": costing.COST_FN, "FP": costing.COST_FP, "TP": costing.COST_TP,
                  "TN": costing.COST_TN, "review": costing.COST_REVIEW},
        "base_rate": costing.BASE_RATE,
        "daily_volume": costing.DAILY_VOLUME,
        "review_capacity_per_day": costing.REVIEW_CAPACITY_PER_DAY,
        "models": model_table,
        "chosen_threshold": t_chosen,
        "binary_at_chosen_threshold_test": binary_metrics(test, t_chosen),
        "leakage_check": leakage_check(test, t_chosen),
        "chosen_band": band,
        "unconstrained_band": band_free,
        "alternatives_on_test": alternatives,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("Done. Weights in weights/best.pt, results in outputs/.")


if __name__ == "__main__":
    main()
