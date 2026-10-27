"""
Assignment 2 training pipeline: is this road snowy enough to send a salt truck?

    uv run python main.py train

Runs start to finish with no manual steps:

1. Download the dataset from Roboflow Universe (UDOT Winter Road Conditions v1).
2. Shrink the images (many are 4K video frames) so training is fast.
3. Describe the dataset: images per class and split, day/night mix, and how
   many validation/test images have a near-identical twin in training.
4. Finetune yolo11n-cls from its pretrained checkpoint, save it to weights/.
5. Score the validation and test images with the finetuned model and with the
   pretrained model used zero-shot.
6. Per-class results, confusion matrix and baselines on the test split.
7. Cost sweep and thresholds (decision.py), chosen on validation.

Everything app.py needs ends up in results/ and weights/.
"""

import json
import os
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import decision
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "datasets" / "udot"  # as downloaded (gitignored)
SMALL = ROOT / "datasets" / "udot_640"  # same images, longest side 640 px
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "road_conditions_yolo11n_cls.pt"

WORKSPACE, PROJECT, VERSION = "udot-classification", "udot-winter-road-conditions", 1
SPLITS = ["train", "valid", "test"]
POSITIVE = decision.load_assumptions()["positive_classes"]  # Snow + Slush = needs treatment

EPOCHS = int(os.environ.get("EPOCHS", 30))
IMAGE_SIZE = 224  # yolo11n-cls default
SEED = 0

NIGHT_BRIGHTNESS = 45  # mean grey level (0-255) below which we call an image "night"
TWIN_MAX_BITS = 25  # images whose 256-bit fingerprints differ in <= 25 bits look the same

# ImageNet labels the pretrained model could use to say "there is snow here".
SNOWY_IMAGENET_LABELS = ["alp", "snowplow", "snowmobile", "ski", "dogsled", "bobsled", "shovel"]


# 1. Download ---------------------------------------------------------------------------


def download() -> None:
    if (RAW / "train").is_dir():
        print(f"Dataset already in {RAW}")
        return
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("folder", location=str(RAW))


# 2. Shrink -----------------------------------------------------------------------------


def shrink(longest_side: int = 640) -> None:
    """The model only ever sees 224 px, so decoding 4K frames every epoch is
    wasted time. Labels and splits are untouched."""
    for src in RAW.glob("*/*/*.jpg"):
        dst = SMALL / src.relative_to(RAW)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(src)
        image.draft("RGB", (longest_side, longest_side))  # let the JPEG decoder skip detail we drop anyway
        image = image.convert("RGB")
        image.thumbnail((longest_side, longest_side))
        image.save(dst, quality=92)


def list_images() -> pd.DataFrame:
    rows = [
        {"split": p.parts[-3], "true_class": p.parts[-2], "file": str(p.relative_to(SMALL))}
        for p in sorted(SMALL.glob("*/*/*.jpg"))
    ]
    df = pd.DataFrame(rows)
    df["needs_treatment"] = df["true_class"].isin(POSITIVE).astype(int)
    return df


# 3. Describe ---------------------------------------------------------------------------


def brightness(path: Path) -> float:
    return float(np.asarray(Image.open(path).convert("L").resize((64, 36))).mean())


def fingerprint(path: Path) -> np.ndarray:
    """256-bit difference hash: is each pixel brighter than its right-hand
    neighbour, on a 17x16 grey thumbnail. Near-identical frames give
    near-identical fingerprints."""
    a = np.asarray(Image.open(path).convert("L").resize((17, 16)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def mark_twins(df: pd.DataFrame) -> pd.Series:
    """True for a validation/test image that has a near-identical image in train."""
    prints = np.stack([fingerprint(SMALL / f) for f in df["file"]])
    train = prints[(df["split"] == "train").to_numpy()]
    twin = np.zeros(len(df), dtype=bool)
    for i in np.where(df["split"] != "train")[0]:
        twin[i] = (train != prints[i]).sum(axis=1).min() <= TWIN_MAX_BITS
    return pd.Series(twin, index=df.index)


def describe(df: pd.DataFrame) -> dict:
    counts = pd.crosstab(df["true_class"], df["split"])[SPLITS]
    sizes = pd.Series([f"{w}x{h}" for w, h in (Image.open(p).size for p in RAW.glob("*/*/*.jpg"))])
    stats = {
        "source": f"{WORKSPACE}/{PROJECT}/{VERSION}",
        "images_per_split": df["split"].value_counts()[SPLITS].to_dict(),
        "images_per_class_and_split": counts.to_dict(orient="index"),
        "original_image_sizes": sizes.value_counts().to_dict(),
        "night_share_per_class": df.groupby("true_class")["night"].mean().round(3).to_dict(),
        "positive_classes": POSITIVE,
        "train_positive_share": float(df.loc[df["split"] == "train", "needs_treatment"].mean()),
        "test_positive_share": float(df.loc[df["split"] == "test", "needs_treatment"].mean()),
        "twin_in_train_share": {
            s: float(df.loc[df["split"] == s, "twin_in_train"].mean()) for s in ["valid", "test"]
        },
        "twin_max_bits": TWIN_MAX_BITS,
        "night_brightness_cutoff": NIGHT_BRIGHTNESS,
    }
    print("\nImages per class and split:\n", counts.to_string())
    print("Night share per class:", stats["night_share_per_class"])
    print("Validation/test images with a near-twin in train:", stats["twin_in_train_share"])
    return stats


# 4. Finetune ---------------------------------------------------------------------------


def finetune() -> None:
    from ultralytics import YOLO

    model = YOLO("yolo11n-cls.pt")  # ImageNet-pretrained checkpoint
    model.train(
        data=str(SMALL),  # uses train/ to learn and valid/ to pick the best epoch
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=64,
        seed=SEED,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="road_conditions",
        exist_ok=True,
    )
    run_dir = Path(model.trainer.save_dir)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", WEIGHTS)
    shutil.copy(run_dir / "results.csv", RESULTS / "training_log.csv")
    print(f"Saved finetuned weights to {WEIGHTS.relative_to(ROOT)}")


# 5. Score ------------------------------------------------------------------------------


def predict_probabilities(model, files: list[str]) -> np.ndarray:
    paths = [str(SMALL / f) for f in files]
    return np.stack(
        [r.probs.data.cpu().numpy() for r in model.predict(paths, imgsz=IMAGE_SIZE, stream=True, verbose=False)]
    )


def score(df: pd.DataFrame) -> pd.DataFrame:
    from ultralytics import YOLO

    df = df[df["split"] != "train"].copy()

    finetuned = YOLO(str(WEIGHTS))
    names = [finetuned.names[i] for i in range(len(finetuned.names))]
    probs = predict_probabilities(finetuned, df["file"].tolist())
    for i, name in enumerate(names):
        df[f"p_{name}"] = probs[:, i]
    df["pred_class"] = [names[i] for i in probs.argmax(axis=1)]
    df["p_treat"] = df[[f"p_{c}" for c in POSITIVE]].sum(axis=1)

    # Zero-shot baseline: the ImageNet model knows no road conditions, so the
    # closest it can say is "alp" or "snowplow". Its snow score is the total
    # probability it puts on those labels.
    pretrained = YOLO("yolo11n-cls.pt")
    snowy = [i for i, n in pretrained.names.items() if n in SNOWY_IMAGENET_LABELS]
    zs = predict_probabilities(pretrained, df["file"].tolist())
    df["zero_shot_score"] = zs[:, snowy].sum(axis=1)
    df["zero_shot_top_label"] = [pretrained.names[i] for i in zs.argmax(axis=1)]
    return df


# 6. Evaluate ---------------------------------------------------------------------------


def per_class(test: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows = []
    for c in classes:
        predicted, actual = test["pred_class"] == c, test["true_class"] == c
        hit = (predicted & actual).sum()
        rows.append(
            {
                "class": c,
                "test_images": int(actual.sum()),
                "precision": hit / max(predicted.sum(), 1),
                "recall": hit / max(actual.sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def accuracy(frame: pd.DataFrame) -> dict:
    return {"images": int(len(frame)), "accuracy": float((frame["pred_class"] == frame["true_class"]).mean())}


def binary_at(scores, y, threshold) -> dict:
    flagged = np.asarray(scores) >= threshold
    y = np.asarray(y).astype(bool)
    return {
        "threshold": float(threshold),
        "precision": float((flagged & y).sum() / max(flagged.sum(), 1)),
        "recall": float((flagged & y).sum() / max(y.sum(), 1)),
    }


def plot_confusion(matrix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, int(matrix.iat[i, j]), ha="center", va="center")
    ax.set_title("Test split confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(scored: pd.DataFrame, stats: dict) -> None:
    test = scored[scored["split"] == "test"]
    classes = sorted(scored["true_class"].unique())

    table = per_class(test, classes)
    table.to_csv(RESULTS / "per_class.csv", index=False)
    matrix = pd.crosstab(test["true_class"], test["pred_class"]).reindex(index=classes, columns=classes, fill_value=0)
    matrix.to_csv(RESULTS / "confusion_matrix.csv")
    plot_confusion(matrix, RESULTS / "confusion_matrix.png")

    majority = max(stats["images_per_class_and_split"], key=lambda c: stats["images_per_class_and_split"][c]["train"])
    metrics = {
        "finetuned": {
            "test_all": accuracy(test),
            "test_without_train_twin": accuracy(test[~test["twin_in_train"]]),
            "test_day": accuracy(test[~test["night"]]),
            "test_night": accuracy(test[test["night"]]),
        },
        "majority_class_baseline": {
            "always_predicts": majority,
            "test_accuracy": float((test["true_class"] == majority).mean()),
        },
        "zero_shot_top_labels_on_test": test["zero_shot_top_label"].value_counts().head(8).to_dict(),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nPer-class results on test:\n", table.round(3).to_string(index=False))
    print("\nConfusion matrix (rows actual, columns predicted):\n", matrix.to_string())
    print("\nAccuracy:", json.dumps(metrics["finetuned"], indent=1))
    print("Majority-class baseline accuracy:", round(metrics["majority_class_baseline"]["test_accuracy"], 3))


def add_binary_metrics(scored: pd.DataFrame, chosen: dict) -> None:
    """Precision/recall of 'needs treatment' for each approach at its own
    validation-chosen threshold, on the test split."""
    test = scored[scored["split"] == "test"]
    y = test["needs_treatment"]
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    metrics["needs_treatment_on_test"] = {
        "finetuned": binary_at(test["p_treat"], y, chosen["threshold"]),
        "zero_shot": binary_at(test["zero_shot_score"], y, chosen["zero_shot_threshold"]),
        "always_treat": binary_at(np.ones(len(y)), y, 0.5),
        "never_treat": {"precision": 0.0, "recall": 0.0},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))


def save_examples(scored: pd.DataFrame, per_class_count: int = 2) -> None:
    """A few test images the dashboard can offer as one-click examples."""
    out = RESULTS / "examples"
    out.mkdir(exist_ok=True)
    test = scored[(scored["split"] == "test") & ~scored["twin_in_train"]]
    for c, group in test.groupby("true_class"):
        for i, f in enumerate(group.sample(per_class_count, random_state=SEED)["file"]):
            image = Image.open(SMALL / f)
            image.thumbnail((480, 480))
            image.save(out / f"{c.lower()}_{i + 1}.jpg", quality=85)


# Main ----------------------------------------------------------------------------------


def main() -> None:
    started = time.time()
    RESULTS.mkdir(exist_ok=True)

    download()
    shrink()
    df = list_images()
    df["night"] = [brightness(SMALL / f) < NIGHT_BRIGHTNESS for f in df["file"]]
    df["twin_in_train"] = mark_twins(df)
    stats = describe(df)
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=int))

    finetune()

    scored = score(df)
    for split in ["valid", "test"]:
        scored[scored["split"] == split].to_csv(RESULTS / f"scores_{split}.csv", index=False)
    evaluate(scored, stats)

    print("\nCosting (thresholds chosen on validation, reported on test):")
    chosen = decision.run()
    add_binary_metrics(scored, chosen)
    save_examples(scored)
    print(f"\nDone in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
