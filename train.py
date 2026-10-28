"""
train.py - the whole training pipeline. Run it with:

    uv run python main.py train

What it does, in order:
  1. Downloads the dataset from Roboflow Universe (settings.py says which one).
  2. Describes it: images per class in train / valid / test.
  3. Finetunes yolo11n-cls on it and copies the best model to weights/best.pt.
  4. Scores every TEST image with three models:
       - our finetuned model
       - the pretrained COCO model, zero-shot (it knows apple / banana / orange,
         but was never taught what "rotten" looks like)
       - a trivial baseline that always answers the most common class
  5. Per-class precision / recall and the confusion matrix.
  6. Costs the mistakes and picks the thresholds (see costing.py).

Everything app.py needs is saved in results/. The app never trains anything.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import costing
import settings as S

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "best.pt"
DATA_DIR = ROOT / "datasets" / f"{S.PROJECT}-{S.VERSION}"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_FRUITS = ["apple", "banana", "orange"]  # fruit classes the COCO model already knows


def is_positive(class_name: str) -> bool:
    return S.POSITIVE_WORD.lower() in class_name.lower()


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_dataset() -> Path:
    if (DATA_DIR / "train").exists():
        print(f"Dataset already downloaded, reusing {DATA_DIR.relative_to(ROOT)}")
        return DATA_DIR
    from roboflow import Roboflow
    from api_key import load_api_key

    print(f"Downloading {S.WORKSPACE}/{S.PROJECT} version {S.VERSION} from Roboflow ...")
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(S.WORKSPACE).project(S.PROJECT).version(S.VERSION)
    version.download("folder", location=str(DATA_DIR))  # one sub-folder per class
    return DATA_DIR


def split_dirs(data_dir: Path) -> dict:
    """Roboflow calls the validation split 'valid'."""
    found = {}
    for name in ["train", "valid", "val", "test"]:
        if (data_dir / name).is_dir():
            found["valid" if name == "val" else name] = data_dir / name
    if "test" not in found:
        raise SystemExit("This dataset version has no test split. Pick a version that has one.")
    return found


def list_images(split_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() in IMAGE_TYPES:
                rows.append({"path": str(image), "true_class": class_dir.name})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Describe the dataset
# ---------------------------------------------------------------------------
def describe_dataset(splits: dict) -> pd.DataFrame:
    from PIL import Image

    counts = {}
    sizes = []
    for split, folder in splits.items():
        images = list_images(folder)
        counts[split] = images["true_class"].value_counts()
        for path in images["path"].head(200):  # a sample is enough for typical size
            with Image.open(path) as im:
                sizes.append({"split": split, "width": im.width, "height": im.height})

    table = pd.DataFrame(counts).fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    table.loc["ALL CLASSES"] = table.sum()
    table["positive"] = ["yes" if is_positive(c) else "no" for c in table.index]
    table.loc["ALL CLASSES", "positive"] = ""
    table.index.name = "class"
    table.to_csv(RESULTS / "dataset_counts.csv")

    size_table = pd.DataFrame(sizes).groupby("split")[["width", "height"]].median()
    size_table.to_csv(RESULTS / "image_sizes.csv")

    print("\nImages per class and split:")
    print(table.to_string())
    print("\nMedian image size (pixels):")
    print(size_table.to_string())
    return table


# ---------------------------------------------------------------------------
# 3. Finetune
# ---------------------------------------------------------------------------
def finetune(data_dir: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(S.BASE_MODEL)  # starts from the pretrained checkpoint
    started = time.time()
    model.train(
        data=str(data_dir),
        epochs=S.EPOCHS,
        imgsz=S.IMAGE_SIZE,
        batch=S.BATCH,
        seed=S.SEED,
        deterministic=True,
        workers=2,
        project=str(ROOT / "runs"),
        name="fruit",
        exist_ok=True,
    )
    minutes = (time.time() - started) / 60
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)
    print(f"\nTraining took {minutes:.1f} min. Best model copied to weights/best.pt")
    return {"device": str(model.trainer.device), "minutes": round(minutes, 1)}


# ---------------------------------------------------------------------------
# 4. Score the test set with the three models
# ---------------------------------------------------------------------------
def score_finetuned(test: pd.DataFrame) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))  # the committed file, exactly what the app uses
    names = model.names
    positive_ids = [i for i, n in names.items() if is_positive(n)]
    rows = []
    for path in test["path"]:
        probs = model(path, verbose=False)[0].probs.data.cpu().numpy()
        row = {f"p_{names[i]}": float(p) for i, p in enumerate(probs)}
        row["pred_class"] = names[int(np.argmax(probs))]
        row["rotten_score"] = float(probs[positive_ids].sum())  # P(any rotten class)
        rows.append(row)
    out = pd.concat([test.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["is_rotten"] = out["true_class"].map(is_positive)
    return out


def score_coco_zero_shot(test: pd.DataFrame, classes: list) -> pd.DataFrame:
    """Pretrained COCO detector, no finetuning.

    It can recognise apple / banana / orange, but has no idea what rotten
    means. So the best it can do is name the fruit and guess it is fresh.
    """
    from ultralytics import YOLO

    coco = YOLO("yolo11n.pt")
    fresh_class = {}
    for fruit in COCO_FRUITS:
        match = [c for c in classes if fruit in c.lower() and not is_positive(c)]
        fresh_class[fruit] = match[0] if match else None

    preds = []
    for path in test["path"]:
        result = coco(path, verbose=False, conf=0.05)[0]
        best_fruit, best_conf = None, 0.0
        for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = coco.names[int(cls_id)]
            if label in COCO_FRUITS and conf > best_conf:
                best_fruit, best_conf = label, conf
        preds.append(fresh_class.get(best_fruit) or "no prediction")
    out = test[["path", "true_class"]].copy()
    out["pred_class"] = preds
    out["rotten_score"] = 0.0  # it can never say "rotten"
    out["is_rotten"] = out["true_class"].map(is_positive)
    return out


def score_trivial(test: pd.DataFrame, majority: str) -> pd.DataFrame:
    """Always answers the most common class in the TRAINING set."""
    out = test[["path", "true_class"]].copy()
    out["pred_class"] = majority
    out["rotten_score"] = 1.0 if is_positive(majority) else 0.0
    out["is_rotten"] = out["true_class"].map(is_positive)
    return out


# ---------------------------------------------------------------------------
# 5. Per-class results
# ---------------------------------------------------------------------------
def per_class_table(preds: pd.DataFrame, classes: list) -> pd.DataFrame:
    rows = []
    for c in classes:
        actual = preds["true_class"] == c
        predicted = preds["pred_class"] == c
        correct = (actual & predicted).sum()
        rows.append({
            "class": c,
            "positive": is_positive(c),
            "test_images": int(actual.sum()),
            "precision": correct / predicted.sum() if predicted.sum() else np.nan,
            "recall": correct / actual.sum() if actual.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def confusion_table(preds: pd.DataFrame, classes: list) -> pd.DataFrame:
    """Rows = what the fruit really is, columns = what the model said."""
    table = pd.crosstab(preds["true_class"], preds["pred_class"])
    table = table.reindex(index=classes, columns=classes, fill_value=0)
    table.index.name = "actual \\ predicted"
    return table


def summary_row(name: str, preds: pd.DataFrame) -> dict:
    sweep = costing.sweep_single_threshold(preds["rotten_score"], preds["is_rotten"])
    best = sweep.loc[sweep["cost_per_1000"].idxmin()]
    flagged = preds["rotten_score"] >= 0.5
    rotten = preds["is_rotten"].astype(bool)
    return {
        "model": name,
        "accuracy_all_classes": (preds["pred_class"] == preds["true_class"]).mean(),
        "rotten_recall_at_0.5": (flagged & rotten).sum() / max(rotten.sum(), 1),
        "fresh_wrongly_binned_at_0.5": (flagged & ~rotten).sum() / max((~rotten).sum(), 1),
        "best_cost_per_1000": best["cost_per_1000"],
    }


def save_confusion_png(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(table.values, cmap="Blues")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, int(table.values[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix, finetuned model, test split")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    data_dir = download_dataset()
    splits = split_dirs(data_dir)
    describe_dataset(splits)

    training_info = finetune(data_dir)

    test = list_images(splits["test"])
    train_counts = list_images(splits["train"])["true_class"].value_counts()
    classes = sorted(train_counts.index)
    majority = train_counts.idxmax()

    print("\nScoring the test split with the finetuned model ...")
    finetuned = score_finetuned(test)
    print("Scoring the test split with the pretrained COCO model (zero-shot) ...")
    zero_shot = score_coco_zero_shot(test, classes)
    trivial = score_trivial(test, majority)

    saved = finetuned.copy()  # save short paths, not your computer's full folder names
    saved["path"] = saved["path"].map(lambda p: Path(p).relative_to(ROOT).as_posix())
    saved.to_csv(RESULTS / "test_predictions.csv", index=False)

    per_class = per_class_table(finetuned, classes)
    per_class.to_csv(RESULTS / "per_class.csv", index=False)
    confusion = confusion_table(finetuned, classes)
    confusion.to_csv(RESULTS / "confusion_matrix.csv")
    save_confusion_png(confusion, RESULTS / "confusion_matrix.png")

    baselines = pd.DataFrame([
        summary_row("Finetuned yolo11n-cls", finetuned),
        summary_row("Pretrained COCO yolo11n, zero-shot", zero_shot),
        summary_row(f"Trivial: always '{majority}'", trivial),
    ])
    baselines.to_csv(RESULTS / "baselines.csv", index=False)

    cost = costing.run_costing(finetuned)

    info = {
        "dataset": f"{S.WORKSPACE}/{S.PROJECT}/{S.VERSION}",
        "positive_classes": [c for c in classes if is_positive(c)],
        "majority_class_in_train": majority,
        "base_model": S.BASE_MODEL,
        "epochs": S.EPOCHS,
        "image_size": S.IMAGE_SIZE,
        "batch": S.BATCH,
        "seed": S.SEED,
        **training_info,
    }
    (RESULTS / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("Per-class results (finetuned model, test split):")
    print(per_class.round(3).to_string(index=False))
    print("\nModel vs baselines:")
    print(baselines.round(3).to_string(index=False))
    print(f"\nCheapest single threshold: {cost['best_single_threshold']:.2f}")
    print(f"Review band: pass below {cost['band_low']:.2f}, bin from {cost['band_high']:.2f}, "
          f"{cost['band_review_share']:.1%} of fruit sent to a person")
    print("Everything is saved in results/. Now run: uv run python main.py app")


if __name__ == "__main__":  # needed on Windows, where training starts helper processes
    main()
