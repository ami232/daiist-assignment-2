"""
Assignment 2: training pipeline for grading spruce boards (premium vs standard).

Run with:  uv run python main.py train

Stages:
  1. Prepare the data: download, hold out a validation set, rotate test to landscape.
  2. Finetune YOLO11 on the defect classes.
  3. Evaluate on test, per class and per board.
  4. Compare against the two baselines on the same test split.
  5. Sweep the threshold and cost it, with the production base rate.
  6. Repeat the evaluation and costing on validation (same camera as training).

SKIP_TRAINING=1 reuses weights/wood_yolo11s.pt instead of training (CPU-friendly).
SMOKE_TEST=1 runs a tiny version only to check that the code works.
"""

import os
import random
import shutil
from pathlib import Path

import json
import math

import numpy as np
import pandas as pd
import yaml
from PIL import Image

import wood_grading
from api_key import load_api_key
from wood_grading import ACCEPTABLE_CLASSES, board_score, positive_ids, share_below

ROOT = Path(__file__).resolve().parent

# Roboflow Universe dataset, used as published (see REPORT.md, "Dataset").
WORKSPACE = "defect-detection-w4mv8"
PROJECT = "wood-defect-detection-9w3x4"
VERSION = 6

RAW_DIR = ROOT / "datasets" / "wood-defects-v6"  # the download, never modified
WORK_DIR = ROOT / "datasets" / "wood-work"  # what YOLO trains and evaluates on

VAL_FRACTION = 0.15  # share of train held out for validation (the dataset has no valid split)
SEED = 42  # fixed so the same images land in validation on every run

# Training settings (see REPORT.md, "Finetuning and baselines").
BASE_MODEL = "yolo11s.pt"  # COCO-pretrained checkpoint; also the zero-shot baseline
IMGSZ = 1280  # defects are tiny: at 640 the smallest are only 8-12 px wide
EPOCHS = 100
PATIENCE = 20  # stop early if validation mAP has not improved for 20 epochs
BATCH = 8  # fits a 16 GB GPU (Colab T4) at imgsz 1280
RUNS_DIR = ROOT / "runs"  # Ultralytics output, not committed
WEIGHTS_PATH = ROOT / "weights" / "wood_yolo11s.pt"  # the committed model the app loads
OUTPUT_DIR = ROOT / "outputs"  # metrics, scores and the cost sweep; committed, read by app.py
CM_CONF = 0.25  # confidence for counting a box in the confusion matrix (Ultralytics' usual value)

# Business figures (see REPORT.md, "Business problem" and "Cost matrix").
COST_FP = 0.93  # € per good board downgraded: (372 - 310) €/m³ × 0.015 m³
COST_FN = 24.0  # € per defective board reaching a premium customer (claim)
REVIEW_SECONDS = 10  # time a grader needs to check one diverted board
GRADER_EUR_PER_HOUR = 20.0  # fully loaded labour cost
COST_REVIEW = REVIEW_SECONDS / 3600 * GRADER_EUR_PER_HOUR  # ≈ €0.06 per board reviewed
BOARDS_PER_MINUTE = 40
BOARDS_PER_DAY = 36_000  # 40/min × 2 shifts × 7.5 h
GRADERS = 2
# Share of board images with any defect on the real line: 18,283 of 20,275 in the
# original dataset (Kodytek et al., 2022). The base rate also needs the share of those
# whose defect is disqualifying, which is measured on test in cost_analysis().
PAPER_DEFECT_SHARE = 18_283 / 20_275
THRESHOLDS = np.round(np.linspace(0, 1, 201), 3)  # 0.000, 0.005, ..., 1.000

# SMOKE_TEST=1 runs a tiny version of the pipeline (1 epoch, 5% of train, small
# images) only to check that the code works end to end. Never used for results.
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"
if SMOKE_TEST:
    IMGSZ, EPOCHS, BATCH = 320, 1, 4


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
def download_dataset() -> Path:
    """Download the dataset once. Later runs reuse the local copy."""
    if (RAW_DIR / "data.yaml").exists():
        print(f"Dataset already downloaded in {RAW_DIR}")
        return RAW_DIR
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov11", location=str(RAW_DIR))
    return RAW_DIR


def link_pair(image: Path, split_dir: Path) -> None:
    """Make an image and its label visible in a split without copying the files."""
    label = image.parent.parent / "labels" / (image.stem + ".txt")
    for src, sub in ((image, "images"), (label, "labels")):
        dst = split_dir / sub / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())


def rotate_pair_to_landscape(image: Path, split_dir: Path) -> bool:
    """
    Rotate a portrait image 90° counter-clockwise so the board lies horizontally,
    as in the training images, and rotate its boxes with it.

    YOLO boxes are (class, x_centre, y_centre, width, height), all as fractions of
    the image. After a 90° counter-clockwise turn, a point (x, y) moves to
    (y, 1 - x), and width and height swap.
    Returns True if the image was rotated, False if it was already landscape.
    """
    label = image.parent.parent / "labels" / (image.stem + ".txt")
    out_img = split_dir / "images" / image.name
    out_lbl = split_dir / "labels" / label.name
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    im = Image.open(image)
    if im.height <= im.width:  # already landscape: keep as is
        out_img.symlink_to(image.resolve())
        out_lbl.symlink_to(label.resolve())
        return False

    im.transpose(Image.Transpose.ROTATE_90).save(out_img, quality=95)
    rotated = []
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = line.split()[:5]
        x, y, w, h = map(float, (x, y, w, h))
        rotated.append(f"{cls} {y:.6f} {1 - x:.6f} {h:.6f} {w:.6f}")
    out_lbl.write_text("\n".join(rotated) + ("\n" if rotated else ""))
    return True


def prepare_data() -> Path:
    """Build WORK_DIR with train / valid / test ready for YOLO and return its data.yaml."""
    raw = download_dataset()
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)  # rebuilt from scratch on every run, so it is always reproducible

    # Hold out a validation set from train, with a fixed seed.
    train_images = sorted((raw / "train" / "images").glob("*"))
    random.Random(SEED).shuffle(train_images)
    n_val = round(len(train_images) * VAL_FRACTION)
    for img in train_images[:n_val]:
        link_pair(img, WORK_DIR / "valid")
    for img in train_images[n_val:]:
        link_pair(img, WORK_DIR / "train")

    # Test: same images, rotated to the training orientation.
    test_images = sorted((raw / "test" / "images").glob("*"))
    n_rotated = sum(rotate_pair_to_landscape(img, WORK_DIR / "test") for img in test_images)

    names = yaml.safe_load((raw / "data.yaml").read_text())["names"]
    data_yaml = WORK_DIR / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(WORK_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(names)),
    }, sort_keys=False))

    print(f"Prepared {WORK_DIR}: train={len(train_images) - n_val}, valid={n_val}, "
          f"test={len(test_images)} ({n_rotated} rotated to landscape)")
    return data_yaml


# ---------------------------------------------------------------------------
# 2. Finetuning
# ---------------------------------------------------------------------------
def train_model(data_yaml: Path) -> Path:
    """Finetune the COCO-pretrained model on the wood defects and save the best weights."""
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)  # starts from COCO weights, not from scratch
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        seed=SEED,
        deterministic=True,
        # Augmentation: Ultralytics defaults, plus vertical flips. A board can pass
        # the camera either way round, and I do not know which way the test images
        # were originally turned, so up-down and left-right must both look normal.
        flipud=0.5,
        fliplr=0.5,
        fraction=0.05 if SMOKE_TEST else 1.0,
        project=str(RUNS_DIR),
        name="wood",
        exist_ok=True,
        plots=True,
    )
    # Ultralytics keeps the checkpoint with the best validation mAP as best.pt.
    best = Path(model.trainer.best)
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_PATH)
    print(f"Best weights copied to {WEIGHTS_PATH}")
    return WEIGHTS_PATH


# ---------------------------------------------------------------------------
# 3. Evaluation on the test split
# ---------------------------------------------------------------------------
def image_labels(label_dir: Path, names: dict[int, str]) -> pd.DataFrame:
    """Ground truth per test image, by my business rule: positive if it has any disqualifying defect."""
    pos = positive_ids(names)
    acceptable = {i for i, n in names.items() if n in ACCEPTABLE_CLASSES}
    rows = []
    for f in sorted(label_dir.glob("*.txt")):
        classes = {int(line.split()[0]) for line in f.read_text().splitlines() if line.strip()}
        if classes & pos:
            group = "positive"
        elif classes & acceptable:
            group = "acceptable defects only"
        else:
            group = "no defect"
        rows.append({"image": f.stem, "y_true": int(bool(classes & pos)), "group": group})
    return pd.DataFrame(rows)


def score_images(model, image_dir: Path, pos_ids: set[int] | None) -> pd.DataFrame:
    """Run the model on every test image and keep one score per board (see wood_grading.board_score)."""
    rows = []
    # conf=0.001 keeps even very unsure detections, so the threshold sweep can explore the full range.
    for result in model.predict(source=str(image_dir), imgsz=IMGSZ, conf=0.001, stream=True, verbose=False):
        rows.append({"image": Path(result.path).stem, "score": board_score(result, pos_ids)})
    return pd.DataFrame(rows)


def evaluate(weights: Path, data_yaml: Path) -> pd.DataFrame:
    """Detection metrics per class, the confusion matrix, and one score per test image."""
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(str(weights))
    names = model.names

    # (a) Box-level metrics: AP50, mAP50-95, and precision / recall per class.
    metrics = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
                        project=str(RUNS_DIR), name="test", exist_ok=True, plots=False, verbose=False)
    per_class = pd.DataFrame(metrics.summary())
    positive_names = {names[i] for i in positive_ids(names)}
    per_class["label"] = per_class["Class"].map(
        lambda c: "positive" if c in positive_names else "acceptable" if c in ACCEPTABLE_CLASSES else "excluded")
    per_class.to_csv(OUTPUT_DIR / "per_class.csv", index=False)

    # (b) Confusion matrix between classes. Rows = predicted, columns = true;
    # the last row/column is "background" (a missed defect or a box on clean wood).
    cm_metrics = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH, conf=CM_CONF,
                           project=str(RUNS_DIR), name="test_cm", exist_ok=True, plots=True, verbose=False)
    labels = [names[i] for i in range(len(names))] + ["background"]
    cm = pd.DataFrame(cm_metrics.confusion_matrix.matrix.astype(int), index=labels, columns=labels)
    cm.index.name = "predicted \\ true"
    cm.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    shutil.copy(RUNS_DIR / "test_cm" / "confusion_matrix_normalized.png", OUTPUT_DIR / "confusion_matrix.png")

    # (c) One score per board, by my business rule, plus the true label.
    test_dir = data_yaml.parent / "test"
    scores = score_images(model, test_dir / "images", positive_ids(names))
    scores = image_labels(test_dir / "labels", names).merge(scores, on="image")
    scores.to_csv(OUTPUT_DIR / "test_scores_finetuned.csv", index=False)

    summary = {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
               "imgsz": IMGSZ, "epochs_max": EPOCHS, "base_model": BASE_MODEL,
               "smoke_test": SMOKE_TEST}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Test mAP50={summary['mAP50']:.3f}  mAP50-95={summary['mAP50-95']:.3f}")
    print(per_class[["Class", "label", "Instances", "Box-P", "Box-R", "mAP50"]].to_string(index=False))
    return scores


# ---------------------------------------------------------------------------
# 4. Baselines, on the same test split
# ---------------------------------------------------------------------------
def roc_auc(y_true: pd.Series, score: pd.Series) -> float:
    """
    Probability that a random defective board scores higher than a random good one
    (0.5 = no better than chance, 1 = perfect ranking). It does not depend on any
    threshold, so it compares models fairly before the costs pick one.
    """
    ranks = score.rank()  # ties get their average rank
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def image_metrics(scores: pd.DataFrame, threshold: float) -> dict:
    """Board-level confusion counts and rates when every board scoring >= threshold is called defective."""
    flagged = scores["score"] >= threshold
    defective = scores["y_true"] == 1
    tp, fp = int((flagged & defective).sum()), int((flagged & ~defective).sum())
    fn, tn = int((~flagged & defective).sum()), int((~flagged & ~defective).sum())
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,  # share of defective boards caught
        "precision": tp / (tp + fp) if tp + fp else 0.0,  # share of flagged boards that really are defective
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,  # share of good boards wrongly flagged
    }


def run_baselines(data_yaml: Path, finetuned: pd.DataFrame, map50_finetuned: float) -> pd.DataFrame:
    """
    Compare the finetuned model with:
    - COCO zero-shot: the pretrained checkpoint with no finetuning. COCO has no wood
      defect classes, so it cannot score any box (mAP = 0 by construction). At board
      level I let *any* detection count as a defect.
    - Trivial: predict nothing. Every board passes as premium (the majority class in test).
    """
    from ultralytics import YOLO

    test_dir = data_yaml.parent / "test"
    truth = finetuned[["image", "y_true", "group"]]

    coco = truth.merge(score_images(YOLO(BASE_MODEL), test_dir / "images", pos_ids=None), on="image")
    coco.to_csv(OUTPUT_DIR / "test_scores_coco.csv", index=False)
    trivial = truth.assign(score=0.0)
    trivial.to_csv(OUTPUT_DIR / "test_scores_trivial.csv", index=False)

    rows = []
    for name, scores, map50 in (("finetuned yolo11s", finetuned, map50_finetuned),
                                ("COCO zero-shot", coco, 0.0),
                                ("trivial: predict nothing", trivial, 0.0)):
        rows.append({"model": name, "mAP50 (boxes)": map50,
                     "ROC AUC (boards)": roc_auc(scores["y_true"], scores["score"]),
                     **image_metrics(scores, threshold=CM_CONF)})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "baselines.csv", index=False)
    print(table.round(3).to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# 5. Cost of each threshold, and the operating point
# ---------------------------------------------------------------------------
def expected_cost(pos: np.ndarray, neg: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  base_rate: float, cost_fn: float = COST_FN, cost_fp: float = COST_FP,
                  cost_review: float = COST_REVIEW) -> tuple[np.ndarray, np.ndarray]:
    """wood_grading.expected_cost with this project's cost figures as defaults."""
    return wood_grading.expected_cost(pos, neg, lo, hi, base_rate, cost_fn, cost_fp, cost_review)


def graders_needed(review_share: float) -> int:
    """Graders needed to keep up with the review band at full line speed."""
    reviews_per_minute = review_share * BOARDS_PER_MINUTE
    return math.ceil(reviews_per_minute * REVIEW_SECONDS / 60 - 1e-9)


def best_two_thresholds(pos, neg, base_rate, max_review_share=1.0, **costs) -> dict:
    """Search every (lo, hi) pair on the grid for the cheapest one within the review capacity."""
    lo, hi = np.meshgrid(THRESHOLDS, THRESHOLDS, indexing="ij")
    cost, review = expected_cost(pos, neg, lo, hi, base_rate, **costs)
    cost = np.where((lo <= hi) & (review <= max_review_share + 1e-12), cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return {"lo": float(THRESHOLDS[i]), "hi": float(THRESHOLDS[j]),
            "cost_per_1000": float(cost[i, j] * 1000), "review_share": float(review[i, j])}


def production_base_rate(data_yaml: Path) -> tuple[float, float]:
    """
    Share of boards on the production line that fail premium:
    P(any defect) × P(disqualifying | any defect).

    The first factor comes from the original paper. The second comes from the train and
    validation labels, because those images come from the same line and camera as the
    paper's figure. The test images come from a different source (see REPORT.md).
    Returns (base_rate, disqualifying_share).
    """
    names = yaml.safe_load(data_yaml.read_text())["names"]
    labels = pd.concat([image_labels(data_yaml.parent / split / "labels", names) for split in ("train", "valid")])
    with_defect = labels["group"] != "no defect"
    disqualifying_share = float(labels.loc[with_defect, "y_true"].mean())
    return PAPER_DEFECT_SHARE * disqualifying_share, disqualifying_share


def plot_cost_curve(sweep: pd.DataFrame, best_t: float, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["cost_per_1000"], color="#1f77b4")
    best = sweep.loc[sweep["threshold"] == best_t].iloc[0]
    ax.scatter([best_t], [best["cost_per_1000"]], color="#d62728", zorder=3,
               label=f"chosen threshold {best_t:.3f} (€{best['cost_per_1000']:.0f} per 1,000 boards)")
    ax.set_xlabel("confidence threshold (board fails if score ≥ threshold)")
    ax.set_ylabel("expected cost per 1,000 boards (€)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_analysis(scores: pd.DataFrame, base_rate: float, disqualifying_share: float,
                  suffix: str = "", label: str = "test") -> dict:
    """
    Sweep the threshold, pick the operating point, test sensitivity, price the alternatives.
    The model's error rates come from `scores` (one split); the base rate is the production one.
    Files are written with `suffix` (e.g. "_val") so several splits can be compared.
    """
    pos = np.sort(scores.loc[scores["y_true"] == 1, "score"].to_numpy())
    neg = np.sort(scores.loc[scores["y_true"] == 0, "score"].to_numpy())

    # (a) One threshold: the model decides alone.
    cost, _ = expected_cost(pos, neg, THRESHOLDS, THRESHOLDS, base_rate)
    sweep = pd.DataFrame({"threshold": THRESHOLDS, "cost_per_1000": cost * 1000,
                          "recall": 1 - share_below(pos, THRESHOLDS),
                          "false_positive_rate": 1 - share_below(neg, THRESHOLDS)})
    sweep.to_csv(OUTPUT_DIR / f"cost_sweep{suffix}.csv", index=False)
    best_t = float(sweep.loc[sweep["cost_per_1000"].idxmin(), "threshold"])
    plot_cost_curve(sweep, best_t, OUTPUT_DIR / f"cost_curve{suffix}.png",
                    f"Cost against threshold, model deciding alone ({label})")

    # (b) Two thresholds with a review band, with and without the limit of the current graders.
    capacity = GRADERS * (60 / REVIEW_SECONDS) / BOARDS_PER_MINUTE  # max share of boards they can review
    two_free = best_two_thresholds(pos, neg, base_rate)
    two_capped = best_two_thresholds(pos, neg, base_rate, max_review_share=capacity)
    for band in (two_free, two_capped):
        band["reviews_per_day"] = round(band["review_share"] * BOARDS_PER_DAY)
        band["graders_needed"] = graders_needed(band["review_share"])

    # (c) Sensitivity: how the operating point moves if my uncertain figures are wrong.
    rows = []
    for ratio in (10, 25, 50):
        for rate in (0.40, round(base_rate, 3), 0.80):
            c, _ = expected_cost(pos, neg, THRESHOLDS, THRESHOLDS, rate, cost_fn=ratio * COST_FP)
            band = best_two_thresholds(pos, neg, rate, max_review_share=capacity, cost_fn=ratio * COST_FP)
            rows.append({"FN/FP ratio": ratio, "base_rate": rate,
                         "best_single_threshold": float(THRESHOLDS[np.argmin(c)]),
                         "single_cost_per_1000": float(c.min() * 1000),
                         "band_lo": band["lo"], "band_hi": band["hi"],
                         "band_cost_per_1000": band["cost_per_1000"]})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"sensitivity{suffix}.csv", index=False)

    # (d) The design against the alternatives the brief asks for.
    alternatives = pd.DataFrame([
        {"option": "do nothing: every board sold as premium", "cost_per_1000": base_rate * COST_FN * 1000,
         "review_share": 0.0},
        {"option": "do nothing: every board sold as standard", "cost_per_1000": (1 - base_rate) * COST_FP * 1000,
         "review_share": 0.0},
        {"option": "a person checks every board", "cost_per_1000": COST_REVIEW * 1000, "review_share": 1.0},
        {"option": "model decides alone", "cost_per_1000": float(sweep["cost_per_1000"].min()),
         "review_share": 0.0},
        {"option": "model + review band (2 graders)", "cost_per_1000": two_capped["cost_per_1000"],
         "review_share": two_capped["review_share"]},
    ])
    alternatives["graders_needed"] = alternatives["review_share"].map(graders_needed)
    alternatives.to_csv(OUTPUT_DIR / f"alternatives{suffix}.csv", index=False)

    result = {
        "costs": {"FP": COST_FP, "FN": COST_FN, "review": round(COST_REVIEW, 4)},
        "line": {"boards_per_minute": BOARDS_PER_MINUTE, "boards_per_day": BOARDS_PER_DAY,
                 "graders": GRADERS, "review_seconds": REVIEW_SECONDS, "review_capacity_share": capacity},
        "evaluated_on": label,
        "base_rate": {"paper_defect_share": PAPER_DEFECT_SHARE,
                      "disqualifying_share_of_defective": disqualifying_share, "production": base_rate,
                      f"share_positive_in_{label}": float(scores["y_true"].mean())},
        "single_threshold": {"threshold": best_t, "cost_per_1000": float(sweep["cost_per_1000"].min())},
        "two_thresholds_unconstrained": two_free,
        "two_thresholds_with_2_graders": two_capped,
    }
    (OUTPUT_DIR / f"costs{suffix}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(alternatives.round(3).to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 6. Same evaluation on validation: same camera and line as training
# ---------------------------------------------------------------------------
def evaluate_validation(weights: Path, data_yaml: Path) -> pd.DataFrame:
    """
    The test split comes from a different source than train (other camera and lighting),
    so I also evaluate on the held-out validation images, which come from the training
    line. Validation picked the best epoch, so these numbers are slightly optimistic.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    metrics = model.val(data=str(data_yaml), split="val", imgsz=IMGSZ, batch=BATCH,
                        project=str(RUNS_DIR), name="val_eval", exist_ok=True, plots=False, verbose=False)
    per_class = pd.DataFrame(metrics.summary())
    positive_names = {names[i] for i in positive_ids(names)}
    per_class["label"] = per_class["Class"].map(
        lambda c: "positive" if c in positive_names else "acceptable" if c in ACCEPTABLE_CLASSES else "excluded")
    per_class.to_csv(OUTPUT_DIR / "per_class_val.csv", index=False)

    val_dir = data_yaml.parent / "valid"
    scores = image_labels(val_dir / "labels", names).merge(
        score_images(model, val_dir / "images", positive_ids(names)), on="image")
    scores.to_csv(OUTPUT_DIR / "val_scores_finetuned.csv", index=False)

    summary = {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
               "ROC AUC (boards)": roc_auc(scores["y_true"], scores["score"]),
               **image_metrics(scores, threshold=CM_CONF)}
    (OUTPUT_DIR / "metrics_val.json").write_text(json.dumps(summary, indent=2))
    print(f"Validation mAP50={summary['mAP50']:.3f}  board AUC={summary['ROC AUC (boards)']:.3f}")
    return scores


def main() -> None:
    data_yaml = prepare_data()
    # SKIP_TRAINING=1 reuses the committed weights, e.g. to rerun the evaluation without a GPU.
    if os.environ.get("SKIP_TRAINING") == "1" and WEIGHTS_PATH.exists():
        weights = WEIGHTS_PATH
    else:
        weights = train_model(data_yaml)
    finetuned = evaluate(weights, data_yaml)
    map50 = json.loads((OUTPUT_DIR / "metrics.json").read_text())["mAP50"]
    run_baselines(data_yaml, finetuned, map50)

    base_rate, disqualifying_share = production_base_rate(data_yaml)
    cost_analysis(finetuned, base_rate, disqualifying_share, suffix="", label="test")
    val_scores = evaluate_validation(weights, data_yaml)
    cost_analysis(val_scores, base_rate, disqualifying_share, suffix="_val", label="validation")


if __name__ == "__main__":
    main()
