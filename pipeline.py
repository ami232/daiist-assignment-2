"""
Code shared by train.ipynb and app.py.

Both stages have to agree on which Roboflow dataset version the model is built
on, what its classes are called, and where the weights and results are saved.
Keeping that in one module means the app can never drift from the training run.
"""

import json
import math
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent

# Roboflow Universe dataset, used as published (not forked)
WORKSPACE = "defect-detection-fxtd5"
PROJECT = "weld-defect-lze1x"
VERSION = 1

# The version-1 export calls its classes "0", "1" and "2". The project lists
# 734 Bad-weld, 1109 Good-weld and 1294 Defect labels, and the export's label
# files hold exactly 734 / 1109 / 1294 boxes of class 0 / 1 / 2.
CLASS_NAMES = ["Bad-weld", "Good-weld", "Defect"]
SPLITS = ["train", "valid", "test"]

DATA_DIR = ROOT / "datasets" / f"{PROJECT}-{VERSION}"
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"

# Written by train.ipynb, read by app.py
SEAMS_CSV = ARTIFACTS_DIR / "seams.csv"
METRICS_CSV = ARTIFACTS_DIR / "detection_metrics.csv"
CONFUSION_CSV = ARTIFACTS_DIR / "detection_confusion_matrix.csv"
OPERATING_POINT_JSON = ARTIFACTS_DIR / "operating_point.json"

# --- The decision and its costs ---------------------------------------------
# One case is one weld seam. The positive class is Bad-weld: the weld needs
# repair. Every figure is in euros per weld; REPORT.md gives the sources.
LABOUR_EUR_PER_HOUR = 40.3    # Eurostat 2025, euro-area industry, incl. employer costs
BASE_RATE = 0.02              # TWI repair-rate survey: mild steel, all products
INSPECT_MINUTES = 5           # one visual check, middle of the 2-8 minute range
REPAIR_MINUTES = 15 + 30 + 10  # grind out, re-weld, re-inspect
REPAIR_MATERIAL_EUR = 10
# A bad weld that reaches the customer: the welded piece comes back, a new one is
# rushed out with priority transport, and the customer trusts the shop less.
# Priced as 1.5 times the shop's average sale.
AVERAGE_SALE_EUR = 600
ESCAPE_MULTIPLE = 1.5
HUMAN_MISS_RATE = 0.15        # share of bad welds a human inspector misses

INSPECT_EUR = INSPECT_MINUTES * LABOUR_EUR_PER_HOUR / 60
REPAIR_EUR = REPAIR_MINUTES * LABOUR_EUR_PER_HOUR / 60 + REPAIR_MATERIAL_EUR
ESCAPE_EUR = ESCAPE_MULTIPLE * AVERAGE_SALE_EUR

# Shop size. Today 3 inspectors, one per shift, check every weld: at 5 minutes a
# weld that is 3 x 90 = 270 welds a day. With the model, 1 reviewer on the day
# shift checks the welds it sends for review; night-shift welds wait until morning.
WELDS_PER_DAY = 270
CURRENT_INSPECTORS = 3
INSPECTORS = 1
SHIFT_HOURS = 7.5

DECISIONS = ["pass", "review", "fail"]
THRESHOLDS = np.round(np.arange(0, 1.001, 0.01), 2)
SCORE_CLASSES = ("Bad-weld", "Defect")


def download_dataset(api_key, data_dir=DATA_DIR):
    """
    Download the YOLO export once, then write a data file with the real class names.

    Uses Roboflow's REST API rather than the roboflow package, whose pinned
    dependencies clash with the libraries Kaggle preinstalls. Roboflow builds an
    export the first time it is asked for, so the link is polled until it is ready.
    """
    if not (data_dir / "train" / "labels").exists():
        url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/yolov11?api_key={api_key}"
        for _ in range(60):
            with urllib.request.urlopen(url, timeout=120) as response:
                link = json.load(response).get("export", {}).get("link")
            if link:
                break
            time.sleep(10)
        else:
            raise RuntimeError("Roboflow did not finish preparing the export within 10 minutes")
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / "export.zip"
        urllib.request.urlretrieve(link, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(data_dir)
        archive.unlink()
    yaml_path = data_dir / "weld.yaml"
    yaml_path.write_text(
        f"path: '{data_dir}'\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"names: {CLASS_NAMES}\n"
    )
    return yaml_path


def video_source(image_name):
    """'SampleV2_1' for a frame cut from one of the dataset's videos, else 'other'."""
    match = re.match(r"(SampleV\d_\d)_mp4", image_name)
    return match.group(1) if match else "other"


def load_labels(data_dir=DATA_DIR):
    """
    Read every YOLO label file.

    Returns two data frames: `images` has one row per image with its box count
    per class, and `boxes` has one row per box with its size as a share of the
    image (YOLO coordinates are already relative to image width and height).
    """
    image_rows, box_rows = [], []
    for split in SPLITS:
        for image_path in sorted((data_dir / split / "images").iterdir()):
            label_path = data_dir / split / "labels" / f"{image_path.stem}.txt"
            lines = label_path.read_text().split("\n") if label_path.exists() else []
            counts = dict.fromkeys(CLASS_NAMES, 0)
            for line in filter(None, (l.split() for l in lines)):
                name = CLASS_NAMES[int(line[0])]
                x, y, w, h = map(float, line[1:5])
                counts[name] += 1
                box_rows.append({"split": split, "image": image_path.name, "class": name,
                                 "x": x, "y": y, "w": w, "h": h, "area_pct": 100 * w * h})
            image_rows.append({"split": split, "image": image_path.name,
                               "source": video_source(image_path.name), **counts})
    return pd.DataFrame(image_rows), pd.DataFrame(box_rows)


def dhash(image_path, size=16):
    """
    Difference hash: shrink to (size+1) x size grey pixels and record whether
    each pixel is brighter than its right-hand neighbour. Two photos of the same
    scene differ in only a few of the size*size bits, even after resizing or
    recompression, so a small Hamming distance flags a near-duplicate.
    """
    grey = np.asarray(Image.open(image_path).convert("L").resize((size + 1, size)), dtype=np.int16)
    return (grey[:, 1:] > grey[:, :-1]).flatten()


def nearest_train_distance(images, data_dir=DATA_DIR):
    """For each valid/test image, the Hamming distance to its closest train image."""
    hashes = np.stack([dhash(data_dir / s / "images" / n) for s, n in zip(images["split"], images["image"])])
    is_train = (images["split"] == "train").to_numpy()
    train_hashes = hashes[is_train]
    distance = np.full(len(images), np.nan)
    for i in np.flatnonzero(~is_train):
        distance[i] = (train_hashes != hashes[i]).sum(axis=1).min()
    return pd.Series(distance, index=images.index, name="train_distance")


def predict_boxes(model, image_paths, conf=0.001):
    """Every box the model predicts above `conf`, one row per box, coordinates relative to the image."""
    rows = []
    results = model.predict([str(p) for p in image_paths], conf=conf, verbose=False, stream=True)
    for path, result in zip(image_paths, results):
        for (x, y, w, h), score, cls in zip(result.boxes.xywhn.tolist(), result.boxes.conf.tolist(),
                                            result.boxes.cls.tolist()):
            rows.append({"image": Path(path).name, "class": result.names[int(cls)],
                         "conf": score, "x": x, "y": y, "w": w, "h": h})
    return pd.DataFrame(rows, columns=["image", "class", "conf", "x", "y", "w", "h"])


def _corners(df):
    return np.stack([df["x"] - df["w"] / 2, df["y"] - df["h"] / 2,
                     df["x"] + df["w"] / 2, df["y"] + df["h"] / 2], axis=1)


def seam_scores(seams, predictions, score_classes=SCORE_CLASSES, min_inside=0.5):
    """
    The model's score for each labelled weld seam.

    A predicted box counts towards a seam when at least `min_inside` of its area
    lies inside the seam's box. The seam's score is the highest confidence among
    the counted boxes of `score_classes` (all classes if None), and 0 when there
    are none, so a seam the model finds nothing wrong with is passed at any
    threshold above 0.
    """
    if score_classes is not None:
        predictions = predictions[predictions["class"].isin(score_classes)]
    by_image = dict(tuple(predictions.groupby("image")))
    scores = np.zeros(len(seams))
    for image, index in seams.groupby("image").indices.items():
        pred = by_image.get(image)
        if pred is None:
            continue
        s, p = _corners(seams.iloc[index]), _corners(pred)
        overlap_w = np.clip(np.minimum(s[:, None, 2], p[None, :, 2]) - np.maximum(s[:, None, 0], p[None, :, 0]), 0, None)
        overlap_h = np.clip(np.minimum(s[:, None, 3], p[None, :, 3]) - np.maximum(s[:, None, 1], p[None, :, 1]), 0, None)
        inside = overlap_w * overlap_h / (pred["w"] * pred["h"]).to_numpy()[None, :]
        conf = np.where(inside >= min_inside, pred["conf"].to_numpy()[None, :], 0)
        scores[index] = conf.max(axis=1)
    return pd.Series(scores, index=seams.index, name="score")


def case_costs(repair=REPAIR_EUR, escape=ESCAPE_EUR, inspect=INSPECT_EUR, miss=HUMAN_MISS_RATE):
    """Euro cost of one weld for each (truth, decision) pair."""
    return {
        ("bad", "pass"): escape,
        ("bad", "review"): inspect + (1 - miss) * repair + miss * escape,
        ("bad", "fail"): repair,
        ("good", "pass"): 0.0,
        ("good", "review"): inspect,
        ("good", "fail"): repair,  # a needless repair
    }


def fixed_rates(decision):
    """Rates for a policy that gives every weld the same decision, e.g. 'pass' for doing nothing."""
    return {truth: {d: float(d == decision) for d in DECISIONS} for truth in ("bad", "good")}


def decide(scores, low, high):
    """Pass below `low`, fail at or above `high`, send for human review in between."""
    scores = np.asarray(scores)
    return np.where(scores >= high, "fail", np.where(scores >= low, "review", "pass"))


def decision_rates(scores, is_bad, low, high):
    """Share of the bad welds, and of the good welds, that end up passed, reviewed or failed."""
    decisions, is_bad = decide(scores, low, high), np.asarray(is_bad)
    return {truth: {d: float(np.mean(decisions[mask] == d)) for d in DECISIONS}
            for truth, mask in (("bad", is_bad), ("good", ~is_bad))}


def cost_per_1000(rates, base_rate=BASE_RATE, *, paid_per_shift=False, **costs):
    """
    Expected cost of 1,000 welds at the real-world base rate.

    The rates come from the test welds, where 37% are bad; reweighting them by
    `base_rate` gives what the same model would cost in a shop where far fewer
    welds are bad.

    By default a review costs only the minutes it takes. With `paid_per_shift`,
    the reviewers are on the payroll instead: whoever reviews at all is paid a
    full shift, so the per-minute review cost is swapped for whole shifts.
    """
    unit = case_costs(**costs)
    share = {"bad": base_rate, "good": 1 - base_rate}
    cost = 1000 * sum(share[t] * rates[t][d] * unit[(t, d)] for t in share for d in DECISIONS)
    if paid_per_shift:
        reviewed = review_share(rates, base_rate)
        per_minute = reviewed * costs.get("inspect", INSPECT_EUR)
        per_shift = reviewers_paid(reviewed) * SHIFT_HOURS * LABOUR_EUR_PER_HOUR / WELDS_PER_DAY
        cost += 1000 * (per_shift - per_minute)
    return cost


def review_share(rates, base_rate=BASE_RATE):
    return base_rate * rates["bad"]["review"] + (1 - base_rate) * rates["good"]["review"]


def sweep(scores, is_bad, base_rate=BASE_RATE, **costs):
    """Cost per 1,000 welds at every single threshold, with no review band."""
    rows = []
    for t in THRESHOLDS:
        rates = decision_rates(scores, is_bad, t, t)
        rows.append({"threshold": t, "cost per 1000": cost_per_1000(rates, base_rate, **costs),
                     "recall": rates["bad"]["fail"], "false alarm rate": rates["good"]["fail"]})
    return pd.DataFrame(rows)


def review_capacity(welds_per_day=WELDS_PER_DAY, inspectors=INSPECTORS):
    """Largest share of the day's welds the available inspectors can check."""
    return min(1.0, inspectors * SHIFT_HOURS * 60 / INSPECT_MINUTES / welds_per_day)


def best_band(scores, is_bad, max_review=1.0, base_rate=BASE_RATE, paid_per_shift=False, **costs):
    """The (low, high) pair with the lowest cost per 1,000 welds whose review share fits `max_review`."""
    best = None
    for i, low in enumerate(THRESHOLDS):
        for high in THRESHOLDS[i:]:
            rates = decision_rates(scores, is_bad, low, high)
            share = review_share(rates, base_rate)
            if share > max_review:
                continue
            cost = cost_per_1000(rates, base_rate, paid_per_shift=paid_per_shift, **costs)
            if best is None or cost < best["cost per 1000"]:
                best = {"low": float(low), "high": float(high), "cost per 1000": cost, "review share": share}
    return best


def staffing(share, welds_per_day=WELDS_PER_DAY):
    """Reviews a day, inspector hours a day, and full-time inspectors needed for a review share."""
    reviews = share * welds_per_day
    hours = reviews * INSPECT_MINUTES / 60
    return {"reviews per day": reviews, "inspector hours per day": hours, "inspectors needed": hours / SHIFT_HOURS}


def reviewers_paid(share, welds_per_day=WELDS_PER_DAY):
    """Whole people on the payroll for a review share: anyone who reviews at all is paid a full shift."""
    return math.ceil(round(staffing(share, welds_per_day)["inspectors needed"], 6))
