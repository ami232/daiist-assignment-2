"""
Assignment 2: training pipeline. Sick-chicken detector on a broiler farm.

Run:  uv run python main.py train

One case = one camera image. The model's "sick score" for an image is its
highest confidence that some chicken in it is diseased (0 if it finds none).
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================ MY DECISIONS
# Dataset (Roboflow Universe, public, used as-is)
WORKSPACE = "annotate-8nzza"
PROJECT = "chicken-sick-healthy"
VERSION = 9
POSITIVE_CLASS = "diseased-chicken"

# Costs in euros (reasoning in REPORT.md)
COSTS = {
    "false_alarm": 1.0,    # healthy bird flagged: ~5 min of a worker at ~EUR 12.50/h
    "miss": 404.0,         # sick bird missed: bird EUR 4 + 5% x EUR 8,000 outbreak
    "inspection": 1.0,     # sick bird flagged: the same 5-min inspection
    "review": 0.07,        # a person checks one image on a screen (~20 s)
    "base_rate": 0.02,     # share of images with a sick bird on the real farm
}
# Pass / review / fail thresholds on the sick score
LOW, HIGH = 0.007, 0.72

# Training settings
BASE_MODEL = "yolo11n.pt"   # pretrained on COCO, also the zero-shot baseline
EPOCHS = int(os.environ.get("EPOCHS", 30))
IMGSZ = 640

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "chicken_yolo11n.pt"
RESULTS = ROOT / "results"   # everything the dashboard reads


# ============================================================ COST MATHS
def cost_per_1000(scores, labels, low, high, costs=COSTS):
    """Expected euros per 1,000 images, plus the share sent to a person.

    score >= high        -> FAIL:   a worker inspects the bird
    low <= score < high  -> REVIEW: a person checks the image first (assumed correct)
    score < low          -> PASS:   nothing happens

    Rates are measured on the test images, separately for sick and healthy,
    then mixed using the farm's base rate (the test set's mix is not the farm's).
    """
    s, y = np.asarray(scores), np.asarray(labels)
    sick, healthy = s[y == 1], s[y == 0]

    caught = np.mean(sick >= high)
    sick_reviewed = np.mean((sick >= low) & (sick < high))
    missed = np.mean(sick < low)
    false_alarm = np.mean(healthy >= high)
    healthy_reviewed = np.mean((healthy >= low) & (healthy < high))

    p = costs["base_rate"]
    cost_if_sick = (caught * costs["inspection"]
                    + sick_reviewed * (costs["review"] + costs["inspection"])
                    + missed * costs["miss"])
    cost_if_healthy = false_alarm * costs["false_alarm"] + healthy_reviewed * costs["review"]

    return {
        "cost_per_1000": 1000 * (p * cost_if_sick + (1 - p) * cost_if_healthy),
        "review_share": p * sick_reviewed + (1 - p) * healthy_reviewed,
        "sick_caught": caught, "sick_reviewed": sick_reviewed, "sick_missed": missed,
        "healthy_false_alarm": false_alarm, "healthy_reviewed": healthy_reviewed,
    }


def sweep(scores, labels, costs=COSTS):
    """Try every threshold from 0 to 1 (model alone, no review band).

    Steps of 0.01, plus every score the model actually gave: many scores are
    tiny (like 0.007), and a 0.01 grid would jump right over them.
    """
    grid = np.round(np.arange(0, 1.001, 0.01), 2)
    thresholds = np.unique(np.concatenate([grid, np.asarray(scores, dtype=float)]))
    return pd.DataFrame([
        {"threshold": t, "cost_per_1000": cost_per_1000(scores, labels, t, t, costs)["cost_per_1000"]}
        for t in thresholds
    ])


# ============================================================ STEP 2: DOWNLOAD
def download():
    from roboflow import Roboflow
    from api_key import load_api_key
    rf = Roboflow(api_key=load_api_key())
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(ROOT / "datasets" / "chicken"))
    folder = Path(ds.location)

    # Rewrite data.yaml with full paths so YOLO always finds the images
    import yaml
    cfg = yaml.safe_load((folder / "data.yaml").read_text())
    test = "test" if (folder / "test" / "images").exists() else "valid"  # fall back if no test split
    cfg.update(path=str(folder), train="train/images", val="valid/images", test=f"{test}/images")
    (folder / "data.yaml").write_text(yaml.safe_dump(cfg))
    return folder, list(cfg["names"])


def labels_of(image):
    """(class id, box width, box height) for every box in an image's label file."""
    txt = image.parent.parent / "labels" / (image.stem + ".txt")
    if not txt.exists():
        return []
    rows = [line.split() for line in txt.read_text().splitlines() if line.strip()]
    return [(int(r[0]), float(r[3]), float(r[4])) for r in rows]


# ============================================================ STEP 3: DESCRIBE
def describe(folder, names):
    sick_id = names.index(POSITIVE_CLASS)
    rows = []
    for split in ["train", "valid", "test"]:
        images = list((folder / split / "images").glob("*"))
        boxes = [b for img in images for b in labels_of(img)]
        row = {"split": split, "images": len(images),
               "images with a sick bird": sum(any(c == sick_id for c, _, _ in labels_of(i)) for i in images)}
        for i, name in enumerate(names):
            sizes = [w * h for c, w, h in boxes if c == i]
            row[f"{name} boxes"] = len(sizes)
            row[f"{name} median size (% of image)"] = round(100 * float(np.median(sizes)), 1) if sizes else None
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================ STEP 4: TRAIN
def train(folder):
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(BASE_MODEL)
    model.train(data=str(folder / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, device=device,
                patience=10, seed=0, project=str(ROOT / "runs"), name="chicken", exist_ok=True)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)


# ============================================================ STEP 5: PER-CLASS RESULTS
def per_class_results(folder, names):
    from ultralytics import YOLO
    m = YOLO(str(WEIGHTS)).val(data=str(folder / "data.yaml"), split="test", imgsz=IMGSZ,
                                project=str(ROOT / "runs"), name="test", exist_ok=True)
    table = pd.DataFrame([{
        "class": names[int(c)],
        "precision": round(float(m.box.p[i]), 3), "recall": round(float(m.box.r[i]), 3),
        "AP50": round(float(m.box.ap50[i]), 3), "AP50-95": round(float(m.box.ap[i]), 3),
    } for i, c in enumerate(m.ap_class_index)])
    overall = {"mAP50": round(float(m.box.map50), 3), "mAP50-95": round(float(m.box.map), 3)}
    picture = Path(m.save_dir) / "confusion_matrix.png"
    if picture.exists():
        shutil.copy(picture, RESULTS / "confusion_matrix.png")
    return table, overall


# ============================================================ STEP 4: BASELINES (image level)
def score_test_images(folder, names):
    """One row per test image: is there really a sick bird, and each model's sick score."""
    from ultralytics import YOLO
    finetuned, coco = YOLO(str(WEIGHTS)), YOLO(BASE_MODEL)
    sick_id = names.index(POSITIVE_CLASS)
    bird_id = [k for k, v in coco.names.items() if v == "bird"][0]  # COCO has no "sick" class

    def score(model, image, class_id):
        boxes = model.predict(str(image), conf=0.001, imgsz=IMGSZ, verbose=False)[0].boxes
        conf = boxes.conf[boxes.cls == class_id]
        return float(conf.max()) if len(conf) else 0.0

    rows = []
    test = "test" if (folder / "test" / "images").exists() else "valid"
    for image in sorted((folder / test / "images").glob("*")):
        rows.append({
            "image": image.name,
            "has_sick_bird": int(any(c == sick_id for c, _, _ in labels_of(image))),
            "finetuned": score(finetuned, image, sick_id),
            "coco_zero_shot": score(coco, image, bird_id),
        })
    df = pd.DataFrame(rows)
    df["trivial_majority"] = float(df["has_sick_bird"].mean() >= 0.5)  # always the most common answer
    return df


def compare_baselines(scores):
    y = scores["has_sick_bird"]
    rows = []
    for model in ["finetuned", "coco_zero_shot", "trivial_majority"]:
        flagged = scores[model] >= 0.5
        tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        best = sweep(scores[model], y)["cost_per_1000"].min()
        rows.append({"model": model,
                     "precision (sick)": round(tp / (tp + fp), 3) if tp + fp else 0.0,
                     "recall (sick)": round(tp / (tp + fn), 3) if tp + fn else 0.0,
                     "accuracy": round(float((flagged == (y == 1)).mean()), 3),
                     "best cost per 1,000 (EUR)": round(best, 1)})
    return pd.DataFrame(rows)


# ============================================================ STEP 6: COST SWEEP
def cost_analysis(scores, overall):
    """Baselines, the threshold sweep and the chosen threshold, saved for the dashboard."""
    baselines = compare_baselines(scores)
    baselines.to_csv(RESULTS / "baselines.csv", index=False)
    curve = sweep(scores["finetuned"], scores["has_sick_bird"])
    curve.to_csv(RESULTS / "cost_curve.csv", index=False)
    best = curve.loc[curve["cost_per_1000"].idxmin()]
    (RESULTS / "summary.json").write_text(json.dumps({
        **overall, "chosen_threshold": float(best["threshold"]),
        "cost_per_1000_at_chosen": round(float(best["cost_per_1000"]), 2)}, indent=2))
    print("\n", baselines.to_string(index=False))
    print(f"\nCheapest threshold: {best['threshold']:.4f} -> EUR {best['cost_per_1000']:.2f} per 1,000 images")


# ============================================================ RUN EVERYTHING
if __name__ == "__main__":
    import sys
    RESULTS.mkdir(exist_ok=True)

    # Only the costs changed? Re-price the saved scores without retraining:
    #   uv run python train.py --analysis-only
    if "--analysis-only" in sys.argv:
        old = json.loads((RESULTS / "summary.json").read_text())
        cost_analysis(pd.read_csv(RESULTS / "test_scores.csv"),
                      {"mAP50": old["mAP50"], "mAP50-95": old["mAP50-95"]})
        sys.exit()

    print("Step 2: downloading the dataset")
    folder, names = download()

    print("Step 3: describing the dataset")
    describe(folder, names).to_csv(RESULTS / "dataset.csv", index=False)
    print("\n", describe(folder, names).to_string(index=False))

    print(f"Step 4: training {BASE_MODEL} for {EPOCHS} epochs")
    train(folder)

    print("Step 5: per-class results on the test split")
    per_class, overall = per_class_results(folder, names)
    per_class.to_csv(RESULTS / "per_class.csv", index=False)
    print("\n", per_class.to_string(index=False), "\n", overall)

    print("Step 4: baselines on the test images")
    scores = score_test_images(folder, names)
    scores.to_csv(RESULTS / "test_scores.csv", index=False)

    print("Step 6: threshold sweep")
    cost_analysis(scores, overall)