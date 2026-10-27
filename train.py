"""
Assignment 2: training pipeline.

Downloads the Plant Health dataset from Roboflow, finetunes yolo11n-cls,
evaluates it against two baselines (CLIP zero-shot, and a majority-class
trivial baseline), sweeps the confidence threshold against an assumed euro
cost matrix, and saves everything app.py needs so the dashboard never has
to retrain or re-run heavy inference.
"""

import json
import os
import shutil
from pathlib import Path

from api_key import load_api_key
from roboflow import Roboflow
from ultralytics import YOLO
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
WEIGHTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["healthy", "powdery mildew", "rust"]  # canonical order we use everywhere

# --- Cost assumptions (euros). Defended in REPORT.md. ---
COST_FALSE_NEGATIVE = 50.0   # missed disease: spreads untreated, costlier fix later
COST_FALSE_POSITIVE = 5.0    # unnecessary spray: chemical + labor on one section
COST_TRUE_POSITIVE = 5.0     # correctly caught disease: same spray cost, but warranted
COST_TRUE_NEGATIVE = 0.0     # correctly left alone
ASSUMED_BASE_RATE = 0.10     # assumed share of field sections actually diseased
                             # NOTE: this is deliberately NOT the test set's own
                             # class balance (which is ~89% diseased) -- the test
                             # set was assembled to have enough disease examples to
                             # evaluate on, but a real field is mostly healthy most
                             # of the time. This assumption is defended in REPORT.md.


def download_dataset() -> Path:
    """Download the Plant Health dataset from Roboflow, clean it up, and
    return the path to its root folder (with train/val/test subfolders)."""
    rf = Roboflow(api_key=load_api_key())
    workspace = rf.workspace("simpleteam")
    project = workspace.project("plant-health-dtdb8")
    dataset = project.version(1).download("folder")
    root = Path(dataset.location)

    # Drop the junk "Unlabeled" folder from every split -- it isn't a real class.
    for split in ["train", "valid", "test"]:
        junk = root / split / "Unlabeled"
        if junk.exists():
            shutil.rmtree(junk)

    # Ultralytics classification training expects "val", not "valid".
    valid_dir = root / "valid"
    val_dir = root / "val"
    if valid_dir.exists() and not val_dir.exists():
        valid_dir.rename(val_dir)

    return root


def train_model(data_root: Path) -> Path:
    """Finetune yolo11n-cls on our data. Returns path to the best weights."""
    model = YOLO("yolo11n-cls.pt")
    results = model.train(
        data=str(data_root),
        epochs=25,
        imgsz=224,
        batch=16,
        patience=8,
        project=str(ROOT / "runs" / "classify"),
        name="leaf_disease",
        seed=0,
        verbose=True,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    committed_path = WEIGHTS_DIR / "best.pt"
    shutil.copy(best_weights, committed_path)
    return committed_path


def list_test_images(data_root: Path) -> list[tuple[Path, str]]:
    """Return [(image_path, true_class_name), ...] for every test image."""
    items = []
    test_dir = data_root / "test"
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        true_label = class_dir.name.strip().lower()
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((img_path, true_label))
    return items


def evaluate_finetuned(weights_path: Path, test_items: list[tuple[Path, str]]) -> list[dict]:
    """Run our trained model on every test image. Returns per-image records
    with the true label and the full probability vector over our 3 classes."""
    model = YOLO(str(weights_path))
    records = []
    for img_path, true_label in test_items:
        result = model.predict(str(img_path), verbose=False)[0]
        names = result.names  # index -> class name, as the model learned it
        probs = result.probs.data.tolist()
        prob_by_name = {names[i].strip().lower(): probs[i] for i in range(len(probs))}
        # Reorder into our canonical CLASS_NAMES order
        prob_vector = {c: prob_by_name.get(c, 0.0) for c in CLASS_NAMES}
        records.append({
            "image": str(img_path.relative_to(ROOT)) if ROOT in img_path.parents else str(img_path),
            "true_label": true_label,
            "probs": prob_vector,
        })
    return records


def evaluate_clip_zero_shot(test_items: list[tuple[Path, str]]) -> list[dict]:
    """Baseline A: CLIP zero-shot, prompted with our class names as plain
    text. Substitutes for 'pretrained COCO model zero-shot' from the brief,
    since COCO's object classes don't apply to a leaf-disease classification
    task -- CLIP is the equivalent 'big pretrained model, no finetuning'
    baseline for classification."""
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts = [f"a photo of a plant leaf that is {c}" for c in CLASS_NAMES]

    records = []
    for img_path, true_label in test_items:
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
        prob_vector = {CLASS_NAMES[i]: probs[i] for i in range(len(CLASS_NAMES))}
        records.append({
            "image": str(img_path.relative_to(ROOT)) if ROOT in img_path.parents else str(img_path),
            "true_label": true_label,
            "probs": prob_vector,
        })
    return records


def majority_class_baseline(train_root: Path, test_items: list[tuple[Path, str]]) -> tuple[str, list[dict]]:
    """Baseline B: always predict whichever class is most common in training."""
    counts = {
        c: len(list((train_root / c).iterdir()))
        for c in CLASS_NAMES
        if (train_root / c).is_dir()
    }
    majority_class = max(counts, key=counts.get)
    records = []
    for img_path, true_label in test_items:
        prob_vector = {c: (1.0 if c == majority_class else 0.0) for c in CLASS_NAMES}
        records.append({
            "image": str(img_path.relative_to(ROOT)) if ROOT in img_path.parents else str(img_path),
            "true_label": true_label,
            "probs": prob_vector,
        })
    return majority_class, records


def confusion_and_per_class(records: list[dict]) -> dict:
    """Compute a confusion matrix and per-class precision/recall from
    per-image probability records (argmax = predicted class)."""
    confusion = {t: {p: 0 for p in CLASS_NAMES} for t in CLASS_NAMES}
    for r in records:
        pred = max(r["probs"], key=r["probs"].get)
        confusion[r["true_label"]][pred] += 1

    per_class = {}
    for c in CLASS_NAMES:
        tp = confusion[c][c]
        fn = sum(confusion[c][p] for p in CLASS_NAMES if p != c)
        fp = sum(confusion[t][c] for t in CLASS_NAMES if t != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        per_class[c] = {"precision": precision, "recall": recall, "support": tp + fn}

    total = sum(confusion[t][p] for t in CLASS_NAMES for p in CLASS_NAMES)
    correct = sum(confusion[c][c] for c in CLASS_NAMES)
    accuracy = correct / total if total > 0 else None

    return {"confusion_matrix": confusion, "per_class": per_class, "accuracy": accuracy}


def disease_probability(probs: dict) -> float:
    """P(disease) = 1 - P(healthy). Used for the pass/review/fail threshold."""
    return 1.0 - probs.get("healthy", 0.0)


def expected_cost_per_1000(tpr: float, fpr: float) -> float:
    """Expected euro cost per 1000 cases, given a TPR/FPR operating point,
    our assumed base rate, and the cost matrix above."""
    fnr = 1 - tpr
    tnr = 1 - fpr
    cost_per_case = (
        ASSUMED_BASE_RATE * (fnr * COST_FALSE_NEGATIVE + tpr * COST_TRUE_POSITIVE)
        + (1 - ASSUMED_BASE_RATE) * (fpr * COST_FALSE_POSITIVE + tnr * COST_TRUE_NEGATIVE)
    )
    return cost_per_case * 1000


def sweep_threshold(finetuned_records: list[dict]) -> dict:
    """Sweep a single confidence threshold on P(disease) from 0 to 1,
    compute TPR/FPR/expected cost at each point, and find the minimum."""
    positives = [r for r in finetuned_records if r["true_label"] != "healthy"]
    negatives = [r for r in finetuned_records if r["true_label"] == "healthy"]

    sweep = []
    best = None
    for i in range(101):
        t = i / 100
        tp = sum(1 for r in positives if disease_probability(r["probs"]) >= t)
        fp = sum(1 for r in negatives if disease_probability(r["probs"]) >= t)
        tpr = tp / len(positives) if positives else 0.0
        fpr = fp / len(negatives) if negatives else 0.0
        cost = expected_cost_per_1000(tpr, fpr)
        point = {"threshold": t, "tpr": tpr, "fpr": fpr, "cost_per_1000": cost}
        sweep.append(point)
        if best is None or cost < best["cost_per_1000"]:
            best = point

    return {"sweep": sweep, "best_threshold": best}


def main():
    print("Downloading dataset from Roboflow...")
    data_root = download_dataset()

    print("Training yolo11n-cls...")
    weights_path = train_model(data_root)

    print("Listing test images...")
    test_items = list_test_images(data_root)

    print("Evaluating finetuned model...")
    finetuned_records = evaluate_finetuned(weights_path, test_items)
    finetuned_metrics = confusion_and_per_class(finetuned_records)

    print("Evaluating CLIP zero-shot baseline (downloads ~600MB on first run)...")
    clip_records = evaluate_clip_zero_shot(test_items)
    clip_metrics = confusion_and_per_class(clip_records)

    print("Evaluating majority-class baseline...")
    majority_class, majority_records = majority_class_baseline(data_root / "train", test_items)
    majority_metrics = confusion_and_per_class(majority_records)

    print("Sweeping confidence threshold against cost matrix...")
    sweep_results = sweep_threshold(finetuned_records)

    print("Saving results to outputs/eval_results.json...")
    output = {
        "class_names": CLASS_NAMES,
        "dataset": {
            "workspace": "simpleteam",
            "project": "plant-health-dtdb8",
            "version": 1,
        },
        "cost_assumptions": {
            "false_negative": COST_FALSE_NEGATIVE,
            "false_positive": COST_FALSE_POSITIVE,
            "true_positive": COST_TRUE_POSITIVE,
            "true_negative": COST_TRUE_NEGATIVE,
            "assumed_base_rate": ASSUMED_BASE_RATE,
        },
        "finetuned": {
            "records": finetuned_records,
            "metrics": finetuned_metrics,
        },
        "baseline_clip_zero_shot": {
            "records": clip_records,
            "metrics": clip_metrics,
        },
        "baseline_majority_class": {
            "majority_class": majority_class,
            "records": majority_records,
            "metrics": majority_metrics,
        },
        "threshold_sweep": sweep_results,
    }
    with open(OUTPUTS_DIR / "eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nDone.")
    print(f"Weights saved to: {weights_path}")
    print(f"Finetuned accuracy: {finetuned_metrics['accuracy']:.3f}")
    print(f"CLIP zero-shot accuracy: {clip_metrics['accuracy']:.3f}")
    print(f"Majority-class accuracy: {majority_metrics['accuracy']:.3f}")
    print(f"Best threshold: {sweep_results['best_threshold']}")


if __name__ == "__main__":
    main()
