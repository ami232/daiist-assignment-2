"""
Assignment 2: training pipeline. Run via:

    uv run python main.py train

Downloads the dataset, finetunes yolo11n on it, and evaluates the result on the
test split against two baselines: the pretrained COCO model used zero-shot, and
a model that predicts nothing. Then prices the finetuned model's test
predictions with the cost matrix in costs.py, sweeps the confidence threshold
and picks the cheapest thresholds. Saves the trained model and everything
app.py needs into weights/, so the app never trains anything.
"""

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import yaml
from matplotlib.figure import Figure
from roboflow import Roboflow
from ultralytics import YOLO

import costs
from api_key import load_api_key
from evaluation import BOX, summarise

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"

# The exact Roboflow Universe dataset this pipeline is built on. REPORT.md must
# name the same workspace, project and version.
WORKSPACE = "dataset-w8sqm"
PROJECT = "tomato-ripeness-jionb"
VERSION = 14
DATASET_DIR = ROOT / "datasets" / f"tomato-ripeness-v{VERSION}"

BASE_MODEL = "yolo11n.pt"
EPOCHS = 50
PATIENCE = 20
IMGSZ = 640
# Fits a 4 GB laptop GPU; 16 ran out of memory. Ultralytics accumulates
# gradients up to 64 images per update either way, so training barely changes.
BATCH = 8
SEED = 0

# Predictions are kept down to this confidence so the app can sweep thresholds.
MIN_CONF = 0.001
# The threshold for the precision, recall and confusion matrix printed here.
REPORT_CONF = 0.25

# COCO has no tomato class. Its nearest classes stand in for "tomato", and each
# of those boxes is given the most common class in the training labels.
COCO_STAND_INS = {"apple", "orange"}


def download_dataset() -> Path:
    """Download the dataset version in YOLOv11 format and return its folder.

    Skips the download if the folder already exists, so re-running the
    pipeline does not fetch the images again.
    """
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    dataset = version.download("yolov11", location=str(DATASET_DIR))
    return Path(dataset.location)


def read_labels(label_dir: Path) -> pd.DataFrame:
    """Labelled boxes from YOLO label files, as normalised xyxy.

    Polygon labels are reduced to their bounding box, which is what Ultralytics
    does with them when training a detector.
    """
    rows = []
    for path in sorted(label_dir.glob("*.txt")):
        for line in path.read_text().splitlines():
            values = line.split()
            if not values:
                continue
            coords = [float(v) for v in values[1:]]
            if len(coords) == 4:
                cx, cy, w, h = coords
                box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            else:
                xs, ys = coords[0::2], coords[1::2]
                box = [min(xs), min(ys), max(xs), max(ys)]
            rows.append([path.stem, int(values[0]), *box])
    return pd.DataFrame(rows, columns=["image", "cls", *BOX])


def finetune(data_yaml: Path, **overrides) -> tuple[Path, dict]:
    """Finetune BASE_MODEL from its pretrained weights.

    Returns the best checkpoint and the settings it was trained with.
    """
    settings = {
        "epochs": EPOCHS, "patience": PATIENCE, "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
        **overrides,
    }
    model = YOLO(BASE_MODEL)
    start = time.time()
    model.train(data=str(data_yaml), project=str(RUNS_DIR), name="train", exist_ok=True, **settings)
    trainer = model.trainer
    return Path(trainer.best), {
        "base_model": BASE_MODEL,
        **settings,
        "epochs_run": trainer.epoch + 1,
        "device": str(trainer.device),
        "minutes": round((time.time() - start) / 60, 1),
    }


def predict(model: YOLO, image_dir: Path, **kwargs) -> pd.DataFrame:
    """Every box the model finds in image_dir down to MIN_CONF, as normalised xyxy."""
    rows = []
    results = model.predict(
        source=str(image_dir), conf=MIN_CONF, imgsz=IMGSZ, stream=True, verbose=False, **kwargs
    )
    for result in results:
        image = Path(result.path).stem
        boxes = result.boxes
        for box, cls, conf in zip(boxes.xyxyn.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            rows.append([image, int(cls), conf, *box])
    return pd.DataFrame(rows, columns=["image", "cls", "conf", *BOX])


def coco_zero_shot(image_dir: Path, stand_in_cls: int) -> pd.DataFrame:
    """The pretrained COCO model's apple and orange boxes, relabelled as stand_in_cls."""
    coco = YOLO(BASE_MODEL)
    ids = [i for i, name in coco.names.items() if name in COCO_STAND_INS]
    preds = predict(coco, image_dir, classes=ids, agnostic_nms=True)
    preds["cls"] = stand_in_cls
    return preds


def evaluate(weights: Path, dataset_dir: Path, names: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Score the finetuned model and both baselines on the test split.

    Returns the summaries, every prediction (tagged by model) and the labels.
    """
    test_images = dataset_dir / "test" / "images"
    truths = read_labels(dataset_dir / "test" / "labels")
    majority = int(read_labels(dataset_dir / "train" / "labels")["cls"].mode()[0])

    preds = {
        "finetuned": predict(YOLO(weights), test_images),
        "coco_zero_shot": coco_zero_shot(test_images, majority),
        "predict_nothing": pd.DataFrame(columns=["image", "cls", "conf", *BOX]).astype(
            {"image": str, "cls": int, "conf": float, **dict.fromkeys(BOX, float)}
        ),
    }
    summaries = {name: summarise(p, truths, names, REPORT_CONF) for name, p in preds.items()}
    summaries["coco_zero_shot"]["stand_in_class"] = names[majority]
    tagged = pd.concat([p.assign(model=name) for name, p in preds.items() if not p.empty])
    return summaries, tagged, truths


def ultralytics_test_metrics(weights: Path, data_yaml: Path, names: list[str]) -> dict:
    """Ultralytics' own test-split metrics for the finetuned model, including mAP50-95.

    Its precision and recall are taken at the confidence that maximises F1, so
    they differ from the REPORT_CONF figures.
    """
    metrics = YOLO(weights).val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
        project=str(RUNS_DIR), name="test", exist_ok=True,
    )
    per_class = []
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class.append({
            "class": names[int(c)], "precision": float(p), "recall": float(r),
            "ap50": float(ap50), "ap50_95": float(ap),
        })
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class": per_class}


def print_results(summaries: dict, names: list[str]) -> None:
    columns = ["mAP50"] + [f"AP50 {n}" for n in names] + [f"recall {n}" for n in names]
    table = pd.DataFrame(
        {
            model: [s["map50"]]
            + [row["ap50"] for row in s["per_class"]]
            + [row["recall"] for row in s["per_class"]]
            for model, s in summaries.items()
        },
        index=columns,
    ).T
    print(f"\nTest split, recall at confidence {REPORT_CONF}:")
    print(table.round(3).to_string())


def cost_analysis() -> dict:
    """Price the finetuned model's test predictions and choose the thresholds.

    Works from the files main() saved in weights/, so it needs no model and can
    be re-run on its own after changing a figure in costs.py:

        uv run --no-sync python -c "import train; train.cost_analysis()"
    """
    names = json.loads((WEIGHTS_DIR / "metrics.json").read_text())["dataset"]["names"]
    preds = pd.read_csv(WEIGHTS_DIR / "test_predictions.csv")
    truths = pd.read_csv(WEIGHTS_DIR / "test_labels.csv")
    tomatoes = costs.tomato_scores(
        preds[preds["model"] == "finetuned"], truths, names.index(costs.POSITIVE_CLASS)
    )

    curve = costs.rates(tomatoes, costs.THRESHOLDS, costs.THRESHOLDS)
    sweep = pd.DataFrame({
        "threshold": costs.THRESHOLDS,
        "fn_rate": curve["fn"],
        "fp_rate": curve["fp"],
        "cost_per_1000": 1000 * costs.expected_cost(curve),
    })
    alone = sweep.loc[sweep["cost_per_1000"].idxmin()]
    lower, upper = costs.best_thresholds(tomatoes)
    banded = costs.rates(tomatoes, lower, upper)
    share = float(costs.review_share(banded))

    summary = {
        "assumptions": {
            "positive_class": costs.POSITIVE_CLASS, "fn_cost": costs.FN_COST, "fp_cost": costs.FP_COST,
            "base_rate": costs.BASE_RATE, "tomatoes_per_day": costs.TOMATOES_PER_DAY,
            "review_seconds": costs.REVIEW_SECONDS, "reviewer_eur_per_hour": costs.REVIEWER_EUR_PER_HOUR,
            "shift_hours": costs.SHIFT_HOURS, "review_cost_per_tomato": costs.REVIEW_COST,
        },
        "test_tomatoes": {"ripe": int(tomatoes["positive"].sum()), "not_ripe": int((~tomatoes["positive"]).sum())},
        # Nothing is ever called ripe, so every ripe tomato is shipped far and spoils.
        "do_nothing": {"cost_per_1000": 1000 * costs.BASE_RATE * costs.FN_COST},
        "person_checks_everything": {
            "cost_per_1000": 1000 * costs.REVIEW_COST, "reviewers_needed": costs.reviewers_needed(1.0),
        },
        "model_alone": {
            "threshold": float(alone["threshold"]), "cost_per_1000": float(alone["cost_per_1000"]),
            "fn_rate": float(alone["fn_rate"]), "fp_rate": float(alone["fp_rate"]),
        },
        "model_with_review": {
            "lower": lower, "upper": upper, "cost_per_1000": 1000 * float(costs.expected_cost(banded)),
            "fn_rate": float(banded["fn"]), "fp_rate": float(banded["fp"]),
            "review_share": share, "reviewers_needed": costs.reviewers_needed(share),
        },
    }
    for option in ["do_nothing", "person_checks_everything", "model_alone", "model_with_review"]:
        summary[option]["cost_per_day"] = summary[option]["cost_per_1000"] * costs.TOMATOES_PER_DAY / 1000

    tomatoes.to_csv(WEIGHTS_DIR / "tomato_scores.csv", index=False)
    sweep.round(5).to_csv(WEIGHTS_DIR / "cost_sweep.csv", index=False)
    (WEIGHTS_DIR / "costs.json").write_text(json.dumps(summary, indent=2))
    plot_cost_curve(sweep, summary, WEIGHTS_DIR / "cost_curve.png")
    print_costs(summary)
    return summary


def plot_cost_curve(sweep: pd.DataFrame, summary: dict, path: Path) -> None:
    fig = Figure(figsize=(8, 4.5), layout="tight")
    ax = fig.subplots()
    ax.plot(sweep["threshold"], sweep["cost_per_1000"], label="Model decides alone")
    ax.axhline(summary["do_nothing"]["cost_per_1000"], ls="--", color="grey", label="Do nothing")
    ax.axhline(
        summary["person_checks_everything"]["cost_per_1000"], ls=":", color="grey",
        label="Person checks everything",
    )
    band = summary["model_with_review"]
    ax.axvspan(band["lower"], band["upper"], alpha=0.15, label=f"Review band {band['lower']:.2f}-{band['upper']:.2f}")
    best = summary["model_alone"]
    ax.scatter([best["threshold"]], [best["cost_per_1000"]], color="black", zorder=3)
    ax.annotate(
        f"cheapest: {best['threshold']:.2f}", (best["threshold"], best["cost_per_1000"]),
        xytext=(6, 8), textcoords="offset points",
    )
    ax.set_xlabel("Confidence threshold for calling a tomato ripe")
    ax.set_ylabel("Expected cost per 1,000 tomatoes (EUR)")
    ax.legend()
    fig.savefig(path, dpi=150)


def print_costs(summary: dict) -> None:
    alone, band = summary["model_alone"], summary["model_with_review"]
    options = {
        "Do nothing": summary["do_nothing"],
        "Person checks every tomato": summary["person_checks_everything"],
        f"Model alone, threshold {alone['threshold']:.2f}": alone,
        f"Model + review, {band['lower']:.2f}-{band['upper']:.2f}": band,
    }
    table = pd.DataFrame(
        {name: {"EUR per 1,000": o["cost_per_1000"], "EUR per day": o["cost_per_day"]} for name, o in options.items()}
    ).T
    print(f"\nExpected cost at a base rate of {costs.BASE_RATE:.0%} ripe:")
    print(table.round(2).to_string())
    print(
        f"The review band sends {band['review_share']:.1%} of tomatoes to a person: "
        f"{band['reviewers_needed']:.1f} people on {costs.SHIFT_HOURS}-hour shifts."
    )


def main() -> None:
    dataset_dir = download_dataset()
    data_yaml = dataset_dir / "data.yaml"
    names = yaml.safe_load(data_yaml.read_text())["names"]

    best, training = finetune(data_yaml)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    weights = WEIGHTS_DIR / "best.pt"
    shutil.copy(best, weights)

    summaries, preds, truths = evaluate(weights, dataset_dir, names)
    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "names": names},
        "training": training,
        "ultralytics_test": ultralytics_test_metrics(weights, data_yaml, names),
        "models": summaries,
    }
    (WEIGHTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    preds.round(5).to_csv(WEIGHTS_DIR / "test_predictions.csv", index=False)
    truths.round(5).to_csv(WEIGHTS_DIR / "test_labels.csv", index=False)

    print_results(summaries, names)
    cost_analysis()
    print(f"\nSaved {weights.name}, metrics.json, the test predictions and the cost sweep to {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
