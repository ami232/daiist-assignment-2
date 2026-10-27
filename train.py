"""
Assignment 2: training pipeline.

    uv run python main.py train                  # the real run (use a GPU, e.g. Colab T4)
    SMOKE_TEST=1 uv run python main.py train     # 1 short epoch on a slice: checks the plumbing
    uv run python train.py --evaluate-only       # reuse weights/best.pt, redo evaluation + costing

Start to finish, with no manual steps:

  1. downloads the public Roboflow dataset (api_key.load_api_key; never forked)
  2. finetunes yolo11 from the pretrained COCO checkpoint (object detection)
  3. evaluates on the TEST split with Ultralytics (AP50, mAP50-95, per-class
     precision/recall, confusion matrix) and compares with two baselines:
       - the pretrained COCO model used zero-shot
       - a trivial "predict nothing" baseline
  4. turns the model's boxes into WELD-LEVEL CASES (weld_data.collect_cases)
  5. sweeps the confidence threshold with the cost model in costing.py. The
     operating point is CHOSEN ON THE VALIDATION SPLIT and REPORTED ON THE TEST
     SPLIT, so the test number is not tuned on.
  6. saves everything the dashboard needs, so app.py never trains:
       weights/best.pt        the model (commit this)
       results/*.json|csv     metrics, cases, sweeps, chosen thresholds (commit these)
       results/report_snippets.md   tables ready to paste into REPORT.md

Settings can be overridden with environment variables: MODEL (default
yolo11n.pt), EPOCHS (100), PATIENCE (25), BATCH (16), IMGSZ (512).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

import costing
from weld_data import (IMGSZ, PROJECT, VERSION, WORKSPACE, collect_cases,
                       count_instances, download_dataset, list_images, prepare_data_yaml,
                       split_dirs)

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"

MODEL_NAME = os.environ.get("MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
PATIENCE = int(os.environ.get("PATIENCE", "25"))
BATCH = int(os.environ.get("BATCH", "16"))
SEED = 0
SMOKE = os.environ.get("SMOKE_TEST") == "1"

COCO_FLAG_CONF = 0.25          # Ultralytics' default confidence for showing a box
FN_COSTS_TO_TEST = [20, 50, 140, 300, 700, 1500]
BASE_RATES_TO_TEST = [0.001, 0.005, 0.01, 0.02, 0.05]


# ---------------------------------------------------------------------------
# 1. Train
# ---------------------------------------------------------------------------

def train_model(data_yaml: Path) -> tuple[Path, dict]:
    """Finetune from the pretrained checkpoint and copy the best weights to weights/."""
    kwargs = dict(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                  patience=PATIENCE, seed=SEED, project=str(RUNS), name="train",
                  exist_ok=True, plots=True)
    if SMOKE:
        kwargs.update(epochs=1, fraction=0.1, workers=0)
        print("SMOKE TEST: 1 epoch on 10% of the training images. Do not submit these weights.")

    model = YOLO(MODEL_NAME)
    started = time.time()
    model.train(**kwargs)
    minutes = (time.time() - started) / 60

    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    target = WEIGHTS_DIR / "best.pt"
    shutil.copy(best, target)
    print(f"Best weights copied to {target.relative_to(ROOT)}")

    epochs_run = None
    log = Path(model.trainer.save_dir) / "results.csv"
    if log.is_file():
        epochs_run = int(len(pd.read_csv(log)))
    config = {
        "model": MODEL_NAME, "epochs_requested": kwargs["epochs"], "epochs_run": epochs_run,
        "patience": PATIENCE, "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
        "training_minutes": round(minutes, 1),
    }
    return target, config


# ---------------------------------------------------------------------------
# 2. Ultralytics evaluation
# ---------------------------------------------------------------------------

def summarise_metrics(metrics, instances: dict[str, int]) -> dict:
    box = metrics.box
    names = metrics.names
    per_class = []
    for i, c in enumerate(box.ap_class_index):
        name = names[int(c)]
        per_class.append({
            "class": name, "instances": instances.get(name, 0),
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]), "map50_95": float(box.ap[i]),
        })
    overall = {"precision": float(box.mp), "recall": float(box.mr),
               "ap50": float(box.map50), "map50_95": float(box.map)}
    matrix = None
    try:
        matrix = np.asarray(metrics.confusion_matrix.matrix).tolist()
    except Exception:  # noqa: BLE001 - the matrix is a nice-to-have, the PNG is saved anyway
        pass
    return {"overall": overall, "per_class": per_class, "confusion_matrix": matrix,
            "confusion_matrix_labels": [names[i] for i in sorted(names)] + ["background"]}


def ultralytics_eval(weights: Path, data_yaml: Path, split: str, instances: dict[str, int]) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, imgsz=IMGSZ, conf=0.001, iou=0.6,
                        plots=True, project=str(RUNS), name=f"eval_{split}", exist_ok=True,
                        verbose=False)
    out = summarise_metrics(metrics, instances)
    # Keep the plots that go into the report.
    for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
        src = Path(metrics.save_dir) / fname
        if src.is_file():
            shutil.copy(src, RESULTS / f"{split}_{fname}")
    return out


# ---------------------------------------------------------------------------
# 3. Baselines
# ---------------------------------------------------------------------------

def coco_zero_shot(images_dir: Path, labels_dir: Path, names: list[str], a: costing.Assumptions) -> dict:
    """The pretrained COCO model, used as it comes.

    COCO has 80 everyday classes (person, car, bench ...) and none of them is a
    weld, so its per-class AP on Good Weld / Bad Weld / Defect is exactly zero by
    construction. To still say something about the DECISION, we read it as
    generously as possible: any COCO box over the confidence limit counts as
    "flag this weld".
    """
    coco = YOLO(MODEL_NAME)
    images = list_images(images_dir)
    seen: Counter = Counter()
    images_with_box = 0
    for start in range(0, len(images), 32):
        for r in coco.predict([str(p) for p in images[start:start + 32]], conf=COCO_FLAG_CONF,
                              imgsz=IMGSZ, verbose=False):
            if len(r.boxes):
                images_with_box += 1
            seen.update(r.names[int(c)] for c in r.boxes.cls)
    cases = collect_cases(coco, images_dir, labels_dir, names, is_positive_pred=lambda _n: True)
    pos, neg, sp = costing.split_scores(cases)
    e = costing.evaluate(pos, neg, sp, COCO_FLAG_CONF, COCO_FLAG_CONF, a)
    return {
        "precision": 0.0, "recall": 0.0, "ap50": 0.0, "map50_95": 0.0,
        "note": "COCO has no weld classes, so weld-class AP is zero by construction.",
        "images_with_any_box": images_with_box, "images": len(images),
        "most_common_coco_classes": seen.most_common(5),
        "flag_threshold": COCO_FLAG_CONF,
        "tpr": e["tpr_fail"], "fpr": e["fpr_fail"],
        "cost_per_1000": e["cost_per_1000"],
    }


# ---------------------------------------------------------------------------
# 4. Costing and the operating point
# ---------------------------------------------------------------------------

def brief(e: dict) -> dict:
    """The numbers worth saving from costing.evaluate()."""
    return {k: e[k] for k in ("t_low", "t_high", "counts", "tpr_fail", "tpr_review", "tpr_pass",
                              "fpr_fail", "fpr_review", "cost_per_1000", "cost_per_day",
                              "review_share", "reviews_per_day", "reviewers", "per_1000")}


def choose_operating_point(cases_val: pd.DataFrame, cases_test: pd.DataFrame,
                           a: costing.Assumptions) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    vp, vn, vs = costing.split_scores(cases_val)
    tp_, tn_, ts_ = costing.split_scores(cases_test)

    single = costing.best_single(vp, vn, vs, a)
    t = single["threshold"]
    sweep_val = single["sweep"]
    sweep_test = costing.sweep_single(tp_, tn_, ts_, a)

    capped = costing.best_band(vp, vn, vs, a, costing.MAX_REVIEWERS)
    free = costing.best_band(vp, vn, vs, a, None)

    def on_both(lo, hi):
        return {"t_low": lo, "t_high": hi,
                "val": brief(costing.evaluate(vp, vn, vs, lo, hi, a)),
                "test": brief(costing.evaluate(tp_, tn_, ts_, lo, hi, a))}

    operating = {
        "assumptions": costing.assumptions_to_dict(a),
        "max_reviewers": costing.MAX_REVIEWERS,
        "ideal_posterior_threshold": costing.ideal_posterior_threshold(a),
        "chosen_on": "validation split",
        "reported_on": "test split",
        "model_alone": {"threshold": t, **on_both(t, t)},
        "model_plus_review": on_both(capped["t_low"], capped["t_high"]),
        "model_plus_review_unconstrained": on_both(free["t_low"], free["t_high"]),
        "options_test": {
            "do_nothing": costing.cost_do_nothing(a),
            "human_everything": costing.cost_human_everything(a),
            "flag_everything": costing.cost_flag_everything(a),
        },
        "dataset_defective_share": {
            "val": float(len(vp) / max(len(vp) + len(vn), 1)),
            "test": float(len(tp_) / max(len(tp_) + len(tn_), 1)),
        },
        "n_cases": {"val": {"defective": len(vp), "good": len(vn), "spurious": len(vs)},
                    "test": {"defective": len(tp_), "good": len(tn_), "spurious": len(ts_)}},
    }
    return operating, sweep_val, sweep_test


def plot_cost_curve(sweep_val, sweep_test, operating, a, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(sweep_val["threshold"], sweep_val["cost_per_1000"], label="Model alone, validation")
    ax.plot(sweep_test["threshold"], sweep_test["cost_per_1000"], "--", label="Model alone, test")
    ax.axhline(costing.cost_do_nothing(a)["cost_per_1000"], color="grey", ls=":", label="Do nothing")
    ax.axhline(costing.cost_human_everything(a)["cost_per_1000"], color="green", ls=":",
               label="A person checks everything")
    t = operating["model_alone"]["threshold"]
    ax.axvline(t, color="black", lw=1)
    ax.scatter([t], [operating["model_alone"]["val"]["cost_per_1000"]], color="black", zorder=5,
               label=f"Chosen threshold {t:.2f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Expected cost per 1,000 welds (EUR)")
    ax.set_title(f"Cost against threshold (base rate {a.base_rate:.1%}, FN cost EUR {a.fn_cost:.0f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Report-ready tables
# ---------------------------------------------------------------------------

def f3(x) -> str:
    return f"{x:.3f}"


def eur(x) -> str:
    return f"EUR {x:,.2f}"


def write_report_snippets(path: Path, test_summary, val_summary, baselines, operating,
                          sensitivity: pd.DataFrame, config: dict, a: costing.Assumptions) -> None:
    t = operating["model_alone"]
    band = operating["model_plus_review"]
    free = operating["model_plus_review_unconstrained"]
    opts = operating["options_test"]
    o = test_summary["overall"]
    L = []
    L.append("# Report snippets (generated by train.py: check every number, then paste)\n")
    if config.get("smoke_test"):
        L.append("**THIS WAS A SMOKE TEST. Do not use these numbers.**\n")

    L.append("## Finetuning and baselines (test split)\n")
    L.append("| Model | Precision | Recall | AP50 | mAP50-95 | Cost per 1,000 welds |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| Fine-tuned {config.get('model', '')} (model alone, threshold {t['threshold']:.2f}) | "
             f"{f3(o['precision'])} | {f3(o['recall'])} | {f3(o['ap50'])} | {f3(o['map50_95'])} | "
             f"{eur(t['test']['cost_per_1000'])} |")
    c = baselines["coco_zero_shot"]
    L.append(f"| Pretrained COCO, zero-shot (any box flags) | 0.000 | 0.000 | 0.000 | 0.000 | "
             f"{eur(c['cost_per_1000'])} |")
    L.append(f"| Trivial: predict nothing | 0.000 | 0.000 | 0.000 | 0.000 | "
             f"{eur(opts['do_nothing']['cost_per_1000'])} |")
    L.append(f"\nCOCO fired on {c['images_with_any_box']} of {c['images']} test images; "
             f"most common COCO classes: {c['most_common_coco_classes']}. "
             f"Precision and recall for the fine-tuned model are the Ultralytics values at its "
             f"best-F1 confidence.\n")

    L.append("## Per-class results (test split)\n")
    L.append("| Class | Test instances | Precision | Recall | AP50 | mAP50-95 |")
    L.append("|---|---|---|---|---|---|")
    for row in test_summary["per_class"]:
        L.append(f"| {row['class']} | {row['instances']} | {f3(row['precision'])} | "
                 f"{f3(row['recall'])} | {f3(row['ap50'])} | {f3(row['map50_95'])} |")
    L.append(f"| **All** | {sum(r['instances'] for r in test_summary['per_class'])} | "
             f"{f3(o['precision'])} | {f3(o['recall'])} | {f3(o['ap50'])} | {f3(o['map50_95'])} |")
    L.append("\nConfusion matrix: results/test_confusion_matrix.png (counts) and "
             "results/test_confusion_matrix_normalized.png. Drawn at confidence 0.001, IoU 0.45.\n")

    L.append("## Operating point\n")
    L.append("- Chosen on the validation split, reported on the test split.")
    L.append(f"- Model alone: threshold **{t['threshold']:.2f}** -> "
             f"{eur(t['test']['cost_per_1000'])} per 1,000 welds on test "
             f"(true-positive rate {t['test']['tpr_fail']:.0%}, false-alarm rate {t['test']['fpr_fail']:.0%}).")
    L.append(f"- Model + human review, at most {costing.MAX_REVIEWERS:.0f} reviewers: pass below "
             f"**{band['t_low']:.2f}**, review from {band['t_low']:.2f} to {band['t_high']:.2f}, "
             f"fail from **{band['t_high']:.2f}** -> {eur(band['test']['cost_per_1000'])} per 1,000 welds, "
             f"{band['test']['review_share']:.1%} sent to a person "
             f"= {band['test']['reviews_per_day']:.0f} reviews/day = {band['test']['reviewers']:.2f} reviewers.")
    L.append(f"- Without the reviewer cap the cheapest band is {free['t_low']:.2f} to {free['t_high']:.2f}: "
             f"{eur(free['test']['cost_per_1000'])} per 1,000, {free['test']['review_share']:.1%} reviewed, "
             f"{free['test']['reviewers']:.2f} reviewers.")
    L.append(f"- Ideal threshold for a perfectly calibrated model: "
             f"{operating['ideal_posterior_threshold']:.2%} probability of a defect.\n")

    L.append("## Four options (test split, assumptions from REPORT.md)\n")
    L.append("| Option | Cost per 1,000 welds | Cost per day | Reviewers needed |")
    L.append("|---|---|---|---|")
    rows = [("Do nothing", opts["do_nothing"]),
            ("A person checks everything", opts["human_everything"]),
            (f"Model decides alone (threshold {t['threshold']:.2f})", t["test"]),
            (f"Model + human review ({band['t_low']:.2f} to {band['t_high']:.2f})", band["test"])]
    for label, d in rows:
        L.append(f"| {label} | {eur(d['cost_per_1000'])} | {eur(d['cost_per_day'])} | {d['reviewers']:.1f} |")

    L.append("\n## Sensitivity (thresholds chosen on validation, cost on test, EUR per 1,000 welds)\n")
    L.append("| FN cost | Base rate | Do nothing | Person checks all | Model alone | Model + review | Reviewed | Reviewers |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in sensitivity.iterrows():
        L.append(f"| EUR {r.fn_cost_eur:.0f} | {r.base_rate_pct:.1f}% | {r.do_nothing:,.0f} | "
                 f"{r.human_everything:,.0f} | {r.model_alone:,.0f} | {r.model_plus_review:,.0f} | "
                 f"{r.review_share_pct:.1f}% | {r.reviewers:.1f} |")

    L.append("\n## Where I trained and with what settings\n")
    dev = config.get("device", "?")
    L.append(f"{config.get('model')} fine-tuned from the COCO-pretrained checkpoint on {dev}; "
             f"{config.get('epochs_run')} epochs run (up to {config.get('epochs_requested')}, "
             f"early stopping patience {config.get('patience')}), image size {config.get('imgsz')}, "
             f"batch {config.get('batch')}, seed {config.get('seed')}, "
             f"{config.get('training_minutes')} minutes. Ultralytics {config.get('ultralytics')}, "
             f"PyTorch {config.get('torch')}.\n")
    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(evaluate_only: bool = False) -> None:
    RESULTS.mkdir(exist_ok=True)
    dataset_dir = download_dataset()
    data_yaml, names = prepare_data_yaml(dataset_dir)
    splits = split_dirs(dataset_dir)
    a = costing.Assumptions()

    # ---- train -------------------------------------------------------------
    weights = WEIGHTS_DIR / "best.pt"
    old_config = {}
    if (RESULTS / "metrics.json").is_file():
        old_config = json.loads((RESULTS / "metrics.json").read_text()).get("config", {})
    if evaluate_only:
        if not weights.is_file():
            raise SystemExit("--evaluate-only needs weights/best.pt. Train first.")
        config = dict(old_config)
    else:
        weights, config = train_model(data_yaml)
    config.update({
        "dataset": f"{WORKSPACE}/{PROJECT}/v{VERSION}",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "smoke_test": SMOKE if not evaluate_only else config.get("smoke_test", False),
        "classes": names,
    })

    # ---- evaluate with Ultralytics ----------------------------------------------
    instances = {s: count_instances(splits[s][1], names) for s in splits}
    val_summary = ultralytics_eval(weights, data_yaml, "val", instances["val"])
    test_summary = ultralytics_eval(weights, data_yaml, "test", instances["test"])

    # ---- weld-level cases ----------------------------------------------------------
    model = YOLO(str(weights))
    cases_val = collect_cases(model, *splits["val"], names)
    cases_test = collect_cases(model, *splits["test"], names)
    cases_val.to_csv(RESULTS / "cases_val.csv", index=False)
    cases_test.to_csv(RESULTS / "cases_test.csv", index=False)

    # ---- baselines ------------------------------------------------------------------
    baselines = {
        "coco_zero_shot": coco_zero_shot(*splits["test"], names, a),
        "predict_nothing": {"precision": 0.0, "recall": 0.0, "ap50": 0.0, "map50_95": 0.0,
                            "note": "No boxes at all: every weld passes."},
    }

    # ---- cost sweep and operating point ---------------------------------------------
    operating, sweep_val, sweep_test = choose_operating_point(cases_val, cases_test, a)
    sweep_val.to_csv(RESULTS / "sweep_val.csv", index=False)
    sweep_test.to_csv(RESULTS / "sweep_test.csv", index=False)
    plot_cost_curve(sweep_val, sweep_test, operating, a, RESULTS / "cost_curve.png")
    sensitivity = costing.sensitivity_table(cases_val, cases_test, a, FN_COSTS_TO_TEST,
                                            BASE_RATES_TO_TEST)
    sensitivity.to_csv(RESULTS / "sensitivity.csv", index=False)

    # ---- save everything the app needs -----------------------------------------------
    metrics = {"config": config, "val": val_summary, "test": test_summary, "baselines": baselines,
               "instances": instances}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / "operating_point.json").write_text(json.dumps(operating, indent=2))
    write_report_snippets(RESULTS / "report_snippets.md", test_summary, val_summary, baselines,
                          operating, sensitivity, config, a)

    # A few test images the dashboard offers as examples.
    examples = RESULTS / "examples"
    examples.mkdir(exist_ok=True)
    for old in examples.glob("*"):
        old.unlink()
    test_images = list_images(splits["test"][0])
    for p in test_images[:: max(1, len(test_images) // 6)][:6]:
        shutil.copy(p, examples / p.name)

    # ---- console summary -----------------------------------------------------------------
    band = operating["model_plus_review"]
    print("\n" + "=" * 72)
    print("DONE" + ("  (SMOKE TEST: ignore these numbers)" if config.get("smoke_test") else ""))
    print(f"  test mAP50 {test_summary['overall']['ap50']:.3f}   "
          f"mAP50-95 {test_summary['overall']['map50_95']:.3f}")
    print(f"  model alone: threshold {operating['model_alone']['threshold']:.2f}, "
          f"EUR {operating['model_alone']['test']['cost_per_1000']:,.0f} per 1,000 welds (test)")
    print(f"  model + review: pass < {band['t_low']:.2f} <= review < {band['t_high']:.2f} <= fail, "
          f"EUR {band['test']['cost_per_1000']:,.0f} per 1,000, "
          f"{band['test']['review_share']:.1%} reviewed")
    print("  Tables for the report: results/report_snippets.md")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="skip training; reuse weights/best.pt and redo evaluation and costing")
    main(evaluate_only=parser.parse_args().evaluate_only)
