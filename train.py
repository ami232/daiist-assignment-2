"""
Assignment 2: training pipeline.  Run with:  uv run python main.py train

Start to finish, no manual steps:

1. Download the dataset from Roboflow Universe (key via api_key.load_api_key).
2. Describe the dataset from its labels (instances per class, split sizes,
   object size, how the classes map to the decision) -> artifacts/dataset_stats.json
3. Finetune yolo11n from the COCO checkpoint -> weights/best.pt
4. Evaluate on the TEST split: per-class P/R/AP50/mAP50-95 + confusion matrix,
   and two baselines (COCO yolo11n zero-shot, trivial predict-nothing /
   predict-everything).
5. Score every test photo (max confidence over the FUNGAL classes), sweep the
   confidence threshold and the two-threshold review band with the cost model
   in costing.py, and choose the operating point.
6. Save everything app.py needs under artifacts/, plus
   artifacts/report_numbers.md with the tables to paste into REPORT.md.

Environment knobs (all optional):
    EPOCHS=30  IMGSZ=416  BATCH=16  MODEL=yolo11n.pt  DEVICE=0|cpu
    SKIP_TRAIN=1   re-use weights/best.pt and only redo evaluation + sweeps
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from api_key import load_api_key
from costing import (
    COST_MODEL,
    alternatives,
    choose_operating_point,
    evaluate_band,
    precision_recall_at,
    roc_auc,
    sweep_band,
    sweep_single_threshold,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
ART = ROOT / "artifacts"

# Roboflow Universe: "Detecting Diseases" (strawberry, tomato and bean plants).
# v5 = 5,493 images, 416x416 stretch, no augmentation, 2904/1416/1173 split.
# https://universe.roboflow.com/artificial-intelligence-82oex/detecting-diseases
RF_WORKSPACE = "artificial-intelligence-82oex"
RF_PROJECT = "detecting-diseases"
RF_VERSION = 5

# The decision is "does this bed get the fungicide application?". A class is
# POSITIVE (fungal) if its name contains one of these words. Everything else
# (spider mites = pest, strawberry angular leaf spot = bacterial, any generic
# or junk label) is NEGATIVE. The mapping that results is printed at the start
# of training and saved to artifacts/dataset_stats.json: check it.
FUNGAL_KEYWORDS = ("mold", "mould", "mildew", "rot", "blight", "rust", "spot", "als", "anthracnose")
NON_FUNGAL_OVERRIDES = ("strawberry angular",)   # strawberry angular leaf spot is bacterial, even though it says "spot"

MODEL_NAME = os.environ.get("MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = int(os.environ.get("IMGSZ", "416"))   # the dataset is already 416x416
BATCH = int(os.environ.get("BATCH", "16"))
DEVICE = os.environ.get("DEVICE", None)   # None -> ultralytics picks GPU if present
SEED = 0
RUN_NAME = "plant_disease_yolo11n"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_PROXY_CLASS = 58   # COCO "potted plant": the only vaguely relevant thing the zero-shot model can see


def log(msg: str) -> None:
    print(f"\n[train] {msg}", flush=True)


def is_fungal(class_name: str) -> bool:
    n = str(class_name).lower()
    if any(w in n for w in NON_FUNGAL_OVERRIDES):
        return False
    # "als" must match as a word (bean angular leaf spot, a fungus), not inside other words
    tokens = n.replace("_", " ").replace("-", " ").split()
    return any(w in n for w in FUNGAL_KEYWORDS if w != "als") or "als" in tokens


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_dataset() -> Path:
    """Download once; re-use on later runs. Returns the path to data.yaml."""
    target = DATA_DIR / f"{RF_PROJECT}-v{RF_VERSION}"
    yaml_path = target / "data.yaml"
    if yaml_path.exists():
        log(f"Dataset already at {target}, skipping download.")
    else:
        from roboflow import Roboflow

        log(f"Downloading {RF_WORKSPACE}/{RF_PROJECT} v{RF_VERSION} from Roboflow Universe ...")
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION)
        version.download("yolov11", location=str(target), overwrite=True)

    # Roboflow writes relative paths that only work from one directory.
    # Make them absolute so ultralytics finds every split from anywhere.
    cfg = yaml.safe_load(yaml_path.read_text())
    cfg["path"] = str(target)
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return yaml_path


def class_names(yaml_path: Path) -> dict[int, str]:
    names = yaml.safe_load(yaml_path.read_text())["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


# ---------------------------------------------------------------------------
# 2. Describe the dataset from its labels
# ---------------------------------------------------------------------------
def describe_dataset(yaml_path: Path) -> dict:
    cfg = yaml.safe_load(yaml_path.read_text())
    names = class_names(yaml_path)
    pos_ids = sorted(k for k, v in names.items() if is_fungal(v))
    root = Path(cfg["path"])

    stats = {
        "classes": names,
        "class_group": {v: ("fungal (positive)" if k in pos_ids else "non-fungal (negative)") for k, v in names.items()},
        "positive_class_ids": pos_ids,
        "positive_classes": [names[k] for k in pos_ids],
        "negative_classes": [v for k, v in names.items() if k not in pos_ids],
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT] if img_dir.exists() else []
        counts = {v: 0 for v in names.values()}
        areas = []
        imgs_with_pos, imgs_without_any, imgs_mixed = 0, 0, 0
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            has_pos, has_neg, n_boxes = False, False, 0
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    c = int(float(parts[0]))
                    w, h = float(parts[3]), float(parts[4])
                    counts[names[c]] += 1
                    areas.append(w * h)
                    n_boxes += 1
                    if c in pos_ids:
                        has_pos = True
                    else:
                        has_neg = True
            imgs_with_pos += has_pos
            imgs_without_any += n_boxes == 0
            imgs_mixed += has_pos and has_neg
        areas = np.array(areas) if areas else np.zeros(1)
        stats["splits"][split] = {
            "images": len(images),
            "instances_per_class": counts,
            "instances_total": int(sum(counts.values())),
            "images_with_positive": imgs_with_pos,
            "share_images_with_positive": imgs_with_pos / max(len(images), 1),
            "images_with_both_groups": imgs_mixed,
            "images_without_any_box": imgs_without_any,
            "box_area_share_median": float(np.median(areas)),
            "box_area_share_p10": float(np.percentile(areas, 10)),
            "box_area_share_p90": float(np.percentile(areas, 90)),
            "share_boxes_under_1pct_of_image": float(np.mean(areas < 0.01)),
            "share_boxes_over_25pct_of_image": float(np.mean(areas > 0.25)),
        }
    return stats


# ---------------------------------------------------------------------------
# 3. Finetune
# ---------------------------------------------------------------------------
def finetune(yaml_path: Path) -> Path:
    from ultralytics import YOLO

    best = WEIGHTS_DIR / "best.pt"
    if os.environ.get("SKIP_TRAIN") == "1" and best.exists():
        log(f"SKIP_TRAIN=1 and {best} exists: skipping training.")
        return best

    log(f"Finetuning {MODEL_NAME} for {EPOCHS} epochs at imgsz={IMGSZ}, batch={BATCH} ...")
    model = YOLO(MODEL_NAME)  # COCO-pretrained checkpoint, auto-downloaded
    kwargs = dict(data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
                  project=str(RUNS_DIR), name=RUN_NAME, exist_ok=True, plots=True, verbose=True)
    if DEVICE is not None:
        kwargs["device"] = DEVICE
    model.train(**kwargs)

    produced = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(produced, best)
    log(f"Best weights copied to {best}")
    return best


# ---------------------------------------------------------------------------
# 4. Detection metrics on the test split, per class
# ---------------------------------------------------------------------------
def _to_list(x):
    try:
        return [float(v) for v in np.asarray(x).ravel()]
    except Exception:  # noqa: BLE001
        return []


def detection_metrics(weights: Path, yaml_path: Path, names: dict, pos_ids: list[int]) -> dict:
    from ultralytics import YOLO

    log("Evaluating finetuned model on the TEST split ...")
    model = YOLO(str(weights))
    kwargs = dict(data=str(yaml_path), split="test", imgsz=IMGSZ, conf=0.001, iou=0.6,
                  plots=True, project=str(RUNS_DIR), name=f"{RUN_NAME}_test", exist_ok=True, verbose=False)
    if DEVICE is not None:
        kwargs["device"] = DEVICE
    res = model.val(**kwargs)
    box = res.box

    class_idx = [int(i) for i in np.asarray(box.ap_class_index).ravel()]
    p, r, ap50, ap = _to_list(box.p), _to_list(box.r), _to_list(box.ap50), _to_list(box.ap)
    per_class = []
    for j, ci in enumerate(class_idx):
        per_class.append({
            "class": names[ci],
            "group": "fungal" if ci in pos_ids else "non-fungal",
            "precision": p[j] if j < len(p) else float("nan"),
            "recall": r[j] if j < len(r) else float("nan"),
            "AP50": ap50[j] if j < len(ap50) else float("nan"),
            "mAP50-95": ap[j] if j < len(ap) else float("nan"),
        })
    # classes with no test instances get no row from ultralytics; say so explicitly
    for ci, n in names.items():
        if ci not in class_idx:
            per_class.append({"class": n, "group": "fungal" if ci in pos_ids else "non-fungal",
                              "precision": float("nan"), "recall": float("nan"),
                              "AP50": float("nan"), "mAP50-95": float("nan")})

    overall = {"precision": float(box.mp), "recall": float(box.mr),
               "mAP50": float(box.map50), "mAP50-95": float(box.map)}

    # Confusion matrix (ultralytics: rows = predicted, cols = true, last = background),
    # computed at its default conf 0.25 / IoU 0.45.
    cm_labels = [names[i] for i in sorted(names)] + ["background"]
    try:
        matrix = np.asarray(res.confusion_matrix.matrix, dtype=float)
    except Exception:  # noqa: BLE001
        matrix = np.full((len(cm_labels), len(cm_labels)), np.nan)
    confusion = {"labels": cm_labels, "rows": "predicted", "cols": "true",
                 "matrix": matrix.tolist(), "note": "ultralytics default conf=0.25, IoU=0.45"}

    plots_dir = RUNS_DIR / f"{RUN_NAME}_test"
    for png in ("confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png"):
        src = plots_dir / png
        if src.exists():
            shutil.copy2(src, ART / png)

    return {"overall": overall, "per_class": per_class, "confusion": confusion}


# ---------------------------------------------------------------------------
# 5. Case-level scores and baselines
# ---------------------------------------------------------------------------
def list_test_images(yaml_path: Path) -> list[Path]:
    cfg = yaml.safe_load(yaml_path.read_text())
    img_dir = Path(cfg["path"]) / "test" / "images"
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)


def image_labels(yaml_path: Path, images: list[Path], pos_ids: list[int]) -> np.ndarray:
    """1 if the photo contains at least one box of a fungal class, else 0."""
    cfg = yaml.safe_load(yaml_path.read_text())
    lbl_dir = Path(cfg["path"]) / "test" / "labels"
    out = []
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        has = False
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5 and int(float(parts[0])) in pos_ids:
                    has = True
                    break
        out.append(int(has))
    return np.array(out)


def score_images(weights: str, images: list[Path], target_classes: list[int]) -> np.ndarray:
    """Per image: the highest confidence of any detection of `target_classes`."""
    from ultralytics import YOLO

    model = YOLO(weights)
    scores = np.zeros(len(images))
    kwargs = dict(conf=0.001, imgsz=IMGSZ, verbose=False, stream=True, classes=list(target_classes))
    if DEVICE is not None:
        kwargs["device"] = DEVICE
    for i, res in enumerate(model.predict([str(p) for p in images], **kwargs)):
        if res.boxes is not None and len(res.boxes):
            scores[i] = float(res.boxes.conf.max())
    return scores


def baselines(scores: np.ndarray, labels: np.ndarray, coco_scores: np.ndarray, chosen_t: float) -> pd.DataFrame:
    rows = []
    pi_test = labels.mean()

    def row(name, pr, note):
        rows.append({"model": name, "test precision": pr["precision"], "test recall": pr["recall"],
                     "test F1": pr["f1"], "TP": pr["TP"], "FP": pr["FP"], "FN": pr["FN"], "TN": pr["TN"],
                     "note": note})

    row("Trivial: predict nothing (never fungal)", precision_recall_at(np.zeros_like(scores), labels, 0.5),
        "every fungal case is missed")
    row("Trivial: majority class (always fungal)", precision_recall_at(np.ones_like(scores), labels, 0.5),
        f"precision = test base rate {pi_test:.2f}")
    row("COCO yolo11n zero-shot (any 'potted plant' >= 0.25)", precision_recall_at(coco_scores, labels, 0.25),
        "COCO has no disease classes; AP on target classes = 0 by construction")
    row(f"Finetuned yolo11n (fungal >= {chosen_t:.2f})", precision_recall_at(scores, labels, chosen_t),
        f"case-level AUC = {roc_auc(scores, labels):.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Report snippet
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report_numbers(stats, det, base_df, single, chosen, alt_df, op_eval, scores, labels, settings) -> None:
    cm = COST_MODEL
    b = chosen["band_chosen"]
    lines = ["# Numbers produced by train.py (paste into REPORT.md)", ""]
    lines += [f"Run settings: {json.dumps(settings)}", ""]

    lines += ["## Class grouping used for the decision", "", "| class | group |", "|---|---|"]
    lines += [f"| {c} | {g} |" for c, g in stats["class_group"].items()]
    lines.append("")

    lines += ["## Dataset", ""]
    rows = []
    for s, d in stats["splits"].items():
        row = {"split": s, "images": d["images"], "images with fungal": d["images_with_positive"],
               "share fungal": round(d["share_images_with_positive"], 2),
               "images with both groups": d["images_with_both_groups"],
               "images with no box": d["images_without_any_box"],
               "median box area": f"{d['box_area_share_median']*100:.1f}%",
               "boxes > 25% of image": f"{d['share_boxes_over_25pct_of_image']*100:.0f}%"}
        rows.append(row)
    lines.append(md_table(pd.DataFrame(rows)))
    lines += ["", "Instances per class:", ""]
    inst = pd.DataFrame({s: d["instances_per_class"] for s, d in stats["splits"].items()})
    inst["group"] = [stats["class_group"][c] for c in inst.index]
    lines.append(md_table(inst.reset_index().rename(columns={"index": "class"}), "{:.0f}"))
    lines.append("")

    lines += ["## Detection metrics on the test split (finetuned yolo11n)", ""]
    lines.append(md_table(pd.DataFrame(det["per_class"])))
    o = det["overall"]
    lines += ["", f"Overall: precision {o['precision']:.3f}, recall {o['recall']:.3f}, "
                  f"mAP50 {o['mAP50']:.3f}, mAP50-95 {o['mAP50-95']:.3f}", ""]
    lines += ["Confusion matrix (rows = predicted, columns = true; conf 0.25):", ""]
    cmdf = pd.DataFrame(det["confusion"]["matrix"], index=det["confusion"]["labels"],
                        columns=det["confusion"]["labels"]).fillna(0).astype(int)
    lines.append(md_table(cmdf.reset_index().rename(columns={"index": "pred \\ true"}), "{:.0f}"))
    lines.append("")

    lines += ["## Case-level baselines on the same test split (positive = photo shows a fungal disease)", ""]
    lines.append(md_table(base_df))
    lines.append("")

    lines += ["## Cost model", "", "```json", json.dumps(cm, indent=2), "```", ""]
    lines += ["## Threshold sweep (model alone, cost per 1000 production cases)", ""]
    sub = single.iloc[::7][["threshold", "cost_per_1000", "TPR", "FPR"]]
    lines.append(md_table(sub, "{:.3f}"))
    lines += ["", f"Minimum: threshold {chosen['model_alone']['threshold']:.2f} -> "
                  f"EUR {chosen['model_alone']['cost_per_1000']:.0f} per 1000 cases", ""]

    lines += ["## Chosen operating point (model + review band)", "",
              f"- t_low = {b['t_low']:.2f}, t_high = {b['t_high']:.2f}",
              f"- cost per 1000 cases: EUR {b['cost_per_1000']:.0f}",
              f"- share sent for review: {b['review_share']*100:.1f}%  "
              f"({op_eval['review_cases_per_day']:.0f} photos/day, {op_eval['review_hours_per_day']:.2f} h/day, "
              f"{op_eval['people_needed']:.2f} people at {cm['REVIEWER_HOURS_PER_DAY']} h each)",
              f"- unconstrained optimum would be [{chosen['band_unconstrained']['t_low']:.2f}, "
              f"{chosen['band_unconstrained']['t_high']:.2f}) at EUR {chosen['band_unconstrained']['cost_per_1000']:.0f} "
              f"with {chosen['band_unconstrained']['review_share']*100:.1f}% reviewed "
              f"({chosen['band_unconstrained']['people_needed']:.2f} people)",
              "", "Expected confusion matrix per 1000 production cases after review:", "",
              f"TP {op_eval['TP_per_1000']:.1f} | FN {op_eval['FN_per_1000']:.1f} | "
              f"FP {op_eval['FP_per_1000']:.1f} | TN {op_eval['TN_per_1000']:.1f}", ""]

    lines += ["## Priced against the alternatives", ""]
    lines.append(md_table(alt_df, "{:.2f}"))
    lines.append("")

    # Sensitivity: how does the single-threshold optimum move with C_FN and base rate?
    lines += ["## Sensitivity of the optimum", "", "| C_FN | base rate | best threshold | cost/1000 |", "|---|---|---|---|"]
    for c_fn in (15, 30, 60, 150):
        for pi in (0.2, 0.4, 0.7):
            cm2 = dict(cm, C_FN=float(c_fn), BASE_RATE=pi)
            sw = sweep_single_threshold(scores, labels, cm2)
            r = sw.loc[sw["cost_per_1000"].idxmin()]
            lines.append(f"| {c_fn} | {pi} | {r['threshold']:.2f} | {r['cost_per_1000']:.0f} |")
    lines.append("")
    (ART / "report_numbers.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    yaml_path = download_dataset()
    stats = describe_dataset(yaml_path)
    (ART / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    names = stats["classes"]
    pos_ids = stats["positive_class_ids"]
    log("Class grouping:\n" + "\n".join(f"    {c:35s} -> {g}" for c, g in stats["class_group"].items()))
    log("Dataset: " + json.dumps({s: d["images"] for s, d in stats["splits"].items()}))
    if not pos_ids or len(pos_ids) == len(names):
        raise SystemExit(
            "The class grouping gives no negatives or no positives. Look at the class names printed "
            "above and adjust FUNGAL_KEYWORDS / NON_FUNGAL_OVERRIDES in train.py."
        )
    t = stats["splits"]["test"]
    if t["images_with_positive"] in (0, t["images"]):
        raise SystemExit(
            f"Test split has {t['images_with_positive']} positive photos out of {t['images']}: the case-level "
            "sweep needs both classes present. Adjust the class grouping in train.py."
        )

    weights = finetune(yaml_path)

    det = detection_metrics(weights, yaml_path, names, pos_ids)
    (ART / "per_class.csv").write_text(pd.DataFrame(det["per_class"]).to_csv(index=False))
    (ART / "confusion_matrix.json").write_text(json.dumps(det["confusion"]))

    log("Scoring test photos with the finetuned model and the COCO zero-shot model ...")
    images = list_test_images(yaml_path)
    labels = image_labels(yaml_path, images, pos_ids)
    scores = score_images(str(weights), images, pos_ids)
    coco_scores = score_images("yolo11n.pt", images, [COCO_PROXY_CLASS])
    pd.DataFrame({"image": [p.name for p in images], "label": labels,
                  "score": scores, "coco_proxy_score": coco_scores}).to_csv(ART / "test_scores.csv", index=False)

    log("Sweeping thresholds with the cost model ...")
    single = sweep_single_threshold(scores, labels)
    band = sweep_band(scores, labels)
    chosen = choose_operating_point(single, band)
    single.to_csv(ART / "cost_sweep.csv", index=False)
    band.to_csv(ART / "band_sweep.csv", index=False)
    b = chosen["band_chosen"]
    op_eval = evaluate_band(scores, labels, b["t_low"], b["t_high"])
    alt_df = alternatives(scores, labels, chosen)
    alt_df.to_csv(ART / "alternatives.csv", index=False)
    base_df = baselines(scores, labels, coco_scores, chosen["model_alone"]["threshold"])
    base_df.to_csv(ART / "baselines.csv", index=False)

    # Cost-vs-threshold figure (the app redraws it live; this one is for the report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(single["threshold"], single["cost_per_1000"], color="#1f4e79")
    t_star = chosen["model_alone"]["threshold"]
    ax.axvline(t_star, color="#c0392b", ls="--", label=f"model alone optimum t={t_star:.2f}")
    ax.axvspan(b["t_low"], b["t_high"], color="#f39c12", alpha=0.25, label=f"review band [{b['t_low']:.2f}, {b['t_high']:.2f})")
    ax.set_xlabel("confidence threshold on 'fungal' detections")
    ax.set_ylabel("expected cost per 1000 photos (EUR)")
    ax.set_title("Cost against threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "cost_vs_threshold.png", dpi=150)
    plt.close(fig)

    settings = {
        "model": MODEL_NAME, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
        "device": DEVICE or "auto", "machine": platform.platform(),
        "roboflow": f"{RF_WORKSPACE}/{RF_PROJECT} v{RF_VERSION}",
        "trained_at": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t0) / 60, 1),
    }
    metrics = {
        "settings": settings,
        "positive_classes": stats["positive_classes"],
        "negative_classes": stats["negative_classes"],
        "positive_class_ids": pos_ids,
        "class_names": names,
        "detection": det["overall"],
        "per_class": det["per_class"],
        "case_level": {"test_base_rate": float(labels.mean()), "n_test_images": int(len(labels)),
                       "auc": roc_auc(scores, labels)},
        "cost_model": COST_MODEL,
        "operating_point": chosen,
        "operating_point_eval": op_eval,
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    write_report_numbers(stats, det, base_df, single, chosen, alt_df, op_eval, scores, labels, settings)
    log(f"Done in {settings['minutes']} min. Artifacts in {ART}. Weights in {weights}.")
    log(f"Chosen band: [{b['t_low']:.2f}, {b['t_high']:.2f}), EUR {b['cost_per_1000']:.0f} per 1000 photos, "
        f"{b['review_share']*100:.1f}% reviewed.")


if __name__ == "__main__":
    main()
