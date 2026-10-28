"""
Evaluate weights/best.pt on the test split and write everything to outputs/.

    uv run python -m src.evaluate                # defaults
    uv run python -m src.evaluate --weights X --out-dir Y

Written to outputs/:

    test_metrics_finetuned.json   per-class P, R, AP50, AP50-95; overall
                                  mAP50 / mAP50-95; image-level TPR/FPR at
                                  0.25 and 0.50 with FPR by ground-truth
                                  class. Both on all test images ("all") and
                                  after dropping test images that have a
                                  near-duplicate in train/valid ("dedup").
    confusion_matrix_{all,dedup}.csv / .png   4 classes + background, at
                                  conf 0.25 (Ultralytics' matrix threshold).
    test_scores.csv               one row per test image: filename, image
                                  score (src.scoring at conf=0.001), label,
                                  ground-truth class of the main box, and
                                  whether it has a twin in train/valid.
    baselines.json                COCO yolo11n zero-shot (class histogram,
                                  scores all 0.0) and the trivial
                                  predict-nothing / flag-everything rules.
    baseline_coco_detections.csv  what the COCO model saw, per detection.

The near-duplicate pairs come from outputs/dataset_stats.json, written by
scripts/describe_dataset.py, so the two scripts agree on which images drop.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import NEGATIVE_CLASSES, POSITIVE_CLASSES  # noqa: E402
from src.data import DATASET_ID, ensure_dataset  # noqa: E402
from src.scoring import decide, image_label, image_score  # noqa: E402

SEED = 42
IMGSZ = 640
SCORE_CONF = 0.001          # keep every candidate for the threshold sweep
MATRIX_CONF = 0.25          # confusion matrix threshold
BASELINE_CONF = 0.25        # what the COCO model "sees" at its usual threshold
REPORT_THRESHOLDS = (0.25, 0.50)
COCO_WEIGHTS = "yolo11n.pt"
WEIGHTS = ROOT / "weights" / "best.pt"
OUT_DIR = ROOT / "outputs"
STATS_JSON = OUT_DIR / "dataset_stats.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------- helpers --

def device_choice() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def seed_everything() -> None:
    import random

    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def list_test_images(location: Path) -> list[Path]:
    img_dir = location / "test" / "images"
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path(img: Path) -> Path:
    return img.parents[1] / "labels" / (img.stem + ".txt")


def main_box_class(label_file: Path, names: list[str]) -> str | None:
    """Class name of the largest box in a YOLO label file, or None."""
    if not label_file.is_file():
        return None
    best, best_area = None, -1.0
    for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            area = vals[2] * vals[3]
        else:
            xs, ys = vals[0::2], vals[1::2]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys)) if xs else 0.0
        if area > best_area and 0 <= cls < len(names):
            best, best_area = names[cls], area
    return best


def duplicate_test_images(stats_json: Path) -> set[str]:
    """Test filenames with a near-duplicate in train or valid."""
    if not stats_json.is_file():
        raise FileNotFoundError(
            f"{stats_json} not found. Run `uv run python scripts/describe_dataset.py` first; "
            "the near-duplicate list comes from there."
        )
    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    dups = set()
    for p in stats.get("duplicate_pairs", []):
        if p["a_split"] == p["b_split"]:
            continue
        if p["a_split"] == "test":
            dups.add(p["a"])
        if p["b_split"] == "test":
            dups.add(p["b"])
    return dups


def write_dedup_yaml(data_yaml: Path, images: list[Path], dups: set[str]) -> Path:
    """A second data.yaml whose test split is a txt list of the kept images."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent
    list_file = root / "test_dedup.txt"
    list_file.write_text(
        "\n".join(str(p.resolve()) for p in images if p.name not in dups) + "\n",
        encoding="utf-8",
    )
    cfg["test"] = list_file.name
    out = root / "data_test_dedup.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


# ------------------------------------------------------- detection metrics --

def per_class_metrics(metrics, names: list[str]) -> dict:
    box = metrics.box
    idx = [int(i) for i in box.ap_class_index]
    per_class = {}
    for name in names:
        per_class[name] = None
    for pos, cls in enumerate(idx):
        per_class[names[cls]] = {
            "precision": round(float(box.p[pos]), 4),
            "recall": round(float(box.r[pos]), 4),
            "ap50": round(float(box.ap50[pos]), 4),
            "ap50_95": round(float(box.ap[pos]), 4),
        }
    return {
        "per_class": per_class,
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "mean_precision": round(float(box.mp), 4),
        "mean_recall": round(float(box.mr), 4),
    }


def save_confusion(matrix, names: list[str], stem: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(names) + ["background"]
    m = np.asarray(matrix, dtype=int)
    with (stem.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["predicted \\ true"] + labels)
        for i, row in enumerate(m):
            w.writerow([labels[i]] + [int(v) for v in row])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, str(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=120)
    plt.close(fig)


def detection_metrics(model, data_yaml: Path, names: list[str], device: str,
                      run_name: str, out_dir: Path, subset: str) -> dict:
    common = dict(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=16, device=device,
                  verbose=False, project=str(ROOT / "runs"), exist_ok=True)
    m_map = model.val(conf=SCORE_CONF, name=f"{run_name}_map", plots=False, **common)
    result = per_class_metrics(m_map, names)
    # plots=True is required: the validator only fills the confusion matrix
    # when plotting is on. Its own PNGs land in runs/, which is git-ignored.
    m_cm = model.val(conf=MATRIX_CONF, name=f"{run_name}_cm", plots=True, **common)
    save_confusion(
        m_cm.confusion_matrix.matrix, names, out_dir / f"confusion_matrix_{subset}",
        f"Test confusion matrix, {subset} (conf {MATRIX_CONF})",
    )
    result["map_conf"] = SCORE_CONF
    result["confusion_matrix_conf"] = MATRIX_CONF
    result["confusion_matrix_files"] = [f"confusion_matrix_{subset}.csv", f"confusion_matrix_{subset}.png"]
    return result


# ------------------------------------------------------ image-level metrics --

def image_level(rows: list[dict], threshold: float) -> dict:
    """Confusion counts and rates when every image with score >= t is flagged."""
    tp = fp = fn = tn = 0
    fp_by_class: Counter = Counter()
    neg_by_class: Counter = Counter()
    for r in rows:
        flagged = decide(r["score"], threshold, threshold) == "fail"
        if r["label"]:
            tp += flagged
            fn += not flagged
        else:
            neg_by_class[r["main_class"]] += 1
            fp += flagged
            tn += not flagged
            fp_by_class[r["main_class"]] += flagged
    positives, negatives = tp + fn, fp + tn
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "positives": positives, "negatives": negatives,
        "tpr": round(tp / positives, 4) if positives else None,
        "fpr": round(fp / negatives, 4) if negatives else None,
        "fpr_by_class": {
            c: {"fp": fp_by_class[c], "negatives": neg_by_class[c],
                "fpr": round(fp_by_class[c] / neg_by_class[c], 4) if neg_by_class[c] else None}
            for c in NEGATIVE_CLASSES
        },
    }


def image_level_block(rows: list[dict]) -> dict:
    return {
        "images": len(rows),
        "positives": sum(1 for r in rows if r["label"]),
        "negatives": sum(1 for r in rows if not r["label"]),
        "at_threshold": {str(t): image_level(rows, t) for t in REPORT_THRESHOLDS},
    }


def score_images(model, images: list[Path], names: list[str], device: str,
                 dups: set[str]) -> list[dict]:
    rows = []
    results = model.predict(source=[str(p) for p in images], conf=SCORE_CONF, imgsz=IMGSZ,
                            device=device, verbose=False, stream=True)
    for img, res in zip(images, results):
        lbl = label_path(img)
        rows.append({
            "filename": img.name,
            "score": round(image_score(res), 6),
            "label": int(image_label(lbl, names)),
            "main_class": main_box_class(lbl, names),
            "dup_in_trainval": int(img.name in dups),
        })
    return rows


def write_scores_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "score", "label", "main_class", "dup_in_trainval"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ baselines --

def coco_baseline(images: list[Path], rows: list[dict], device: str, out_dir: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(COCO_WEIGHTS)
    coco_names = model.names
    hist: Counter = Counter()
    images_with_any = 0
    det_rows = []
    results = model.predict(source=[str(p) for p in images], conf=BASELINE_CONF, imgsz=IMGSZ,
                            device=device, verbose=False, stream=True)
    for img, res in zip(images, results):
        n = 0
        for cls, conf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            name = coco_names[int(cls)]
            hist[name] += 1
            n += 1
            det_rows.append({"filename": img.name, "coco_class": name, "conf": round(conf, 4)})
        images_with_any += n > 0
    with (out_dir / "baseline_coco_detections.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "coco_class", "conf"], lineterminator="\n")
        w.writeheader()
        w.writerows(det_rows)
    # Every image scores 0.0: COCO has no package class and we do not invent a mapping.
    zero_rows = [{**r, "score": 0.0} for r in rows]
    return {
        "weights": COCO_WEIGHTS,
        "note": "COCO has no package class and no mapping is invented, so every image "
                "scores 0.0 and passes. Detection AP against the dataset's classes is 0 by "
                "construction and is not computed.",
        "detection_conf": BASELINE_CONF,
        "images_with_any_detection": images_with_any,
        "detections_total": sum(hist.values()),
        "class_histogram": dict(hist.most_common()),
        "image_level": {
            "all": image_level_block(zero_rows),
            "dedup": image_level_block([r for r in zero_rows if not r["dup_in_trainval"]]),
        },
    }


def trivial_baselines(rows: list[dict]) -> dict:
    out = {}
    for name, score in (("predict_nothing_all_pass", 0.0), ("flag_everything_all_fail", 1.0)):
        fixed = [{**r, "score": score} for r in rows]
        out[name] = {
            "rule": "every image scores 0.0 (pass)" if score == 0.0 else "every image scores 1.0 (fail)",
            "image_level": {
                "all": image_level_block(fixed),
                "dedup": image_level_block([r for r in fixed if not r["dup_in_trainval"]]),
            },
        }
    return out


# ----------------------------------------------------------------------- main --

def main(weights: Path = WEIGHTS, out_dir: Path = OUT_DIR, stats_json: Path = STATS_JSON) -> dict:
    import yaml
    from ultralytics import YOLO

    weights, out_dir = Path(weights), Path(out_dir)
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} not found. Train first (uv run python main.py train).")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything()
    device = device_choice()

    data_yaml = ensure_dataset()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = [str(n) for n in cfg["names"]]
    location = data_yaml.parent
    images = list_test_images(location)
    dups = duplicate_test_images(stats_json)
    dedup_yaml = write_dedup_yaml(data_yaml, images, dups)
    print(f"test images: {len(images)}, with a twin in train/valid: {len(dups)}; device {device}")

    model = YOLO(str(weights))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != names:
        raise SystemExit(f"model classes {model_names} differ from dataset classes {names}")

    # Image-level scores first: they are what the sweep and the dashboard use.
    rows = score_images(model, images, names, device, dups)
    write_scores_csv(rows, out_dir / "test_scores.csv")
    rows_dedup = [r for r in rows if not r["dup_in_trainval"]]

    finetuned = {
        "dataset": DATASET_ID,
        "weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
        "device": device,
        "seed": SEED,
        "imgsz": IMGSZ,
        "score_conf": SCORE_CONF,
        "map_conf": SCORE_CONF,
        "confusion_matrix_conf": MATRIX_CONF,
        "conf_note": (
            f"Per-class P/R/AP and mAP come from a validation pass at conf {SCORE_CONF} "
            f"(the standard mAP setting). The confusion matrix comes from a second pass "
            f"at conf {MATRIX_CONF}. Image scores keep every detection at conf {SCORE_CONF}."
        ),
        "positive_classes": list(POSITIVE_CLASSES),
        "dedup_excluded": sorted(dups),
        "all": {
            "detection": detection_metrics(model, data_yaml, names, device, "eval_all", out_dir, "all"),
            "image_level": image_level_block(rows),
        },
        "dedup": {
            "detection": detection_metrics(model, dedup_yaml, names, device, "eval_dedup", out_dir, "dedup"),
            "image_level": image_level_block(rows_dedup),
        },
    }
    (out_dir / "test_metrics_finetuned.json").write_text(json.dumps(finetuned, indent=2), encoding="utf-8")

    baselines = {
        "dataset": DATASET_ID,
        "dedup_excluded": sorted(dups),
        "coco_zero_shot": coco_baseline(images, rows, device, out_dir),
        "trivial": trivial_baselines(rows),
    }
    (out_dir / "baselines.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")

    for subset in ("all", "dedup"):
        d = finetuned[subset]
        il = d["image_level"]["at_threshold"]
        print(f"[{subset}] mAP50 {d['detection']['map50']}  mAP50-95 {d['detection']['map50_95']}  "
              f"image-level n={d['image_level']['images']}  "
              + "  ".join(f"t={t}: TPR {v['tpr']} FPR {v['fpr']}" for t, v in il.items()))
    print(f"wrote test_metrics_finetuned.json, test_scores.csv, confusion_matrix_*.{{csv,png}}, "
          f"baselines.json, baseline_coco_detections.csv to {out_dir}")
    return {"finetuned": finetuned, "baselines": baselines}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stats-json", type=Path, default=STATS_JSON)
    a = parser.parse_args()
    main(weights=a.weights, out_dir=a.out_dir, stats_json=a.stats_json)
