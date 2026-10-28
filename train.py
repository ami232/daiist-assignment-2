"""
Assignment 2: training pipeline (run with `uv run python main.py train`).

Stages, start to finish, no manual steps:
  1. Download the public Roboflow dataset named in config.py.
  2. Describe it: images and instances per class per split, object sizes.
  3. Finetune yolo11n from COCO-pretrained weights; save weights/best.pt.
  4. Evaluate on the test split, per class, with the confusion matrix.
  5. Baselines on the same test split: COCO yolo11n zero-shot, and a
     trivial "predict nothing" model.
  6. Score every test image, sweep the threshold, cost it, and choose the
     binary threshold and the pass/review/fail band.
  7. Save everything the dashboard needs into artifacts/.

Optional environment overrides (handy for a quick dry run):
  EPOCHS=1            train for fewer epochs
  DATASET_DIR=path    use an already-downloaded YOLO dataset instead
  REUSE_WEIGHTS=1     skip training if weights/best.pt already exists
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

import config as C
import costing

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}


# --------------------------------------------------------------------------- 1
def download_dataset() -> Path:
    override = os.environ.get("DATASET_DIR")
    if override:
        return Path(override).resolve()
    target = C.DATASETS_DIR / f"{C.RF_PROJECT}-v{C.RF_VERSION}"
    if (target / "data.yaml").exists():
        print(f"Dataset already at {target}")
        return target
    from roboflow import Roboflow
    from api_key import load_api_key

    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(C.RF_WORKSPACE).project(C.RF_PROJECT).version(C.RF_VERSION)
    version.download(C.RF_FORMAT, location=str(target), overwrite=True)
    return target


def write_clean_yaml(ds: Path) -> tuple[Path, list[str]]:
    """Roboflow's data.yaml uses relative paths that break depending on cwd.
    Rewrite it with an absolute root so training works from anywhere."""
    original = yaml.safe_load((ds / "data.yaml").read_text())
    names = original["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    clean = {
        "path": str(ds),
        "train": f"{SPLITS['train']}/images",
        "val": f"{SPLITS['val']}/images",
        "test": f"{SPLITS['test']}/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    out = ds / "data_clean.yaml"
    out.write_text(yaml.safe_dump(clean, sort_keys=False))
    return out, list(names)


# --------------------------------------------------------------------------- 2
def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT)


def label_file(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def read_labels(img: Path) -> list[tuple[int, float, float]]:
    """(class_id, width, height) per box, normalised to the image."""
    f = label_file(img)
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append((int(parts[0]), float(parts[3]), float(parts[4])))
    return out


def describe_dataset(ds: Path, names: list[str]) -> dict:
    stats = {"splits": {}, "object_size": {}}
    areas = {n: [] for n in names}
    for split, folder in SPLITS.items():
        imgs = images_in(ds / folder / "images")
        counts = {n: 0 for n in names}
        imgs_with = {n: 0 for n in names}
        empty = 0
        for img in imgs:
            boxes = read_labels(img)
            if not boxes:
                empty += 1
            seen = set()
            for cls, w, h in boxes:
                counts[names[cls]] += 1
                areas[names[cls]].append(w * h)
                seen.add(names[cls])
            for n in seen:
                imgs_with[n] += 1
        stats["splits"][split] = {
            "images": len(imgs),
            "instances": counts,
            "images_containing": imgs_with,
            "images_without_labels": empty,
        }
    for n, a in areas.items():
        a = np.array(a) if a else np.array([0.0])
        stats["object_size"][n] = {
            "median_area_pct_of_image": round(float(np.median(a) * 100), 3),
            "p10_area_pct": round(float(np.percentile(a, 10) * 100), 3),
            "p90_area_pct": round(float(np.percentile(a, 90) * 100), 3),
            "share_small_lt_1pct": round(float((a < 0.01).mean()), 3),
        }
    test = stats["splits"]["test"]
    stats["test_base_rate_positive_images"] = round(
        test["images_containing"][C.POSITIVE_CLASS] / max(test["images"], 1), 3
    )
    return stats


# --------------------------------------------------------------------------- 3
def finetune(data_yaml: Path) -> Path:
    epochs = int(os.environ.get("EPOCHS", C.EPOCHS))
    model = YOLO(C.BASE_MODEL)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=C.IMGSZ, batch=C.BATCH,
        patience=C.PATIENCE, seed=C.SEED, project=str(C.ROOT / "runs"),
        name="finetune", exist_ok=True, plots=True,
    )
    best = Path(model.trainer.best)
    C.WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, C.BEST_WEIGHTS)
    # Ultralytics' AMP check can drop a stray checkpoint into weights/; keep only ours.
    for stray in C.WEIGHTS_DIR.glob("*.pt"):
        if stray != C.BEST_WEIGHTS:
            stray.unlink()
    print(f"Saved {C.BEST_WEIGHTS}")
    return C.BEST_WEIGHTS


# --------------------------------------------------------------------------- 4
def evaluate_detection(weights: Path, data_yaml: Path, names: list[str]) -> dict:
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml), split="test", imgsz=C.IMGSZ, batch=C.BATCH,
        project=str(C.ROOT / "runs"), name="test_eval", exist_ok=True, plots=True,
    )
    per_class = []
    for i, cls_idx in enumerate(m.box.ap_class_index):
        per_class.append({
            "class": names[int(cls_idx)],
            "precision": round(float(m.box.p[i]), 4),
            "recall": round(float(m.box.r[i]), 4),
            "AP50": round(float(m.box.ap50[i]), 4),
            "mAP50-95": round(float(m.box.ap[i]), 4),
        })
    cm = m.confusion_matrix.matrix.astype(int).tolist()  # rows = predicted, cols = true
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = Path(m.save_dir) / f
        if src.exists():
            shutil.copy(src, C.ARTIFACTS_DIR / f)
    return {
        "overall": {"AP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)},
        "per_class": per_class,
        # Ultralytics builds this matrix at the validation confidence (0.001), so it
        # counts every faint box the model emits. Good for AP, misleading as a picture
        # of the errors we would act on: see confusion_at_threshold() below.
        "confusion_matrix": {"labels": names + ["background"], "matrix": cm,
                             "orientation": "rows = predicted, columns = true",
                             "conf": 0.001},  # Ultralytics' default val confidence
    }


def confusion_at_threshold(weights: Path, data_yaml: Path, names: list[str], conf: float) -> dict:
    """Box-level confusion matrix counting only boxes at or above the chosen threshold,
    i.e. the boxes that would actually drive a decision in production."""
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml), split="test", imgsz=C.IMGSZ, batch=C.BATCH, conf=conf,
        project=str(C.ROOT / "runs"), name="test_eval_at_threshold", exist_ok=True, plots=True,
    )
    src = Path(m.save_dir) / "confusion_matrix.png"
    if src.exists():
        shutil.copy(src, C.ARTIFACTS_DIR / "confusion_matrix_at_threshold.png")
    return {"labels": names + ["background"],
            "matrix": m.confusion_matrix.matrix.astype(int).tolist(),
            "orientation": "rows = predicted, columns = true",
            "conf": conf}


# --------------------------------------------------------------------------- 5/6
def image_scores(model: YOLO, imgs: list[Path], class_names: set[str]) -> np.ndarray:
    """Max confidence of any box whose class is in class_names, 0 if none."""
    scores = []
    paths = [str(p) for p in imgs]
    # Small chunks: passing all 700 paths at once makes Ultralytics build one
    # giant batch and run out of GPU memory.
    for i in range(0, len(paths), C.BATCH):
        for res in model.predict(paths[i:i + C.BATCH], conf=0.001, imgsz=C.IMGSZ, verbose=False):
            best = 0.0
            for cls, conf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                if model.names[int(cls)] in class_names:
                    best = max(best, conf)
            scores.append(best)
    return np.array(scores)


def auc(scores, labels) -> float:
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarise(name, scores, labels, params) -> dict:
    sweep = costing.sweep_binary(scores, labels, params)
    row = sweep.loc[sweep.cost_per_1000.idxmin()]
    return {
        "model": name,
        "image_AUC": round(auc(scores, labels), 4),
        "best_threshold": float(row.threshold),
        "recall_at_best": round(float(row.recall), 4),
        "false_alarm_rate_at_best": round(float(row.false_alarm_rate), 4),
        "cost_per_1000_at_best": round(float(row.cost_per_1000), 2),
    }


def plot_cost_curve(sweep: pd.DataFrame, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.threshold, sweep.cost_per_1000, lw=2)
    ax.axvline(chosen, ls="--", color="crimson", label=f"chosen t = {chosen:.2f}")
    ax.set_xlabel("Confidence threshold on 'head' (no helmet)")
    ax.set_ylabel("Expected cost per 1,000 snapshots (€)")
    ax.set_title("Cost vs threshold (model decides alone)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    ds = download_dataset()
    data_yaml, names = write_clean_yaml(ds)
    assert C.POSITIVE_CLASS in names, f"{C.POSITIVE_CLASS} not in {names}"

    print("\n== Describing dataset")
    stats = describe_dataset(ds, names)
    print(json.dumps(stats, indent=2))
    (C.ARTIFACTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    print("\n== Finetuning")
    if os.environ.get("REUSE_WEIGHTS") and C.BEST_WEIGHTS.exists():
        print(f"REUSE_WEIGHTS set: skipping training, using {C.BEST_WEIGHTS}")
        weights = C.BEST_WEIGHTS
    else:
        weights = finetune(data_yaml)

    print("\n== Detection metrics on test split")
    det = evaluate_detection(weights, data_yaml, names)
    det["baselines"] = {
        "coco_zero_shot": {"AP50": 0.0, "mAP50-95": 0.0,
                           "note": "COCO has no 'head' or 'helmet' class, so it cannot produce either label."},
        "predict_nothing": {"AP50": 0.0, "mAP50-95": 0.0,
                            "note": "No boxes at all: every violation is missed."},
    }
    print(json.dumps(det, indent=2))

    print("\n== Image-level scores on test split")
    test_imgs = images_in(ds / SPLITS["test"] / "images")
    pos_id = names.index(C.POSITIVE_CLASS)
    labels = np.array([any(c == pos_id for c, _, _ in read_labels(p)) for p in test_imgs])
    ft_scores = image_scores(YOLO(str(weights)), test_imgs, {C.POSITIVE_CLASS})
    # Best a COCO model can do zero-shot: it sees people, not helmets.
    coco_scores = image_scores(YOLO(C.BASE_MODEL), test_imgs, {"person"})
    trivial_scores = np.zeros(len(test_imgs))  # predicts nothing -> always PASS

    pd.DataFrame({
        "image": [p.name for p in test_imgs],
        "is_violation": labels.astype(int),
        # Full precision on purpose: app.py re-runs the cost sweep from this file,
        # and rounded scores can move an image across a threshold, so the app
        # would pick a different "cheapest" threshold than this script does.
        "score_finetuned": ft_scores,
        "score_coco_person": coco_scores,
    }).to_csv(C.ARTIFACTS_DIR / "test_scores.csv", index=False)

    params = C.cost_params()
    comparison = [
        summarise("Finetuned yolo11n", ft_scores, labels, params),
        summarise("COCO yolo11n zero-shot (person as proxy)", coco_scores, labels, params),
        summarise("Trivial: predict nothing", trivial_scores, labels, params),
    ]
    print(pd.DataFrame(comparison).to_string(index=False))

    print("\n== Cost sweep and operating point")
    sweep = costing.sweep_binary(ft_scores, labels, params)
    sweep.to_csv(C.ARTIFACTS_DIR / "cost_sweep.csv", index=False)
    t_bin = float(sweep.loc[sweep.cost_per_1000.idxmin(), "threshold"])
    plot_cost_curve(sweep, t_bin, C.ARTIFACTS_DIR / "cost_curve.png")
    det["confusion_matrix_at_threshold"] = confusion_at_threshold(weights, data_yaml, names, t_bin)
    print(json.dumps(det["confusion_matrix_at_threshold"], indent=2))

    capacity_share = C.REVIEW_CAPACITY_PER_DAY / C.CASES_PER_DAY
    band = costing.best_band(ft_scores, labels, params, max_review_share=capacity_share)
    band_unconstrained = costing.best_band(ft_scores, labels, params, max_review_share=None)
    binary = costing.evaluate(ft_scores, labels, t_bin, t_bin, params)
    alts = costing.alternatives(params)

    operating = {
        "positive_class": C.POSITIVE_CLASS,
        "assumptions": {**params, "cases_per_day": C.CASES_PER_DAY,
                        "seconds_per_review": C.SECONDS_PER_REVIEW,
                        "review_capacity_per_day": C.REVIEW_CAPACITY_PER_DAY},
        "binary_threshold": t_bin,
        "band": {"t_low": band["t_low"], "t_high": band["t_high"]},
        "band_unconstrained": {"t_low": band_unconstrained["t_low"], "t_high": band_unconstrained["t_high"],
                               "cost_per_1000": round(1000 * band_unconstrained["cost_per_case"], 2),
                               "review_share": round(band_unconstrained["review_share"], 4)},
        "cost_per_1000": {
            **{k: round(1000 * v, 2) for k, v in alts.items()},
            "Model decides alone": round(1000 * binary["cost_per_case"], 2),
            "Model + human review band": round(1000 * band["cost_per_case"], 2),
        },
        "review_share": round(band["review_share"], 4),
        "reviews_per_day": round(band["review_share"] * C.CASES_PER_DAY, 1),
        "reviewer_hours_per_day": round(band["review_share"] * C.CASES_PER_DAY * C.SECONDS_PER_REVIEW / 3600, 2),
        "test_images": int(len(labels)),
        "test_positives": int(labels.sum()),
    }
    print(json.dumps(operating, indent=2))

    (C.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(
        {"detection": det, "image_level_comparison": comparison}, indent=2))
    (C.ARTIFACTS_DIR / "operating_point.json").write_text(json.dumps(operating, indent=2))
    print(f"\nDone. Commit {C.BEST_WEIGHTS.relative_to(C.ROOT)} and artifacts/.")


if __name__ == "__main__":
    main()
