"""Assignment 2 training pipeline.

Download the dataset, finetune YOLO, evaluate on the test split against two
baselines, sweep the cost curve, and save everything app.py needs to the
outputs/ folder, so the app never retrains.

QUICK=1      smoke test: tiny subset, 2 epochs, writes to outputs_quick/
SKIP_TRAIN=1 reuse existing weights (weights/best.pt, or best_quick.pt if QUICK=1)
"""
import gc
import json
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- dataset
WORKSPACE = "roboflow-universe-projects"
PROJECT = "mask-wearing-iskms"
VERSION = 6

# --------------------------------------------------------------- training
QUICK = os.environ.get("QUICK") == "1"
SKIP_TRAIN = os.environ.get("SKIP_TRAIN") == "1"
MODEL = "yolo11n.pt" if QUICK else "yolo11s.pt"
EPOCHS = 2 if QUICK else 40
FRACTION = 0.1 if QUICK else 1.0
IMGSZ = 640
BATCH = 16
SEED = 0

DATA_DIR = Path("data")
WEIGHTS_DIR = Path("weights")
OUT_DIR = Path("outputs_quick" if QUICK else "outputs")

# ------------------------------------------- cost assumptions (match REPORT.md)
BASE_RATE = 0.02            # share of entries without a mask (assumption)
C_FN = 300.0                # missed no-mask: 2% chance of a EUR 15,000 batch hold
C_FP = 1.0                  # false alarm: supervisor 30 s + operator wait
C_TP = 0.10                 # correct block: ~15 s delay while the mask goes on
C_TN = 0.0                  # correct pass: nothing happens
SUPERVISOR_EUR_PER_HOUR = 25.0
REVIEW_SECONDS = 30         # time for a person to check one photo
C_REVIEW = REVIEW_SECONDS / 3600 * SUPERVISOR_EUR_PER_HOUR
HUMAN_MISS = 0.02           # share of no-mask photos a person misses when checking
DAILY_CASES = 1000          # gate entries per day (assumption)
REVIEW_MINUTES_AVAILABLE = 60   # supervisor minutes per day available for review
PRODUCTIVE_MINUTES_PER_PERSON = 420
MAX_REVIEW_SHARE = REVIEW_MINUTES_AVAILABLE * 60 / REVIEW_SECONDS / DAILY_CASES

THRESHOLDS = np.unique(np.concatenate([
    [0.0, 0.001, 0.002, 0.005],
    np.round(np.arange(0.01, 1.0, 0.01), 2),
    [1.01],  # 1.01 means "never flag"
]))


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eval_device():
    """Evaluation uses CUDA if available, otherwise CPU (MPS runs out of memory)."""
    return 0 if torch.cuda.is_available() else "cpu"


def download_dataset() -> Path:
    ds = (
        Roboflow(api_key=load_api_key())
        .workspace(WORKSPACE)
        .project(PROJECT)
        .version(VERSION)
        .download("yolov8", location=str(DATA_DIR), overwrite=False)
    )
    return Path(ds.location) / "data.yaml"


def finetune(data_yaml: Path) -> Path:
    name = "mask_quick" if QUICK else "mask"
    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
        fraction=FRACTION, device=pick_device(), project="runs", name=name,
        exist_ok=True, plots=False,
    )
    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    target = WEIGHTS_DIR / ("best_quick.pt" if QUICK else "best.pt")
    shutil.copy(best, target)
    return target


# ------------------------------------------------------------- scoring
def class_names(data_yaml: Path):
    names = yaml.safe_load(open(data_yaml))["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def max_conf(result, cls_id):
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    keep = cls == cls_id
    return float(conf[keep].max()) if keep.any() else 0.0


def read_truth(label_path: Path, nomask_id: int):
    n_no = n_mask = 0
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if parts:
                if int(float(parts[0])) == nomask_id:
                    n_no += 1
                else:
                    n_mask += 1
    return n_no, n_mask


def predict_chunks(model, sources, chunk=8, **kw):
    """Predict a few images per call, so the batch (and GPU memory) stays small."""
    for i in range(0, len(sources), chunk):
        for r in model.predict(source=sources[i:i + chunk], verbose=False, **kw):
            yield r


def score_split(weights: Path, names, split: str, with_coco: bool) -> pd.DataFrame:
    """One row per image of a split: ground truth plus each model's score."""
    nomask_id, mask_id = names.index("NO-Mask"), names.index("Mask")
    split_dir = DATA_DIR / split
    images = sorted(p for p in (split_dir / "images").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sources = [str(p) for p in images]

    finetuned = {}
    for r in predict_chunks(YOLO(str(weights)), sources, conf=0.001, iou=0.6,
                            imgsz=IMGSZ, device=eval_device()):
        finetuned[Path(r.path).name] = (max_conf(r, nomask_id), max_conf(r, mask_id))

    coco = {}  # pretrained COCO model, zero-shot: only knows 'person' (class 0)
    if with_coco:
        for r in predict_chunks(YOLO(MODEL), sources, conf=0.25, classes=[0],
                                imgsz=IMGSZ, device=eval_device()):
            coco[Path(r.path).name] = max_conf(r, 0)

    rows = []
    for p in images:
        n_no, n_mask = read_truth(split_dir / "labels" / f"{p.stem}.txt", nomask_id)
        rows.append({
            "image": p.name, "y_true": int(n_no > 0), "n_nomask": n_no, "n_mask": n_mask,
            "score_nomask": finetuned[p.name][0], "score_mask": finetuned[p.name][1],
            "score_coco_person": coco.get(p.name, 0.0),
        })
    return pd.DataFrame(rows)


def box_metrics(weights: Path, data_yaml: Path) -> dict:
    """Box-level metrics from ultralytics on the test split."""
    model = YOLO(str(weights))
    common = dict(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH, iou=0.6,
                  device=eval_device(), project="runs", name="eval",
                  exist_ok=True, verbose=False)
    m = model.val(conf=0.001, plots=False, **common)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[m.names[int(c)]] = {
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i]),
        }
    m25 = model.val(conf=0.25, plots=True, **common)
    return {
        "map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class,
        "confusion_matrix_conf0.25": m25.confusion_matrix.matrix.tolist(),
        "confusion_matrix_labels": list(m.names.values()) + ["background"],
    }


def image_level(y, pred) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


# ------------------------------------------------------------- costing
def outcome_rates(y, s, t_low, t_high) -> dict:
    """score < t_low -> pass; t_low <= score < t_high -> review; score >= t_high -> fail."""
    pos = y == 1
    fail, review = s >= t_high, (s >= t_low) & (s < t_high)
    passed = s < t_low
    return {"tpr": fail[pos].mean(), "rev_pos": review[pos].mean(), "fnr": passed[pos].mean(),
            "fpr": fail[~pos].mean(), "rev_neg": review[~pos].mean(), "tnr": passed[~pos].mean()}


def cost_per_case(r) -> float:
    reviewed_pos = C_REVIEW + HUMAN_MISS * C_FN + (1 - HUMAN_MISS) * C_TP
    pos = r["tpr"] * C_TP + r["rev_pos"] * reviewed_pos + r["fnr"] * C_FN
    neg = r["fpr"] * C_FP + r["rev_neg"] * C_REVIEW + r["tnr"] * C_TN
    return BASE_RATE * pos + (1 - BASE_RATE) * neg


def review_share(r) -> float:
    return BASE_RATE * r["rev_pos"] + (1 - BASE_RATE) * r["rev_neg"]


def scenario(y, s, lo, hi) -> dict:
    r = outcome_rates(y, s, lo, hi)
    share = review_share(r)
    reviews_per_day = share * DAILY_CASES
    minutes = reviews_per_day * REVIEW_SECONDS / 60
    return {
        "t_low": float(lo), "t_high": float(hi),
        "cost_per_1000": float(1000 * cost_per_case(r)),
        "cost_per_day": float(DAILY_CASES * cost_per_case(r)),
        "review_share": float(share), "reviews_per_day": float(reviews_per_day),
        "review_minutes_per_day": float(minutes),
        "people_needed": float(minutes / PRODUCTIVE_MINUTES_PER_PERSON),
        **{k: float(v) for k, v in r.items()},
    }


def sweep_single(y, s) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        r = outcome_rates(y, s, t, t)
        rows.append({"threshold": float(t), "cost_per_1000": float(1000 * cost_per_case(r)),
                     "recall": float(r["tpr"]), "false_alarm_rate": float(r["fpr"])})
    return pd.DataFrame(rows)


def best_hybrid(y, s):
    best = None
    for lo in THRESHOLDS:
        for hi in THRESHOLDS:
            if hi < lo:
                continue
            r = outcome_rates(y, s, lo, hi)
            if review_share(r) > MAX_REVIEW_SHARE:
                continue
            c = cost_per_case(r)
            if best is None or c < best[0]:
                best = (c, lo, hi)
    return best[1], best[2]


def main():
    np.random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    data_yaml = download_dataset()
    names = class_names(data_yaml)
    if SKIP_TRAIN:
        weights = WEIGHTS_DIR / ("best_quick.pt" if QUICK else "best.pt")
    else:
        weights = finetune(data_yaml)
    print(f"Using weights: {weights}")
    gc.collect()
    torch.cuda.empty_cache()

    # Thresholds are CHOSEN on the validation split and only REPORTED on the test split.
    val = score_split(weights, names, "valid", with_coco=False)
    test = score_split(weights, names, "test", with_coco=True)
    val.to_csv(OUT_DIR / "valid_scores.csv", index=False)
    test.to_csv(OUT_DIR / "test_scores.csv", index=False)
    yv, sv = val["y_true"].to_numpy(), val["score_nomask"].to_numpy()
    y, s = test["y_true"].to_numpy(), test["score_nomask"].to_numpy()

    baselines = {
        "finetuned (conf>=0.25)": image_level(y, s >= 0.25),
        "COCO zero-shot, person = NO-Mask": image_level(y, test["score_coco_person"] >= 0.25),
        "predict nothing": image_level(y, np.zeros_like(y)),
    }

    sweep_val, sweep = sweep_single(yv, sv), sweep_single(y, s)
    sweep_val.to_csv(OUT_DIR / "cost_sweep_valid.csv", index=False)
    sweep.to_csv(OUT_DIR / "cost_sweep.csv", index=False)
    best_t = float(sweep_val.loc[sweep_val["cost_per_1000"].idxmin(), "threshold"])
    lo, hi = best_hybrid(yv, sv)

    def make_designs(yy, ss):
        return {
            "do nothing (pass everyone)": scenario(yy, ss, 1.01, 1.01),
            "person checks everything": scenario(yy, ss, 0.0, 1.01),
            "model alone (best single threshold)": scenario(yy, ss, best_t, best_t),
            "model + human review (best two thresholds)": scenario(yy, ss, lo, hi),
        }
    designs, designs_valid = make_designs(y, s), make_designs(yv, sv)

    config = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (int, float, str, bool))}
    config.update({"best_single_threshold": best_t, "review_low": float(lo),
                   "review_high": float(hi), "thresholds_chosen_on": "valid",
                   "class_names": names, "weights": str(weights),
                   "valid_images": int(len(val)), "valid_positive_images": int(yv.sum()),
                   "test_images": int(len(test)), "test_positive_images": int(y.sum())})
    results = {"box_metrics": box_metrics(weights, data_yaml),
               "image_level_baselines": baselines, "designs": designs,
               "designs_on_valid": designs_valid}
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(7, 4))
    for sw, label in [(sweep_val, "validation (used to choose)"), (sweep, "test (reported)")]:
        pl = sw[sw["threshold"] <= 1.0]
        plt.plot(pl["threshold"], pl["cost_per_1000"], label=label)
    plt.axvline(best_t, color="red", linestyle="--", label=f"chosen threshold {best_t:.2f}")
    plt.xlabel("confidence threshold"); plt.ylabel("expected cost per 1,000 cases (EUR)")
    plt.yscale("log"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "cost_curve.png", dpi=150)

    print("\n=== Box metrics (test) ===")
    print(json.dumps(results["box_metrics"], indent=2))
    print("\n=== Image-level baselines on test (conf 0.25) ===")
    for k, v in baselines.items():
        print(f"{k}: {v}")
    print(f"\n=== Thresholds chosen on VALIDATION ({int(yv.sum())} positive images), "
          f"reported on TEST ({int(y.sum())} positive images) ===")
    print(f"base rate {BASE_RATE:.0%}, {DAILY_CASES} cases/day")
    for k in designs:
        d, dv = designs[k], designs_valid[k]
        print(f"{k}: t=({d['t_low']}, {d['t_high']}) cost/1000 test=EUR {d['cost_per_1000']:.0f} "
              f"(valid EUR {dv['cost_per_1000']:.0f}) review={d['review_share']:.1%} "
              f"people={d['people_needed']:.2f}")
    print(f"\nSaved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
