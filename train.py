"""
Assignment 2 training pipeline: potato sorting on a conveyor.

    uv run python main.py train

Start to finish, no manual steps:
  1. download Roboflow Universe dataset ms-workspace-m1gci/potato-sorting-on-conveyor v9
  2. describe it (instances per class, split sizes, object size, where the frames come from)
  3. finetune yolo11s from the COCO checkpoint (resumes if a run was interrupted)
  4. evaluate on the test split: finetuned model, COCO yolo11s zero-shot, predict-nothing
  5. turn boxes into one decision per object on the belt and save the scores
  6. sweep the reject threshold and the pass/review/reject band against the cost model
Everything the dashboard needs lands in artifacts/; the weights land in weights/.
"""

import bisect
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import costing

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "potato-v9"
RUNS_DIR = ROOT / "runs"
ART_DIR = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "potato_yolo11s.pt"

WORKSPACE = "ms-workspace-m1gci"
PROJECT = "potato-sorting-on-conveyor"
VERSION = 9

def _device() -> str:
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


DEVICE = _device()
TRAIN_CFG = dict(
    model="yolo11s.pt",   # COCO-pretrained checkpoint we finetune from
    epochs=60,
    imgsz=640,            # frames are 512x512; upsampling to 640 gives the ~13 px blemishes more pixels
    batch=32,
    close_mosaic=10,
    flipud=0.5,           # top-down camera: vertical flips are as plausible as horizontal ones
    seed=0,
    deterministic=True,
    workers=2,
    device=DEVICE,        # trained on a Colab T4; falls back to CPU (slow: ~17 min/epoch for yolo11n)
    patience=15,
)
RUN_NAME = "potato-yolo11s"
PRED_CONF = 0.001         # keep low-confidence boxes so the threshold sweep can see them
CM_CONF = 0.25            # confidence for the box-level confusion matrix


# 1. data ---------------------------------------------------------------------------

def download_dataset() -> Path:
    if (DATA_DIR / "data.yaml").exists() and (DATA_DIR / "test" / "images").is_dir():
        print(f"Dataset already at {DATA_DIR}")
    else:
        from roboflow import Roboflow
        from api_key import load_api_key

        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download("yolov11", location=str(DATA_DIR), overwrite=True)

    # Roboflow writes relative paths that only resolve from inside the folder; pin them.
    cfg = yaml.safe_load((DATA_DIR / "data.yaml").read_text())
    cfg["path"] = str(DATA_DIR)
    cfg["train"], cfg["val"], cfg["test"] = "train/images", "valid/images", "test/images"
    out = DATA_DIR / "data_abs.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def read_labels(split: str) -> dict:
    """image stem -> array of [cls, x1, y1, x2, y2] in normalised coordinates."""
    out = {}
    for f in sorted((DATA_DIR / split / "labels").glob("*.txt")):
        rows = [list(map(float, line.split()[:5])) for line in f.read_text().splitlines() if line.strip()]
        arr = np.array(rows, dtype=float).reshape(-1, 5)
        c, x, y, w, h = arr.T
        out[f.stem] = np.stack([c, x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)
    return out


def frame_source(name: str) -> str:
    if name.startswith("BELANA_auto"):
        return "conveyor camera A (BELANA_auto)"
    if name.startswith("kartoffel_au"):
        return "conveyor camera B (kartoffel_auto)"
    return "other (web/phone photos)"


def describe_dataset(names: list) -> dict:
    stats = {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "splits": {}}
    for split in ["train", "valid", "test"]:
        labels = read_labels(split)
        per_class = {}
        for k, name in enumerate(names):
            boxes = [b[b[:, 0] == k] for b in labels.values()]
            allb = np.concatenate(boxes) if boxes else np.zeros((0, 5))
            side = np.sqrt((allb[:, 3] - allb[:, 1]) * (allb[:, 4] - allb[:, 2]))
            per_class[name] = {
                "instances": int(len(allb)),
                "images_with": int(sum(len(b) > 0 for b in boxes)),
                "rel_size_median": float(np.median(side)) if len(side) else None,
                "rel_size_p10": float(np.percentile(side, 10)) if len(side) else None,
                "rel_size_p90": float(np.percentile(side, 90)) if len(side) else None,
            }
        sources = {}
        for p in (DATA_DIR / split / "images").glob("*"):
            s = frame_source(p.name)
            sources[s] = sources.get(s, 0) + 1
        stats["splits"][split] = {
            "images": len(labels),
            "objects_per_image_mean": float(np.mean([len(b) for b in labels.values()])),
            "sources": sources,
            "per_class": per_class,
        }
    # Conveyor frames carry a capture timestamp. How close is each test frame to a train frame?
    train_ts = sorted(t for p in (DATA_DIR / "train" / "images").glob("*") if (t := capture_time(p.name)))
    for split in ["valid", "test"]:
        gaps = []
        for p in (DATA_DIR / split / "images").glob("*"):
            t = capture_time(p.name)
            if t is None or not train_ts:
                continue
            i = bisect.bisect(train_ts, t)
            gaps.append(min(abs((train_ts[j] - t).total_seconds()) for j in (i - 1, i) if 0 <= j < len(train_ts)))
        stats[f"{split}_frames_within_2s_of_train"] = int(sum(g < 2 for g in gaps))
        stats[f"{split}_frames_with_timestamp"] = len(gaps)
        stats[f"{split}_median_gap_to_train_s"] = float(np.median(gaps)) if gaps else None
    return stats


def capture_time(name: str):
    m = re.search(r"_(\d{8})_(\d{6})_(\d{6})", name)
    return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S%f") if m else None


# 2. training -----------------------------------------------------------------------

def train_model(data_yaml: Path) -> Path:
    from ultralytics import YOLO

    run_dir = RUNS_DIR / "detect" / RUN_NAME
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    done = results_csv.exists() and len(pd.read_csv(results_csv)) >= TRAIN_CFG["epochs"]
    if not done and last.exists():
        # Ultralytics strips the optimizer from last.pt when a run ends (incl. early stopping)
        import torch
        done = torch.load(last, map_location="cpu", weights_only=False).get("optimizer") is None

    if done and best.exists():
        print(f"Training run already complete: {best}")
    elif last.exists():
        print(f"Resuming interrupted run from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        cfg = dict(TRAIN_CFG)
        model = YOLO(cfg.pop("model"))
        model.train(data=str(data_yaml), project=str(RUNS_DIR / "detect"), name=RUN_NAME,
                    exist_ok=True, plots=True, **cfg)

    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    print(f"Saved weights to {WEIGHTS}")
    return WEIGHTS


# 3. detection metrics on the test split --------------------------------------------

def detection_metrics(weights: Path, data_yaml: Path, names: list) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=TRAIN_CFG["imgsz"], batch=8,
                               device=DEVICE, plots=True, project=str(RUNS_DIR / "eval"),
                               name="finetuned-test", exist_ok=True, verbose=False)
    per_class = {}
    for i, k in enumerate(m.box.ap_class_index):
        # P and R are Ultralytics' values at the confidence that maximises mean F1
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(k)]] = {"precision": float(p), "recall": float(r),
                                    "ap50": float(ap50), "ap50_95": float(ap)}
    # AP needs every box down to conf 0.001, but Ultralytics also builds the confusion matrix at the
    # validation conf, which would count thousands of near-zero boxes. A second pass at 0.25 gives a
    # matrix that shows what the model actually says. Rows = predicted, columns = true.
    cm = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=TRAIN_CFG["imgsz"], batch=8,
                                device=DEVICE, conf=CM_CONF, plots=True, project=str(RUNS_DIR / "eval"),
                                name="finetuned-test-cm", exist_ok=True, verbose=False)
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr),
            "per_class": per_class,
            "confusion_conf": CM_CONF,
            "confusion_matrix": cm.confusion_matrix.matrix.astype(int).tolist(),
            "confusion_labels": list(names) + ["background"]}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def ap50_agnostic(preds: dict, gts: dict) -> tuple:
    """Class-agnostic AP50 and max recall: does the model find the objects at all, label ignored?"""
    scores, tps, n_gt = [], [], 0
    for stem, g in gts.items():
        n_gt += len(g)
        p = preds.get(stem, np.zeros((0, 6)))
        p = p[np.argsort(-p[:, 5])]
        taken = np.zeros(len(g), bool)
        ious = iou_matrix(p[:, 1:5], g[:, 1:5])
        for i in range(len(p)):
            j = int(np.argmax(ious[i])) if len(g) else -1
            ok = j >= 0 and ious[i, j] >= 0.5 and not taken[j]
            if ok:
                taken[j] = True
            scores.append(p[i, 5]); tps.append(ok)
    if not scores:
        return 0.0, 0.0
    order = np.argsort(-np.array(scores)); tp = np.array(tps)[order]
    ctp = np.cumsum(tp); recall = ctp / max(n_gt, 1); precision = ctp / np.arange(1, len(tp) + 1)
    mrec = np.concatenate([[0], recall, [1]]); mpre = np.concatenate([[1], precision, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), float(recall[-1])


def predict_split(weights: str, split: str) -> dict:
    """image stem -> array of [cls, x1, y1, x2, y2, conf] in normalised coordinates."""
    from ultralytics import YOLO

    model = YOLO(weights)
    out = {}
    paths = sorted((DATA_DIR / split / "images").glob("*.jpg"))
    for r in model.predict([str(p) for p in paths], imgsz=TRAIN_CFG["imgsz"], conf=PRED_CONF,
                           device=DEVICE, stream=True, verbose=False, max_det=300):
        b = r.boxes
        out[Path(r.path).stem] = np.concatenate(
            [b.cls.cpu().numpy()[:, None], b.xyxyn.cpu().numpy(), b.conf.cpu().numpy()[:, None]], axis=1)
    return out


def coco_baseline(names: list) -> dict:
    """COCO yolo11s (same architecture) with no finetuning. COCO has no potato, stone or blemish class, so its
    per-class AP on our labels is zero by construction. We also measure whether it finds the
    objects at all (class-agnostic AP50) and what it thinks they are."""
    from ultralytics import YOLO

    preds = predict_split("yolo11s.pt", "test")
    coco_names = YOLO("yolo11s.pt").names
    hist = {}
    for p in preds.values():
        for c in p[p[:, 5] >= 0.25, 0].astype(int):
            hist[coco_names[c]] = hist.get(coco_names[c], 0) + 1
    ap50, rec = ap50_agnostic(preds, read_labels("test"))
    zero = {"precision": 0.0, "recall": 0.0, "ap50": 0.0, "ap50_95": 0.0}
    return {"map50": 0.0, "map50_95": 0.0, "per_class": {n: dict(zero) for n in names},
            "agnostic_ap50": ap50, "agnostic_max_recall": rec,
            "coco_labels_at_conf_0.25": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:8])}


# 4. one decision per object on the belt --------------------------------------------

def build_units(preds: dict, names: list) -> pd.DataFrame:
    """Turn boxes into the things a sorter actually acts on.

    A unit is one object on the belt: every ground-truth potato, stone or cut piece.
    The dataset boxes the blemish ('bad'), not the bad potato, so a potato is a defect unit
    when a ground-truth blemish centre lies inside it. Each unit is scored by the most
    confident predicted blemish/cut whose centre falls inside it, and by the most confident
    predicted stone overlapping it (IoU >= 0.3). Reject predictions that land on no object are
    kept as 'phantom' units: the ejector would still fire on them.
    """
    BAD, CUT, POT, STONE = (names.index(n) for n in ["bad", "cut", "potato", "stone"])
    rows = []
    for stem, g in read_labels("test").items():
        p = preds.get(stem, np.zeros((0, 6)))
        objs = g[np.isin(g[:, 0], [POT, STONE, CUT])]
        cuts = objs[objs[:, 0] == CUT]
        if len(cuts):   # a cut piece is often also boxed as a potato: keep it once, as 'cut'
            dup = (iou_matrix(objs[:, 1:5], cuts[:, 1:5]).max(1) > 0.5) & (objs[:, 0] == POT)
            objs = objs[~dup]
        blems = g[g[:, 0] == BAD]
        bx = (blems[:, 1] + blems[:, 3]) / 2; by = (blems[:, 2] + blems[:, 4]) / 2

        units, used = [], np.zeros(len(blems), bool)
        for o in objs:
            cls = int(o[0])
            if cls == STONE:
                truth = "stone"
            elif cls == CUT:
                truth = "defect"
            else:
                hit = (bx >= o[1]) & (bx <= o[3]) & (by >= o[2]) & (by <= o[4])
                used |= hit
                truth = "defect" if hit.any() else "good"
            units.append((o[1:5], truth))
        for b in blems[~used]:   # blemish with no potato box around it: still a defect unit
            units.append((b[1:5], "defect"))

        ubox = np.array([u[0] for u in units], float).reshape(-1, 4)
        area = (ubox[:, 2] - ubox[:, 0]) * (ubox[:, 3] - ubox[:, 1])
        s_def = np.zeros(len(units)); s_st = np.zeros(len(units)); phantoms = []
        for q in p[np.isin(p[:, 0], [BAD, CUT, STONE])]:
            qx, qy = (q[1] + q[3]) / 2, (q[2] + q[4]) / 2
            inside = (qx >= ubox[:, 0]) & (qx <= ubox[:, 2]) & (qy >= ubox[:, 1]) & (qy <= ubox[:, 3])
            ious = iou_matrix(q[None, 1:5], ubox)[0]
            if int(q[0]) == STONE:
                cand = np.where(inside & (ious >= 0.3))[0]
                if len(cand):
                    j = cand[np.argmax(ious[cand])]; s_st[j] = max(s_st[j], q[5]); continue
            else:
                cand = np.where(inside)[0]
                if len(cand):   # smallest enclosing object = the potato the spot is on
                    j = cand[np.argmin(area[cand])]; s_def[j] = max(s_def[j], q[5]); continue
            phantoms.append(q)
        for k, (_, truth) in enumerate(units):
            rows.append(dict(image=stem, truth=truth, phantom=False, score_defect=s_def[k],
                             score_stone=s_st[k], score=max(s_def[k], s_st[k])))
        # merge overlapping stray boxes so one stray detection is one ejection, not five
        kept = []
        for q in sorted(phantoms, key=lambda q: -q[5]):
            if all(iou_matrix(q[None, 1:5], k[None, 1:5])[0, 0] < 0.3 for k in kept):
                kept.append(q)
        for q in kept:
            st = int(q[0]) == STONE
            rows.append(dict(image=stem, truth="good", phantom=True, score_defect=0.0 if st else q[5],
                             score_stone=q[5] if st else 0.0, score=q[5]))
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def operating_point(units: pd.DataFrame, sweep: pd.DataFrame, band: dict) -> dict:
    best = sweep.loc[sweep.cost_per_1000.idxmin()]
    t = float(best.threshold)
    real = units[~units.phantom]
    op = {"threshold": t, "cost_per_1000": float(best.cost_per_1000),
          "t_low": band["t_low"], "t_high": band["t_high"],
          "band_cost_per_1000": band["cost_per_1000"], "review_share": band["review_share"]}
    for grp in ["defect", "stone"]:
        s = real.loc[real.truth == grp, "score"]
        k = int((s >= t).sum())
        op[f"recall_{grp}_test"] = k / max(len(s), 1)
        op[f"recall_{grp}_ci95"] = wilson(k, len(s))
        op[f"n_{grp}_test"] = int(len(s))
    good = units[units.truth == "good"]
    n_good = int((real.truth == "good").sum())
    op["false_rejects_per_good_test"] = float((good.score >= t).sum() / max(n_good, 1))
    return op


# 5. orchestration ------------------------------------------------------------------

def main() -> None:
    ART_DIR.mkdir(exist_ok=True)
    data_yaml = download_dataset()
    names = list(yaml.safe_load(data_yaml.read_text())["names"])

    stats = describe_dataset(names)
    (ART_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    weights = train_model(data_yaml)
    shutil.copy(RUNS_DIR / "detect" / RUN_NAME / "results.csv", ART_DIR / "training_curve.csv")

    print("Evaluating the finetuned model on the test split")
    finetuned = detection_metrics(weights, data_yaml, names)
    print("Evaluating COCO yolo11s zero-shot on the test split")
    coco = coco_baseline(names)
    zero = {"precision": 0.0, "recall": 0.0, "ap50": 0.0, "ap50_95": 0.0}
    trivial = {"map50": 0.0, "map50_95": 0.0, "per_class": {n: dict(zero) for n in names},
               "note": "predicts no boxes: every object passes, nothing is ever rejected"}

    print("Scoring every object on the test belt")
    preds = predict_split(str(weights), "test")
    finetuned["agnostic_ap50"], finetuned["agnostic_max_recall"] = ap50_agnostic(preds, read_labels("test"))
    units = build_units(preds, names)
    units.to_csv(ART_DIR / "test_units.csv", index=False)

    print("Sweeping thresholds against the cost model")
    sweep = costing.sweep_single(units)
    sweep.to_csv(ART_DIR / "cost_sweep.csv", index=False)
    # Shipped band: cheapest one the available reviewers can actually keep up with.
    band = costing.best_band(units, max_review_share=costing.max_review_share())
    band_free = costing.best_band(units)
    op = operating_point(units, sweep, band)
    real = units[~units.phantom]
    metrics = {
        "dataset": f"{WORKSPACE}/{PROJECT}/{VERSION}",
        "train_cfg": TRAIN_CFG,
        "finetuned": finetuned, "coco_zero_shot": coco, "predict_nothing": trivial,
        "test_units": {t: int((real.truth == t).sum()) for t in ["good", "defect", "stone"]},
        "phantom_units": int(units.phantom.sum()),
        "costs": costing.COSTS, "human": costing.HUMAN, "base_rates": costing.BASE_RATES,
        "line": costing.LINE,
        "operating_point": op,
        "band_detail": band,
        "band_unconstrained": band_free,
        "staffing_unconstrained": costing.staffing(band_free["review_share"]),
        "max_review_share": costing.max_review_share(),
        "alternatives": costing.alternatives(units, op["threshold"], band),
        "staffing": costing.staffing(op["review_share"]),
        "sensitivity": costing.sensitivity(units),
    }
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    costing.plot_sweep(sweep, op["threshold"], ART_DIR / "cost_vs_threshold.png")
    for f in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = RUNS_DIR / "eval" / "finetuned-test-cm" / f
        if src.exists():
            shutil.copy(src, ART_DIR / f)
    print(json.dumps({k: metrics[k] for k in ["test_units", "operating_point", "alternatives"]}, indent=2))


if __name__ == "__main__":
    main()
