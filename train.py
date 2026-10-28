"""
Assignment 2: training pipeline.   Run with:  uv run python main.py train

What it does, start to finish, with no manual steps:
  1. Downloads the Roboflow dataset (workspace / project / version below).
  2. Describes it (instances per class, split sizes, box size, leakage check).
  3. Finetunes yolo11n from the pretrained COCO checkpoint -> weights/best.pt
  4. Evaluates on the TEST split: per-class box metrics, confusion matrix,
     and image-level metrics ("does this metre have ANY defect?") for the
     finetuned model and both baselines (COCO zero-shot, predict nothing /
     predict everything).
  5. Costs every decision: sweeps the pass / human-review / fail thresholds,
     picks the cheapest pair, compares the four strategies, and runs a
     sensitivity analysis.
  6. Saves everything app.py needs into results/. app.py trains nothing.

Handy environment variables (defaults train the real model):
  EPOCHS=2 FRACTION=0.05   quick smoke test of the whole pipeline
                           (also IMGSZ=320 BATCH=4 on a small machine)
  SKIP_TRAIN=1             reuse weights/best.pt and only redo steps 4-6
                           (use it when you change a cost below)

THE DECISIONS THAT ARE YOURS are in the CONFIG block. Every figure there has
to be defended in REPORT.md, without this code in front of you.
"""

import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================== CONFIG ====================================
# --- Data (goes in REPORT.md exactly like this) ---------------------------
WORKSPACE = "deepti-raj-7uqio"
PROJECT = "textile-defects-tqim4"
VERSION = 2

# --- Training settings (record where you trained and these in REPORT.md) --
MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 60))
FRACTION = float(os.environ.get("FRACTION", 1.0))
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 16))
PATIENCE = 15
SEED = 0

# --- Business problem: ONE unit = ONE metre of denim = ONE image ----------
# Positive = "this metre has at least one defect" (any class). Fail = scrap.
C_FP = 10.0          # EUR: scrap a good metre = what a metre of denim costs
FN_MULTIPLE = 3.0    # a missed defect costs this many times a false positive
C_TP = 0.0           # EUR: a real defect correctly scrapped is a sunk cost
C_TN = 0.0           # EUR: a good metre correctly passed
BASE_RATE = 0.05     # assumed share of metres with a defect in a real factory
                     # (the dataset has ~70% because it was built for training)

# --- Human review ---------------------------------------------------------
HOURLY_COST = 14.29  # EUR/h: 1,500 gross x 14 pays + 30.65% employer SS / 1,920 h
REVIEW_SECONDS = 30  # seconds to look at one flagged metre. NOTE: if ~97% of metres end up
                     # in review this is really normal inspection, and published figures
                     # for manual fabric inspection are much faster (~10 yards/min = ~7 s/m).
                     # Decide which you mean and defend it.
HUMAN_TPR = 0.90     # ASSUMPTION: share of real defects a reviewer catches
HUMAN_FPR = 0.02     # ASSUMPTION: share of good metres a reviewer wrongly scraps
METRES_PER_DAY = 20_000   # ASSUMPTION: daily volume of the line
SHIFT_HOURS = 8           # productive hours per reviewer per day

# Optional nicer names for the report and dashboard, keyed by dataset name.
# Fill in only AFTER you have checked the labels yourself, e.g.
#   {"Knot": "Stain (labelled Knot)", "Stain": "Line (labelled Stain)"}
DISPLAY_NAMES = {}
# ==========================================================================

BASE = dict(
    c_fp=C_FP,
    c_fn=C_FP * FN_MULTIPLE,
    c_tp=C_TP,
    c_tn=C_TN,
    base_rate=BASE_RATE,
    c_review=HOURLY_COST / 3600 * REVIEW_SECONDS,
    h_tpr=HUMAN_TPR,
    h_fpr=HUMAN_FPR,
)
# Candidate thresholds. A metre is PASSED if its score < low, FAILED if its
# score >= high, and sent to a person in between. 1.01 means "never".
THRESHOLDS = np.round(np.concatenate([np.linspace(0, 1, 101), [1.01]]), 4)


# ------------------------------ small helpers -----------------------------
def jdump(obj, path):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    Path(path).write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_labels(path):
    """Return [(class_id, w, h)] from a YOLO label file (w, h normalised)."""
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                rows.append((int(float(p[0])), float(p[3]), float(p[4])))
    return rows


def list_images(split_dir):
    return sorted(p for p in (Path(split_dir) / "images").iterdir() if p.suffix.lower() in IMG_EXT)


# ------------------------------ 1. dataset --------------------------------
def ensure_dataset():
    dataset_dir = DATASETS / f"{PROJECT}-v{VERSION}"
    if not (dataset_dir / "train" / "images").exists():
        from roboflow import Roboflow

        DATASETS.mkdir(exist_ok=True)
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(dataset_dir)
        )
    original = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    names = original["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    # Roboflow's data.yaml uses "../train/images", which breaks depending on
    # where you run from. Write our own with absolute paths.
    data_yaml = dataset_dir / "data_local.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": {i: n for i, n in enumerate(names)},
            }
        )
    )
    return dataset_dir, data_yaml, names


# ------------------------------ 2. describe -------------------------------
def dataset_stats(dataset_dir, names):
    out = {"classes": names, "splits": {}}
    bases = {}
    for split in ("train", "valid", "test"):
        imgs = list_images(dataset_dir / split)
        inst, img_with, areas, empty = Counter(), Counter(), [], 0
        for img in imgs:
            labels = read_labels(dataset_dir / split / "labels" / f"{img.stem}.txt")
            empty += not labels
            for c, w, h in labels:
                inst[names[c]] += 1
                areas.append(w * h)
            for c in {c for c, _, _ in labels}:
                img_with[names[c]] += 1
        bases[split] = [i.name.split(".rf.")[0] for i in imgs]
        q = np.percentile(areas, [10, 50, 90]) if areas else [0, 0, 0]
        out["splits"][split] = {
            "images": len(imgs),
            "images_without_defect": empty,
            "share_without_defect": empty / max(len(imgs), 1),
            "instances_per_class": {n: inst.get(n, 0) for n in names},
            "images_with_class": {n: img_with.get(n, 0) for n in names},
            "box_area_share_p10_p50_p90": [float(x) for x in q],
        }
    train_valid = set(bases["train"]) | set(bases["valid"])
    shared_test = sum(b in train_valid for b in bases["test"])
    out["leakage_check"] = {
        "test_images_whose_base_name_is_also_in_train_or_valid": shared_test,
        "share_of_test": shared_test / max(len(bases["test"]), 1),
        "train_valid_shared_base_names": len(set(bases["train"]) & set(bases["valid"])),
        "note": "Base name = file name before Roboflow's '.rf.<hash>'. A high share "
                "suggests near-duplicate images across splits and an inflated test score.",
    }
    return out


# ------------------------------ 3. train ----------------------------------
def pretrained_path():
    CACHE.mkdir(exist_ok=True)
    return str(CACHE / MODEL)


def train(data_yaml):
    from ultralytics import YOLO

    model = YOLO(pretrained_path())
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        fraction=FRACTION,
        device=pick_device(),
        project=str(RUNS),
        name="train",
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    WEIGHTS.mkdir(exist_ok=True)
    for old in WEIGHTS.glob("*.pt"):
        old.unlink()
    shutil.copy(best, WEIGHTS / "best.pt")
    curve = Path(model.trainer.save_dir) / "results.csv"
    if curve.exists():
        shutil.copy(curve, RESULTS / "training_curve.csv")
    return WEIGHTS / "best.pt"


# ------------------------------ 4. evaluate -------------------------------
def evaluate_boxes(weights, data_yaml, names):
    """Per-class box metrics and the class-level confusion matrix on TEST."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    common = dict(data=str(data_yaml), split="test", imgsz=IMGSZ, device=pick_device(),
                  verbose=False)
    wide = model.val(conf=0.001, iou=0.6, plots=False, **common)   # for AP (full PR curve)
    # Ultralytics only fills the confusion matrix when plots=True, so this run needs it.
    at25 = model.val(conf=0.25, iou=0.6, plots=True, **common)     # for the confusion matrix
    box = wide.box
    per_class = []
    for i, ci in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({
            "class": names[int(ci)],
            "instances_in_test": int(wide.nt_per_class[int(ci)]),
            "precision": float(p), "recall": float(r),
            "ap50": float(ap50), "map50_95": float(ap),
        })
    overall = {"precision": float(box.mp), "recall": float(box.mr),
               "ap50": float(box.map50), "map50_95": float(box.map)}
    cm = at25.confusion_matrix.matrix
    return overall, per_class, {
        "labels": names + ["background"],
        "orientation": "rows = predicted class, columns = true class",
        "conf": 0.25,
        "matrix": cm.tolist(),
    }


SCORE_CONF = 0.01   # boxes below this confidence are not kept (same resolution as the threshold grid)


def image_scores(weights, split_dir, dataset_names, label):
    """One row per image: its score = highest box confidence of ANY class.

    Images are predicted ONE AT A TIME on purpose. Passing the whole list in a
    single call makes Ultralytics treat it as one huge batch, its NMS time limit
    (2 s + 0.05 s per image) is exceeded, and the images after that point come
    back with NO boxes, which silently turns real defects into score 0.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows = []
    for img in list_images(split_dir):
        r = model.predict(str(img), imgsz=IMGSZ, conf=SCORE_CONF, iou=0.6, batch=1,
                          device=pick_device(), verbose=False)[0]
        gt = read_labels(Path(split_dir) / "labels" / f"{img.stem}.txt")
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
        if len(confs):
            k = int(confs.argmax())
            score, top = float(confs[k]), model.names[int(r.boxes.cls[k])]
        else:
            score, top = 0.0, ""
        rows.append({
            "image": img.name,
            "y_true": int(len(gt) > 0),
            "n_gt_boxes": len(gt),
            "gt_classes": "|".join(sorted({dataset_names[c] for c, _, _ in gt})),
            "score": round(score, 5),
            "top_class": top,
        })
    write_csv(RESULTS / f"{label}_scores.csv", rows)
    defective = [r for r in rows if r["y_true"] == 1]
    missed = sum(r["score"] == 0.0 for r in defective)
    print(f"[{label}] {len(rows)} images scored; {missed} of {len(defective)} defective ones "
          f"have no box at all (score 0)")
    return rows


# ------------------------------ 5. cost -----------------------------------
def confusion_at(y, s, t):
    pred = s >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "accuracy": (tp + tn) / len(y)}


def wilson_upper(k, n, z=1.96):
    """Upper end of the 95% interval for a rate seen k times in n cases.
    With 0 false fails in 52 clean metres the true rate is NOT 0: it could be ~7%."""
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (centre + margin) / denom))


def expected_cost(pass_p, fail_p, rev_p, pass_n, fail_n, rev_n, P):
    """EUR and review share per 1,000 metres.

    p = DEFECTIVE metres, n = CLEAN metres: the share of each that is passed,
    failed (scrapped) or sent to a person. The BASE RATE re-weights the two so
    the answer reflects a real factory, not the 70%-defective dataset.
    """
    cost_pos = (pass_p * P["c_fn"] + fail_p * P["c_tp"]
                + rev_p * (P["c_review"] + P["h_tpr"] * P["c_tp"] + (1 - P["h_tpr"]) * P["c_fn"]))
    cost_neg = (pass_n * P["c_tn"] + fail_n * P["c_fp"]
                + rev_n * (P["c_review"] + P["h_fpr"] * P["c_fp"] + (1 - P["h_fpr"]) * P["c_tn"]))
    cost = 1000 * (P["base_rate"] * cost_pos + (1 - P["base_rate"]) * cost_neg)
    review = P["base_rate"] * rev_p + (1 - P["base_rate"]) * rev_n
    return cost, review


def single_cost(c, P):
    """Plain pass/fail with no human: EUR per 1,000 metres."""
    tpr = c["recall"]
    fpr = c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else 0.0
    return float(expected_cost(1 - tpr, tpr, 0.0, 1 - fpr, fpr, 0.0, P)[0])


def cost_grid(sp, sn, P, T=THRESHOLDS):
    """EUR per 1,000 metres for every (low, high) pair.

    Rows index `low` (pass if score < low), columns index `high` (fail if
    score >= high). low <= high; the gap between them goes to a person.
    The diagonal low == high is a plain pass/fail with no human review.
    """
    def parts(s):
        passed = (s[None, :] < T[:, None]).mean(1)[:, None]    # (K,1) by low
        failed = (s[None, :] >= T[:, None]).mean(1)[None, :]   # (1,K) by high
        return passed, failed, np.maximum(0.0, 1 - passed - failed)
    cost, review = expected_cost(*parts(sp), *parts(sn), P)
    valid = np.triu(np.ones(cost.shape, dtype=bool))
    return np.where(valid, cost, np.nan), np.where(valid, review, np.nan)


def best_pair(cost):
    i, j = np.unravel_index(np.nanargmin(cost), cost.shape)
    return int(i), int(j)


def pessimistic_cost(sp, sn, low, high, P):
    """Same decision, but charge the clean metres a false-scrap rate at the
    upper end of its 95% interval instead of the lucky rate seen in a small test."""
    ub = wilson_upper(int((sn >= high).sum()), len(sn))
    pass_n = float((sn < low).mean()); fail_seen = float((sn >= high).mean())
    rev_n = 1 - pass_n - fail_seen
    extra = max(0.0, ub - fail_seen)          # extra clean metres that would be scrapped
    take = min(extra, rev_n); rev_n -= take   # they come from the review band first,
    pass_n = max(0.0, pass_n - (extra - take))  # then from the passed ones
    fail_n = fail_seen + extra
    pass_p = float((sp < low).mean()); fail_p = float((sp >= high).mean())
    cost, _ = expected_cost(pass_p, fail_p, 1 - pass_p - fail_p, pass_n, fail_n, rev_n, P)
    return float(cost), ub


def sensitivity(spv, snv, sp, sn):
    """Thresholds are CHOSEN on validation and the cost is MEASURED on test."""
    rows = []

    def run(name, value, P):
        cv, _ = cost_grid(spv, snv, P)
        i, j = best_pair(cv)
        k = int(np.nanargmin(np.diag(cv)))
        ct, rt = cost_grid(sp, sn, P)
        rows.append({"parameter": name, "value": value,
                     "best_single_threshold": THRESHOLDS[k],
                     "cost_single_per_1000": round(float(np.diag(ct)[k]), 2),
                     "review_low": THRESHOLDS[i], "review_high": THRESHOLDS[j],
                     "cost_with_review_per_1000": round(float(ct[i, j]), 2),
                     "review_share": round(float(rt[i, j]), 4)})

    for m in (1, 2, 3, 5, 10):
        run("fn_multiple", m, {**BASE, "c_fn": m * BASE["c_fp"]})
    for b in (0.01, 0.02, 0.05, 0.10, 0.20):
        run("base_rate", b, {**BASE, "base_rate": b})
    for c in (0.05, 0.12, 0.25, 0.5, 1.0, 2.0):
        run("review_cost_eur", c, {**BASE, "c_review": c})
    for v in (5, 10, 30, 50):
        run("metre_value_eur", v, {**BASE, "c_fp": v, "c_fn": v * FN_MULTIPLE})
    for h in (0.7, 0.8, 0.9, 0.95, 1.0):
        run("human_tpr", h, {**BASE, "h_tpr": h})
    return rows


def review_load(review_share):
    cases = METRES_PER_DAY * review_share
    seconds = cases * REVIEW_SECONDS
    return {"metres_per_day": METRES_PER_DAY, "review_share": float(review_share),
            "review_cases_per_day": float(cases),
            "reviewer_hours_per_day": float(seconds / 3600),
            "reviewers_needed": float(seconds / (SHIFT_HOURS * 3600))}


# ------------------------------ main --------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    dataset_dir, data_yaml, names = ensure_dataset()
    stats = dataset_stats(dataset_dir, names)
    jdump(stats, RESULTS / "dataset_stats.json")
    print("Dataset described:", {k: v["images"] for k, v in stats["splits"].items()})

    if os.environ.get("SKIP_TRAIN") == "1" and (WEIGHTS / "best.pt").exists():
        weights = WEIGHTS / "best.pt"
        print("SKIP_TRAIN=1: reusing", weights)
    else:
        weights = train(data_yaml)

    overall, per_class, class_cm = evaluate_boxes(weights, data_yaml, names)
    write_csv(RESULTS / "per_class.csv", per_class)
    jdump(class_cm, RESULTS / "class_confusion.json")

    te_rows = image_scores(weights, dataset_dir / "test", names, "test")
    va_rows = image_scores(weights, dataset_dir / "valid", names, "valid")
    coco_rows = image_scores(CACHE / MODEL, dataset_dir / "test", names, "coco")
    y = np.array([r["y_true"] for r in te_rows]); s_ft = np.array([r["score"] for r in te_rows])
    yv = np.array([r["y_true"] for r in va_rows]); s_va = np.array([r["score"] for r in va_rows])
    s_coco = np.array([r["score"] for r in coco_rows])
    sp, sn = s_ft[y == 1], s_ft[y == 0]
    spv, snv = s_va[yv == 1], s_va[yv == 0]

    # --- CHOOSE the thresholds on VALIDATION, MEASURE the cost on TEST ---
    # (choosing and measuring on the same images would flatter the result)
    cost_v, _ = cost_grid(spv, snv, BASE)
    i, j = best_pair(cost_v)
    diag_v = np.diag(cost_v); k = int(np.nanargmin(diag_v))
    cost, review = cost_grid(sp, sn, BASE)
    diag = np.diag(cost)

    sweep = []
    for n_, t in enumerate(THRESHOLDS):
        c = confusion_at(y, s_ft, t)
        sweep.append({"threshold": t, **c, "cost_per_1000_test": round(float(diag[n_]), 2),
                      "cost_per_1000_valid": round(float(diag_v[n_]), 2)})
    write_csv(RESULTS / "cost_sweep.csv", sweep)
    write_csv(RESULTS / "sensitivity.csv", sensitivity(spv, snv, sp, sn))

    lo, hi, ks = float(THRESHOLDS[i]), float(THRESHOLDS[j]), float(THRESHOLDS[k])
    pess_pair, ub_pair = pessimistic_cost(sp, sn, lo, hi, BASE)
    pess_single, ub_single = pessimistic_cost(sp, sn, ks, ks, BASE)
    op = {
        "chosen_on": "validation", "measured_on": "test",
        "review_low": lo, "review_high": hi,
        "cost_per_1000_valid": float(cost_v[i, j]),
        "cost_per_1000": float(cost[i, j]), "review_share": float(review[i, j]),
        "best_single_threshold": ks, "single_cost_per_1000": float(diag[k]),
        "clean_images": {"valid": int(len(snv)), "test": int(len(sn))},
        "clean_scrapped_rate_test": {"seen": float((sn >= hi).mean()), "upper95": ub_pair},
        "pessimistic_cost_per_1000": pess_pair,
        "single": {"clean_scrapped_rate_seen": float((sn >= ks).mean()), "upper95": ub_single,
                   "pessimistic_cost_per_1000": pess_single},
        "params": BASE,
    }
    op["review_load"] = review_load(op["review_share"])
    jdump(op, RESULTS / "operating_point.json")

    # --- four strategies, same base rate, same units (measured on test) ---
    last = len(THRESHOLDS) - 1
    strategies = {
        "do_nothing (pass everything)": (last, last),
        "person checks everything": (0, last),
        "model alone (best single threshold)": (k, k),
        "model + human review (chosen)": (i, j),
    }
    table = []
    for name, (a, b) in strategies.items():
        share = float(review[a, b])
        table.append({"strategy": name, "cost_per_1000_metres": round(float(cost[a, b]), 2),
                      "cost_per_day": round(float(cost[a, b]) * METRES_PER_DAY / 1000, 2),
                      **{kk: round(v, 3) for kk, v in review_load(share).items()
                         if kk in ("review_share", "review_cases_per_day", "reviewers_needed")}})
    write_csv(RESULTS / "strategies.csv", table)

    # --- baselines on the same test split (image level) ---
    kc = int(np.nanargmin([single_cost(confusion_at(y, s_coco, t), BASE) for t in THRESHOLDS]))
    baselines = []

    def add(name, c, ap50):
        baselines.append({"model": name, **{x: round(c[x], 4) for x in
                          ("precision", "recall", "accuracy")},
                          "tp": c["tp"], "fp": c["fp"], "tn": c["tn"], "fn": c["fn"],
                          "cost_per_1000_metres": round(single_cost(c, BASE), 2),
                          "box_ap50": ap50})
    add(f"finetuned @ single threshold {ks} (chosen on validation)", confusion_at(y, s_ft, ks), overall["ap50"])
    add("finetuned @ 0.25", confusion_at(y, s_ft, 0.25), overall["ap50"])
    add("COCO zero-shot @ 0.25 (any object = defect)", confusion_at(y, s_coco, 0.25), 0.0)
    add(f"COCO zero-shot @ {THRESHOLDS[kc]} (picked ON TEST: best case for the baseline)",
        confusion_at(y, s_coco, THRESHOLDS[kc]), 0.0)
    add("trivial: predict nothing", confusion_at(y, s_ft, 1.01), 0.0)
    add("trivial: predict defect everywhere (majority class)", confusion_at(y, s_ft, 0.0), 0.0)
    write_csv(RESULTS / "baselines.csv", baselines)

    jdump({"overall_box_metrics_test": overall, "test_images": len(y),
           "test_images_with_defect": int(y.sum()), "weights": str(weights.relative_to(ROOT)),
           "training": {"model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                        "patience": PATIENCE, "seed": SEED, "fraction": FRACTION,
                        "device": str(pick_device())},
           "class_names": names, "display_names": DISPLAY_NAMES, "params": BASE,
           "metres_per_day": METRES_PER_DAY, "shift_hours": SHIFT_HOURS,
           "review_seconds": REVIEW_SECONDS,
           "thresholds": THRESHOLDS, "units": "1 metre of denim = 1 image"},
          RESULTS / "metrics.json")

    print("\nOverall (test, box level):", {k_: round(v, 3) for k_, v in overall.items()})
    for row in per_class:
        print("  ", row)
    print(f"\nChosen ON VALIDATION: pass < {lo}, review in between, fail >= {hi}")
    print(f"  measured on test: {op['cost_per_1000']:.2f} EUR per 1,000 m "
          f"(valid: {op['cost_per_1000_valid']:.2f}), review share {op['review_share']:.1%}")
    print(f"  pessimistic (clean scrapped rate at its 95% upper bound "
          f"{ub_pair:.1%}): {pess_pair:.2f} EUR per 1,000 m")
    print("  review load:", {k_: round(v, 2) for k_, v in op["review_load"].items()})
    for row in table:
        print("  ", row)
    print("\nSaved everything app.py needs to results/ and the model to weights/best.pt")


if __name__ == "__main__":
    main()
