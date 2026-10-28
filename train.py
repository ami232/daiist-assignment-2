"""
Assignment 2: training pipeline (car damage pre-inspection for insurance).

Run with:  uv run python main.py train

What it does, start to finish:
  1. Download the dataset version from Roboflow Universe.
  2. Describe it: images per split, empty (undamaged) images, instances per
     class, box size relative to the image.
  3. Finetune YOLO11 from the pretrained COCO checkpoint.
  4. Evaluate on the TEST split: per-class detection metrics, plus a per-car
     decision ("does this car have damage?") compared against two baselines.
  5. Put euros on each outcome, sweep the threshold, pick the cheapest one,
     and pick a pass / review / fail band that fits the review team.
  6. Save weights/best.pt and weights/results.json for app.py.

Set SKIP_TRAINING=1 to reuse weights/best.pt and only redo steps 4-6
(handy when you change a cost). The real pipeline always trains.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS = WEIGHTS_DIR / "best.pt"
RESULTS = WEIGHTS_DIR / "results.json"

# ---------------------------------------------------------------- dataset ----
# Public Roboflow Universe datasets, in order of preference. The first one that
# is a detection project with a test split gets used, and the run prints which.
# Put the one it used (workspace / project / version) in REPORT.md.
#   1. car-damages: 2.3k images, dent / scratch / severe + a "No Damage" class
#      for clean cars, so false positives can actually be measured.
#   2. car-damage-detection (Celebal): small (~370) but has "Undamaged".
#   3-5. bigger damage-only sets: no clean cars, false positives not measurable.
DATASET_CANDIDATES = [
    ("project-p5nyc", "car-damages-v3gyz"),
    ("celebalworkspace", "car-damage-detection-dy5hu"),
    ("jijo", "car-damage-new"),
    ("clickits", "car_damage_detection-zztxm"),
    ("accdamrep", "car-damage-detection-t0g92-afbxe"),
]
VERSION = None   # None = newest version that has a test split; a number pins it
DATA_DIR = None  # set once the dataset is chosen
# Some datasets label clean cars with a box ("No Damage", "Undamaged").
# Classes listed here do NOT count as damage for the per-car decision.
NOT_DAMAGE = ["no damage", "no-damage", "undamaged", "undamged", "nodamaged"]

# --------------------------------------------------------------- training ----
BASE_MODEL = "yolo11n.pt"   # pretrained on COCO; yolo11s.pt is bigger/slower
EPOCHS = 50
IMGSZ = 640
BATCH = 16
PATIENCE = 15               # stop early if val mAP stops improving
SEED = 0

# ------------------------------------------------------- business numbers ----
# YOUR DECISIONS. Placeholders so the code runs; replace each one with a figure
# you can defend in REPORT.md. Units: euros per car inspected.
# Positive class = "car has pre-existing damage".
COST_FN = 600.0     # damage missed -> later claimed as new -> insurer pays repair
COST_FP = 25.0      # clean car flagged -> re-shoot/inspection + some customers walk
COST_TP = 0.0       # damage correctly recorded
COST_TN = 0.0       # clean car correctly passed
COST_REVIEW = 2.0   # one underwriter look at the photos (wage x minutes)
BASE_RATE = 0.15    # share of cars arriving with pre-existing damage

CARS_PER_DAY = 600            # pre-inspections the insurer handles per day
REVIEW_MINUTES = 3             # time for one human check
REVIEWER_MINUTES_PER_DAY = 420 # 7 productive hours per reviewer
REVIEWERS_AVAILABLE = 3        # people you actually have for this

THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)


# ======================================================== cost arithmetic ====
# These functions are imported by app.py, so the dashboard and this script
# always compute cost the same way.

def rates(scores, labels, low, high):
    """Share of damaged and of clean cars that land in fail / review / pass.

    score >= high        -> fail   (damage recorded, excluded from cover)
    low <= score < high  -> review (an underwriter looks)
    score < low          -> pass   (policy issued as is)
    With low == high there is no review band: the model decides alone.
    """
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    out = {}
    for name, mask in (("pos", labels), ("neg", ~labels)):
        s = scores[mask]
        n = max(len(s), 1)
        fail = float((s >= high).sum()) / n
        passed = float((s < low).sum()) / n
        out[name] = {"fail": fail, "review": 1 - fail - passed, "pass": passed}
    return out


def cost_per_car(scores, labels, low, high, c):
    """Expected euros per car at the PRODUCTION base rate.

    The test split's share of damaged cars is not the real-world share, so we
    take the model's behaviour on each class from the test set (the rates) and
    re-weight it by the base rate we believe in. Reviewed cars are assumed to
    be decided correctly, at the price of the review.
    """
    r = rates(scores, labels, low, high)
    p, n = r["pos"], r["neg"]
    pos = p["fail"] * c["tp"] + p["pass"] * c["fn"] + p["review"] * (c["review"] + c["tp"])
    neg = n["fail"] * c["fp"] + n["pass"] * c["tn"] + n["review"] * (c["review"] + c["tn"])
    return c["base_rate"] * pos + (1 - c["base_rate"]) * neg


def review_share(scores, labels, low, high, base_rate):
    r = rates(scores, labels, low, high)
    return base_rate * r["pos"]["review"] + (1 - base_rate) * r["neg"]["review"]


def reviewers_needed(share, c):
    return share * c["cars_per_day"] * c["review_minutes"] / c["reviewer_minutes_per_day"]


def alternatives(c):
    """Cost per car of the three non-model options the brief asks about."""
    return {
        "do_nothing": c["base_rate"] * c["fn"] + (1 - c["base_rate"]) * c["tn"],
        "human_checks_all": c["review"] + c["base_rate"] * c["tp"] + (1 - c["base_rate"]) * c["tn"],
    }


def roc_auc(scores, labels):
    """Chance a random damaged car scores higher than a random clean car."""
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


# ============================================================ the pipeline ===

def costs():
    return {
        "fn": COST_FN, "fp": COST_FP, "tp": COST_TP, "tn": COST_TN,
        "review": COST_REVIEW, "base_rate": BASE_RATE,
        "cars_per_day": CARS_PER_DAY, "review_minutes": REVIEW_MINUTES,
        "reviewer_minutes_per_day": REVIEWER_MINUTES_PER_DAY,
        "reviewers_available": REVIEWERS_AVAILABLE,
    }


def choose_dataset(rf):
    """First candidate that is a detection-compatible project with a test split."""
    for ws, proj in DATASET_CANDIDATES:
        try:
            project = rf.workspace(ws).project(proj)
        except Exception as error:  # noqa: BLE001 - deleted/private project: skip it
            print(f"  skip {ws}/{proj}: not reachable ({error})")
            continue
        # Instance-segmentation polygons work too: YOLO turns them into boxes.
        if project.type not in ("object-detection", "instance-segmentation"):
            print(f"  skip {ws}/{proj}: project type is {project.type}")
            continue
        for v in sorted(project.versions(), key=lambda v: int(v.version), reverse=True):
            if VERSION is not None and int(v.version) != VERSION:
                continue
            if (v.splits or {}).get("test", 0) > 0:
                return ws, proj, int(v.version), v
        print(f"  skip {ws}/{proj}: no version with a test split")
    raise SystemExit("None of DATASET_CANDIDATES is usable. Add another one.")


def download_dataset():
    """Pick the dataset, download it once (datasets/ is gitignored), fix data.yaml."""
    global DATA_DIR
    from roboflow import Roboflow
    from api_key import load_api_key

    print("== Choosing dataset ==")
    ws, proj, ver, version = choose_dataset(Roboflow(api_key=load_api_key()))
    print(f"USING: workspace={ws} project={proj} version={ver} splits={version.splits}")
    DATA_DIR = ROOT / "datasets" / f"{proj}-v{ver}"
    if not (DATA_DIR / "data.yaml").exists():
        version.download("yolov11", location=str(DATA_DIR), overwrite=True)

    # Roboflow's data.yaml uses relative paths that break depending on where
    # you run from. Rewrite it with an absolute root.
    cfg = yaml.safe_load((DATA_DIR / "data.yaml").read_text())
    for split in ("train", "valid", "test"):
        if not (DATA_DIR / split / "images").is_dir():
            raise SystemExit(f"Dataset has no '{split}' split. Pick a version that has train/valid/test.")
    fixed = {"path": str(DATA_DIR), "train": "train/images", "val": "valid/images",
             "test": "test/images", "nc": cfg["nc"], "names": cfg["names"]}
    data_yaml = DATA_DIR / "data_fixed.yaml"
    data_yaml.write_text(yaml.safe_dump(fixed))
    return data_yaml, {"workspace": ws, "project": proj, "version": ver,
                       "url": f"https://universe.roboflow.com/{ws}/{proj}/dataset/{ver}"}


def label_rows(label_file: Path):
    """Yield (class_id, relative_box_area) from one YOLO label file.
    Handles plain boxes and polygons (converted to their bounding box)."""
    for line in label_file.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = int(float(parts[0])), list(map(float, parts[1:]))
        if len(vals) == 4:
            w, h = vals[2], vals[3]
        else:
            xs, ys = vals[0::2], vals[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        yield cls, w * h


def damage_ids(names):
    return [i for i, n in enumerate(names) if n.strip().lower() not in NOT_DAMAGE]


def is_damaged(label_file: Path, dmg_ids) -> int:
    """Ground truth for the per-car decision: any box of a damage class."""
    if not label_file.exists():
        return 0
    return int(any(cls in dmg_ids for cls, _ in label_rows(label_file)))


def describe_dataset(names) -> dict:
    stats = {}
    for split in ("train", "valid", "test"):
        images = sorted((DATA_DIR / split / "images").glob("*"))
        per_class = {n: 0 for n in names}
        areas, empty, damaged = [], 0, 0
        for img in images:
            lbl = DATA_DIR / split / "labels" / f"{img.stem}.txt"
            rows = list(label_rows(lbl)) if lbl.exists() else []
            empty += not rows
            damaged += is_damaged(lbl, damage_ids(names))
            for cls, area in rows:
                per_class[names[cls]] += 1
                areas.append(area)
        stats[split] = {
            "images": len(images),
            "empty_images": empty,
            "damaged_cars": damaged,
            "clean_cars": len(images) - damaged,
            "instances": per_class,
            "median_box_area_pct": round(100 * float(np.median(areas)), 2) if areas else None,
            "small_box_share_pct": round(100 * float(np.mean(np.array(areas) < 0.01)), 1) if areas else None,
        }
    return stats


def device() -> str:
    import torch
    if os.environ.get("YOLO_DEVICE"):
        return os.environ["YOLO_DEVICE"]
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def finetune(data_yaml: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                patience=PATIENCE, seed=SEED, device=device(),
                project=str(ROOT / "runs"), name="finetune", exist_ok=True)
    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)


def detection_metrics(data_yaml: Path, names) -> dict:
    """Box-level metrics on the test split, overall and per class."""
    from ultralytics import YOLO

    def val(conf):
        return YOLO(str(WEIGHTS)).val(data=str(data_yaml), split="test", imgsz=IMGSZ, conf=conf,
                                      device=device(), plots=True, verbose=False,
                                      project=str(ROOT / "runs"), name="test", exist_ok=True)

    m = val(0.001)        # AP needs every prediction, down to very low confidence
    cm = getattr(val(0.25), "confusion_matrix", None)   # matrix at a usable cut-off
    per_class = []
    for i, cls in enumerate(m.ap_class_index):
        per_class.append({"class": names[int(cls)],
                          "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                          "ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])})
    return {
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class": per_class,
        # Ultralytics layout: rows = predicted, columns = true, last = background.
        # Background row = damage missed; background column = damage invented.
        "confusion_matrix": cm.matrix.astype(int).tolist() if cm is not None else None,
        "confusion_labels": list(names) + ["background"],
    }


def car_scores(weights: str, dmg_ids, keep_classes=None):
    """One score per test image: the highest box confidence the model gives.
    That is the per-car decision score the thresholds act on."""
    from ultralytics import YOLO

    model = YOLO(weights)
    images = sorted((DATA_DIR / "test" / "images").glob("*"))
    scores, labels = [], []
    for img in images:
        lbl = DATA_DIR / "test" / "labels" / f"{img.stem}.txt"
        labels.append(is_damaged(lbl, dmg_ids))
        res = model.predict(str(img), conf=0.001, imgsz=IMGSZ, device=device(), verbose=False)[0]
        conf = res.boxes.conf.cpu().numpy()
        if keep_classes is not None:
            conf = conf[np.isin(res.boxes.cls.cpu().numpy().astype(int), keep_classes)]
        scores.append(float(conf.max()) if len(conf) else 0.0)
    return scores, labels


def sweep(scores, labels, c) -> dict:
    """Single threshold (model decides alone) and the best pass/review/fail band."""
    single = [cost_per_car(scores, labels, t, t, c) for t in THRESHOLDS]
    best_t = float(THRESHOLDS[int(np.argmin(single))])

    best_free, best_fit = None, None
    for i, low in enumerate(THRESHOLDS):
        for high in THRESHOLDS[i:]:
            cost = cost_per_car(scores, labels, low, high, c)
            share = review_share(scores, labels, low, high, c["base_rate"])
            row = {"low": float(low), "high": float(high), "cost": cost,
                   "review_share": share, "reviewers": reviewers_needed(share, c)}
            if best_free is None or cost < best_free["cost"]:
                best_free = row
            if row["reviewers"] <= c["reviewers_available"] and (best_fit is None or cost < best_fit["cost"]):
                best_fit = row
    return {"thresholds": THRESHOLDS.tolist(), "cost_single": single,
            "best_single_threshold": best_t, "best_single_cost": min(single),
            "best_band_unconstrained": best_free, "best_band_with_staff": best_fit}


def main() -> None:
    data_yaml, dataset_ref = download_dataset()
    names = yaml.safe_load(data_yaml.read_text())["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    print("\n== Dataset ==")
    dataset = describe_dataset(names)
    print(json.dumps(dataset, indent=2))

    if os.environ.get("SKIP_TRAINING") == "1" and WEIGHTS.exists():
        print("\nSKIP_TRAINING=1: reusing weights/best.pt")
    else:
        print("\n== Finetuning ==")
        finetune(data_yaml)

    print("\n== Detection metrics on test ==")
    det = detection_metrics(data_yaml, names)
    print(json.dumps({k: v for k, v in det.items() if k != "confusion_matrix"}, indent=2))

    print("\n== Per-car decision on test ==")
    c = costs()
    dmg = damage_ids(names)
    ft_scores, labels = car_scores(str(WEIGHTS), dmg, keep_classes=dmg)
    # COCO has no "damage" class. The best a zero-shot COCO model can do is
    # flag cars it is confident about detecting at all (car/truck/bus...),
    # so we score with its top confidence on any class and see if that
    # separates damaged from clean cars. Expect it not to.
    coco_scores, _ = car_scores(BASE_MODEL, dmg)
    nothing_scores = [0.0] * len(labels)   # trivial: never flag damage

    n_pos, n_neg = int(sum(labels)), len(labels) - int(sum(labels))
    if n_neg == 0:
        print("WARNING: no undamaged cars in the test split. False positives cannot be "
              "measured, so the cost curve only reflects missed damage.")

    baselines = {}
    for name, s in (("finetuned", ft_scores), ("coco_zero_shot", coco_scores),
                    ("predict_nothing", nothing_scores)):
        sw = sweep(s, labels, c)
        baselines[name] = {"auc": roc_auc(s, labels),
                           "best_threshold": sw["best_single_threshold"],
                           "cost_per_car": sw["best_single_cost"]}
        if name == "finetuned":
            decision = sw
    # "Predict nothing" is a fixed rule, not something to tune: it never flags
    # damage, so its cost is every damaged car slipping through.
    never = cost_per_car(nothing_scores, labels, 1.01, 1.01, c)
    baselines["predict_nothing"].update(best_threshold=None, cost_per_car=never)
    baselines["predict_nothing"]["map50"] = 0.0
    baselines["coco_zero_shot"]["map50"] = 0.0   # cannot output damage classes
    baselines["finetuned"]["map50"] = det["map50"]
    print(json.dumps(baselines, indent=2))

    alts = alternatives(c)
    print(f"\nModel alone, t={decision['best_single_threshold']}: "
          f"€{decision['best_single_cost']:.2f}/car")
    print(f"Best band that fits {c['reviewers_available']} reviewers: {decision['best_band_with_staff']}")
    print(f"Do nothing: €{alts['do_nothing']:.2f}/car | Human checks all: €{alts['human_checks_all']:.2f}/car")

    results = {
        "dataset_ref": dataset_ref,
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ,
                     "batch": BATCH, "device": device()},
        "class_names": names, "damage_class_ids": dmg, "dataset": dataset, "detection": det,
        "test_cars": {"scores": ft_scores, "labels": labels, "damaged": n_pos, "clean": n_neg},
        "baselines": baselines, "costs": c, "decision": decision, "alternatives": alts,
    }
    RESULTS.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {WEIGHTS.relative_to(ROOT)} and {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
