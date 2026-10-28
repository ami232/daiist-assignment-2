"""
Assignment 2: training pipeline.   Run with:  uv run python main.py train

No manual steps. Start to finish this script:

  1. downloads the (public, version-pinned) Roboflow dataset into datasets/
  2. finetunes yolo11n from the pretrained COCO checkpoint
  3. saves the best weights to weights/best.pt
  4. evaluates on the TEST split: per-class metrics + confusion matrix
  5. scores every validation/test image once and stores the predictions, so the
     dashboard never needs to retrain or re-predict the whole dataset
  6. compares against two baselines (COCO zero-shot, predict-nothing)
  7. sweeps the confidence threshold with the cost matrix in ASSUMPTIONS below.
     Thresholds are CHOSEN on the validation split and REPORTED on the test split.
  8. writes everything app.py needs to outputs/

The numbers in ASSUMPTIONS must match the cost matrix in REPORT.md.
"""

import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# =============================================================================
# SETTINGS
# =============================================================================
WORKSPACE, PROJECT, VERSION = "fruit-ripening", "banana-ripening-process", 2
BASE_MODEL = "yolo11n.pt"      # COCO-pretrained checkpoint we finetune from
EPOCHS = int(os.environ.get("EPOCHS", "20"))   # the real run uses 20; set EPOCHS=1 for a quick smoke test
IMGSZ, BATCH, SEED = 416, 32, 0
POSITIVE_CLASS = "rotten"      # the class we treat as "bad". One case = one image.

# Cost and operating assumptions. Every figure needs a reason in REPORT.md.
ASSUMPTIONS = dict(
    base_rate=0.03,    # share of hands that are really rotten (an ASSUMPTION; the dataset is ~38%)
    c_miss=25.0,       # EUR: a rotten hand is shipped (carton credited)
    c_fa=0.50,         # EUR: a good hand is wrongly failed (sold cheaper)
    hands_day=10_000,  # hands photographed per day (one every ~3 s over 8 h)
    check_s=10,        # seconds for a person to check one hand
    wage_h=26.4,       # EUR per hour (Eurostat 2025 Spain average, incl. employer costs)
    h_catch=0.95,      # a reviewer taking a close look catches this share of rotten hands
    h_false=0.01,      # PLACEHOLDER: a reviewer wrongly fails this share of good hands. Set your own.
    all_catch=0.95,    # person checking EVERY hand (10 s each) catches this share
    all_false=0.01,    # ... and wrongly fails this share of good hands
)
REVIEW_LOW = 0.10      # lower edge of the human-review band (a design choice, see REPORT.md)
PRODUCTIVE_H = 7       # productive hours per person per day (assumption)
PRED_CONF = 0.05       # lowest confidence stored; thresholds are swept from here

OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights"
DATA = ROOT / "datasets" / f"{PROJECT}-{VERSION}"   # datasets/ is git-ignored


# =============================================================================
# 1. DATA
# =============================================================================
def download_dataset():
    if (DATA / "data.yaml").exists():
        print(f"Dataset already at {DATA}, reusing it.")
        return
    from roboflow import Roboflow
    from api_key import load_api_key

    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DATA)
    )


def dataset_stats(names):
    """Instances per class, split sizes and typical object size, from the labels."""
    stats = {"images": {}, "instances": {}, "images_with_class": {}, "median_box_area_frac": {}}
    for split in ["train", "valid", "test"]:
        files = sorted((DATA / split / "labels").glob("*.txt"))
        inst, imgs, areas = Counter(), Counter(), defaultdict(list)
        for f in files:
            seen = set()
            for line in f.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                inst[c] += 1
                seen.add(c)
                areas[c].append(float(parts[3]) * float(parts[4]))
            for c in seen:
                imgs[c] += 1
        stats["images"][split] = len(files)
        stats["instances"][split] = {names[c]: inst[c] for c in sorted(names)}
        stats["images_with_class"][split] = {names[c]: imgs[c] for c in sorted(names)}
        stats["median_box_area_frac"][split] = {
            names[c]: (round(float(np.median(areas[c])), 3) if areas[c] else None)
            for c in sorted(names)
        }
    stats["note"] = "train includes the 3x augmented copies Roboflow generated for version 2"
    return stats


# =============================================================================
# 2. TRAINING
# =============================================================================
def train_model():
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        seed=SEED, project="runs", name="full", exist_ok=True,
    )
    return Path(model.trainer.save_dir) / "weights" / "best.pt"


def finalize_weights(best):
    """Keep exactly one model in weights/. check.py loads every .pt in there."""
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS / "best.pt")
    for p in WEIGHTS.rglob("*.pt"):
        if p.name != "best.pt":   # e.g. the yolo26n.pt Ultralytics fetches for its AMP check
            p.unlink()


# =============================================================================
# 3. EVALUATION ON THE TEST SPLIT (detection metrics, per class)
# =============================================================================
def detection_metrics(model):
    m = model.val(data=str(DATA / "data.yaml"), split="test", imgsz=IMGSZ,
                  project="runs", name="test_eval", exist_ok=True, verbose=False)
    per_class = []
    for idx, cid in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(idx)
        per_class.append({"class": m.names[int(cid)], "precision": float(p), "recall": float(r),
                          "ap50": float(ap50), "ap50_95": float(ap)})
    result = {
        "overall": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                    "map50": float(m.box.map50), "map50_95": float(m.box.map)},
        "per_class": per_class,
        "note": "precision/recall are at the confidence that maximises F1; computed on the test split",
    }
    try:  # keep the picture for the report/dashboard
        OUT.mkdir(exist_ok=True)
        shutil.copy(Path(m.save_dir) / "confusion_matrix_normalized.png",
                    OUT / "confusion_matrix_normalized.png")
    except Exception as e:  # noqa: BLE001
        print("could not copy confusion matrix image:", e)
    # Class-level confusion matrix at a realistic confidence (the default val
    # run uses a tiny confidence, which floods the background column).
    try:
        m2 = model.val(data=str(DATA / "data.yaml"), split="test", imgsz=IMGSZ, conf=0.25,
                       plots=False, project="runs", name="test_cm", exist_ok=True, verbose=False)
        result["confusion_matrix_conf025"] = {
            "matrix": m2.confusion_matrix.matrix.astype(int).tolist(),
            "labels": [m.names[i] for i in sorted(m.names)] + ["background"],
            "orientation": "rows = predicted, columns = true",
        }
    except Exception as e:  # noqa: BLE001
        print("could not build the conf=0.25 confusion matrix:", e)
    return result


# =============================================================================
# 4. PREDICTIONS (stored once, so the dashboard never re-predicts the dataset)
# =============================================================================
def dump_predictions(model, split):
    out = []
    for r in model.predict(source=str(DATA / split / "images"), conf=PRED_CONF, iou=0.7,
                           imgsz=IMGSZ, stream=True, verbose=False):
        stem = Path(r.path).stem
        lbl = DATA / split / "labels" / f"{stem}.txt"
        gt = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    gt.append([int(parts[0])] + [round(float(v), 4) for v in parts[1:5]])
        b = r.boxes
        pred = [[int(c), round(float(s), 4)] + [round(float(v), 4) for v in xywhn]
                for c, s, xywhn in zip(b.cls.tolist(), b.conf.tolist(), b.xywhn.tolist())]
        out.append({"image": os.path.basename(r.path), "gt": gt, "pred": pred})
    return out


def coco_banana_scores(split="test"):
    """Baseline: the pretrained COCO model, never finetuned. It has a 'banana'
    class but no ripeness classes, so all it can say is 'a banana is here'."""
    from ultralytics import YOLO

    coco = YOLO(BASE_MODEL)
    banana = [k for k, v in coco.names.items() if v == "banana"][0]
    scores = {}
    for r in coco.predict(source=str(DATA / split / "images"), conf=PRED_CONF,
                          imgsz=IMGSZ, stream=True, verbose=False):
        scores[os.path.basename(r.path)] = max(
            [s for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()) if int(c) == banana],
            default=0.0)
    return scores


# =============================================================================
# 5. COSTING (pure functions of the stored predictions)
# =============================================================================
def image_scores(items, pos_id):
    """One case = one image. Truth: the image has a positive-class label.
    Score: the highest positive-class confidence the model produced."""
    y = np.array([any(g[0] == pos_id for g in im["gt"]) for im in items])
    s = np.array([max([p[1] for p in im["pred"] if p[0] == pos_id], default=0.0) for im in items])
    return y, s


def cost_per_1000(y, s, lo, hi, A):
    """Pass if s < lo, human review if lo <= s < hi, fail if s >= hi.
    Returns (expected cost per 1,000 hands, share of hands sent to a person)."""
    r, g = s[y], s[~y]
    chk = A["check_s"] * A["wage_h"] / 3600          # EUR per human check
    n_rot, n_good = 1000 * A["base_rate"], 1000 * (1 - A["base_rate"])
    r_fail, r_pass = (r >= hi).mean(), (r < lo).mean()
    r_rev = 1 - r_fail - r_pass
    g_fail, g_pass = (g >= hi).mean(), (g < lo).mean()
    g_rev = 1 - g_fail - g_pass
    cost = (n_rot * (r_pass * A["c_miss"] + r_rev * (chk + (1 - A["h_catch"]) * A["c_miss"]))
            + n_good * (g_fail * A["c_fa"] + g_rev * (chk + A["h_false"] * A["c_fa"])))
    share = (n_rot * r_rev + n_good * g_rev) / 1000
    return float(cost), float(share)


def do_nothing_cost(A):
    return 1000 * A["base_rate"] * A["c_miss"]


def fail_everything_cost(A):
    return 1000 * (1 - A["base_rate"]) * A["c_fa"]


def check_everything_cost(A):
    chk = A["check_s"] * A["wage_h"] / 3600
    n_rot, n_good = 1000 * A["base_rate"], 1000 * (1 - A["base_rate"])
    return 1000 * chk + n_rot * (1 - A["all_catch"]) * A["c_miss"] + n_good * A["all_false"] * A["c_fa"]


def people_needed(share, A):
    hands = share * A["hands_day"]
    return hands, hands * A["check_s"] / (PRODUCTIVE_H * 3600)


def counts(y, s, lo, hi):
    r, g = s[y], s[~y]
    return {"rotten_images": {"pass": int((r < lo).sum()), "review": int(((r >= lo) & (r < hi)).sum()),
                              "fail": int((r >= hi).sum())},
            "good_images": {"pass": int((g < lo).sum()), "review": int(((g >= lo) & (g < hi)).sum()),
                            "fail": int((g >= hi).sum())}}


def analyse(preds, banana_scores, A):
    names = {int(k): v for k, v in preds["names"].items()}
    pos_id = [k for k, v in names.items() if v == POSITIVE_CLASS][0]
    yv, sv = image_scores(preds["valid"], pos_id)
    yt, st = image_scores(preds["test"], pos_id)
    ts = np.round(np.arange(0.05, 0.96, 0.01), 2)

    # --- choose operating points on VALIDATION ---
    cost_v = [cost_per_1000(yv, sv, t, t, A)[0] for t in ts]
    t_single = float(ts[int(np.argmin(cost_v))])
    his = ts[ts > REVIEW_LOW]
    hi_best = float(his[int(np.argmin([cost_per_1000(yv, sv, REVIEW_LOW, h, A)[0] for h in his]))])

    # --- report on TEST ---
    cost_t = [cost_per_1000(yt, st, t, t, A)[0] for t in ts]
    c_single, _ = cost_per_1000(yt, st, t_single, t_single, A)
    c_band, share = cost_per_1000(yt, st, REVIEW_LOW, hi_best, A)
    hands, people = people_needed(share, A)
    flag = st >= t_single
    tp, fp = int((flag & yt).sum()), int((flag & ~yt).sum())
    fn, tn = int((~flag & yt).sum()), int((~flag & ~yt).sum())

    strategies = {
        "do_nothing": {"cost_per_1000": do_nothing_cost(A)},
        "fail_everything": {"cost_per_1000": fail_everything_cost(A)},
        "person_checks_every_hand": {"cost_per_1000": check_everything_cost(A),
                                     "share_to_person": 1.0, **dict(zip(["hands_per_day", "people"], people_needed(1.0, A)))},
        "model_alone": {"cost_per_1000": c_single, "threshold": t_single},
        "model_plus_review": {"cost_per_1000": c_band, "low": REVIEW_LOW, "high": hi_best,
                              "share_to_person": share, "hands_per_day": hands, "people": people},
    }
    for v in strategies.values():
        v["cost_per_day"] = v["cost_per_1000"] * A["hands_day"] / 1000

    # --- COCO zero-shot baseline: "flag the image if COCO sees a banana" ---
    coco_rows = []
    sb = np.array([banana_scores.get(im["image"], 0.0) for im in preds["test"]])
    for t in [0.05, 0.10, 0.25, 0.50, 0.75]:
        fl = sb >= t
        rec, fpr = float((fl & yt).sum() / yt.sum()), float((fl & ~yt).sum() / (~yt).sum())
        c = 1000 * A["base_rate"] * (1 - rec) * A["c_miss"] + 1000 * (1 - A["base_rate"]) * fpr * A["c_fa"]
        coco_rows.append({"banana_conf": t, "recall": rec, "false_alarm_rate": fpr, "cost_per_1000": float(c)})

    # --- sensitivity (model alone, re-optimised on validation) ---
    def best_t(B):
        cs = [cost_per_1000(yv, sv, t, t, B)[0] for t in ts]
        i = int(np.argmin(cs))
        return {"best_threshold": float(ts[i]), "cost_per_1000": float(cs[i]),
                "do_nothing": float(do_nothing_cost(B))}
    sens_base = {str(br): best_t({**A, "base_rate": br}) for br in [0.005, 0.01, 0.03, 0.05, 0.10]}
    sens_miss = {str(cm): best_t({**A, "c_miss": float(cm)}) for cm in [1, 2, 5, 10, 25, 50, 100]}
    sens_catch = {str(hc): cost_per_1000(yt, st, REVIEW_LOW, hi_best, {**A, "h_catch": hc})[0]
                  for hc in [0.70, 0.80, 0.90, 0.95, 0.99]}
    tradeoff = []
    for lo in [0.05, 0.10, 0.20, 0.30]:
        c, sh = cost_per_1000(yt, st, lo, hi_best, A)
        h, p = people_needed(sh, A)
        tradeoff.append({"low": lo, "cost_per_1000": c, "share_to_person": sh,
                         "hands_per_day": h, "people": p})

    return {
        "positive_class": POSITIVE_CLASS,
        "assumptions": A, "productive_hours_per_person": PRODUCTIVE_H,
        "chosen": {"single_threshold": t_single, "review_low": REVIEW_LOW, "review_high": hi_best,
                   "chosen_on": "validation", "reported_on": "test"},
        "image_level_test_at_single_threshold": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                                                 "recall": tp / (tp + fn), "false_alarm_rate": fp / (fp + tn)},
        "image_level_test_band_outcomes": counts(yt, st, REVIEW_LOW, hi_best),
        "strategies_test": strategies,
        "coco_flag_if_banana_test": coco_rows,
        "trivial_baselines": {"predict_nothing_detection": {"precision": 0.0, "recall": 0.0, "map50": 0.0},
                              "majority_class_image_level": "no rotten -> same as do_nothing"},
        "sweep": {"thresholds": ts.tolist(), "cost_valid_single": cost_v, "cost_test_single": cost_t},
        "sensitivity": {"base_rate": sens_base, "miss_cost": sens_miss,
                        "reviewer_catch_rate_cost_per_1000": sens_catch, "review_low_tradeoff": tradeoff},
        "scores": {"valid": {"y": yv.tolist(), "s": [round(float(v), 4) for v in sv]},
                   "test": {"y": yt.tolist(), "s": [round(float(v), 4) for v in st]}},
    }


def plot_cost_curve(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts, c = res["sweep"]["thresholds"], res["sweep"]["cost_valid_single"]
    ch = res["chosen"]
    plt.figure(figsize=(7, 4))
    plt.plot(ts, c, label="model alone (validation)")
    plt.plot(ts, res["sweep"]["cost_test_single"], alpha=0.6, label="model alone (test)")
    plt.axhline(res["strategies_test"]["do_nothing"]["cost_per_1000"], ls="--", c="gray", label="do nothing")
    plt.axvline(ch["single_threshold"], c="red", ls=":", label=f"chosen t = {ch['single_threshold']:.2f}")
    plt.xlabel("confidence threshold")
    plt.ylabel("expected cost per 1,000 hands (EUR)")
    plt.title("Cost vs threshold")
    plt.legend(loc="upper left")
    OUT.mkdir(exist_ok=True)
    plt.savefig(OUT / "cost_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
def main():
    t0 = time.time()
    download_dataset()
    t1 = time.time()
    best = train_model()
    train_s = time.time() - t1
    finalize_weights(best)

    from ultralytics import YOLO
    import torch

    model = YOLO(str(WEIGHTS / "best.pt"))
    names = {int(k): v for k, v in model.names.items()}
    OUT.mkdir(exist_ok=True)

    print("\n=== Dataset description ===")
    ds = dataset_stats(names)
    print("\n=== Test-split detection metrics ===")
    det = detection_metrics(model)
    print("\n=== Storing predictions ===")
    preds = {"names": names, "valid": dump_predictions(model, "valid"),
             "test": dump_predictions(model, "test")}
    json.dump(preds, open(OUT / "predictions.json", "w"))
    print("\n=== Baselines and costing ===")
    banana = coco_banana_scores("test")
    res = analyse(preds, banana, ASSUMPTIONS)

    import ultralytics
    res.update({
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, **ds},
        "detection_test": det,
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                     "seed": SEED, "train_seconds": round(train_s),
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                     "ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "names": names,
    })
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    plot_cost_curve(res)

    st = res["strategies_test"]
    print("\n" + "=" * 70)
    print(f"Finished in {(time.time() - t0) / 60:.1f} min (training {train_s / 60:.1f} min on {res['training']['device']})")
    o = det["overall"]
    print(f"TEST  mAP50 {o['map50']:.3f}  mAP50-95 {o['map50_95']:.3f}")
    for pc in det["per_class"]:
        print(f"   {pc['class']:12s} P {pc['precision']:.3f}  R {pc['recall']:.3f}  AP50 {pc['ap50']:.3f}  AP50-95 {pc['ap50_95']:.3f}")
    ch = res["chosen"]
    print(f"\nChosen on validation: single t = {ch['single_threshold']:.2f}; review band = [{ch['review_low']:.2f}, {ch['review_high']:.2f})")
    print("Expected cost per 1,000 hands, TEST split:")
    for k, v in st.items():
        extra = f"  ({v['share_to_person']:.1%} to a person = {v['hands_per_day']:.0f}/day = {v['people']:.2f} people)" if "people" in v else ""
        print(f"   {k:26s} EUR {v['cost_per_1000']:8.2f}{extra}")
    print("\nSaved: weights/best.pt, outputs/results.json, outputs/predictions.json, outputs/cost_curve.png")


if __name__ == "__main__":
    main()
