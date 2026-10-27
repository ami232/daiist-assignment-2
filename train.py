"""Assignment 2 training pipeline: PCB defect detection."""
import json
import shutil
from pathlib import Path

import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

# Dataset on Roboflow Universe (recorded in REPORT.md)
WORKSPACE = "object-detection-dt-wzpc6"
PROJECT = "pcb-dataset-defect"
VERSION = 1

# Training settings
BASE_MODEL = "yolo11s.pt"  # pretrained COCO checkpoint we finetune from
IMGSZ = 1024               # larger than the default 640 because defects are tiny
EPOCHS = 100
PATIENCE = 20              # stop early if validation stops improving
BATCH = 8

IOU_MATCH = 0.5            # a box "finds" a defect if it overlaps it this much


# ---------- 1. Data ----------

def download_data() -> Path:
    """Download the dataset and point data.yaml at the right folders."""
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DATA_DIR)
    )
    yaml_path = DATA_DIR / "data.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    cfg.update(path=str(DATA_DIR), train="train/images",
               val="valid/images", test="test/images")
    yaml_path.write_text(yaml.safe_dump(cfg))
    return yaml_path


def load_test_set() -> list:
    """Return (image path, list of true boxes) for every test image.
    Boxes are (class, x1, y1, x2, y2) in 0-1 image coordinates."""
    items = []
    for img in sorted((DATA_DIR / "test" / "images").iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = DATA_DIR / "test" / "labels" / f"{img.stem}.txt"
        boxes = []
        for line in label.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, x, y, w, h = int(p[0]), *map(float, p[1:5])
            boxes.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        items.append((img, boxes))
    return items


# ---------- 2. Training ----------

def train_model(data_yaml: Path) -> None:
    """Finetune YOLO from the pretrained checkpoint and save the best weights."""
    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), imgsz=IMGSZ, epochs=EPOCHS,
                patience=PATIENCE, batch=BATCH, seed=0,
                project=str(ROOT / "runs"), name="pcb", exist_ok=True)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)
    print(f"Saved best weights to {WEIGHTS}")


# ---------- 3. Evaluation and baselines ----------

def iou(a, b) -> float:
    """Overlap between two boxes (x1, y1, x2, y2): 0 = none, 1 = identical."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_test_set(model: YOLO, items: list) -> tuple[list, list]:
    """Class-agnostic scores used by the business analysis.

    defect_scores: for each true defect, the highest confidence of any box
        overlapping it (0 if the model missed it). Each defect is treated as
        its own board, since real defective boards usually have one defect.
    false_alarm_scores: for each board, the highest confidence of any box that
        overlaps no true defect (0 if none). Stand-in for a defect-free board.
    """
    defect_scores, false_alarm_scores = [], []
    for img, truth in items:
        r = model.predict(str(img), imgsz=IMGSZ, conf=0.001, verbose=False)[0]
        preds = list(zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist()))
        matched = set()
        for _, *tbox in truth:
            best = 0.0
            for j, (pbox, conf) in enumerate(preds):
                if iou(tbox, pbox) >= IOU_MATCH:
                    best = max(best, conf)
                    matched.add(j)
            defect_scores.append(best)
        unmatched = [conf for j, (_, conf) in enumerate(preds) if j not in matched]
        false_alarm_scores.append(max(unmatched, default=0.0))
    return defect_scores, false_alarm_scores


def summarise(defect_scores, false_alarm_scores, threshold=0.25) -> dict:
    """Share of defects found and of boards with a false alarm, at one threshold."""
    found = sum(s >= threshold for s in defect_scores) / len(defect_scores)
    false = sum(s >= threshold for s in false_alarm_scores) / len(false_alarm_scores)
    return {"defects_found": found, "boards_with_false_alarm": false}


def evaluate_finetuned(model: YOLO, data_yaml: Path) -> dict:
    """Standard YOLO metrics on the test split, overall and per class."""
    m = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                  project=str(ROOT / "runs"), name="test", exist_ok=True)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[model.names[int(c)]] = {
            "precision": float(p), "recall": float(r),
            "ap50": float(ap50), "ap50_95": float(ap)}
    return {
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class": per_class,
        "confusion_matrix": m.confusion_matrix.matrix.tolist(),
        "confusion_labels": list(model.names.values()) + ["background"],
    }


def evaluate(data_yaml: Path | None = None) -> dict:
    """Compare the finetuned model with both baselines on the same test split."""
    if data_yaml is None:
        data_yaml = download_data()
    items = load_test_set()
    n_defects = sum(len(t) for _, t in items)

    finetuned = YOLO(str(WEIGHTS))
    ft = evaluate_finetuned(finetuned, data_yaml)
    ft_defects, ft_false = score_test_set(finetuned, items)
    ft["at_0.25"] = summarise(ft_defects, ft_false)

    coco = YOLO(BASE_MODEL)  # pretrained, never saw a circuit board
    co_defects, co_false = score_test_set(coco, items)
    co = {"map50": 0.0, "map50_95": 0.0,  # COCO has no defect classes at all
          "at_0.25": summarise(co_defects, co_false)}

    trivial = {"map50": 0.0, "map50_95": 0.0,  # predicts nothing
               "at_0.25": {"defects_found": 0.0, "boards_with_false_alarm": 0.0}}

    results = {
        "test_images": len(items), "test_defects": n_defects,
        "finetuned": ft, "coco_zero_shot": co, "trivial": trivial,
        "scores": {"defect": ft_defects, "false_alarm": ft_false},
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nTest set: {len(items)} images, {n_defects} defects")
    print(f"{'model':16s} {'mAP50':>7s} {'mAP50-95':>9s} {'found@0.25':>11s} {'false@0.25':>11s}")
    for name, r in [("finetuned", ft), ("coco_zero_shot", co), ("trivial", trivial)]:
        a = r["at_0.25"]
        print(f"{name:16s} {r['map50']:7.3f} {r['map50_95']:9.3f} "
              f"{a['defects_found']:11.1%} {a['boards_with_false_alarm']:11.1%}")
    print("\nPer class (finetuned):")
    for c, v in ft["per_class"].items():
        print(f"  {c:16s} P={v['precision']:.3f} R={v['recall']:.3f} AP50={v['ap50']:.3f}")
    return results

# ---------- 4. Cost sweep (assumptions justified in REPORT.md) ----------

COST_MISS = 100.0             # defective board shipped: compensation to the client
COST_SCRAP = 5.0              # good board binned automatically: cost of making it
COST_REVIEW = 0.33            # inspector checks one board: ~1 min at ~€20/h
BASE_RATE = 0.05              # share of boards that are actually defective
BOARDS_PER_DAY = 10_000
REVIEWS_PER_INSPECTOR = 420   # 7 productive hours x 60 boards per hour
PER = 1000                    # costs are reported per 1,000 boards
THRESHOLDS = [round(i / 100, 2) for i in range(102)]  # 0.00 ... 1.01 (1.01 = never)


def share_at_or_above(scores: list, t: float) -> float:
    return sum(s >= t for s in scores) / len(scores)


def outcome(D: dict, F: dict, low: float, high: float,
            cost_miss: float = COST_MISS, base_rate: float = BASE_RATE) -> dict:
    """Expected result per 1,000 boards with two thresholds on a board's score:
    score >= high -> rejected automatically
    low <= score < high -> checked by an inspector (who catches real defects)
    score < low -> shipped
    D and F hold, for each threshold, the share of defective / clean boards
    scoring at or above it."""
    bad, good = PER * base_rate, PER * (1 - base_rate)
    shipped_defects = bad * (1 - D[low])
    reviews = bad * (D[low] - D[high]) + good * (F[low] - F[high])
    scrapped_good = good * F[high]
    cost = shipped_defects * cost_miss + reviews * COST_REVIEW + scrapped_good * COST_SCRAP
    return {"low": low, "high": high, "cost": cost,
            "shipped_defects": shipped_defects, "reviews": reviews,
            "scrapped_good": scrapped_good, "review_share": reviews / PER}


def plot_curve(curve: list, review_curve: list, chosen: dict) -> None:
    """Cost against threshold, saved as an image for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([o["low"] for o in curve], [o["cost"] for o in curve],
            label="Model alone (one threshold)")
    ax.plot([o["low"] for o in review_curve], [o["cost"] for o in review_curve],
            label=f"With review (upper threshold fixed at {chosen['high']:.2f})")
    ax.axvline(chosen["low"], linestyle="--", color="grey")
    ax.scatter([chosen["low"]], [chosen["cost"]], zorder=3,
               label=f"Chosen: review from {chosen['low']:.2f}, reject from {chosen['high']:.2f}")
    ax.set_xlabel("Threshold (model confidence)")
    ax.set_ylabel("Expected cost per 1,000 boards (€)")
    ax.set_ylim(0, max(o["cost"] for o in curve) * 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "cost_curve.png", dpi=150)
    plt.close(fig)


def cost_sweep(results: dict | None = None) -> dict:
    """Pick the cheapest thresholds and price them against the alternatives."""
    if results is None:
        results = json.loads((ARTIFACTS / "results.json").read_text())
    defect, false_alarm = results["scores"]["defect"], results["scores"]["false_alarm"]
    D = {t: share_at_or_above(defect, t) for t in THRESHOLDS}
    F = {t: share_at_or_above(false_alarm, t) for t in THRESHOLDS}

    def best(cost_miss=COST_MISS, base_rate=BASE_RATE, with_review=True):
        options = [outcome(D, F, lo, hi, cost_miss, base_rate)
                   for lo in THRESHOLDS for hi in THRESHOLDS
                   if (hi >= lo if with_review else hi == lo)]
        return min(options, key=lambda o: o["cost"])

    chosen = best()
    alone = best(with_review=False)
    curve = [outcome(D, F, t, t) for t in THRESHOLDS]
    review_curve = [outcome(D, F, t, chosen["high"]) for t in THRESHOLDS if t <= chosen["high"]]

    alternatives = {
        "do_nothing": PER * BASE_RATE * COST_MISS,      # ship every board
        "inspect_everything": PER * COST_REVIEW,         # assumes inspectors catch every defect
        "model_alone": alone["cost"],
        "model_plus_review": chosen["cost"],
    }
    reviews_per_day = chosen["review_share"] * BOARDS_PER_DAY
    staffing = {
        "reviews_per_day": reviews_per_day,
        "inspectors_needed": reviews_per_day / REVIEWS_PER_INSPECTOR,
        "inspectors_to_check_everything": BOARDS_PER_DAY / REVIEWS_PER_INSPECTOR,
    }
    sensitivity = [{"cost_miss": cm, "base_rate": br, **best(cm, br)}
                   for cm in (30, 100, 500) for br in (0.02, 0.05, 0.10)]

    sweep = {
        "assumptions": {"cost_miss": COST_MISS, "cost_scrap": COST_SCRAP,
                        "cost_review": COST_REVIEW, "base_rate": BASE_RATE,
                        "boards_per_day": BOARDS_PER_DAY,
                        "reviews_per_inspector": REVIEWS_PER_INSPECTOR},
        "chosen": chosen, "model_alone": alone, "curve": curve,
        "alternatives": alternatives, "staffing": staffing, "sensitivity": sensitivity,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "cost_sweep.json").write_text(json.dumps(sweep, indent=2))
    plot_curve(curve, review_curve, chosen)

    print(f"\nChosen: review from {chosen['low']:.2f}, reject automatically from {chosen['high']:.2f}")
    print(f"  per 1,000 boards: €{chosen['cost']:.2f} | defects shipped {chosen['shipped_defects']:.2f} | "
          f"reviews {chosen['reviews']:.1f} | good boards binned {chosen['scrapped_good']:.2f}")
    print(f"Model alone: threshold {alone['low']:.2f}, €{alone['cost']:.2f} per 1,000 boards")
    print("\nAlternatives (€ per 1,000 boards):")
    for k, v in alternatives.items():
        print(f"  {k:20s} €{v:8.2f}")
    print(f"\nStaffing: {staffing['reviews_per_day']:.0f} reviews/day -> "
          f"{staffing['inspectors_needed']:.1f} inspectors "
          f"(checking everything would need {staffing['inspectors_to_check_everything']:.1f})")
    print("\nSensitivity (best thresholds if the assumptions change):")
    print(f"  {'miss €':>7s} {'base':>5s} {'low':>5s} {'high':>5s} {'€/1000':>8s} {'review':>7s}")
    for s in sensitivity:
        print(f"  {s['cost_miss']:7.0f} {s['base_rate']:5.0%} {s['low']:5.2f} {s['high']:5.2f} "
              f"{s['cost']:8.2f} {s['review_share']:7.1%}")
    return sweep


# ---------- Pipeline ----------

def main() -> None:
    data_yaml = download_data()
    train_model(data_yaml)
    results = evaluate(data_yaml)
    cost_sweep(results)


if __name__ == "__main__":
    main()