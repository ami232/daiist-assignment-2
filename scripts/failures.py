"""
Show the test images the model gets most wrong.

From outputs/test_scores.csv: the 2 positives with the lowest image score and
the 3 negatives with the highest. Each is copied to outputs/failures/ with
its top-3 predictions at conf 0.001 drawn (class and confidence) and the
ground-truth boxes in white. Writes outputs/failures/failures.md and
failures.csv: filename, ground truth, score, what the model predicted.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CLASS_NAMES, ensure_dataset  # noqa: E402
from src.config import POSITIVE_CLASSES  # noqa: E402

OUT_DIR = ROOT / "outputs"
FAIL_DIR = OUT_DIR / "failures"
WEIGHTS = ROOT / "weights" / "best.pt"
N_LOW_POS, N_HIGH_NEG, TOP_K, CONF = 2, 3, 3, 0.001
COLORS = {"Box": (200, 160, 60), "Package": (200, 120, 200),
          "Box_broken": (40, 60, 230), "Open_package": (40, 160, 240)}


def gt_boxes(label_file: Path, w: int, h: int) -> list[tuple[str, int, int, int, int]]:
    out = []
    if not label_file.is_file():
        return out
    for line in label_file.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:5])
        out.append((CLASS_NAMES[cls], int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                    int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return out


def draw(img, preds, gts):
    for name, x0, y0, x1, y1 in gts:
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(img, f"GT {name}", (x0 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Labels stack top-left of the image in rank order, so overlapping boxes
    # (which share corners) never hide each other's text.
    for rank, (name, conf, x0, y0, x1, y1) in enumerate(preds, 1):
        color = COLORS.get(name, (0, 255, 0))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        text = f"#{rank} {name} {conf:.2f}  [{x0},{y0}-{x1},{y1}]"
        pos = (6, 22 * rank)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"#{rank}", (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"#{rank}", (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


def main() -> int:
    from ultralytics import YOLO

    data_yaml = ensure_dataset()
    img_dir = data_yaml.parent / "test" / "images"
    lbl_dir = data_yaml.parent / "test" / "labels"
    with (OUT_DIR / "test_scores.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
    pos = sorted((r for r in rows if r["label"] == "1"), key=lambda r: r["score"])[:N_LOW_POS]
    neg = sorted((r for r in rows if r["label"] == "0"), key=lambda r: -r["score"])[:N_HIGH_NEG]
    picks = [("missed positive (lowest score)", r) for r in pos] + [("false alarm (highest-scoring negative)", r) for r in neg]

    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    table = []
    for kind, r in picks:
        path = img_dir / r["filename"]
        res = model.predict(str(path), conf=CONF, imgsz=640, verbose=False)[0]
        order = res.boxes.conf.argsort(descending=True)[:TOP_K]
        preds = []
        for i in order.tolist():
            name = res.names[int(res.boxes.cls[i])]
            x0, y0, x1, y1 = (int(v) for v in res.boxes.xyxy[i].tolist())
            preds.append((name, float(res.boxes.conf[i]), x0, y0, x1, y1))
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gts = gt_boxes(lbl_dir / (Path(r["filename"]).stem + ".txt"), w, h)
        cv2.imwrite(str(FAIL_DIR / r["filename"]), draw(img, preds, gts))
        best_pos = max((p for p in preds if p[0] in POSITIVE_CLASSES), key=lambda p: p[1], default=None)
        table.append({
            "kind": kind,
            "filename": r["filename"],
            "ground_truth": ", ".join(f"{g[0]}" for g in gts) or "none",
            "main_class": r["main_class"],
            "score": f"{r['score']:.4f}",
            "top3_predictions": "; ".join(f"{p[0]} {p[1]:.3f}" for p in preds) or "nothing",
            "best_positive_pred": f"{best_pos[0]} {best_pos[1]:.3f}" if best_pos else "none",
            "dup_in_trainval": r["dup_in_trainval"],
        })

    with (FAIL_DIR / "failures.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    md = ["# Worst test cases", "",
          f"Lowest-scoring {N_LOW_POS} positives and highest-scoring {N_HIGH_NEG} negatives. "
          f"Predictions at conf {CONF}, top {TOP_K} boxes drawn; ground truth in white.", "",
          "| Kind | File | Ground truth | Score | Top-3 predictions |", "|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['kind']} | `{t['filename']}` | {t['ground_truth']} | {t['score']} | {t['top3_predictions']} |")
    (FAIL_DIR / "failures.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for t in table:
        print(f"{t['kind']:42s} {t['filename'][:34]:34s} gt={t['ground_truth']:28s} score={t['score']}  preds: {t['top3_predictions']}")
    print(f"wrote {len(table)} images, failures.md and failures.csv to {FAIL_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
