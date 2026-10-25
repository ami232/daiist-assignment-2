"""Evaluate weights/best.pt on test, compare with baselines, dump predictions for the sweep.

Run:  uv run python evaluate.py
Outputs in outputs/: eval_results.json, predictions_test.json
Confusion matrices and curves land in runs/eval_test/ (made by ultralytics).
"""
import json
import sys
import traceback
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

DATA_DIR = Path("datasets/fire-smoke-lwjte-v1").resolve()
DATA_YAML = DATA_DIR / "data_local.yaml"
WEIGHTS = Path("weights/best.pt")
SETTINGS = json.loads(Path("weights/train_settings.json").read_text())
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def stage(label):
    print(f"\n>>> {label}", flush=True)


def make_data_yaml() -> None:
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    cfg = {"path": str(DATA_DIR), "train": "train/images", "val": "valid/images",
           "test": "test/images", "names": names}
    DATA_YAML.write_text(yaml.safe_dump(cfg))


def per_class_metrics(model, split):
    m = model.val(data=str(DATA_YAML), split=split, imgsz=640, batch=16, device=DEVICE,
                  conf=0.001, iou=0.7, plots=True, project=str(Path("runs").resolve()),
                  name=f"eval_{split}", exist_ok=True, verbose=False)
    rows = {}
    for i, c in enumerate(m.box.ap_class_index):
        rows[model.names[int(c)]] = {
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i]),
        }
    overall = {"precision": float(m.box.mp), "recall": float(m.box.mr),
               "ap50": float(m.box.map50), "map50_95": float(m.box.map)}
    return {"overall": overall, "per_class": rows}


def dump_predictions(model, split):
    """Raw predictions at very low confidence plus ground truth, for the cost sweep.
    gt rows: [class, cx, cy, w, h] normalised. pred rows: [class, conf, x1, y1, x2, y2] normalised."""
    img_dir, lbl_dir = DATA_DIR / split / "images", DATA_DIR / split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    data = {}
    for res in model.predict(source=[str(p) for p in images], conf=0.001, iou=0.7, imgsz=640,
                             max_det=100, device=DEVICE, stream=True, verbose=False):
        lf = lbl_dir / f"{Path(res.path).stem}.txt"
        gt = ([[float(v) for v in line.split()[:5]] for line in lf.read_text().splitlines() if line.strip()]
              if lf.exists() else [])
        b = res.boxes
        pred = [[int(c), float(s), *map(float, xy)]
                for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]
        data[Path(res.path).name] = {"gt": gt, "pred": pred}
    payload = {"names": {int(k): v for k, v in model.names.items()}, "images": data}
    (OUT / f"predictions_{split}.json").write_text(json.dumps(payload))
    return data


def coco_zero_shot(test_data):
    """Pretrained COCO model of the same size, used as is."""
    coco = YOLO(SETTINGS["model"])
    related = {i: n for i, n in coco.names.items() if "fire" in n or "smoke" in n}
    paths = [str(DATA_DIR / "test" / "images" / name) for name in test_data]
    frames_with_related = 0
    for res in coco.predict(source=paths, conf=0.25, imgsz=640, device=DEVICE, stream=True, verbose=False):
        if any(int(c) in related for c in res.boxes.cls.tolist()):
            frames_with_related += 1
    return {"coco_classes_matching_fire_or_smoke": related,
            "test_frames_with_such_a_detection": frames_with_related,
            "ap50": 0.0,
            "note": "COCO has no fire or smoke class, so box-level AP is 0 by construction."}


def main():
    make_data_yaml()
    model = YOLO(str(WEIGHTS))

    stage("1/3 metrics on test")
    test = per_class_metrics(model, "test")
    o = test["overall"]
    print("\n=== TEST, finetuned model ===", flush=True)
    print(f"all    P {o['precision']:.3f}  R {o['recall']:.3f}  AP50 {o['ap50']:.3f}  mAP50-95 {o['map50_95']:.3f}")
    for name, r in test["per_class"].items():
        print(f"{name:6s} P {r['precision']:.3f}  R {r['recall']:.3f}  AP50 {r['ap50']:.3f}  mAP50-95 {r['map50_95']:.3f}", flush=True)
    (OUT / "eval_results.json").write_text(
        json.dumps({"test": test, "weights": str(WEIGHTS), "settings": SETTINGS}, indent=2))

    stage("2/3 predictions on test (for the cost sweep)")
    test_data = dump_predictions(model, "test")
    print(f"saved predictions_test.json ({len(test_data)} images)", flush=True)

    stage("3/3 baselines")
    n = len(test_data)
    n_bg = sum(1 for v in test_data.values() if not v["gt"])
    baselines = {
        "predict_nothing": {"ap50": 0.0, "recall": 0.0,
                            "note": "no boxes means no true positives, so AP is 0 by definition"},
        "always_alert_frame_level": {"test_frames": n, "frames_without_fire_or_smoke": n_bg,
                                     "accuracy": (n - n_bg) / n},
        "coco_zero_shot": coco_zero_shot(test_data),
    }
    (OUT / "eval_results.json").write_text(
        json.dumps({"test": test, "baselines": baselines, "weights": str(WEIGHTS), "settings": SETTINGS}, indent=2))
    print(json.dumps(baselines, indent=2))
    print("\nDONE. Saved outputs/eval_results.json and outputs/predictions_test.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)