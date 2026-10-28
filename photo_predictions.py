"""Build validation exports; the shared inference helper also serves frozen test evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/.matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/.ultralytics"))
INFERENCE = {"imgsz": 640, "batch": 1, "rect": False, "confidence": .001,
             "nms_iou": .7, "max_det": 300}
OUT = ROOT / "results/validation_analysis"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=lambda scalar: scalar.item()) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_labels(bundle):
    """Name architectures as well as epoch counts when comparing saved runs."""
    labels = {}
    for key, metadata in bundle["models"].items():
        architecture = metadata.get("model_architecture", "yolo11n").replace("yolo", "YOLO")
        label = f"{architecture} · {metadata['completed_epochs']} épocas"
        if label in labels:
            raise ValueError(f"Ambiguous model label: {label}")
        labels[label] = key
    return labels


def records_from_model(weights, files, cache, classes, device, coco=False, split="valid"):
    from ultralytics import YOLO
    expected_hash = digest(weights)
    if cache.exists():
        saved = read(cache)
        if (saved["weights_sha256"] == expected_hash and saved["inference"] == INFERENCE
                and saved["evaluation_split"] == split
                and {r["image"] for r in saved["images"]} == {p.relative_to(ROOT).as_posix() for p in files}
                and len(saved["images"]) == len(files)):
            print(f"Using verified predictions: {cache.relative_to(ROOT)}", flush=True)
            return saved
        raise ValueError(f"Cache does not match this model/data: {cache}")
    model = YOLO(str(weights))
    if not coco:
        assert [model.names[i] for i in range(len(model.names))] == classes
    records = []
    raw_predictions = 0
    car_images = 0
    for result in model.predict(source=str(files[0].parent), stream=True, device=device,
                                imgsz=640, batch=1, rect=False, conf=.001, iou=.7,
                                max_det=300, verbose=False):
        path = Path(result.path)
        truth = []
        for line in (path.parent.parent / "labels" / (path.stem + ".txt")).read_text().splitlines():
            if line.strip():
                cls, cx, cy, width, height = map(float, line.split())
                truth.append({"class_id": int(cls), "xyxy": [cx-width/2, cy-height/2, cx+width/2, cy+height/2]})
        predictions = []
        contains_car = False
        for cls, confidence, xyxy in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                         result.boxes.xyxyn.cpu().tolist()):
            source_name = model.names[int(cls)]
            raw_predictions += 1
            contains_car |= source_name == "car"
            if coco:
                # Exact semantic matches only: 'car' does not mean 'Undamaged'.
                if source_name not in classes:
                    continue
                cls = classes.index(source_name)
            predictions.append({"class_id": int(cls), "confidence": float(confidence), "xyxy": xyxy})
        car_images += contains_car
        records.append({"image": path.relative_to(ROOT).as_posix(), "truth": truth, "predictions": predictions,
                        "condition_ground_truth_known": bool(truth),
                        "has_damage_label": any(b["class_id"] == 0 for b in truth),
                        "max_damage_confidence": max((b["confidence"] for b in predictions if b["class_id"] == 0), default=0.)})
        if len(records) % 50 == 0:
            print(f"{weights.name}: {len(records)}/{len(files)} {split} photos", flush=True)
    assert len(records) == len(files) and len({r["image"] for r in records}) == len(files)
    assert {r["image"] for r in records} == {p.relative_to(ROOT).as_posix() for p in files}
    saved = {"evaluation_split": split, "test_evaluated": split == "test", "weights_sha256": expected_hash,
             "classes": classes, "inference": INFERENCE, "device": device, "images": records}
    if coco:
        saved["zero_shot"] = {"source_classes": model.names, "mapping": "Exact class-name matches only",
                              "raw_prediction_count": raw_predictions, "photos_with_coco_car": car_images,
                              "mapped_target_prediction_count": sum(len(r["predictions"]) for r in records),
                              "limitation": "COCO contains neither Damage-detected nor Undamaged. Car detections are not condition labels."}
    assert digest(weights) == expected_hash
    write(cache, saved)
    return saved


def main():
    from train import choose_device
    from ultralytics.utils.downloads import attempt_download_asset
    config = read(ROOT / "dataset_config.json")
    files = sorted(p for p in (ROOT / config["data_dir"] / "valid/images").iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    assert len(files) == config["measured_export_images"]["valid"]
    catalog = read(ROOT / "results/experiment_catalog.json")
    runs = {key: ROOT / directory for key, directory in catalog.items()}
    device = choose_device(["cuda", "mps", "cpu"])
    model_data = {}
    truths = None
    for key, directory in runs.items():
        summary = read(directory / "summary.json")
        weights = ROOT / summary["weights"]
        assert digest(weights) == summary["weights_sha256"]
        saved = records_from_model(weights, files, directory / "review/validation_predictions.json", config["classes"], device)
        by_image = {r["image"]: r for r in saved["images"]}
        if truths is None:
            truths = {image: {k: r[k] for k in ("truth", "has_damage_label", "condition_ground_truth_known")}
                      for image, r in by_image.items()}
        assert all(by_image[image]["truth"] == truths[image]["truth"] for image in truths)
        model_data[key] = {"weights": summary["weights"], "weights_sha256": summary["weights_sha256"],
                           "model_architecture": summary.get("model_architecture", "yolo11n"),
                           "pretrained_weights_sha256": summary.get("pretrained_weights_sha256"),
                           "best_epoch": summary["best_epoch"], "completed_epochs": summary["completed_epochs"],
                           "detection_metrics": summary["metrics"], "per_class": summary["per_class"],
                           "scores": {image: r["max_damage_confidence"] for image, r in by_image.items()}}
    pretrained = ROOT / "data/pretrained/yolo11n.pt"
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    pretrained = Path(attempt_download_asset(str(pretrained)))
    zero = records_from_model(pretrained, files, OUT / "coco_zero_shot_predictions.json", config["classes"], device, coco=True)
    assert zero["zero_shot"]["mapped_target_prediction_count"] == 0
    baseline = {"class_order": config["classes"], "per_class": [
        {"Class": name, "AP50": 0., "AP50_95": 0., "recall": 0., "precision": None}
        for name in config["classes"]], "mAP50": 0., "mAP50_95": 0.,
        "precision_note": "Undefined without target predictions; do not present as measured positive predictive value."}
    write(OUT / "photo_predictions.json", {
        "evaluation_split": "valid", "test_evaluated": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "inference": INFERENCE,
        "unit": "one_photo_as_proxy_for_one_return", "truths": truths, "models": model_data,
        "baselines": {"coco_zero_shot": {**baseline, "inference_ran": True, **zero["zero_shot"]},
                      "predict_nothing": {**baseline, "inference_ran": False, "definition": "No boxes and no damage alerts."}},
        "unknown_condition_policy": "Keep all 366 images for detection; exclude the empty-label case from condition/cost rates.",
    })
    print("VALIDATION PREDICTIONS COMPLETE: results/validation_analysis/photo_predictions.json", flush=True)


if __name__ == "__main__":
    main()
