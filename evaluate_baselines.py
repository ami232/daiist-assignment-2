import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "missing-pills-v1"
OUT = ROOT / "analysis"


def main():
    OUT.mkdir(exist_ok=True)

    config = yaml.safe_load((DATA / "data.yaml").read_text())
    names = config["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    else:
        names = {int(key): value for key, value in names.items()}

    images = sorted(
        path for path in (DATA / "test" / "images").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )

    instances = Counter()
    positive_images = 0

    for path in images:
        label = DATA / "test" / "labels" / f"{path.stem}.txt"
        classes = [
            names[int(line.split()[0])]
            for line in label.read_text().splitlines()
            if line.strip()
        ]
        instances.update(classes)
        positive_images += int(
            any(name in {"nopill_front", "nopill_back"} for name in classes)
        )

    # Original COCO weights, not the finetuned best.pt.
    model = YOLO(str(ROOT / "yolo11n.pt"))
    shared_classes = set(names.values()) & set(model.names.values())

    if shared_classes:
        raise RuntimeError(
            f"Review the class mapping before evaluation: {shared_classes}"
        )

    raw_predictions = []
    predicted_classes = Counter()

    results = model.predict(
        source=[str(path) for path in images],
        imgsz=640,
        conf=0.001,
        device="cpu",
        stream=True,
        verbose=False,
    )

    for result in results:
        detections = []
        for box in result.boxes:
            name = model.names[int(box.cls.item())]
            predicted_classes[name] += 1
            detections.append({
                "class": name,
                "confidence": float(box.conf.item()),
                "xyxy": box.xyxy[0].tolist(),
            })
        raw_predictions.append({
            "image": Path(result.path).name,
            "detections": detections,
        })

    (OUT / "coco_raw_predictions.json").write_text(
        json.dumps(raw_predictions, indent=2), encoding="utf-8"
    )

    # Neither baseline produces a semantically valid target-class box.
    per_class = []
    summaries = []

    for baseline in ("COCO zero-shot: no matching classes", "Predict nothing"):
        for name in names.values():
            per_class.append({
                "baseline": baseline,
                "class": name,
                "ground_truth_instances": instances[name],
                "target_predictions": 0,
                "precision": None,
                "recall": 0.0,
                "AP50": 0.0,
                "AP50_95": 0.0,
            })

        summaries.append({
            "baseline": baseline,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "image_TP": 0,
            "image_FP": 0,
            "image_FN": positive_images,
            "image_TN": len(images) - positive_images,
        })

    pd.DataFrame(per_class).to_csv(
        OUT / "baseline_per_class.csv", index=False
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "baseline_summary.csv", index=False)

    print("\nCOCO classes predicted, before task mapping:")
    print(dict(predicted_classes.most_common(10)))
    print("\nBASELINE RESULTS")
    print(summary.to_string(index=False))
    print("\nPrecision is undefined when there are no target predictions.")
    print("COCO results use an empty semantic mapping to the target classes.")


if __name__ == "__main__":
    main()