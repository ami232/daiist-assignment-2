import hashlib
import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "missing-pills-v1"
OUT = ROOT / "analysis"
WEIGHTS = ROOT / "weights" / "best.pt"

EMPTY = {"nopill_front", "nopill_back"}
POCKETS = EMPTY | {"pill_front", "pill_back"}


def main():
    OUT.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS))

    rows = []
    records = []

    for split in ("valid", "test"):
        paths = sorted(
            path for path in (DATA / split / "images").iterdir()
            if path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp", ".bmp"
            }
        )

        results = model.predict(
            source=[str(path) for path in paths],
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            device="cpu",
            stream=True,
            verbose=False,
        )

        for result in results:
            path = Path(result.path)
            label = DATA / split / "labels" / f"{path.stem}.txt"

            ground_truth = []
            for line in label.read_text().splitlines():
                if line.strip():
                    values = line.split()
                    ground_truth.append({
                        "class": model.names[int(values[0])],
                        "xywhn": [float(value) for value in values[1:]],
                    })

            predictions = []
            for box in result.boxes:
                predictions.append({
                    "class": model.names[int(box.cls.item())],
                    "confidence": float(box.conf.item()),
                    "xyxyn": box.xyxyn[0].tolist(),
                })

            score = max(
                (p["confidence"] for p in predictions if p["class"] in EMPTY),
                default=0.0,
            )
            pocket_confidence = max(
                (p["confidence"] for p in predictions if p["class"] in POCKETS),
                default=0.0,
            )

            rows.append({
                "split": split,
                "image": path.name,
                "positive": int(
                    any(g["class"] in EMPTY for g in ground_truth)
                ),
                "score": score,
                "max_pocket_confidence": pocket_confidence,
            })
            records.append({
                "split": split,
                "image": path.name,
                "ground_truth": ground_truth,
                "predictions": predictions,
            })

        print(f"Finished: {split} ({len(paths)} images)")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "image_scores.csv", index=False)

    cache = {
        "weights_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "inference": {
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
        },
        "images": records,
    }
    (OUT / "prediction_cache.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )

    print("\nIMAGES AND POSITIVE CASES")
    print(table.groupby("split")["positive"].agg(["count", "sum"]))
    print("\nSCORE SUMMARY")
    print(table.groupby(["split", "positive"])["score"].agg(
        ["min", "median", "max"]
    ))


if __name__ == "__main__":
    main()