import json
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "missing-pills-v1"
OUT = ROOT / "analysis"


def main():
    OUT.mkdir(exist_ok=True)

    # Replace Colab paths with paths valid on this computer.
    config = yaml.safe_load((DATA / "data.yaml").read_text())
    config.update({
        "path": str(DATA.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    })

    config_path = DATA / "evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )

    model = YOLO(str(ROOT / "weights" / "best.pt"))
    metrics = model.val(
        data=str(config_path),
        split="test",
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        conf=0.001,
        plots=True,
        project=str(ROOT / "runs"),
        name="test_evaluation",
    )

    overall = {
        "split": "test",
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    (OUT / "test_overall.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    rows = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        rows.append({
            "class": model.names[int(class_id)],
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "AP50": float(metrics.box.ap50[index]),
            "AP50_95": float(metrics.box.ap[index]),
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "test_per_class.csv", index=False)

    print("\nOVERALL TEST RESULTS")
    print(json.dumps(overall, indent=2))
    print("\nPER-CLASS TEST RESULTS")
    print(table.to_string(index=False))
    print(f"\nPlots saved to: {metrics.save_dir}")


if __name__ == "__main__":
    main()