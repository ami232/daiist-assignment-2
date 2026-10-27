"""Dataset statistics for REPORT.md. Run: uv run python stats.py"""
import json

import pandas as pd
import yaml

from dataset import DATASET_DIR, get_dataset

SPLITS = ["train", "valid", "test"]


def read_boxes(label_file):
    """Return (class_id, width, height) per object, all normalised to the image."""
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
        if len(vals) == 4:  # x, y, w, h
            w, h = vals[2], vals[3]
        else:  # polygon: take its bounding box
            xs, ys = vals[0::2], vals[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((cls, w, h))
    return boxes


def main():
    names = yaml.safe_load(get_dataset().read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    rows, images = [], {}
    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        n_images = n_empty = 0
        for img in sorted(img_dir.iterdir()):
            n_images += 1
            lbl = lbl_dir / f"{img.stem}.txt"
            boxes = read_boxes(lbl) if lbl.exists() else []
            n_empty += not boxes
            for cls, w, h in boxes:
                rows.append({"split": split, "class": names[cls], "area": w * h})
        images[split] = {"images": n_images, "no_objects": n_empty}

    df = pd.DataFrame(rows)
    print("\nImages per split")
    print(pd.DataFrame(images).T)
    print("\nInstances per class and split")
    counts = df.pivot_table(index="class", columns="split", values="area",
                            aggfunc="count", fill_value=0)[SPLITS]
    print(counts)
    print("\nBox area as % of the image (all splits)")
    summary = df.groupby("class")["area"].agg(
        median_pct=lambda s: 100 * s.median(),
        p10_pct=lambda s: 100 * s.quantile(0.10),
        p90_pct=lambda s: 100 * s.quantile(0.90),
        below_1pct=lambda s: 100 * (s < 0.01).mean(),
    ).round(2)
    print(summary)

    out = DATASET_DIR.parent.parent / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "dataset_stats.json").write_text(json.dumps({
        "images": images,
        "instances": counts.to_dict(),
        "box_area_pct": summary.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()