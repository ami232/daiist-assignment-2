from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "missing-pills-v1"
OUTPUT = ROOT / "analysis"

EMPTY = {"nopill_front", "nopill_back"}
FILLED = {"pill_front", "pill_back"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

config = yaml.safe_load((DATA / "data.yaml").read_text())
names = config["names"]

if isinstance(names, list):
    names = dict(enumerate(names))
else:
    names = {int(key): value for key, value in names.items()}

class_rows = []
split_rows = []

for split in ("train", "valid", "test"):
    images = sorted(
        path
        for path in (DATA / split / "images").iterdir()
        if path.suffix.lower() in IMAGE_TYPES
    )

    instances = Counter()
    images_per_class = Counter()
    areas = defaultdict(list)

    incomplete = 0
    complete = 0
    unknown = 0

    for image in images:
        label = DATA / split / "labels" / f"{image.stem}.txt"

        if not label.exists():
            raise FileNotFoundError(f"Missing annotation: {label}")

        present = set()

        for line in label.read_text().splitlines():
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(f"Unexpected detection format: {label}")

            class_id = int(values[0])
            _, _, width, height = map(float, values[1:])

            if class_id not in names:
                raise ValueError(f"Unknown class in {label}")

            if not (0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid box size in {label}")

            name = names[class_id]
            instances[name] += 1
            areas[name].append(100 * width * height)
            present.add(name)

        images_per_class.update(present)

        if present & EMPTY:
            incomplete += 1
        elif present & FILLED:
            complete += 1
        else:
            unknown += 1

    split_rows.append({
        "split": split,
        "images": len(images),
        "incomplete": incomplete,
        "complete": complete,
        "unknown": unknown,
    })

    for name in names.values():
        class_rows.append({
            "split": split,
            "class": name,
            "instances": instances[name],
            "images_with_class": images_per_class[name],
            "median_box_area_pct": (
                round(median(areas[name]), 3)
                if areas[name] else None
            ),
        })

OUTPUT.mkdir(exist_ok=True)

splits = pd.DataFrame(split_rows)
classes = pd.DataFrame(class_rows)

splits.to_csv(OUTPUT / "dataset_splits.csv", index=False)
classes.to_csv(OUTPUT / "dataset_classes.csv", index=False)

print("\nIMAGES PER SPLIT")
print(splits.to_string(index=False))

print("\nOBJECTS AND RELATIVE SIZE PER CLASS")
print(classes.to_string(index=False))

print("\nSaved results in:", OUTPUT)