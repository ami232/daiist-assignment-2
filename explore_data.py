from pathlib import Path
from collections import Counter

import yaml
from roboflow import Roboflow

from api_key import load_api_key

DATA_DIR = Path("data")

rf = Roboflow(api_key=load_api_key())
version = rf.workspace("trash-32e51").project("search-rescue").version(14)
version.download("yolov11", location=str(DATA_DIR))

names = yaml.safe_load(open(DATA_DIR / "data.yaml"))["names"]
print("\nClasses:", names)

for split in ["train", "valid", "test"]:
    label_dir = DATA_DIR / split / "labels"
    n_images = len(list((DATA_DIR / split / "images").glob("*")))
    counts, areas = Counter(), []
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            cls, x, y, w, h = line.split()[:5]
            counts[int(cls)] += 1
            areas.append(float(w) * float(h))
    print(f"\n{split}: {n_images} images")
    for cls_id, n in sorted(counts.items()):
        print(f"  {names[cls_id]}: {n} instances")
    if areas:
        areas.sort()
        print(f"  median box area as share of image: {areas[len(areas)//2]:.3%}")