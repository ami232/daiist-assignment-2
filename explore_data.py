import statistics
from collections import Counter
from pathlib import Path

import yaml
from roboflow import Roboflow

from api_key import load_api_key

rf = Roboflow(api_key=load_api_key())
project = rf.workspace("sayed-gamall").project("fire-smoke-detection-yolov11")
dataset = project.version(2).download("yolov11", location="datasets/fire-smoke-v2")

root = Path(dataset.location)
names = yaml.safe_load((root / "data.yaml").read_text())["names"]
if isinstance(names, list):
    names = {i: n for i, n in enumerate(names)}

for split in ["train", "valid", "test"]:
    n_images = len(list((root / split / "images").glob("*")))
    counts = Counter()
    sizes = {i: [] for i in names}
    images_with_objects = 0
    for f in (root / split / "labels").glob("*.txt"):
        lines = [l.split() for l in f.read_text().splitlines() if l.strip()]
        if lines:
            images_with_objects += 1
        for parts in lines:
            c = int(parts[0])
            nums = [float(x) for x in parts[1:]]
            if len(nums) == 4:
                # box format: x_center, y_center, width, height
                w, h = nums[2], nums[3]
            else:
                # outline format: a list of x, y points
                xs, ys = nums[0::2], nums[1::2]
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            counts[c] += 1
            sizes[c].append(w * h)

    print(f"\n=== {split} ===")
    print(f"images: {n_images}, with objects: {images_with_objects}, empty: {n_images - images_with_objects}")
    for c, name in names.items():
        med = statistics.median(sizes[c]) * 100 if sizes[c] else 0
        print(f"{name}: {counts[c]} objects, typical size = {med:.1f}% of the image")