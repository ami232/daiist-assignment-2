"""Download v10 of mediiot/fire-smoke-znprn and count instances per class."""
from collections import Counter
from pathlib import Path

import yaml
from roboflow import Roboflow

from api_key import load_api_key

WORKSPACE = "mediiot"
PROJECT = "fire-smoke-znprn"
VERSION = 10
DEST = Path("datasets") / f"{PROJECT}-v{VERSION}"

if not DEST.exists():
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DEST)
    )

names = yaml.safe_load((DEST / "data.yaml").read_text())["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
print("Classes:", names)

for split in ["train", "valid", "test"]:
    labels = list((DEST / split / "labels").glob("*.txt"))
    counts, empty = Counter(), 0
    for f in labels:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        for l in lines:
            counts[names[int(l.split()[0])]] += 1
    print(f"\n{split}: {len(labels)} images, {empty} with no objects (background)")
    for n in names:
        print(f"  {n:10s} {counts[n]:6d}")