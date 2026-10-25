"""Download any Roboflow dataset version, count classes, and check train/test leakage.
Usage: uv run python check_dataset.py <workspace> <project> <version>
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from roboflow import Roboflow

from api_key import load_api_key

ws, proj, ver = sys.argv[1], sys.argv[2], int(sys.argv[3])
DEST = Path("datasets") / f"{proj}-v{ver}"
if not DEST.exists():
    Roboflow(api_key=load_api_key()).workspace(ws).project(proj).version(ver).download(
        "yolov11", location=str(DEST)
    )

names = yaml.safe_load((DEST / "data.yaml").read_text())["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
print("Classes:", names)

for split in ["train", "valid", "test"]:
    lf = list((DEST / split / "labels").glob("*.txt"))
    cnt, empty = Counter(), 0
    for f in lf:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        empty += not lines
        cnt.update(names[int(l.split()[0])] for l in lines)
    print(f"{split}: {len(lf)} images, {empty} background | {dict(cnt)}")


def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8)), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def load(split):
    return np.stack([dhash(f) for f in sorted((DEST / split / "images").glob("*.*"))])


train = load("train")
for split in ["test"]:
    h = load(split)
    d = np.array([np.unpackbits(train ^ x, axis=1).sum(axis=1).min() for x in h])
    print(f"\n{split} vs train: near-identical (<=4): {(d <= 4).mean():.0%}, "
          f"very similar (<=10): {(d <= 10).mean():.0%}, median {np.median(d):.0f}")