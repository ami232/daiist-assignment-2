"""How many valid/test images have a near-duplicate in train? (perceptual hash)"""
from pathlib import Path

import numpy as np
from PIL import Image

DEST = Path("datasets/fire-smoke-znprn-v10")


def dhash(path):
    im = Image.open(path).convert("L").resize((9, 8))
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())  # 64 bits


def load(split):
    files = sorted((DEST / split / "images").glob("*.*"))
    return files, np.stack([dhash(f) for f in files])


_, train = load("train")
for split in ["test", "valid"]:
    files, h = load(split)
    dist = np.array([np.unpackbits(train ^ x, axis=1).sum(axis=1).min() for x in h])
    print(f"\n{split}: {len(files)} images, distance to nearest train image (0 = identical)")
    print(f"  <= 4  (near-identical): {(dist <= 4).mean():.0%}")
    print(f"  <= 10 (very similar)  : {(dist <= 10).mean():.0%}")
    print(f"  median distance       : {np.median(dist):.0f}")