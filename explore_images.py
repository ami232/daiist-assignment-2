"""Box sizes, split-overlap check, and sample sheets per class."""
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

DEST = Path("datasets/fire-smoke-lwjte-v1")
OUT = Path("report_assets_lwjte")
OUT.mkdir(exist_ok=True)
random.seed(0)

names = yaml.safe_load((DEST / "data.yaml").read_text())["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]

areas = {n: [] for n in names}
samples = {n: [] for n in names}   # (image path, boxes of that class)
sources = {}                       # split -> set of source image names

for split in ["train", "valid", "test"]:
    sources[split] = set()
    for lf in (DEST / split / "labels").glob("*.txt"):
        sources[split].add(lf.stem.split(".rf.")[0])
        per_class = {}
        for line in lf.read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, w, h = line.split()[:5]
            n = names[int(c)]
            areas[n].append(float(w) * float(h))
            per_class.setdefault(n, []).append((float(x), float(y), float(w), float(h)))
        if split == "train":
            img = next((lf.parent.parent / "images").glob(lf.stem + ".*"))
            for n, bxs in per_class.items():
                samples[n].append((img, bxs))

print("Box area as a share of the image (all splits):")
for n in names:
    a = sorted(areas[n])
    print(f"  {n:8s} median {a[len(a)//2]:.3f}   10th pct {a[len(a)//10]:.4f}   90th pct {a[9*len(a)//10]:.3f}")

print("\nSource images shared between splits:")
print("  train & test :", len(sources["train"] & sources["test"]))
print("  train & valid:", len(sources["train"] & sources["valid"]))

for n in names:
    tiles = []
    for img, bxs in random.sample(samples[n], min(6, len(samples[n]))):
        im = Image.open(img).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        for x, y, w, h in bxs:
            d.rectangle([(x - w/2) * W, (y - h/2) * H, (x + w/2) * W, (y + h/2) * H],
                        outline="red", width=4)
        tiles.append(im.resize((320, 320)))
    sheet = Image.new("RGB", (960, 640))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 3) * 320, (i // 3) * 320))
    sheet.save(OUT / f"class_{n}.jpg")
print("\nSaved sample sheets to report_assets/")
