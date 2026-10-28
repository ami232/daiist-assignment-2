import sys
from pathlib import Path
from statistics import median

root = Path(sys.argv[1])
names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

for split in ['train', 'valid', 'test']:
    n_images = len(list((root / split / 'images').glob('*')))
    counts = [0] * len(names)
    sizes = [[] for _ in names]
    for f in (root / split / 'labels').glob('*.txt'):
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, w, h = int(p[0]), float(p[3]), float(p[4])
            counts[c] += 1
            sizes[c].append((w * h) ** 0.5)
    print(f"\n{split}: {n_images} images, {sum(counts)} instances")
    for i, n in enumerate(names):
        med = f"{median(sizes[i]) * 100:.1f}%" if sizes[i] else "-"
        print(f"  {n:16s} {counts[i]:5d}   median defect size: {med} of image side")