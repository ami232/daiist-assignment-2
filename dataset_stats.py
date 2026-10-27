"""Count images, defects per class and defect size for each split."""
from pathlib import Path
import yaml

DATA = Path("data")
names = yaml.safe_load((DATA / "data.yaml").read_text())["names"]

for split in ["train", "valid", "test"]:
    label_files = list((DATA / split / "labels").glob("*.txt"))
    counts = {name: 0 for name in names}
    areas = []
    for f in label_files:
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, w, h = int(parts[0]), float(parts[3]), float(parts[4])
            counts[names[cls]] += 1
            areas.append(w * h * 100)  # % of the image the box covers

    print(f"\n=== {split}: {len(label_files)} images, {sum(counts.values())} defects ===")
    for name, n in counts.items():
        print(f"  {name:16s} {n}")
    areas.sort()
    print(f"  defect size: median {areas[len(areas)//2]:.3f}% of image, "
          f"largest {areas[-1]:.3f}%")