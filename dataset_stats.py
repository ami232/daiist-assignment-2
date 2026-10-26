import collections
from pathlib import Path

import yaml

root = Path("_explore/whwcy_v5")
names = yaml.safe_load((root / "data.yaml").read_text())["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
print("Clases:", names)

for split in ("train", "valid", "test"):
    counts = collections.Counter()
    areas = collections.defaultdict(list)
    empty = 0
    files = list((root / split / "labels").glob("*.txt"))
    for f in files:
        lines = [l.split() for l in f.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        for p in lines:
            c = int(p[0])
            counts[c] += 1
            areas[c].append(float(p[3]) * float(p[4]))
    print(f"\n{split}: {len(files)} imagenes, {empty} sin ningun defecto")
    for c in sorted(counts):
        mean_area = 100 * sum(areas[c]) / len(areas[c])
        print(f"   {names[c]:<18} {counts[c]:>6} instancias | area media {mean_area:.2f}% de la imagen")

# pistas de aumentos: nombres de archivo repetidos con sufijo .rf.
stems = collections.Counter(p.name.split(".rf.")[0] for p in (root / "train" / "images").glob("*"))
print("\nImagenes originales distintas en train:", len(stems), "de", sum(stems.values()))