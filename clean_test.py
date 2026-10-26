import collections
from pathlib import Path

root = Path("_explore/whwcy_v5")

def base(path):
    stem = path.name.split(".rf.")[0]
    return stem.replace("_rotated", "")

train_bases = {base(p) for p in (root / "train" / "images").glob("*")}

total = clean = 0
clean_files = []
counts_all = collections.Counter()
counts_clean = collections.Counter()

for img in sorted((root / "test" / "images").glob("*")):
    label = root / "test" / "labels" / (img.stem + ".txt")
    classes = [l.split()[0] for l in label.read_text().splitlines() if l.strip()] if label.exists() else []
    total += 1
    counts_all.update(classes)
    if base(img) not in train_bases:
        clean += 1
        clean_files.append(img.name)
        counts_clean.update(classes)

print(f"Test total: {total} | sin solape con train: {clean} ({100*clean/total:.0f}%)")
print("\nInstancias por clase (id): todo el test -> test limpio")
for c in sorted(counts_all, key=int):
    print(f"  clase {c}: {counts_all[c]} -> {counts_clean.get(c, 0)}")

Path("_explore/clean_test_files.txt").write_text("\n".join(clean_files))