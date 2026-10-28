"""
Dataset description and train/test leakage check.

    uv run python dataset_stats.py [path/to/dataset]

Reads the YOLO labels and images of the Roboflow export and writes tables and
figures to artifacts/dataset/. The published split leaks: many test photos have
a copy (augmented, renamed or a burst neighbour) in train/valid. Instead of
shrinking the test set, we drop those copies from train/valid, so the full test
split becomes leak-free. The dropped files are listed in
artifacts/dataset/excluded_train_val.csv; train.py calls describe() after
downloading and trains on what is left.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "fruit-v1"
OUT = ROOT / "artifacts" / "dataset"
SPLITS = ["train", "valid", "test"]

AUG_PREFIX = re.compile(r"^(rotated_by_\d+_|vertical_flip_|horizontal_flip_|saltandpepper_|translation_)")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", re.IGNORECASE)

# COCO size bins (<32², 32²–96², >96² px) expressed as a share of a 640x640 image
SMALL_MAX = 32**2 / 640**2
MEDIUM_MAX = 96**2 / 640**2
TINY_MAX = 8**2 / 640**2  # boxes under 8x8 px: almost certainly annotation errors
HASH_MAX_DIST = 3  # dHash bits that may differ for two images to count as the same photo

SPLIT_COLORS = {"train": "#2a78d6", "valid": "#eb6834", "test": "#1baf7a"}
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def base_name(filename: str) -> str:
    """Original photo name: strip the external-augmentation prefix and Roboflow's .rf.<hash> suffix."""
    return AUG_PREFIX.sub("", RF_SUFFIX.sub("", filename))


def name_family(base: str) -> str:
    """Rough source of an image, from its naming pattern (digits collapsed)."""
    if base.startswith("Screen-Shot"):
        return "Screen-Shot (screenshots)"
    if base.startswith("resized_image"):
        return "resized_image_N"
    if base.startswith("IMG"):
        return "IMG* (phone camera)"
    if re.fullmatch(r"\d{8}_\d{6}_jpg", base):
        return "YYYYMMDD_HHMMSS (phone camera)"
    if re.fullmatch(r"\d+_(jpg|png|jpeg)", base, re.IGNORECASE):
        return "N (numbered)"
    return "other"


def read_labels(path: Path) -> list[tuple[int, float, float]]:
    """(class_id, rel_width, rel_height) per object; polygons are turned into their bounding box."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, coords = int(parts[0]), list(map(float, parts[1:]))
        if len(coords) == 4:
            w, h = coords[2], coords[3]
        else:
            xs, ys = coords[0::2], coords[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        rows.append((cls, w, h))
    return rows


def dhash(path: Path) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def load(dataset: Path, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    images, boxes = [], []
    for split in SPLITS:
        for img in sorted((dataset / split / "images").iterdir()):
            labels = read_labels(dataset / split / "labels" / (img.stem + ".txt"))
            base = base_name(img.name)
            classes = sorted({c for c, _, _ in labels})
            images.append({
                "split": split, "file": img.name, "base": base,
                "augmented": bool(AUG_PREFIX.match(img.name)), "family": name_family(base),
                "n_objects": len(labels), "classes": ",".join(names[c] for c in classes),
                "n_classes": len(classes),
                "has_fresh": any(names[c].startswith("fresh") for c in classes),
                "has_rotten": any(names[c].startswith("rotten") for c in classes),
                "path": img,
            })
            boxes += [{"split": split, "file": img.name, "class": names[c], "rel_w": w, "rel_h": h}
                      for c, w, h in labels]
    images, boxes = pd.DataFrame(images), pd.DataFrame(boxes)
    with ThreadPoolExecutor() as pool:  # image decoding releases the GIL
        images["dhash"] = list(pool.map(dhash, images.pop("path")))
    boxes["rel_area"] = boxes.rel_w * boxes.rel_h
    boxes["size_bin"] = pd.cut(boxes.rel_area, [0, SMALL_MAX, MEDIUM_MAX, np.inf],
                               labels=["small", "medium", "large"], include_lowest=True)
    return images, boxes


def leakage(images: pd.DataFrame) -> pd.DataFrame:
    """Flag test images whose original photo (by name or by near-identical pixels) is also in train/valid."""
    test = images[images.split == "test"].copy()
    other = images[images.split != "test"]

    other_classes = other.groupby("base")["classes"].agg(set)
    test["name_match"] = test.base.isin(other_classes.index)
    test["name_match_same_classes"] = [
        r.name_match and r.classes in other_classes[r.base] for r in test.itertuples()
    ]

    other_hashes = np.array(other.dhash.tolist(), dtype=np.uint64)
    dists = [int(np.bitwise_count(other_hashes ^ np.uint64(h)).min()) for h in test.dhash]
    test["hash_min_dist"] = dists
    test["hash_match"] = test.hash_min_dist <= HASH_MAX_DIST
    test["leaked"] = test.name_match | test.hash_match
    return test


def train_val_exclusions(images: pd.DataFrame) -> pd.Series:
    """True for train/valid images that are a copy of some test photo (same base name or near-identical pixels)."""
    other = images[images.split != "test"]
    test = images[images.split == "test"]
    test_hashes = np.array(test.dhash.tolist(), dtype=np.uint64)
    near = [int(np.bitwise_count(test_hashes ^ np.uint64(h)).min()) <= HASH_MAX_DIST for h in other.dhash]
    return other.base.isin(set(test.base)) | pd.Series(near, index=other.index)


def style(ax, title: str):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_instances(per_class: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    x = np.arange(len(per_class))
    width = 0.27
    for i, split in enumerate(SPLITS):
        bars = ax.bar(x + (i - 1) * width, per_class[split], width - 0.02,
                      color=SPLIT_COLORS[split], label=split)
        ax.bar_label(bars, fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, per_class.index, rotation=15)
    ax.set_ylabel("instances (boxes)", color=INK_2)
    ax.legend(frameon=False, fontsize=9)
    style(ax, "Instances per class and split")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_box_sizes(boxes: pd.DataFrame, names: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    data = [boxes.loc[boxes["class"] == n, "rel_area"] * 100 for n in names]
    ax.boxplot(data, tick_labels=names, showfliers=False, widths=0.5, whis=(5, 95),
               medianprops={"color": INK, "linewidth": 2},
               boxprops={"color": "#2a78d6"}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    ax.set_yscale("log")
    ax.set_ylabel("box area, % of image (log)", color=INK_2)
    for y, label in [(SMALL_MAX * 100, "small / medium (32² px)"), (MEDIUM_MAX * 100, "medium / large (96² px)")]:
        ax.axhline(y, color=INK_2, linestyle="--", linewidth=0.8)
        ax.text(0.45, y * 1.08, label, color=INK_2, fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    style(ax, "Object size relative to the image (box = p25–p75, whiskers = p5–p95)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def describe(dataset: Path = DEFAULT_DATASET) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    names = yaml.safe_load((dataset / "data.yaml").read_text())["names"]
    images, boxes = load(dataset, names)

    split_sizes = images.groupby("split").agg(
        images=("file", "size"), instances=("n_objects", "sum"),
        empty_images=("n_objects", lambda s: int((s == 0).sum())),
        multi_class_images=("n_classes", lambda s: int((s > 1).sum())),
        augmented_copies=("augmented", "sum"),
    ).reindex(SPLITS)
    split_sizes["share_of_images"] = (split_sizes.images / split_sizes.images.sum()).round(3)

    per_class = boxes.pivot_table(index="class", columns="split", values="file",
                                  aggfunc="size", fill_value=0).reindex(index=names, columns=SPLITS, fill_value=0)
    per_class["total"] = per_class.sum(axis=1)
    per_class["share"] = (per_class.total / per_class.total.sum()).round(3)

    images_per_class = pd.DataFrame({
        split: [images[(images.split == split) & images.classes.str.contains(rf"\b{n}\b")].shape[0] for n in names]
        for split in SPLITS}, index=names)

    size = boxes.groupby("class").rel_area.describe(percentiles=[0.25, 0.5, 0.75]).reindex(names)
    size = (size[["count", "25%", "50%", "75%"]] * [1, 100, 100, 100]).round(2)
    size.columns = ["boxes", "p25_area_pct", "median_area_pct", "p75_area_pct"]
    bins = boxes.pivot_table(index="class", columns="size_bin", values="file",
                             aggfunc="size", fill_value=0, observed=False).reindex(names)
    size = size.join((bins.div(bins.sum(axis=1), axis=0) * 100).round(1).add_prefix("pct_"))
    size["tiny_boxes_under_8px"] = boxes[boxes.rel_area < TINY_MAX].groupby("class").size() \
        .reindex(names, fill_value=0)

    objects_per_image = images.groupby("split").n_objects.describe()[["mean", "50%", "max"]].reindex(SPLITS).round(2)

    mixing = pd.DataFrame({
        "multi_class": images.groupby("split").n_classes.apply(lambda s: int((s > 1).sum())),
        "fresh_and_rotten": images.groupby("split").apply(lambda d: int((d.has_fresh & d.has_rotten).sum()),
                                                          include_groups=False),
    }).reindex(SPLITS)

    families = images.pivot_table(index="family", columns="split", values="file",
                                  aggfunc="size", fill_value=0).reindex(columns=SPLITS, fill_value=0)

    test = leakage(images)
    leak_summary = {
        "test_images": int(len(test)),
        "name_match": int(test.name_match.sum()),
        "name_match_same_classes": int(test.name_match_same_classes.sum()),
        "hash_match": int(test.hash_match.sum()),
        "hash_match_without_name_match": int((test.hash_match & ~test.name_match).sum()),
        "leaked_total": int(test.leaked.sum()),
        "dedup_test_images": int((~test.leaked).sum()),
        "hash_max_dist_bits": HASH_MAX_DIST,
    }
    leak_by_family = test.groupby("family").agg(images=("file", "size"), leaked=("leaked", "sum"))

    excluded = train_val_exclusions(images)
    dropped = images.loc[excluded[excluded].index, ["split", "file"]]
    kept = images.drop(dropped.index)
    leak_summary["excluded_train"] = int((dropped.split == "train").sum())
    leak_summary["excluded_valid"] = int((dropped.split == "valid").sum())
    kept_boxes = boxes.merge(kept[["split", "file"]], on=["split", "file"])
    clean_split = kept.groupby("split").agg(images=("file", "size"), instances=("n_objects", "sum")).reindex(SPLITS)
    clean_per_class = kept_boxes.pivot_table(index="class", columns="split", values="file", aggfunc="size",
                                             fill_value=0).reindex(index=names, columns=SPLITS, fill_value=0)

    split_sizes.to_csv(OUT / "split_sizes.csv")
    per_class.to_csv(OUT / "instances_per_class.csv")
    images_per_class.to_csv(OUT / "images_per_class.csv")
    size.to_csv(OUT / "box_size_per_class.csv")
    objects_per_image.to_csv(OUT / "objects_per_image.csv")
    mixing.to_csv(OUT / "class_mixing.csv")
    families.to_csv(OUT / "source_families.csv")
    leak_by_family.to_csv(OUT / "leakage_by_family.csv")
    test.drop(columns="dhash").to_csv(OUT / "test_leakage_detail.csv", index=False)
    dropped.to_csv(OUT / "excluded_train_val.csv", index=False)
    clean_split.to_csv(OUT / "clean_split_sizes.csv")
    clean_per_class.to_csv(OUT / "clean_instances_per_class.csv")
    (OUT / "leakage_summary.json").write_text(json.dumps(leak_summary, indent=2))

    plot_instances(per_class[SPLITS], OUT / "instances_per_class.png")
    plot_box_sizes(boxes, names, OUT / "box_size_per_class.png")

    for title, table in [("Split sizes", split_sizes), ("Instances per class", per_class),
                         ("Images containing each class", images_per_class),
                         ("Box size per class (% of image area)", size),
                         ("Objects per image", objects_per_image), ("Class mixing", mixing),
                         ("Source families (by filename)", families),
                         ("Leakage by family (test)", leak_by_family),
                         ("After dropping test copies from train/valid", clean_split),
                         ("Instances per class after cleaning", clean_per_class)]:
        print(f"\n== {title} ==\n{table.to_string()}")
    print(f"\n== Leakage summary ==\n{json.dumps(leak_summary, indent=2)}")
    return leak_summary


if __name__ == "__main__":
    describe(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET)
