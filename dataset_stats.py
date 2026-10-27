"""
Describe the dataset BEFORE training on it.

    uv run python dataset_stats.py

Downloads the Roboflow version (cached in datasets/) and prints, for REPORT.md:

  1. images per split, and how many distinct source images train really holds
     (the Roboflow export triples train with augmented copies)
  2. instances per class, per split
  3. object size relative to the image, per class
  4. how Defect boxes sit relative to the weld boxes (decides what a "case" is)
  5. near-duplicates and shared source images between train and val/test
     (leakage would make the test numbers look better than they are)

It also saves everything to results/dataset_stats.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from weld_data import (PROJECT, VERSION, WORKSPACE, download_dataset, is_positive_name,
                       list_images, norm_name, read_class_names, read_gt, split_dirs)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NEAR_DUPLICATE_BITS = 6   # of 64: images this similar are treated as the same picture


def source_name(path: Path) -> str:
    """Roboflow names augmented copies <original>.rf.<hash>.jpg: strip the hash."""
    return path.name.split(".rf.")[0]


def dhash(img_gray: np.ndarray) -> int:
    """64-bit difference hash: survives resizing and small brightness changes."""
    small = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_to_all(h: int, others: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(others, np.uint64(h))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def hash_image(path: Path, flip: bool = False) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return dhash(cv2.flip(img, 1) if flip else img)


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def main() -> None:
    dataset_dir = download_dataset()
    names = read_class_names(dataset_dir)
    splits = split_dirs(dataset_dir)
    stats: dict = {"dataset": f"{WORKSPACE}/{PROJECT}/v{VERSION}", "classes": names, "splits": {}}

    print("\n" + "=" * 72)
    print(f"DATASET  {WORKSPACE} / {PROJECT} / v{VERSION}")
    print(f"CLASSES  {names}")
    print("=" * 72)

    # ---- 1 + 2: sizes and instances -----------------------------------------
    sizes_by_class: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
    for split, (images_dir, labels_dir) in splits.items():
        images = list_images(images_dir)
        counts = Counter()
        unlabeled = 0
        images_with_positive = 0
        images_with_good_only = 0
        for img in images:
            gt = read_gt(labels_dir / f"{img.stem}.txt")
            if len(gt) == 0:
                unlabeled += 1
            labels = [names[int(c)] for c in gt[:, 0]]
            counts.update(labels)
            if any(is_positive_name(lbl) for lbl in labels):
                images_with_positive += 1
            elif labels:
                images_with_good_only += 1
            for row in gt:
                sizes_by_class[names[int(row[0])]].append((row[3] - row[1], row[4] - row[2]))
        unique_sources = len({source_name(p) for p in images})
        stats["splits"][split] = {
            "images": len(images),
            "unique_source_images": unique_sources,
            "images_without_labels": unlabeled,
            "images_with_a_defective_weld": images_with_positive,
            "images_with_only_good_welds": images_with_good_only,
            "instances": {n: counts.get(n, 0) for n in names},
        }
        print(f"\n[{split}]  {len(images)} images  ({unique_sources} distinct source images)")
        print(f"    images with no labels at all:       {unlabeled}")
        print(f"    images with >=1 defective weld:     {images_with_positive}")
        print(f"    images with only good welds:        {images_with_good_only}")
        for n in names:
            print(f"    {n:<12} {counts.get(n, 0):>6} instances")

    # ---- 3: object size relative to the image ---------------------------------
    print("\nOBJECT SIZE relative to the image (all splits)")
    print(f"    {'class':<12} {'median w':>9} {'median h':>9} {'median area':>12} {'10th pct area':>14}")
    stats["object_size"] = {}
    for n in names:
        wh = np.array(sizes_by_class[n]).reshape(-1, 2)
        if len(wh) == 0:
            continue
        area = wh[:, 0] * wh[:, 1]
        stats["object_size"][n] = {"median_width": float(np.median(wh[:, 0])),
                                   "median_height": float(np.median(wh[:, 1])),
                                   "median_area_share": float(np.median(area)),
                                   "p10_area_share": pct(area, 10), "p90_area_share": pct(area, 90)}
        print(f"    {n:<12} {np.median(wh[:, 0]):>8.1%} {np.median(wh[:, 1]):>8.1%} "
              f"{np.median(area):>11.2%} {pct(area, 10):>13.2%}")

    # ---- 4: are Defect boxes inside weld boxes? --------------------------------
    inside = total = 0
    weld_names = {n for n in names if norm_name(n) in {"goodweld", "badweld"}}
    defect_names = {n for n in names if norm_name(n) == "defect"}
    for _, (images_dir, labels_dir) in splits.items():
        for img in list_images(images_dir):
            gt = read_gt(labels_dir / f"{img.stem}.txt")
            welds = np.array([r[1:5] for r in gt if names[int(r[0])] in weld_names]).reshape(-1, 4)
            for r in gt:
                if names[int(r[0])] in defect_names:
                    total += 1
                    cx, cy = (r[1] + r[3]) / 2, (r[2] + r[4]) / 2
                    if len(welds) and ((welds[:, 0] <= cx) & (cx <= welds[:, 2])
                                       & (welds[:, 1] <= cy) & (cy <= welds[:, 3])).any():
                        inside += 1
    if total:
        share = inside / total
        stats["defect_boxes_inside_a_weld_box_share"] = share
        print(f"\nDEFECT boxes whose centre lies inside a Good Weld / Bad Weld box: "
              f"{inside}/{total} ({share:.0%})")
        print("    High share = Defect marks a flaw ON a seam, not a separate seam.")
        print("    Then each labelled object is still treated as one case (see REPORT.md).")

    # ---- 5: leakage between splits ---------------------------------------------
    print("\nLEAKAGE CHECK (train vs val/test)")
    train_images = list_images(splits["train"][0])
    train_sources = {source_name(p) for p in train_images}
    train_hashes = np.array([h for h in (hash_image(p) for p in train_images) if h is not None],
                            dtype=np.uint64)
    stats["leakage"] = {}
    for split in ("val", "test"):
        if split not in splits:
            continue
        imgs = list_images(splits[split][0])
        shared_source = sum(1 for p in imgs if source_name(p) in train_sources)
        near = 0
        for p in imgs:
            distances = []
            for flip in (False, True):
                h = hash_image(p, flip)
                if h is not None:
                    distances.append(hamming_to_all(h, train_hashes).min())
            if distances and min(distances) <= NEAR_DUPLICATE_BITS:
                near += 1
        stats["leakage"][split] = {"images": len(imgs), "same_source_name_as_train": shared_source,
                                   "near_duplicate_of_a_train_image": near}
        print(f"    {split:<5} {len(imgs):>4} images:  {shared_source} share a source name with train,"
              f"  {near} look like a train image (hash distance <= {NEAR_DUPLICATE_BITS}/64, flips included)")
    print("    Non-zero numbers mean the test score is optimistic.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "dataset_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
