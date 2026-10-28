"""
Describe the pinned dataset from the files on disk, before training on it.

Writes:
    outputs/dataset_stats.json   every number below, machine-readable
    outputs/dataset_stats.md     the same, as tables
    outputs/box_area_hist.png    box-area distribution per class

Covers image counts per split, instances per class per split, image-level
positive/negative balance (positive = at least one Box_broken or Open_package
box), box-area distribution per class, images with zero labels, image
resolution distribution, a perceptual-hash near-duplicate check within and
across splits (Hamming distance <= 4 on a 64-bit pHash), and what Roboflow's
version metadata says about preprocessing, augmentation and the split.

Run:

    uv run python scripts/describe_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import (  # noqa: E402
    CLASS_NAMES,
    DATASET_ID,
    NEGATIVE_CLASSES,
    POSITIVE_CLASSES,
    SPLITS,
    get_dataset,
    get_version_metadata,
)

SEED = 42
OUT_DIR = ROOT / "outputs"
HASH_BITS = 64
DUP_THRESHOLD = 4
EXAMPLE_PAIRS = 3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash(gray: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash: 32x32 DCT, top-left 8x8, threshold at median."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    return (dct > np.median(dct)).astype(np.uint8)


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:
            xs, ys = vals[0::2], vals[1::2]
            if not xs:
                continue
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((cls, cx, cy, w, h))
    return boxes


def scan(location: Path, names: list[str]) -> dict:
    per_split: dict[str, dict] = {}
    areas: dict[str, list[float]] = defaultdict(list)
    hashes: list[np.ndarray] = []
    hash_index: list[tuple[str, str]] = []
    resolutions: Counter = Counter()
    zero_label_files: dict[str, list[str]] = defaultdict(list)
    pos_idx = {names.index(c) for c in POSITIVE_CLASSES}
    neg_idx = {names.index(c) for c in NEGATIVE_CLASSES}

    for split in SPLITS:
        img_dir = location / split / "images"
        lbl_dir = location / split / "labels"
        info = {
            "images": 0,
            "instances": 0,
            "instances_per_class": {n: 0 for n in names},
            "images_positive": 0,
            "images_negative": 0,
            "images_zero_labels": 0,
            "images_positive_only": 0,
            "images_mixed": 0,
            "boxes_per_image": [],
        }
        if not img_dir.is_dir():
            per_split[split] = info
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            resolutions[f"{w}x{h}"] += 1
            hashes.append(phash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            hash_index.append((split, img_path.name))

            boxes = read_labels(lbl_dir / (img_path.stem + ".txt"))
            info["images"] += 1
            info["instances"] += len(boxes)
            info["boxes_per_image"].append(len(boxes))
            classes_here = set()
            for cls, _, _, bw, bh in boxes:
                if 0 <= cls < len(names):
                    info["instances_per_class"][names[cls]] += 1
                    areas[names[cls]].append(max(bw, 0) * max(bh, 0))
                classes_here.add(cls)
            if not boxes:
                info["images_zero_labels"] += 1
                zero_label_files[split].append(img_path.name)
            elif classes_here & pos_idx:
                info["images_positive"] += 1
                if classes_here & neg_idx:
                    info["images_mixed"] += 1
                else:
                    info["images_positive_only"] += 1
            else:
                info["images_negative"] += 1
        bpi = info.pop("boxes_per_image")
        info["boxes_per_image"] = {
            "mean": round(float(np.mean(bpi)), 3) if bpi else 0,
            "max": int(max(bpi)) if bpi else 0,
        }
        per_split[split] = info

    return {
        "per_split": per_split,
        "areas": areas,
        "hashes": np.array(hashes, dtype=np.uint8),
        "hash_index": hash_index,
        "resolutions": dict(resolutions.most_common()),
        "zero_label_files": dict(zero_label_files),
    }


def near_duplicates(hashes: np.ndarray, index: list[tuple[str, str]]) -> dict:
    """All pairs with Hamming distance <= DUP_THRESHOLD, grouped by split pair."""
    n = len(index)
    if n < 2:
        return {"pairs": [], "within": {}, "across": {}, "total": 0}
    a = hashes.astype(np.int32)
    same = a @ a.T + (1 - a) @ (1 - a).T  # matching bits
    dist = HASH_BITS - same
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    keep = d <= DUP_THRESHOLD
    ii, jj = iu[0][keep], iu[1][keep]
    pairs = sorted(
        (
            {
                "a_split": index[i][0], "a": index[i][1],
                "b_split": index[j][0], "b": index[j][1],
                "distance": int(dist[i, j]),
            }
            for i, j in zip(ii, jj)
        ),
        key=lambda p: (p["distance"], p["a_split"], p["a"], p["b"]),
    )
    within: Counter = Counter()
    across: Counter = Counter()
    for p in pairs:
        if p["a_split"] == p["b_split"]:
            within[p["a_split"]] += 1
        else:
            across["/".join(sorted((p["a_split"], p["b_split"])))] += 1
    exact = sum(1 for p in pairs if p["distance"] == 0)
    return {
        "threshold": DUP_THRESHOLD,
        "total": len(pairs),
        "exact": exact,
        "within": {s: within.get(s, 0) for s in SPLITS},
        "across": dict(sorted(across.items())),
        "within_examples": [p for p in pairs if p["a_split"] == p["b_split"]][:EXAMPLE_PAIRS],
        "across_examples": [p for p in pairs if p["a_split"] != p["b_split"]][:EXAMPLE_PAIRS],
        "pairs": pairs,
    }


def area_stats(areas: dict[str, list[float]]) -> dict:
    out = {}
    for name, vals in areas.items():
        arr = np.array(vals)
        out[name] = {
            "count": int(arr.size),
            "median": round(float(np.median(arr)), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "min": round(float(arr.min()), 5),
            "max": round(float(arr.max()), 4),
        }
    return out


def plot_areas(areas: dict[str, list[float]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    bins = np.linspace(0, 1, 41)
    for ax, name in zip(axes.flat, CLASS_NAMES):
        vals = areas.get(name, [])
        ax.hist(vals, bins=bins, color="#4C78A8", edgecolor="white")
        ax.set_title(f"{name} (n={len(vals)})")
        ax.set_xlabel("box area / image area")
        ax.set_ylabel("boxes")
        if vals:
            ax.axvline(float(np.median(vals)), color="#E45756", linestyle="--", label="median")
            ax.legend(loc="upper right")
    fig.suptitle(f"Box area distribution per class ({DATASET_ID})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def split_method(meta: dict, per_split: dict) -> str:
    """What the metadata lets us say about how the split was made."""
    sizes = meta.get("splits") or {}
    total = sum(sizes.values()) or 1
    ratios = {k: round(v / total, 3) for k, v in sizes.items()}
    note = (
        "The Roboflow version metadata has no field recording how images were "
        "assigned to splits. "
    )
    if ratios.get("train") == 0.7 and ratios.get("valid") == 0.2 and ratios.get("test") == 0.1:
        note += (
            "The split is exactly 70/20/10, which is Roboflow's default ratio when "
            "it splits for you, so a Roboflow random split is likely; an "
            "author-supplied split would rarely land on exactly those numbers. "
            "This is an inference, not a stated fact."
        )
    else:
        note += f"The split ratio is {ratios}, which is not Roboflow's default 70/20/10."
    return note


def render_md(stats: dict) -> str:
    L = [f"# Dataset description: `{DATASET_ID}`", ""]
    L += [
        "Computed from the YOLOv11 export on disk by `scripts/describe_dataset.py`.",
        f"Positive (compromised) classes: {', '.join(POSITIVE_CLASSES)}. "
        f"Negative (intact) classes: {', '.join(NEGATIVE_CLASSES)}.",
        "",
        "## Roboflow version metadata",
        "",
        f"- Version name: {stats['metadata']['name']}",
        f"- Preprocessing: {stats['metadata']['preprocessing_text']}",
        f"- Augmentation: {stats['metadata']['augmentation_text']}",
        f"- Split: {stats['metadata']['split_method']}",
        f"- Licence field in data.yaml: {stats['metadata']['license']}",
        "",
        "## Images and instances per split",
        "",
        "| Split | Images | Instances | " + " | ".join(CLASS_NAMES) + " | Boxes/img mean | Boxes/img max |",
        "|---|---|---|" + "---|" * len(CLASS_NAMES) + "---|---|",
    ]
    for split in SPLITS:
        s = stats["per_split"][split]
        L.append(
            f"| {split} | {s['images']} | {s['instances']} | "
            + " | ".join(str(s["instances_per_class"][c]) for c in CLASS_NAMES)
            + f" | {s['boxes_per_image']['mean']} | {s['boxes_per_image']['max']} |"
        )
    L += [
        "",
        "## Image-level balance (the one that matters)",
        "",
        "| Split | Positive (any Box_broken/Open_package) | of which mixed with negatives | Negative (only Box/Package) | Zero labels | Positive share |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        s = stats["per_split"][split]
        labelled = s["images_positive"] + s["images_negative"]
        share = f"{100 * s['images_positive'] / labelled:.1f}%" if labelled else "n/a"
        L.append(
            f"| {split} | {s['images_positive']} | {s['images_mixed']} | {s['images_negative']} | "
            f"{s['images_zero_labels']} | {share} |"
        )
    L += ["", "## Box area as a fraction of image area", "",
          "| Class | Boxes | Median | p10 | p90 | Min | Max |", "|---|---|---|---|---|---|---|"]
    for name in CLASS_NAMES:
        a = stats["box_area"].get(name)
        if a:
            L.append(f"| {name} | {a['count']} | {100*a['median']:.2f}% | {100*a['p10']:.2f}% | "
                     f"{100*a['p90']:.2f}% | {100*a['min']:.3f}% | {100*a['max']:.2f}% |")
    L += ["", "Histogram: `outputs/box_area_hist.png`", ""]
    L += ["## Images with zero labels", ""]
    zl = stats["zero_label_files"]
    if any(zl.values()):
        for split, files in zl.items():
            L.append(f"- {split}: {len(files)}: " + ", ".join(files[:10]) + (" ..." if len(files) > 10 else ""))
    else:
        L.append("- None in any split.")
    L += ["", "## Image resolution", "", "| Resolution | Images |", "|---|---|"]
    for res, n in stats["resolutions"].items():
        L.append(f"| {res} | {n} |")
    d = stats["duplicates"]
    L += [
        "",
        f"## Near-duplicates (pHash, Hamming distance <= {d['threshold']})",
        "",
        f"- Total pairs: {d['total']} (exact hash matches: {d['exact']})",
        "- Within split: " + ", ".join(f"{s} {n}" for s, n in d["within"].items()),
        "- Across splits: " + (", ".join(f"{k} {n}" for k, n in d["across"].items()) or "none"),
        "",
    ]
    for title, key in (("Within-split examples", "within_examples"), ("Across-split examples", "across_examples")):
        L.append(f"### {title}")
        L.append("")
        if d[key]:
            for p in d[key]:
                L.append(f"- d={p['distance']}: `{p['a_split']}/{p['a']}` and `{p['b_split']}/{p['b']}`")
        else:
            L.append("- none")
        L.append("")
    return "\n".join(L)


def main() -> int:
    np.random.seed(SEED)
    data_yaml = get_dataset()
    location = data_yaml.parent
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = [str(n) for n in cfg["names"]]
    if names != CLASS_NAMES:
        raise SystemExit(f"data.yaml names {names} differ from src.data.CLASS_NAMES {CLASS_NAMES}")

    meta = get_version_metadata()
    pre = meta.get("preprocessing") or {}
    aug = meta.get("augmentation") or {}
    pre_text = "; ".join(f"{k}={v}" for k, v in pre.items()) or "none recorded"
    aug_text = "; ".join(f"{k}={v}" for k, v in aug.items()) or "none (augmentation field is empty)"

    scanned = scan(location, names)
    dups = near_duplicates(scanned["hashes"], scanned["hash_index"])

    stats = {
        "dataset": DATASET_ID,
        "seed": SEED,
        "positive_classes": list(POSITIVE_CLASSES),
        "negative_classes": list(NEGATIVE_CLASSES),
        "metadata": {
            "name": meta.get("name"),
            "created": meta.get("created"),
            "images": meta.get("images"),
            "splits": meta.get("splits"),
            "preprocessing": pre,
            "augmentation": aug,
            "preprocessing_text": pre_text,
            "augmentation_text": aug_text,
            "split_method": split_method(meta, scanned["per_split"]),
            "license": (cfg.get("roboflow") or {}).get("license"),
        },
        "per_split": scanned["per_split"],
        "box_area": area_stats(scanned["areas"]),
        "zero_label_files": scanned["zero_label_files"],
        "resolutions": scanned["resolutions"],
        "duplicates": {k: v for k, v in dups.items() if k != "pairs"},
        "duplicate_pairs": dups["pairs"],
    }

    OUT_DIR.mkdir(exist_ok=True)
    plot_areas(scanned["areas"], OUT_DIR / "box_area_hist.png")
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (OUT_DIR / "dataset_stats.md").write_text(render_md(stats), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'dataset_stats.json'}, dataset_stats.md, box_area_hist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
