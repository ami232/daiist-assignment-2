"""Download the selected public version and audit it before any training.

Run: uv run python audit_dataset.py
No images, annotations or supplied splits are changed by this script.
"""

from __future__ import annotations

import contextlib
import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "dataset_config.json"
DATA = ROOT / "data" / "car-damage-v1"
OUTPUT = ROOT / "audit" / "car-damage-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_dataset() -> Path:
    """Use the original public project directly; never fork the dataset."""
    from api_key import load_api_key

    config = json.loads(CONFIG.read_text())
    if (DATA / "data.yaml").is_file():
        print("Using the existing local dataset export.", flush=True)
        return DATA
    from roboflow import Roboflow

    key = load_api_key()
    print("Requesting the selected public Roboflow version...", flush=True)
    # SDK progress/debug messages may contain signed links. Keep them out of logs.
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            project = Roboflow(api_key=key).workspace(config["workspace"]).project(
                config["project"]
            )
            project.version(config["version"]).download(
                config["export_format"], location=str(DATA)
            )
    except Exception as error:
        # Do not expose exception messages containing a signed export URL or key.
        raise RuntimeError(
            f"Roboflow download failed ({type(error).__name__}). "
            "Check dataset access and the selected export format. Credentials "
            "and signed links have not been printed."
        ) from None
    if not (DATA / "data.yaml").is_file():
        raise RuntimeError("The download did not produce the expected data.yaml.")
    print("Dataset downloaded. The supplied split and labels are unchanged.", flush=True)
    return DATA


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=1)
def phash_basis():
    import numpy as np

    basis = np.cos(np.pi * np.arange(8)[:, None] * (np.arange(32)[None, :] + .5) / 32)
    basis *= np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    return basis


def perceptual_hash(image) -> int:
    """64-bit pHash; matches are candidates, not proof of duplicate identity."""
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    basis = phash_basis()
    low = (basis @ pixels.astype(float) @ basis.T).ravel()
    bits = low > np.median(low[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class HashTree:
    """BK tree for Hamming-distance candidate retrieval."""

    def __init__(self):
        self.root = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def find(self, value: int, radius: int):
        if self.root is None:
            return []
        matches = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                matches.extend((distance, index) for index in node[1])
            stack.extend(
                child for edge, child in node[2].items()
                if distance - radius <= edge <= distance + radius
            )
        return matches


def audit_dataset(dataset: Path) -> dict:
    import numpy as np
    import yaml
    from PIL import Image

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = yaml.safe_load((dataset / "data.yaml").read_text())
    names = metadata["names"]
    names = [names[key] for key in sorted(names, key=int)] if isinstance(names, dict) else names
    config = json.loads(CONFIG.read_text())
    expected = set(config["classes"])
    optional = set(config.get("allow_unused_class_names", []))
    if expected - set(names) or set(names) - expected - optional:
        raise ValueError("Downloaded class names differ from the selected public project.")
    manifest, instances, invalid, missing_labels = [], [], [], []
    class_stats = {}
    geometry = defaultdict(list)
    exact = defaultdict(list)
    pixel = defaultdict(list)
    hashes = []
    orientations = []
    split_totals = {}
    for split in ("train", "valid", "test"):
        directory = dataset / split / "images"
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing exported split directory: {split}/images")
        files = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        split_totals[split] = len(files)
        counts, image_counts = Counter(), Counter()
        print(f"Auditing {split}: {len(files)} images...", flush=True)
        for path in files:
            relative = str(path.relative_to(dataset))
            label = dataset / split / "labels" / (path.stem + ".txt")
            labels = []
            if not label.is_file():
                missing_labels.append(relative)
            else:
                for line_no, line in enumerate(label.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        values = [float(value) for value in line.split()]
                        if len(values) != 5:
                            raise ValueError("Expected five YOLO detection fields")
                        raw_id, x, y, width, height = values
                        class_id = int(raw_id)
                        if not np.isfinite(values).all() or class_id != raw_id or not 0 <= class_id < len(names):
                            raise ValueError("Invalid class or nonfinite coordinates")
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                            raise ValueError("Coordinates outside valid normalized ranges")
                        if x - width / 2 < -0.002 or y - height / 2 < -0.002 or x + width / 2 > 1.002 or y + height / 2 > 1.002:
                            raise ValueError("Bounding box extends outside the image")
                        name = names[class_id]
                        labels.append(class_id)
                        counts[name] += 1
                        geometry[name].append(width * height)
                        instances.append(dict(image=relative, split=split, class_id=class_id,
                                              class_name=name, x=x, y=y, width=width,
                                              height=height, relative_area=width * height))
                    except (ValueError, OverflowError) as error:
                        invalid.append(dict(image=relative, line=line_no, reason=str(error)))
            image_counts.update(names[class_id] for class_id in set(labels))
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    width_px, height_px = image.size
                    pixel_hash = hashlib.sha256(
                        f"{width_px}x{height_px}:".encode() + image.tobytes()
                    ).hexdigest()
                    all_views = [image, image.transpose(Image.Transpose.ROTATE_90),
                                 image.transpose(Image.Transpose.ROTATE_180),
                                 image.transpose(Image.Transpose.ROTATE_270)]
                    mirrored = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    all_views += [mirrored, mirrored.transpose(Image.Transpose.ROTATE_90),
                                  mirrored.transpose(Image.Transpose.ROTATE_180),
                                  mirrored.transpose(Image.Transpose.ROTATE_270)]
                    views = [perceptual_hash(view) for view in all_views]
            except Exception as error:
                invalid.append(dict(image=relative, line=0, reason=f"Unreadable image ({type(error).__name__})"))
                continue
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            record = dict(image=relative, split=split, filename=path.name,
                          width_px=width_px, height_px=height_px, annotation_count=len(labels),
                          classes="|".join(names[index] for index in sorted(set(labels))),
                          sha256=file_hash, pixel_sha256=pixel_hash, phash=f"{views[0]:016x}")
            index = len(manifest)
            manifest.append(record)
            hashes.append(views[0])
            orientations.append(views)
            exact[file_hash].append(index)
            pixel[pixel_hash].append(index)
        class_stats[split] = {
            name: {"instances": counts[name], "images_containing_class": image_counts[name]}
            for name in names
        }

    def duplicate_groups(mapping):
        return [dict(images=[manifest[index]["image"] for index in indexes],
                     classes=[manifest[index]["classes"] for index in indexes])
                for indexes in mapping.values()
                if len({manifest[index]["split"] for index in indexes}) > 1]

    exact_cross = duplicate_groups(exact)
    pixel_cross = duplicate_groups(pixel)
    candidates = []
    for source_split, target_split in (("train", "valid"), ("train", "test"), ("valid", "test")):
        tree = HashTree()
        for index, item in enumerate(manifest):
            if item["split"] == source_split:
                tree.add(hashes[index], index)
        for index, item in enumerate(manifest):
            if item["split"] != target_split:
                continue
            best = {}
            for view_no, value in enumerate(orientations[index]):
                for distance, other in tree.find(value, radius=4):
                    if other not in best or distance < best[other][0]:
                        best[other] = (distance, view_no)
            for other, (distance, view_no) in sorted(best.items(), key=lambda pair: pair[1])[:5]:
                candidates.append(dict(image_a=manifest[other]["image"], image_b=item["image"],
                                       split_a=source_split, split_b=target_split,
                                       class_a=manifest[other]["classes"], class_b=item["classes"],
                                       phash_distance=distance, transform_b=view_no,
                                       same_pixels=manifest[other]["pixel_sha256"] == item["pixel_sha256"]))
    sizes = {
        name: {"count": len(values),
               "area_fraction_p10": float(np.percentile(values, 10)),
               "area_fraction_median": float(np.median(values)),
               "area_fraction_p90": float(np.percentile(values, 90)),
               "fraction_under_1pct_area": float(np.mean(np.array(values) < 0.01))}
        for name, values in geometry.items() if values
    }
    summary = dict(dataset=str(dataset.relative_to(ROOT)), classes=names,
                   split_images=split_totals, class_statistics=class_stats,
                   object_sizes=sizes, total_annotations=len(instances),
                   missing_label_images=missing_labels, invalid_entries=invalid,
                   empty_label_images=sum(item["annotation_count"] == 0 for item in manifest),
                   exact_cross_split_groups=exact_cross,
                   identical_pixel_cross_split_groups=pixel_cross,
                   near_duplicate_candidate_pairs=len(candidates),
                   near_duplicate_method="64-bit pHash distance <= 4, checking eight image orientations; at most five matches per target image and split pair. Candidates require visual review.",
                   sample_independence_confirmed=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(OUTPUT / "image_manifest.csv", manifest, list(manifest[0]) if manifest else ["image"])
    write_csv(OUTPUT / "instances.csv", instances, ["image", "split", "class_id", "class_name", "x", "y", "width", "height", "relative_area"])
    write_csv(OUTPUT / "near_duplicate_candidates.csv", candidates,
              ["image_a", "image_b", "split_a", "split_b", "class_a", "class_b", "phash_distance", "transform_b", "same_pixels"])
    make_plots(summary, instances)
    make_contact_sheet(dataset, candidates, manifest)
    config_path = CONFIG
    config = json.loads(config_path.read_text())
    config["download_verified"] = True
    config["label_audit_complete"] = not invalid and not missing_labels
    config["split_independence_review_complete"] = False
    config["measured_export_images"] = split_totals
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("split_images", "class_statistics", "total_annotations", "near_duplicate_candidate_pairs")}, indent=2))
    print(f"Cross-split duplicate groups: {len(exact_cross)} exact-file; {len(pixel_cross)} identical-pixel.")
    print(f"Invalid entries: {len(invalid)}; missing labels: {len(missing_labels)}.")
    print(f"Audit outputs saved to {OUTPUT.relative_to(ROOT) if OUTPUT.is_relative_to(ROOT) else OUTPUT}. No training has run.")
    return summary


def make_plots(summary: dict, instances: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(3)
    bar_width = .8 / len(summary["classes"])
    for offset, name in enumerate(summary["classes"]):
        values = [summary["class_statistics"][split][name]["instances"] for split in ("train", "valid", "test")]
        bars = axes[0].bar(positions + (offset - (len(summary["classes"]) - 1) / 2) * bar_width,
                           values, bar_width, label=name)
        axes[0].bar_label(bars, fontsize=8)
        areas = [item["relative_area"] * 100 for item in instances if item["class_name"] == name]
        axes[1].hist(areas, bins=np.linspace(0, 100, 21), alpha=.55, label=name)
    axes[0].set(xticks=positions, xticklabels=["Train", "Validation", "Test"], ylabel="Annotated instances", title="Class support in the supplied splits")
    axes[0].legend()
    axes[1].set(xlabel="Bounding-box area (% of image)", ylabel="Annotated instances", title="Object size relative to image")
    axes[1].legend()
    fig.savefig(OUTPUT / "dataset_composition.png", dpi=160)
    plt.close(fig)


def make_contact_sheet(dataset: Path, candidates: list[dict], manifest: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ordered = sorted(candidates, key=lambda item: (not item["same_pixels"], item["phash_distance"], item["split_b"]))
    selected, seen = [], set()
    for pair in ordered:
        if pair["image_b"] not in seen:
            selected.append(pair)
            seen.add(pair["image_b"])
        if len(selected) == 8:
            break
    if selected:
        fig, axes = plt.subplots(len(selected), 2, figsize=(11, len(selected) * 3.3), squeeze=False, layout="constrained")
        for row, pair in enumerate(selected):
            for col, suffix in enumerate(("a", "b")):
                with Image.open(dataset / pair[f"image_{suffix}"]) as image:
                    axes[row, col].imshow(image)
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{pair[f'split_{suffix}']} | {pair[f'class_{suffix}']}\n{Path(pair[f'image_{suffix}']).name[:65]}\npHash distance={pair['phash_distance']}; identical pixels={pair['same_pixels']}", fontsize=8)
        fig.savefig(OUTPUT / "duplicate_review.png", dpi=110)
        plt.close(fig)
    selected = []
    class_names = list(dict.fromkeys(name for item in manifest for name in item["classes"].split("|") if name))
    for name in class_names:
        selected.extend([item for item in manifest if item["split"] == "train" and name in item["classes"].split("|")][:max(1, 12 // len(class_names))])
    selected = selected[:12]
    if selected:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), layout="constrained")
        for axis, item in zip(axes.ravel(), selected):
            with Image.open(dataset / item["image"]) as image:
                axis.imshow(image)
            axis.set_title(f"{item['classes']}\n{item['filename'][:42]}", fontsize=8)
            axis.axis("off")
        for axis in axes.ravel()[len(selected):]:
            axis.axis("off")
        fig.savefig(OUTPUT / "training_examples.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    CONFIG, DATA, OUTPUT = args.config.resolve(), args.data_dir.resolve(), args.output.resolve()
    try:
        audit_dataset(prepare_dataset())
    except Exception as error:
        from api_key import resolve_api_key
        key, _ = resolve_api_key()
        message = str(error)
        if key:
            message = message.replace(key, "[REDACTED]")
        print(f"Dataset audit stopped: {message}", file=sys.stderr)
        sys.exit(1)
