import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WORKSPACE = "cuizh"
PROJECT = "water-rescue-dataset"
VERSION = 1
DATASET_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
RESULTS_DIR = ROOT / "results"
SPLITS = ["train", "valid", "test"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SMALL_AREA = 0.01


def download_dataset() -> Path:
    if (DATASET_DIR / "data.yaml").is_file():
        return DATASET_DIR
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download("yolov11", location=str(DATASET_DIR))
    return DATASET_DIR


def class_names(dataset_dir: Path) -> list[str]:
    config = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    names = config["names"]
    if isinstance(names, dict):
        return [names[key] for key in sorted(names)]
    return list(names)


def write_local_yaml(dataset_dir: Path) -> Path:
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(class_names(dataset_dir))},
    }
    target = dataset_dir / "local.yaml"
    target.write_text(yaml.safe_dump(config))
    return target


def read_boxes(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        values = np.array(parts[1:], dtype=float)
        if len(values) == 4:
            cx, cy, width, height = values
        else:
            xs, ys = values[0::2], values[1::2]
            width, height = xs.max() - xs.min(), ys.max() - ys.min()
            cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
        boxes.append((int(float(parts[0])), float(cx), float(cy), float(width), float(height)))
    return boxes


def base_name(stem: str) -> str:
    return re.split(r"\.rf\.", stem)[0]


def source_of(name: str) -> str:
    if name.startswith("IMG_"):
        return "phone"
    if name.startswith("DJI_"):
        return "drone"
    if name.startswith("WIN_"):
        return "webcam"
    return "web_or_video"


def collect(dataset_dir: Path):
    names = class_names(dataset_dir)
    image_rows, box_rows = [], []
    for split in SPLITS:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = label_dir / f"{image.stem}.txt"
            boxes = read_boxes(label) if label.is_file() else []
            source = source_of(image.name)
            image_rows.append(
                {
                    "split": split,
                    "image": image.name,
                    "base": base_name(image.stem),
                    "source": source,
                    "boxes": len(boxes),
                }
            )
            for class_id, _, _, width, height in boxes:
                box_rows.append(
                    {
                        "split": split,
                        "image": image.name,
                        "source": source,
                        "class": names[class_id],
                        "w": width,
                        "h": height,
                        "area": width * height,
                    }
                )
    return names, pd.DataFrame(image_rows), pd.DataFrame(box_rows)


def summarize(names, images, boxes) -> dict:
    instances = (
        boxes.pivot_table(index="class", columns="split", values="area", aggfunc="count", fill_value=0)
        .reindex(index=names, columns=SPLITS, fill_value=0)
        .astype(int)
    )
    instances["total"] = instances.sum(axis=1)

    images_per_class = (
        boxes.drop_duplicates(["split", "image", "class"])
        .pivot_table(index="class", columns="split", values="image", aggfunc="count", fill_value=0)
        .reindex(index=names, columns=SPLITS, fill_value=0)
        .astype(int)
    )

    split_sizes = images.groupby("split").size().reindex(SPLITS, fill_value=0)
    empty_images = images[images["boxes"] == 0].groupby("split").size().reindex(SPLITS, fill_value=0)

    size = boxes.groupby("class").agg(
        median_width=("w", "median"),
        median_height=("h", "median"),
        median_area=("area", "median"),
        p10_area=("area", lambda s: s.quantile(0.10)),
        p90_area=("area", lambda s: s.quantile(0.90)),
        small_share=("area", lambda s: (s < SMALL_AREA).mean()),
    ).reindex(names)

    sources_all = (
        boxes.pivot_table(index="class", columns="source", values="area", aggfunc="count", fill_value=0)
        .reindex(index=names, fill_value=0)
        .astype(int)
    )
    sources_test = (
        boxes[boxes["split"] == "test"]
        .pivot_table(index="class", columns="source", values="area", aggfunc="count", fill_value=0)
        .reindex(index=names, fill_value=0)
        .astype(int)
    )

    bases = {split: set(images[images["split"] == split]["base"]) for split in SPLITS}
    overlap = {
        "train_valid": len(bases["train"] & bases["valid"]),
        "train_test": len(bases["train"] & bases["test"]),
        "valid_test": len(bases["valid"] & bases["test"]),
    }

    return {
        "class_names": names,
        "split_images": split_sizes.to_dict(),
        "split_empty_images": empty_images.to_dict(),
        "instances_per_class": instances.to_dict(orient="index"),
        "images_per_class": images_per_class.to_dict(orient="index"),
        "object_size_relative_to_image": size.round(4).to_dict(orient="index"),
        "instances_by_source_all_splits": sources_all.to_dict(orient="index"),
        "instances_by_source_test": sources_test.to_dict(orient="index"),
        "base_name_overlap_between_splits": overlap,
        "small_object_area_threshold": SMALL_AREA,
    }


def show(summary: dict) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nImages per split")
    print(pd.Series(summary["split_images"]).to_string())
    print("\nImages with no labelled object (background)")
    print(pd.Series(summary["split_empty_images"]).to_string())
    print("\nInstances per class and split")
    print(pd.DataFrame(summary["instances_per_class"]).T.to_string())
    print("\nImages containing each class")
    print(pd.DataFrame(summary["images_per_class"]).T.to_string())
    print("\nObject size as a fraction of image area / width / height")
    print(pd.DataFrame(summary["object_size_relative_to_image"]).T.to_string())
    print("\nInstances by image source, all splits")
    print(pd.DataFrame(summary["instances_by_source_all_splits"]).T.to_string())
    print("\nInstances by image source, test split only")
    print(pd.DataFrame(summary["instances_by_source_test"]).T.to_string())
    print("\nShared base image names between splits")
    print(pd.Series(summary["base_name_overlap_between_splits"]).to_string())


def describe_dataset(dataset_dir: Path) -> dict:
    names, images, boxes = collect(dataset_dir)
    summary = summarize(names, images, boxes)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(summary, indent=2))
    show(summary)
    return summary
