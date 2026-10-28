
from pathlib import Path
 
import pandas as pd
import yaml
from roboflow import Roboflow
 
from api_key import load_api_key
 
WORKSPACE = "project-irggm"
PROJECT = "-w4yds"
VERSION = 1
DATA_DIR = Path("data") / f"fruit-rot-v{VERSION}"
OUT_DIR = Path("results")
SPLITS = ["train", "valid", "test"]
IMG_SIZE = 640  # this version was resized (stretched) to 640x640
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def download() -> None:
    """Download the dataset once; skip if it is already on disk."""
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset already in {DATA_DIR}, skipping download.")
        return
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download("yolov11", location=str(DATA_DIR))
 
 
def read_boxes(label_file: Path) -> list[tuple[int, float, float]]:
    """Return (class_id, width, height) per object, normalised to 0-1."""
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls = int(float(values[0]))
        nums = [float(v) for v in values[1:]]
        if len(nums) == 4:  # normal YOLO box: x_center y_center w h
            w, h = nums[2], nums[3]
        else:  # polygon: take its bounding box
            xs, ys = nums[0::2], nums[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((cls, w, h))
    return boxes
 
 
def collect(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    objects, images = [], []
    for split in SPLITS:
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            boxes = read_boxes(lbl_dir / f"{img.stem}.txt")
            classes = [names[c] for c, _, _ in boxes]
            images.append({
                "split": split,
                "image": img.name,
                "n_objects": len(boxes),
                "has_rotten": any("rotten" in c for c in classes),
            })
            for (c, w, h), name in zip(boxes, classes):
                objects.append({
                    "split": split,
                    "class": name,
                    "rel_area": w * h,
                    "area_px": w * h * IMG_SIZE * IMG_SIZE,
                })
    return pd.DataFrame(objects), pd.DataFrame(images)
 
 
def size_bucket(area_px: float) -> str:
    # COCO convention: small < 32x32 px, medium < 96x96 px, large otherwise
    if area_px < 32 ** 2:
        return "small"
    if area_px < 96 ** 2:
        return "medium"
    return "large"
 
 
def main() -> None:
    download()
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    objects, images = collect(names)
    OUT_DIR.mkdir(exist_ok=True)
 
    print("\n=== 1. Split sizes ===")
    splits = images.groupby("split").agg(
        images=("image", "count"),
        empty_images=("n_objects", lambda s: int((s == 0).sum())),
        objects=("n_objects", "sum"),
        images_with_rotten=("has_rotten", "sum"),
    ).reindex(SPLITS)
    splits["share_images_with_rotten"] = (splits["images_with_rotten"] / splits["images"]).round(3)
    print(splits.to_string())
    splits.to_csv(OUT_DIR / "split_sizes.csv")
 
    print("\n=== 2. Instances per class and split ===")
    per_class = pd.crosstab(objects["class"], objects["split"]).reindex(columns=SPLITS, fill_value=0)
    per_class["total"] = per_class.sum(axis=1)
    print(per_class.to_string())
    per_class.to_csv(OUT_DIR / "instances_per_class.csv")
 
    print("\n=== 3. Objects per image ===")
    print(images.groupby("split")["n_objects"].describe().reindex(SPLITS).round(2).to_string())
 
    print("\n=== 4. Object size relative to the image (share of image area) ===")
    size = objects.groupby("class")["rel_area"].describe(percentiles=[0.1, 0.5, 0.9]).round(3)
    print(size.to_string())
    size.to_csv(OUT_DIR / "object_size_per_class.csv")
 
    objects["size_bucket"] = objects["area_px"].apply(size_bucket)
    buckets = pd.crosstab(objects["class"], objects["size_bucket"]).reindex(
        columns=["small", "medium", "large"], fill_value=0
    )
    print("\nSize buckets (COCO: small < 32x32 px, medium < 96x96 px, at 640x640):")
    print(buckets.to_string())
    buckets.to_csv(OUT_DIR / "object_size_buckets.csv")
 
    print(f"\nCSV files saved in {OUT_DIR}/")
 
 
if __name__ == "__main__":
    main()
