import json
from pathlib import Path

import cv2
import numpy as np

from dataset import IMAGE_SUFFIXES, RESULTS_DIR, SPLITS, base_name, class_names
from evaluate import load_ground_truth

HASH_SIZE = 32
HASH_BLOCK = 8
DUPLICATE_MAX_DISTANCE = 10
SWEEP_THRESHOLDS = [2, 5, 10]
EXAMPLE_PAIRS = 8
THUMB = 256
DISTANCE_BINS = [(0, 2), (3, 5), (6, 10), (11, 16), (17, 24), (25, 64)]
POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def image_files(dataset_dir: Path, split: str) -> list[Path]:
    image_dir = dataset_dir / split / "images"
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_image(path: Path, flag: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flag)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image


def phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:HASH_BLOCK, :HASH_BLOCK].flatten()[1:]
    bits = np.append(coefficients > np.median(coefficients), False)
    return np.packbits(bits)


def split_hashes(dataset_dir: Path, split: str):
    files = image_files(dataset_dir, split)
    plain = np.zeros((len(files), HASH_BLOCK), dtype=np.uint8)
    mirrored = np.zeros_like(plain)
    for i, path in enumerate(files):
        gray = read_image(path, cv2.IMREAD_GRAYSCALE)
        plain[i] = phash(gray)
        mirrored[i] = phash(cv2.flip(gray, 1))
    return files, plain, mirrored


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(a), len(b)), dtype=np.int32)
    for start in range(0, len(a), 64):
        chunk = a[start : start + 64]
        xor = np.bitwise_xor(chunk[:, None, :], b[None, :, :])
        out[start : start + 64] = POPCOUNT[xor].sum(axis=2)
    return out


def nearest(query_plain: np.ndarray, query_mirrored: np.ndarray, reference: np.ndarray):
    distances = np.minimum(hamming(query_plain, reference), hamming(query_mirrored, reference))
    return distances.min(axis=1), distances.argmin(axis=1)


def histogram(distances: np.ndarray) -> dict:
    return {
        f"{low}-{high}": int(((distances >= low) & (distances <= high)).sum())
        for low, high in DISTANCE_BINS
    }


def write_examples(files: dict, distances: np.ndarray, matches: np.ndarray) -> bool:
    candidates = [i for i in np.where(distances <= DUPLICATE_MAX_DISTANCE)[0]]
    candidates.sort(key=lambda i: -distances[i])
    rows = []
    for i in candidates[:EXAMPLE_PAIRS]:
        left = cv2.resize(read_image(files["test"][i], cv2.IMREAD_COLOR), (THUMB, THUMB))
        right = cv2.resize(read_image(files["train"][matches[i]], cv2.IMREAD_COLOR), (THUMB, THUMB))
        row = cv2.hconcat([left, right])
        cv2.putText(
            row,
            f"test vs train, distance {int(distances[i])}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        rows.append(row)
    if not rows:
        return False
    sheet = cv2.vconcat(rows)
    ok, encoded = cv2.imencode(".jpg", sheet)
    if ok:
        encoded.tofile(str(RESULTS_DIR / "duplicate_examples.jpg"))
    return ok


def find_duplicates(dataset_dir: Path) -> dict[str, int]:
    names = class_names(dataset_dir)
    files, plain, mirrored = {}, {}, {}
    for split in SPLITS:
        files[split], plain[split], mirrored[split] = split_hashes(dataset_dir, split)

    test_train, test_train_match = nearest(plain["test"], mirrored["test"], plain["train"])
    test_valid, _ = nearest(plain["test"], mirrored["test"], plain["valid"])
    valid_train, _ = nearest(plain["valid"], mirrored["valid"], plain["train"])

    combined = np.minimum(test_train, test_valid)
    distances = {str(files["test"][i]): int(combined[i]) for i in range(len(files["test"]))}
    flagged_mask = combined <= DUPLICATE_MAX_DISTANCE

    other_names = {base_name(f.stem) for f in files["train"]} | {base_name(f.stem) for f in files["valid"]}
    shares_name = np.array([base_name(f.stem) in other_names for f in files["test"]])

    truth = load_ground_truth(dataset_dir, "test")
    per_threshold = {}
    for threshold in SWEEP_THRESHOLDS:
        flagged = {path for path, value in distances.items() if value <= threshold}
        instances = {name: 0 for name in names}
        for path, item in truth.items():
            if path in flagged:
                for class_id in item["classes"]:
                    instances[names[int(class_id)]] += 1
        per_threshold[str(threshold)] = {
            "test_images_flagged": len(flagged),
            "test_instances_flagged_per_class": instances,
        }

    summary = {
        "method": "64-bit DCT perceptual hash, compared against both the image and its mirror",
        "default_max_hamming_distance": DUPLICATE_MAX_DISTANCE,
        "test_images": len(files["test"]),
        "valid_images": len(files["valid"]),
        "valid_images_within_default_distance_of_train": int((valid_train <= DUPLICATE_MAX_DISTANCE).sum()),
        "test_nearest_train_distance_bins": histogram(test_train),
        "by_threshold": per_threshold,
        "shared_file_name_check_at_default": {
            "test_images_sharing_a_name_with_train_or_valid": int(shares_name.sum()),
            "of_which_content_duplicates": int((shares_name & flagged_mask).sum()),
            "content_duplicates_without_shared_name": int((~shares_name & flagged_mask).sum()),
        },
        "flagged_test_images_at_default": sorted(Path(p).name for p, v in distances.items() if v <= DUPLICATE_MAX_DISTANCE),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "duplicates.json").write_text(json.dumps(summary, indent=2))
    write_examples(files, test_train, test_train_match)
    show(summary)
    return distances


def show(summary: dict) -> None:
    print("\nDuplicate check: test images against train and valid (perceptual hash, 64 bits)")
    print(f"  nearest-train distance, test images, by bin: {summary['test_nearest_train_distance_bins']}")
    print(f"  valid images within default distance of train: {summary['valid_images_within_default_distance_of_train']} of {summary['valid_images']}")
    print(f"  shared file name check: {summary['shared_file_name_check_at_default']}")
    for threshold, row in summary["by_threshold"].items():
        print(
            f"  distance <= {threshold}: {row['test_images_flagged']} of {summary['test_images']} test images flagged, "
            f"instances {row['test_instances_flagged_per_class']}"
        )
