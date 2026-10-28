from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "missing-pills-v1"
OUT = ROOT / "analysis"


def main():
    exact_groups = defaultdict(list)
    source_groups = defaultdict(list)

    for split in ("train", "valid", "test"):
        for path in sorted((DATA / split / "images").iterdir()):
            if path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".webp", ".bmp"
            }:
                continue

            # Compare decoded pixels, regardless of the filename.
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                digest = sha256(
                    str(image.size).encode() + image.tobytes()
                ).hexdigest()

            record = {"split": split, "filename": path.name}
            exact_groups[digest].append(record)

            # Filename families are clues, not proof of duplication.
            stem = path.stem.split(".rf.")[0]
            match = re.match(
                r"^(my_photo[-_]\d+|test[-_]\d+)(?=[_.-]|$)",
                stem,
                flags=re.IGNORECASE,
            )
            if match:
                source_groups[match.group(1).lower()].append(record)

    OUT.mkdir(exist_ok=True)

    for label, groups in (
        ("identical_pixels", exact_groups),
        ("possible_source_family", source_groups),
    ):
        rows = []
        crossing_groups = 0

        for group, records in groups.items():
            if len({record["split"] for record in records}) < 2:
                continue

            crossing_groups += 1
            for record in records:
                rows.append({"group": group, **record})

        table = pd.DataFrame(
            rows, columns=["group", "split", "filename"]
        )
        table.to_csv(OUT / f"split_audit_{label}.csv", index=False)

        print(f"\n{label}: {crossing_groups} groups across splits")
        if not table.empty:
            print(table.head(15).to_string(index=False))

    print(
        "\nFilename families need visual checking. "
        "Zero matches does not prove the splits are independent."
    )


if __name__ == "__main__":
    main()