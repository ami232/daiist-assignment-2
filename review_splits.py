from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "missing-pills-v1"
OUT = ROOT / "analysis"

rows = pd.read_csv(OUT / "split_audit_possible_source_family.csv")
family = rows[rows["group"] == "my_photo-100"]

if family.empty:
    raise ValueError("No matching family found.")

canvas = Image.new("RGB", (960, 840), "white")
draw = ImageDraw.Draw(canvas)

for column, split in enumerate(("train", "valid", "test")):
    examples = family[family["split"] == split].head(3)
    draw.text((column * 320 + 10, 10), split.upper(), fill="black")

    for row, record in enumerate(examples.itertuples()):
        path = DATA / split / "images" / record.filename

        with Image.open(path) as image:
            thumbnail = ImageOps.contain(
                ImageOps.exif_transpose(image).convert("RGB"),
                (310, 230),
            )

        x = column * 320 + 5
        y = row * 270 + 40
        canvas.paste(thumbnail, (x, y))
        draw.text((x, y + 235), f"my_photo-100 / {row + 1}", fill="black")

output = OUT / "related_images_example.jpg"
canvas.save(output)
print(f"Saved: {output}")