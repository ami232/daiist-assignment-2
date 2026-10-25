import json
from pathlib import Path
from PIL import Image, ImageDraw

D = json.loads(Path("outputs/predictions_test.json").read_text())["images"]
IMG = Path("datasets/fire-smoke-lwjte-v1/test/images")
top = {n: max([p[1] for p in v["pred"]] + [0.0]) for n, v in D.items()}
print("frames with no box >= 0.25:", sum(t < 0.25 for t in top.values()), "of", len(top))
worst = sorted(top, key=top.get)[:8]
sheet = Image.new("RGB", (4 * 320, 2 * 320))
for k, name in enumerate(worst):
    im = Image.open(IMG / name).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for c, x, y, w, h in D[name]["gt"]:
        d.rectangle([(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H], outline="lime", width=4)
    for c, s, x1, y1, x2, y2 in D[name]["pred"]:
        if s >= 0.25:
            d.rectangle([x1*W, y1*H, x2*W, y2*H], outline="red", width=3)
    sheet.paste(im.resize((320, 320)), ((k % 4) * 320, (k // 4) * 320))
sheet.save("outputs/least_confident_frames.jpg")