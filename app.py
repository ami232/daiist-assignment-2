"""
Assignment 2: Gradio dashboard.

Everything here comes from what `main.py train` already produced:
  weights/best.pt            the finetuned model
  outputs/*.json, *.csv      metrics, dataset counts, saved confidence scores
Nothing is trained when the app starts.

Run with:  uv run python main.py app
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from costing import BASE_RATE, COST_FN, THRESHOLDS, confusion_figure, read_scores
from review import (COST_REVIEW, COST_WRONG_FAIL, HUMAN_ACCURACY, LISTINGS_PER_DAY,
                    staffing, three_way_cost, zones)

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
OUT = ROOT / "outputs"
IMGSZ = 512

# Pink and purple look.
PINK = "#E0218A"
PURPLE = "#7B2CBF"
LILAC = "#B08CC9"
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
)

# ---------------------------------------------------------------- load results
needed = [WEIGHTS, OUT / "test_metrics.json", OUT / "three_way.json",
          OUT / "dataset_stats.json", OUT / "image_scores_test.csv"]
missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
if missing:
    raise SystemExit(f"Missing {missing}. Run `uv run python main.py train` first.")

METRICS = json.loads((OUT / "test_metrics.json").read_text())
THREE = json.loads((OUT / "three_way.json").read_text())
STATS = json.loads((OUT / "dataset_stats.json").read_text())
T_CAUGHT, T_SPUR = read_scores(OUT / "image_scores_test.csv")
CLASS_NAMES = STATS["class_names"]
CHOSEN_LOW = THREE["chosen"]["low"]
CHOSEN_HIGH = THREE["chosen"]["high"]
DO_NOTHING = 1000 * BASE_RATE * COST_FN

if torch.cuda.is_available():
    DEVICE = "0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
MODEL = YOLO(str(WEIGHTS))


def fix_band(low, high):
    """The high threshold can never be below the low one."""
    low = float(low)
    return low, max(low, float(high))


# ------------------------------------------------------------------ tab 1: photo
REVIEW_COLOUR = (150, 70, 220)   # violet
FAIL_COLOUR = (255, 20, 147)     # hot pink
try:
    LABEL_FONT = ImageFont.load_default(size=16)
except TypeError:  # older Pillow without the size option
    LABEL_FONT = ImageFont.load_default()

def prepare_photo(image):
    """Make an uploaded photo look like the training photos.

    The dataset was preprocessed on Roboflow: stretched to 512x512, contrast
    equalized, converted to greyscale. Photos that are already greyscale (the
    dataset's own) are left alone. Colour photos get a similar treatment; the
    equalization settings are an approximation, because Roboflow does not
    publish its exact ones.
    """
    rgb = np.asarray(image.convert("RGB"))
    channel_gap = max(np.abs(rgb[..., 0].astype(int) - rgb[..., 1]).mean(),
                      np.abs(rgb[..., 1].astype(int) - rgb[..., 2]).mean())
    if channel_gap < 2:
        return image.convert("RGB"), False
    resized = cv2.resize(rgb, (IMGSZ, IMGSZ))
    grey = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    return Image.fromarray(grey).convert("RGB"), True


def predict_image(image, low, high):
    low, high = fix_band(low, high)
    if image is None:
        return None, "Upload a photo of a garment to see the verdict."
    image, converted = prepare_photo(image)
    result = MODEL.predict(source=image, conf=0.01, imgsz=IMGSZ, device=DEVICE,
                           max_det=100, verbose=False)[0]
    boxes = result.boxes
    confs = boxes.conf.cpu().numpy() if len(boxes) else np.array([])
    xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
    classes = boxes.cls.cpu().numpy() if len(boxes) else np.array([])
    top = float(confs.max()) if len(confs) else 0.0

    drawn = image.convert("RGB").copy()
    draw = ImageDraw.Draw(drawn)
    for box, conf, cls in zip(xyxy, confs, classes):
        if conf < low:
            continue
        colour = FAIL_COLOUR if conf >= high else REVIEW_COLOUR
        draw.rectangle([float(v) for v in box], outline=colour, width=3)
        label = f"{CLASS_NAMES[int(cls)]} {conf:.2f}"
        x0, y0 = float(box[0]), float(box[1])
        text_xy = (x0, max(0.0, y0 - 20))
        draw.rectangle(draw.textbbox(text_xy, label, font=LABEL_FONT), fill=colour)
        draw.text(text_xy, label, fill=(255, 255, 255), font=LABEL_FONT)

    if top >= high:
        text = (f"## FAIL\nThe most confident box is {top:.2f}, at or above the high threshold "
                f"({high:.2f}). The seller is asked to declare the defect or retake the photos.")
    elif top >= low:
        text = (f"## HUMAN REVIEW\nThe most confident box is {top:.2f}, between the thresholds "
                f"({low:.2f} and {high:.2f}). A moderator looks at the listing.")
    else:
        text = (f"## PASS\nThe most confident box is {top:.2f}, below the low threshold "
                f"({low:.2f}). The listing goes live.")
    text += "\n\nBoxes shown: violet = review zone, pink = fail zone. Boxes below the low threshold are hidden."
    if converted:
        text += ("\n\nThis colour photo was converted to greyscale, resized to 512x512 and contrast-equalized "
                 "first, because the model was trained only on photos prepared that way (approximate).")
    return drawn, text



# ------------------------------------------------------------ tab 2: cost/review
def cost_figure(low, high):
    plt.close("all")
    low, high = fix_band(low, high)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    lows = [float(t) for t in THRESHOLDS if t <= high]
    ax.plot(lows, [three_way_cost(T_CAUGHT, T_SPUR, t, high)[0] for t in lows],
            color=PURPLE, label=f"low threshold varies, high = {high:.2f}")
    if abs(high - CHOSEN_HIGH) > 1e-9:
        lows2 = [float(t) for t in THRESHOLDS if t <= CHOSEN_HIGH]
        ax.plot(lows2, [three_way_cost(T_CAUGHT, T_SPUR, t, CHOSEN_HIGH)[0] for t in lows2],
                color=LILAC, linestyle="--", label=f"high = chosen {CHOSEN_HIGH:.2f}")

    chosen_cost = three_way_cost(T_CAUGHT, T_SPUR, CHOSEN_LOW, CHOSEN_HIGH)[0]
    ax.scatter([CHOSEN_LOW], [chosen_cost], color=PINK, marker="*", s=240, zorder=5,
               label=f"chosen point (low {CHOSEN_LOW:.2f}, high {CHOSEN_HIGH:.2f}): EUR {chosen_cost:.0f}")
    current_cost = three_way_cost(T_CAUGHT, T_SPUR, low, high)[0]
    ax.scatter([low], [current_cost], color=PURPLE, s=70, zorder=6,
               label=f"current sliders: EUR {current_cost:.0f}")
    ax.axhline(DO_NOTHING, color="#444444", linestyle=":", label=f"do nothing: EUR {DO_NOTHING:.0f}")
    ax.set_xlabel("low threshold")
    ax.set_ylabel("expected cost per 1,000 listings (EUR)")
    ax.set_title("Cost against threshold (test split)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def refresh_cost(low, high):
    low, high = fix_band(low, high)
    cost, review_share = three_way_cost(T_CAUGHT, T_SPUR, low, high)
    d = zones(T_CAUGHT, low, high)
    g = zones(T_SPUR, low, high)
    n_def, n_good = 1000 * BASE_RATE, 1000 * (1 - BASE_RATE)
    matrix = pd.DataFrame({
        "Per 1,000 listings": ["Defective (really has a defect)", "Good (no defect, proxy)"],
        "Total": [round(n_def, 1), round(n_good, 1)],
        "Passed": [round(n_def * d[0], 1), round(n_good * g[0], 1)],
        "Sent to review": [round(n_def * d[1], 1), round(n_good * g[1], 1)],
        "Failed": [round(n_def * d[2], 1), round(n_good * g[2], 1)],
    })
    staff = staffing(review_share)
    summary = (
        f"### Current thresholds: pass below {low:.2f}, review {low:.2f} to {high:.2f}, fail from {high:.2f}\n"
        f"- **Expected cost: EUR {cost:.2f} per 1,000 listings** (doing nothing: EUR {DO_NOTHING:.0f}; "
        f"chosen thresholds: EUR {THREE['chosen']['test_cost_per_1000']:.2f})\n"
        f"- **Sent to a person: {review_share:.1%} of listings**, which is about {staff['cases_per_day']} "
        f"cases a day at {LISTINGS_PER_DAY:,} listings a day, or **{staff['moderators_needed']} moderators**\n"
        f"- In the table, a defective listing that is *passed* is a missed defect (EUR {COST_FN:.0f}); "
        f"a good listing that is *failed* is a wrong fail (EUR {COST_WRONG_FAIL:.0f}); each review costs EUR {COST_REVIEW:.2f}, "
        f"and a moderator is right {HUMAN_ACCURACY:.0%} of the time.\n"
        f"- Base rate assumed: {BASE_RATE:.0%} of listings have an undeclared defect. "
        f"The good-listing row is a proxy (stray boxes on defective photos), so real false alarms are probably higher."
    )
    rows = [[f"three-way, current (low {low:.2f}, high {high:.2f})", round(cost, 2), f"{review_share:.1%}"]]
    for name, s in THREE["strategies"].items():
        rows.append([name + " (saved)", s["cost_per_1000"], f"{s['review_share']:.1%}"])
    strategies = pd.DataFrame(rows, columns=["Strategy", "Cost per 1,000 (EUR)", "Sent to a person"])
    return summary, matrix, cost_figure(low, high), strategies


# ------------------------------------------------------------- tab 3: per class
def class_table():
    inst = STATS["splits"]["test"]["instances_per_class"]
    rows = []
    for name, v in METRICS["finetuned"]["per_class"].items():
        rows.append([name, inst[name], round(v["precision"], 3), round(v["recall"], 3),
                     round(v["ap50"], 3), round(v["map50_95"], 3)])
    o = METRICS["finetuned"]["overall"]
    rows.append(["ALL", sum(inst.values()), round(o["precision"], 3), round(o["recall"], 3),
                 round(o["ap50"], 3), round(o["map50_95"], 3)])
    return pd.DataFrame(rows, columns=["Class", "Test instances", "Precision", "Recall", "AP50", "mAP50-95"])


def baseline_table():
    o = METRICS["finetuned"]["overall"]
    coco = METRICS["baseline_coco_zero_shot"]
    nothing = METRICS["baseline_predict_nothing"]
    return pd.DataFrame([
        ["Finetuned yolo11n", round(o["ap50"], 3),
         f"precision {o['precision']:.3f}, recall {o['recall']:.3f}"],
        ["COCO model, zero-shot", 0.0,
         f"boxes on {coco['images_with_any_box']} of {coco['test_images']} test images, all with wrong labels"],
        ["Predict nothing", 0.0,
         f"catches {nothing['defective_images_caught']} of {nothing['defective_images_in_test']} defective images"],
    ], columns=["Model", "AP50 on our classes", "On the test split"])


def dataset_table():
    rows = []
    for split, s in STATS["splits"].items():
        rows.append([split, s["images"], s["images_with_defects"]] +
                    [s["instances_per_class"][n] for n in CLASS_NAMES])
    return pd.DataFrame(rows, columns=["Split", "Images", "With defects"] + [f"{n} (instances)" for n in CLASS_NAMES])


# ------------------------------------------------------------------------- app
with gr.Blocks(title="Second-hand clothing defect check", theme=THEME) as demo:
    gr.Markdown(
        "# Second-hand clothing: defect check\n"
        "A model looks at the seller's photos. Each listing is **passed**, **sent to a human moderator**, "
        "or **failed**. Move the two thresholds and watch the cost change."
    )
    with gr.Row():
        low = gr.Slider(0.02, 0.98, value=CHOSEN_LOW, step=0.01,
                        label="Low threshold: below this, the listing passes")
        high = gr.Slider(0.02, 0.98, value=CHOSEN_HIGH, step=0.01,
                         label="High threshold: from here, the listing fails (in between: human review)")
        reset = gr.Button("Reset to chosen thresholds", variant="primary")

    with gr.Tabs():
        with gr.Tab("Try a photo"):
            with gr.Row():
                image_in = gr.Image(type="pil", label="Garment photo")
                image_out = gr.Image(label="Model output")
            verdict = gr.Markdown()
            for trigger in (image_in.change, low.change, high.change):
                trigger(predict_image, [image_in, low, high], [image_out, verdict])

        with gr.Tab("Cost and review"):
            summary = gr.Markdown()
            matrix = gr.Dataframe(label="Confusion matrix per 1,000 listings (test split)", interactive=False)
            curve = gr.Plot(label="Cost against threshold")
            strategies = gr.Dataframe(label="Four ways to run the check", interactive=False)
            for trigger in (low.change, high.change):
                trigger(refresh_cost, [low, high], [summary, matrix, curve, strategies])
            demo.load(refresh_cost, [low, high], [summary, matrix, curve, strategies])

        with gr.Tab("Per-class results"):
            gr.Markdown("### Finetuned model on the test split (positive = a defect of that type)")
            gr.Dataframe(value=class_table(), interactive=False)
            gr.Markdown("### Against the two baselines")
            gr.Dataframe(value=baseline_table(), interactive=False)
            gr.Markdown("### Dataset (Roboflow Universe: defect-5hwax / fabric-defect-ieswu, version 16)")
            gr.Dataframe(value=dataset_table(), interactive=False)
            cm_file = OUT / "confusion_matrix.json"
            if cm_file.exists():
                cm = json.loads(cm_file.read_text())
                gr.Markdown(
                    f"### Confusion matrix on the test split, at threshold {cm['threshold']:.2f}\n"
                    "Rows are what the model predicted, columns are what was really there. "
                    "*background* means no defect: a count in the background column is a stray box "
                    "(a false alarm), and a count in the background row is a defect the model missed."
                )
                gr.Plot(value=confusion_figure(
                    np.array(cm["matrix"]), cm["labels"], f"Test split, threshold {cm['threshold']:.2f}"))

    reset.click(lambda: (CHOSEN_LOW, CHOSEN_HIGH), None, [low, high])


if __name__ == "__main__":
    demo.launch()