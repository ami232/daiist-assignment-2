"""
Assignment 2: Gradio dashboard.

Loads weights/best.pt and artifacts/results.json written by train.py. Nothing
is trained here: threshold changes are recomputed from the saved test-split
slot scores.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw
from ultralytics import YOLO

import config
import costing

ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / "artifacts" / "results.json").read_text())
MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
POS_ID = RESULTS["names"].index(RESULTS["positive_class"])
MAX_DET = 1000
IMGSZ = RESULTS["train_settings"]["imgsz"]  # predict at the size the model was trained at

TEST = RESULTS["cases"]["test"]
SCORES = np.array(TEST["finetuned"])
LABELS = np.array(TEST["labels"])
PHANTOMS = np.array(TEST["finetuned_phantoms"])
VAL = RESULTS["cases"]["val"]
VAL_SCORES = np.array(VAL["finetuned"])
VAL_LABELS = np.array(VAL["labels"])
VAL_PHANTOMS = np.array(VAL["finetuned_phantoms"])
THRESHOLDS = np.array(RESULTS["thresholds"])
CHOSEN = RESULTS["operating_point"]
SLOTS_PER_DAY = config.IMAGES_PER_DAY * config.SLOTS_PER_IMAGE
COLOURS = {costing.REVIEW: "#f59e0b", costing.FAIL: "#dc2626"}


def ordered(t_low, t_high):
    return min(t_low, t_high), max(t_low, t_high)


def predict(image, t_low, t_high):
    """Draw gap detections at or above t_low, coloured by how each one is routed."""
    if image is None:
        return None, ""
    t_low, t_high = ordered(t_low, t_high)
    r = MODEL.predict(image, imgsz=IMGSZ, conf=max(t_low, 0.001), max_det=MAX_DET,
                      verbose=False)[0]
    gaps = r.boxes[r.boxes.cls == POS_ID]
    confs = gaps.conf.cpu().numpy()
    routed = costing.route(confs, t_low, t_high)
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    width = max(2, out.width // 400)
    for box, conf, k in zip(gaps.xyxy.cpu().numpy(), confs, routed):
        if k == costing.PASS:  # only when t_low is 0
            continue
        draw.rectangle(box.tolist(), outline=COLOURS[k], width=width)
        draw.text((box[0] + width, box[1] + width), f"{conf:.2f}", fill=COLOURS[k])
    text = (f"<span style='color:{COLOURS[costing.FAIL]}'>■</span> "
            f"**{int((routed == costing.FAIL).sum())}** gaps sent to restock (≥ {t_high:.2f}) · "
            f"<span style='color:{COLOURS[costing.REVIEW]}'>■</span> "
            f"**{int((routed == costing.REVIEW).sum())}** sent to a person ({t_low:.2f}–{t_high:.2f}) · "
            f"{int((r.boxes.cls != POS_ID).sum())} products detected")
    return out, text


def cost_curve(t_low, t_high, base_rate):
    costs = costing.sweep(SCORES, LABELS, THRESHOLDS, PHANTOMS, base_rate)
    val_costs = costing.sweep(VAL_SCORES, VAL_LABELS, THRESHOLDS, VAL_PHANTOMS, base_rate)
    fig = go.Figure(go.Scatter(x=THRESHOLDS, y=costs, mode="lines", name="test"))
    fig.add_scatter(x=THRESHOLDS, y=val_costs, mode="lines", line_dash="dot",
                    name="validation (threshold chosen here)")
    fig.add_vline(x=CHOSEN["single_threshold"], line_dash="dash",
                  annotation_text=f"chosen {CHOSEN['single_threshold']:.2f}")
    fig.add_vrect(x0=t_low, x1=t_high, opacity=0.15, line_width=0, annotation_text="review band")
    fig.update_layout(xaxis_title="Confidence threshold", yaxis_title="€ per 1,000 slots")
    return fig


def operating_view(t_low, t_high, base_rate):
    t_low, t_high = ordered(t_low, t_high)
    table = pd.DataFrame(costing.confusion_table(SCORES, LABELS, t_low, t_high, PHANTOMS),
                         index=["actual: filled slot", "actual: gap", "no slot (phantom)"],
                         columns=["pass", "review", "fail"]).reset_index(names="")
    cost = costing.cost_per_1000(SCORES, LABELS, t_low, t_high, PHANTOMS, base_rate)
    share = costing.review_share(SCORES, LABELS, t_low, t_high, PHANTOMS, base_rate)
    nothing = costing.cost_per_1000(SCORES, LABELS, np.inf, np.inf, (), base_rate)
    saved = (nothing - cost) * SLOTS_PER_DAY / 1000
    reviewers = costing.reviewers_needed(share)
    summary = (f"**Expected cost:** €{cost:,.2f} per 1,000 slots "
               f"(€{saved:,.0f}/day {'saved' if saved >= 0 else 'lost'} against doing nothing)  \n"
               f"**Sent to review:** {share:.2%} of slots → "
               f"{costing.reviews_per_day(share):,.0f} reviews/day → "
               f"{reviewers:.2f} reviewers")
    if config.MAX_REVIEWERS is not None and reviewers > config.MAX_REVIEWERS:
        summary += (f"  \n⚠️ Over the staffing cap of {config.MAX_REVIEWERS:.2f} reviewers "
                    f"({config.MAX_REVIEWERS * config.REVIEWER_SECONDS_PER_DAY / 3600:.1f} h of review a day)")
    options = pd.DataFrame(costing.alternatives(SCORES, LABELS, CHOSEN["single_threshold"],
                                                t_low, t_high, PHANTOMS, base_rate))
    return table, summary, cost_curve(t_low, t_high, base_rate), options


finetuned = RESULTS["box_metrics"]["finetuned"]

with gr.Blocks(title="Shelf gap detector") as demo:
    gr.Markdown("# Empty shelf slot detector")
    with gr.Row():
        t_low = gr.Slider(0, 1, value=CHOSEN["t_low"], step=0.01, label="Below this: pass")
        t_high = gr.Slider(0, 1, value=CHOSEN["t_high"], step=0.01, label="At or above this: fail (restock)")
        base_rate = gr.Slider(0.001, 0.5, value=config.BASE_RATE, step=0.001, label="Assumed base rate (share of slots empty)")

    with gr.Tab("Predict"):
        with gr.Row():
            image = gr.Image(type="pil", label="Shelf image")
            annotated = gr.Image(label="Gap detections")
        decision = gr.Markdown()

    with gr.Tab("Operating point"):
        summary = gr.Markdown()
        table = gr.Dataframe(label="Test split: actual vs outcome (slots)")
        plot = gr.Plot()
        options = gr.Dataframe(label="Against the alternatives")

    with gr.Tab("Per-class results"):
        gr.Markdown(f"**mAP50:** {finetuned['mAP50']:.3f} · **mAP50-95:** {finetuned['mAP50-95']:.3f} "
                    f"· positive class: `{RESULTS['positive_class']}`")
        gr.Dataframe(pd.DataFrame(finetuned["per_class"]), label="Finetuned, test split (box level)")
        gr.Markdown(f"**COCO zero-shot:** {RESULTS['box_metrics']['coco_zero_shot']}  \n"
                    f"**Trivial baseline:** {RESULTS['box_metrics']['trivial']}")
        gr.Dataframe(pd.DataFrame(RESULTS["slot_level_baselines"]).T.reset_index(names="model"),
                     label="Slot level, at the chosen single threshold")
        cm_png = ROOT / "artifacts" / "box_confusion_matrix.png"
        if cm_png.exists():
            gr.Image(str(cm_png), label="Box-level confusion matrix")

    sliders = [t_low, t_high, base_rate]
    for s in sliders:
        s.change(operating_view, sliders, [table, summary, plot, options])
    for c in (image, t_low, t_high):
        c.change(predict, [image, t_low, t_high], [annotated, decision])
    demo.load(operating_view, sliders, [table, summary, plot, options])

if __name__ == "__main__":
    demo.launch()
