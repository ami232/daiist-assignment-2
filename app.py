from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"
APP_PORT = int(os.environ.get("APP_PORT", "7860"))

if not WEIGHTS.exists():
    raise FileNotFoundError("weights/best.pt is missing. Run `uv run python main.py train` first.")

required = [
    ARTIFACTS / "operating_point.json",
    ARTIFACTS / "cost_sweep.csv",
    ARTIFACTS / "test_image_scores.csv",
    ARTIFACTS / "per_class_metrics.csv",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Training artifacts are missing: " + ", ".join(missing) + ". Run `uv run python main.py train` first.")

model = YOLO(str(WEIGHTS))
operating = json.loads((ARTIFACTS / "operating_point.json").read_text(encoding="utf-8"))
cost_sweep = pd.read_csv(ARTIFACTS / "cost_sweep.csv")
scores = pd.read_csv(ARTIFACTS / "test_image_scores.csv")
per_class = pd.read_csv(ARTIFACTS / "per_class_metrics.csv")

costs = operating["cost_assumptions"]
fp_cost = float(costs["false_positive_cost_eur"])
fn_cost = float(costs["false_negative_cost_eur"])
review_cost = float(costs["human_review_cost_eur"])
cases_per_day = float(costs["cases_per_day"])
review_seconds = 15.0
best_single = operating["best_model_only_threshold"]
best_band = operating["best_assisted_review_band"]
positive_class = operating["business_framing"]["positive_class"]


def assisted_state(low: float, high: float):
    if low >= high:
        high = min(0.99, low + 0.01)
    actual = scores["actual"].to_numpy().astype(int)
    score = scores["score"].to_numpy()
    auto_negative = score < low
    auto_positive = score >= high
    review = (~auto_negative) & (~auto_positive)

    fn = int(np.sum(auto_negative & (actual == 1)))
    fp = int(np.sum(auto_positive & (actual == 0)))
    tp_auto = int(np.sum(auto_positive & (actual == 1)))
    tn_auto = int(np.sum(auto_negative & (actual == 0)))
    review_pos = int(np.sum(review & (actual == 1)))
    review_neg = int(np.sum(review & (actual == 0)))
    tp = tp_auto + review_pos
    tn = tn_auto + review_neg
    review_count = int(np.sum(review))
    n = max(len(scores), 1)
    review_share = review_count / n

    total_cost = fp * fp_cost + fn * fn_cost + review_count * review_cost
    cost_per_1000 = total_cost / n * 1000
    daily_reviews = review_share * cases_per_day
    daily_hours = daily_reviews * review_seconds / 3600

    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted negative", "Predicted positive"],
        y=["Actual negative", "Actual positive"],
        title="Assisted confusion matrix (reviewed cases resolved correctly)",
        labels={"x": "System outcome", "y": "Ground truth"},
    )
    summary = (
        f"**Expected cost / 1,000 cases:** €{cost_per_1000:.2f}  \n"
        f"**Sent to human review:** {review_share:.1%}  \n"
        f"**Daily reviews at {cases_per_day:.0f} cases/day:** {daily_reviews:.1f}  \n"
        f"**Daily review time:** {daily_hours:.2f} hours  \n"
        f"**FP:** {fp} | **FN:** {fn} | **TP:** {tp} | **TN:** {tn}"
    )
    return fig, summary


def cost_curve():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cost_sweep["threshold"], y=cost_sweep["cost_per_1000_eur"], mode="lines", name="Model-only cost"))
    fig.add_vline(x=float(best_single["threshold"]), line_dash="dash", annotation_text=f"Chosen {float(best_single['threshold']):.2f}")
    fig.update_layout(title="Expected business cost vs confidence threshold", xaxis_title="Confidence threshold", yaxis_title="Expected cost per 1,000 cases (€)")
    return fig


def predict_image(image, threshold):
    if image is None:
        return None, "Upload an image first."
    result = model.predict(image, conf=float(threshold), verbose=False)[0]
    plotted_bgr = result.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    rows = []
    if result.boxes is not None:
        for cls, conf in zip(result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            rows.append({"class": model.names[int(cls)], "confidence": round(float(conf), 3)})
    text = "\n".join(f"- {r['class']}: {r['confidence']:.3f}" for r in rows) if rows else "No detections above the current threshold."
    return plotted_rgb, text


per_class_display = per_class.copy()
for col in ["precision", "recall", "ap50", "map50_95"]:
    if col in per_class_display:
        per_class_display[col] = per_class_display[col].round(4)

with gr.Blocks(title="Valorant Enemy Detection - Assignment 2") as demo:
    gr.Markdown(
        "# Valorant Enemy Detection — Assignment 2\n"
        f"Positive business class: **{positive_class}**. The model is already trained; this dashboard never retrains it."
    )

    with gr.Tab("Upload & predict"):
        threshold = gr.Slider(0.01, 0.99, value=float(best_single["threshold"]), step=0.01, label="Detection confidence threshold")
        image_input = gr.Image(type="numpy", label="Upload Valorant frame")
        run_button = gr.Button("Run prediction")
        image_output = gr.Image(label="Detections")
        prediction_text = gr.Markdown()
        run_button.click(predict_image, inputs=[image_input, threshold], outputs=[image_output, prediction_text])
        threshold.change(predict_image, inputs=[image_input, threshold], outputs=[image_output, prediction_text])

    with gr.Tab("Pass / review / flag"):
        gr.Markdown("Scores below the low threshold are **auto-pass**. Scores between thresholds go to **human review**. Scores at or above the high threshold are **auto-flagged**.")
        with gr.Row():
            low = gr.Slider(0.01, 0.95, value=float(best_band["low_threshold"]), step=0.01, label="Pass → review threshold")
            high = gr.Slider(0.05, 0.99, value=float(best_band["high_threshold"]), step=0.01, label="Review → flag threshold")
        cm_plot = gr.Plot()
        review_summary = gr.Markdown()
        for control in (low, high):
            control.change(assisted_state, inputs=[low, high], outputs=[cm_plot, review_summary])
        demo.load(assisted_state, inputs=[low, high], outputs=[cm_plot, review_summary])

    with gr.Tab("Cost curve"):
        gr.Plot(value=cost_curve())

    with gr.Tab("Per-class results"):
        gr.Dataframe(value=per_class_display, interactive=False, label="Held-out test metrics")

    with gr.Tab("Assumptions"):
        gr.Markdown(
            f"""
### Cost assumptions used by the pipeline

- False positive cost: **€{fp_cost:.2f}**
- False negative cost: **€{fn_cost:.2f}**
- Human-review cost: **€{review_cost:.2f}**
- Cases per day: **{cases_per_day:.0f}**

These are explicit assumptions, not facts from the dataset. They must be defended in the written comprehension check.
"""
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=APP_PORT)
