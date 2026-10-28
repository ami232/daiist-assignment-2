"""
Assignment 2: Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file. main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via uv run python main.py app, this file must build and launch a Gradio app
(a demo that calls .launch()) using only what your training stage already
produced. It must never train anything itself. Load the committed weights in
weights/, never whatever the training run left in runs/.

At minimum, the dashboard must let you:

- Upload an image and see the model's predictions at the current threshold.
- Move the two edges of your escalation band and watch the confusion matrix,
  the expected cost per 1,000 units, and the share of cases escalated to a
  human all change together.
- See the cost-against-threshold curve with your chosen operating point marked
  on it.
- See per-class performance, not just the overall number.

How you structure the code beyond that is your call to make and be able to
defend: file layout, function boundaries, naming.
"""


import hashlib
import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw
from ultralytics import YOLO

from review_policy import evaluate_policy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
WEIGHTS = ROOT / "weights" / "best.pt"

saved = json.loads((OUT / "review_policy.json").read_text())
binary = json.loads((OUT / "chosen_threshold.json").read_text())
cache_info = json.loads((OUT / "prediction_cache.json").read_text())

if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != cache_info["weights_sha256"]:
    raise RuntimeError("Saved predictions do not match weights/best.pt.")

CONFIG = saved["config"]
SCORES = pd.read_csv(OUT / "image_scores.csv")
SWEEP = pd.read_csv(OUT / "threshold_sweep.csv")
CLASSES = pd.read_csv(OUT / "test_per_class.csv")
COMPARISON = pd.read_csv(OUT / "policy_comparison.csv")
MODEL = YOLO(str(WEIGHTS))

EMPTY = {"nopill_front", "nopill_back"}
POCKETS = EMPTY | {"pill_front", "pill_back"}


def validate(lower, upper):
    if lower > upper:
        raise gr.Error(
            "The pass/review threshold must not exceed the review/fail threshold."
        )


def policy_views(lower, upper, split):
    validate(lower, upper)
    frame = SCORES[SCORES["split"] == split]
    result = evaluate_policy(frame, lower, upper, CONFIG)

    capacity = (
        "Within available review capacity"
        if result["within_capacity"]
        else "Exceeds available review capacity"
    )

    summary = (
        f"### €{result['cost_per_1000']:.2f} per 1,000 cases\n"
        f"**Sent for review:** "
        f"{result['operational_review_share']:.2%} of the operational mix · "
        f"{result['reviews_per_day']:.1f} cases/day\n\n"
        f"**Review workload:** {result['review_hours_per_day']:.2f} hours/day · "
        f"{result['review_FTE']:.2f} FTE · {capacity}\n\n"
        f"Observed review referrals in this split: "
        f"{result['dataset_review_count']} / {len(frame)}."
    )

    routing = pd.DataFrame({
        "Actual": ["Complete", "Incomplete"],
        "Pass": [result["negative_pass"], result["positive_pass"]],
        "Human review": [
            result["negative_review"], result["positive_review"]
        ],
        "Fail / hold": [
            result["negative_fail"], result["positive_fail"]
        ],
    })

    matrix = [
        [
            result["expected_TN_per_1000"],
            result["expected_FP_per_1000"],
        ],
        [
            result["expected_FN_per_1000"],
            result["expected_TP_per_1000"],
        ],
    ]
    confusion = go.Figure(go.Heatmap(
        z=matrix,
        x=["Release", "Hold"],
        y=["Actually complete", "Actually incomplete"],
        texttemplate="%{z:.2f}",
        colorscale="Blues",
    ))
    confusion.update_layout(
        title="Expected final confusion matrix per 1,000 cases",
        xaxis_title="Final decision, including assumed human outcomes",
        yaxis={"autorange": "reversed"},
        height=360,
    )

    curve = go.Figure(go.Scatter(
        x=SWEEP["threshold"],
        y=SWEEP["cost_per_1000"],
        name="Validation: model alone",
    ))
    curve.add_vline(
        x=binary["chosen_threshold"], line_dash="dash",
        line_color="black", annotation_text="Selected binary threshold",
    )
    curve.add_vrect(
        x0=lower, x1=upper,
        fillcolor="orange", opacity=0.15, line_width=0,
    )
    curve.add_vline(x=lower, line_color="green")
    curve.add_vline(x=upper, line_color="red")
    curve.update_layout(
        title="Validation cost curve and current review interval",
        xaxis_title="Confidence threshold",
        yaxis_title="Expected cost per 1,000 cases (€)",
        height=360,
    )

    return summary, routing, confusion, curve


def render_prediction(image, detections, lower, upper):
    validate(lower, upper)
    if image is None or detections is None:
        return None, "Upload an inspection image."

    score = max(
        (d["confidence"] for d in detections if d["class"] in EMPTY),
        default=0.0,
    )
    pocket_confidence = max(
        (d["confidence"] for d in detections if d["class"] in POCKETS),
        default=0.0,
    )

    if pocket_confidence < CONFIG["pocket_guard"]:
        decision = "HUMAN REVIEW — no pocket detected at confidence ≥ 0.25"
    elif score >= upper:
        decision = "FAIL — hold for correction"
    elif score >= lower:
        decision = "HUMAN REVIEW — intermediate score"
    else:
        decision = "PASS — release"

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    shown = 0

    for detection in detections:
        if detection["confidence"] < lower:
            continue

        color = "red" if detection["class"] in EMPTY else "blue"
        box = detection["xyxy"]
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{detection['class']} {detection['confidence']:.2f}",
            fill=color,
        )
        shown += 1

    message = (
        f"### {decision}\n"
        f"Empty-pocket score: **{score:.3f}**. "
        f"This is a confidence score, not a calibrated probability.\n\n"
        f"Showing {shown} boxes at confidence ≥ {lower:.3f}. "
        "The input check uses pocket detections at confidence ≥ 0.25."
    )
    return annotated, message


def inspect_image(image, lower, upper):
    validate(lower, upper)
    if image is None:
        return None, None, "Upload an inspection image."

    result = MODEL.predict(
        image.convert("RGB"),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device="cpu",
        verbose=False,
    )[0]

    detections = [
        {
            "class": MODEL.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "xyxy": box.xyxy[0].tolist(),
        }
        for box in result.boxes
    ]
    annotated, message = render_prediction(
        image, detections, lower, upper
    )
    return detections, annotated, message


initial = policy_views(saved["lower"], saved["upper"], "test")

with gr.Blocks(title="Missing Pills — Inspection Decisions") as demo:
    gr.Markdown(
        "# Missing Pills — Inspection Decisions\n"
        "Detect visible empty pockets and explore the cost of release, "
        "hold and human-review decisions."
    )
    gr.Markdown(
        "Positive = an inspection image containing at least one empty pocket. "
        "Costs assume **2% incomplete cases**, **10,000 cases/day**, "
        "€1 per FP, €20 per FN, €0.50 per TP and €0 per TN. "
        "Human review assumes 99% sensitivity and 99.5% specificity."
    )

    with gr.Row():
        lower = gr.Slider(
            0, 1, value=saved["lower"], step=0.001,
            label="Pass / human-review threshold",
        )
        upper = gr.Slider(
            0, 1, value=saved["upper"], step=0.001,
            label="Human-review / fail threshold",
        )

    gr.Markdown(
        "Below the lower threshold: pass. Between thresholds: review. "
        "At or above the upper threshold: fail / hold. "
        "Equal thresholds give an empty confidence-review interval. "
        "No pocket detected at confidence ≥ 0.25 always triggers review."
    )
    split = gr.Radio(
        ["valid", "test"], value="test", label="Evaluation split"
    )
    gr.Markdown(
        "The saved policy was selected on validation. Changing sliders "
        "on test is exploratory and does not replace the saved policy."
    )

    with gr.Tab("Decision costs"):
        summary = gr.Markdown(initial[0])
        gr.Markdown("### Observed routing counts in the selected split")
        routing = gr.Dataframe(value=initial[1], interactive=False)
        gr.Markdown(
            "Routing counts are observed. The matrix below is an "
            "operational projection with assumed human errors, not "
            "measured reviewer performance."
        )
        confusion = gr.Plot(initial[2])
        curve = gr.Plot(initial[3])
        gr.Markdown(
            "The curve is the model-only validation sweep. Green and red "
            "lines show the current thresholds; orange marks the review "
            "interval. The current policy cost is shown above."
        )

    with gr.Tab("Inspect an image"):
        detections = gr.State(None)
        with gr.Row():
            uploaded_image = gr.Image(type="pil", label="Upload image")
            annotated_image = gr.Image(label="Predictions", interactive=False)
        inspect_button = gr.Button("Inspect image", variant="primary")
        decision = gr.Markdown("Upload an inspection image.")
        gr.Markdown(
            "Prototype for visible pill presence. Performance on other "
            "capture conditions has not been established."
        )

    with gr.Tab("Per-class results"):
        gr.Markdown(
            "### Test detection metrics\n"
            "Precision and recall use Ultralytics' diagnostic F1 operating "
            "point, not the current business thresholds."
        )
        gr.Dataframe(value=CLASSES, interactive=False)
        gr.Markdown("### Saved policy comparison on test")
        gr.Dataframe(
            value=COMPARISON[COMPARISON["split"] == "test"][
                ["policy", "cost_per_1000", "reviews_per_day", "review_FTE"]
            ],
            interactive=False,
        )

    for control in (lower, upper, split):
        control.change(
            policy_views,
            inputs=[lower, upper, split],
            outputs=[summary, routing, confusion, curve],
            queue=False,
        )

    inspect_button.click(
        inspect_image,
        inputs=[uploaded_image, lower, upper],
        outputs=[detections, annotated_image, decision],
        queue=False,
    )

    uploaded_image.change(
        inspect_image,
        inputs=[uploaded_image, lower, upper],
        outputs=[detections, annotated_image, decision],
        queue=False,
    )

    for control in (lower, upper):
        control.change(
            render_prediction,
            inputs=[uploaded_image, detections, lower, upper],
            outputs=[annotated_image, decision],
            queue=False,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("APP_PORT", "7860")),
        inbrowser=True,
    )