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


"""Interactive Gradio dashboard for road-damage detection."""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "road_damage_yolo11n.pt"
ARTIFACTS = ROOT / "artifacts"
METRICS_PATH = ARTIFACTS / "metrics.json"
VALIDATION_PATH = ARTIFACTS / "validation_predictions.json"

CLASSES = ["Damaged road", "Potholes"]
FN_COST = 100.0
FP_COST = 20.0
REVIEW_COST = 10.0


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_records(path):
    """Accept a list of records or a dictionary containing the records."""
    data = read_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("predictions", "records", "results", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        f"Could not find a list of prediction records in {path}"
    )


def get_score(record):
    """Get the maximum image-level detection confidence."""
    if record.get("score") is not None:
        return float(record["score"])

    scores = record.get("class_scores", {})
    if isinstance(scores, dict) and scores:
        return max(float(value) for value in scores.values())

    return 0.0


def get_truth(record):
    """Read the image-level ground-truth label."""
    for key in ("truth", "ground_truth", "actual", "has_target"):
        if key in record:
            value = record[key]

            if isinstance(value, str):
                return value.strip().lower() in (
                    "true", "1", "yes", "positive"
                )

            return bool(value)

    raise ValueError(
        "A validation prediction is missing its image-level truth label. "
        f"Available fields: {list(record.keys())}"
    )


def evaluate_band(records, low, high):
    """
    Evaluate an image-level decision policy.

    score < low: no flag
    low <= score < high: human review
    score >= high: automatic flag

    Reviewed images incur the review cost and are excluded from
    the automatic-decision confusion matrix.
    """
    low, high = sorted((float(low), float(high)))

    tp = fp = tn = fn = reviewed = 0

    for record in records:
        score = get_score(record)
        actual = get_truth(record)

        if low <= score < high:
            reviewed += 1
        elif score >= high:
            if actual:
                tp += 1
            else:
                fp += 1
        else:
            if actual:
                fn += 1
            else:
                tn += 1

    total = len(records)
    cost = FN_COST * fn + FP_COST * fp + REVIEW_COST * reviewed

    return {
        "images": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "reviewed": reviewed,
        "review_share": reviewed / total if total else 0.0,
        "cost_total": cost,
        "cost_per_1000": cost * 1000 / total if total else 0.0,
        "low": low,
        "high": high,
    }


def make_cost_curve(records, low, high):
    """Plot validation cost for a single threshold without human review."""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []

    for threshold in thresholds:
        total_cost = 0.0

        for record in records:
            score = get_score(record)
            actual = get_truth(record)
            predicted_positive = score >= threshold

            if actual and not predicted_positive:
                total_cost += FN_COST
            elif not actual and predicted_positive:
                total_cost += FP_COST

        total_cost = (
            total_cost * 1000 / len(records) if records else 0.0
        )
        costs.append(total_cost)

    # Mark the current upper threshold: the automatic-flag boundary.
    current = min(
        range(len(thresholds)),
        key=lambda i: abs(thresholds[i] - high),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=costs,
            mode="lines",
            name="Single-threshold cost",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[high],
            y=[costs[current]],
            mode="markers",
            name=f"Current upper threshold ({high:.2f})",
            marker=dict(size=11),
        )
    )
    fig.add_vline(
        x=low,
        line_dash="dash",
        annotation_text="Review starts",
    )
    fig.add_vline(
        x=high,
        line_dash="dash",
        annotation_text="Auto-flag starts",
    )

    fig.update_layout(
        title="Validation cost versus confidence threshold",
        xaxis_title="Confidence threshold",
        yaxis_title="Cost per 1,000 images (€)",
        template="plotly_white",
        height=420,
        margin=dict(l=55, r=25, t=65, b=55),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def dashboard_update(low, high):
    """Update all decision-policy outputs together."""
    low, high = sorted((float(low), float(high)))
    stats = evaluate_band(VALIDATION_RECORDS, low, high)

    summary = (
        f"**Validation-set estimate**\n\n"
        f"- Images evaluated: {stats['images']}\n"
        f"- False negatives: {stats['fn']}\n"
        f"- False positives: {stats['fp']}\n"
        f"- True positives: {stats['tp']}\n"
        f"- True negatives: {stats['tn']}\n"
        f"- Images sent to review: {stats['reviewed']}\n"
        f"- Human-review share: {stats['review_share']:.1%}\n"
        f"- Total estimated cost: €{stats['cost_total']:.2f}\n"
        f"- **Estimated cost per 1,000 images: "
        f"€{stats['cost_per_1000']:.2f}**\n\n"
        f"Current band: {low:.2f} to {high:.2f}. "
        f"Assumed costs: FN €{FN_COST:.0f}, FP €{FP_COST:.0f}, "
        f"review €{REVIEW_COST:.0f}."
    )

    matrix = pd.DataFrame(
        [
            [stats["tn"], stats["fp"]],
            [stats["fn"], stats["tp"]],
        ],
        index=["Actual negative", "Actual positive"],
        columns=["Automatically not flagged", "Automatically flagged"],
    )

    curve = make_cost_curve(VALIDATION_RECORDS, low, high)
    return summary, matrix, curve


def predict_image(image, low, high):
    """Run the saved detector on an uploaded image."""
    if image is None:
        return None, "Upload a road image to begin.", pd.DataFrame(
            columns=["Class", "Confidence", "Bounding box"]
        )

    low, high = sorted((float(low), float(high)))

    # Use the lower threshold to show detections relevant to the
    # current policy, including detections in the review band.
    result = MODEL.predict(
        source=image,
        conf=max(0.001, low),
        verbose=False,
    )[0]

    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated = Image.fromarray(annotated_rgb)

    detections = []
    scores = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = result.names.get(class_id, str(class_id))

            scores.append(confidence)
            coordinates = [
                round(float(value), 1)
                for value in box.xyxy[0].tolist()
            ]

            detections.append(
                {
                    "Class": class_name,
                    "Confidence": round(confidence, 4),
                    "Bounding box": str(coordinates),
                }
            )

    image_score = max(scores, default=0.0)

    if image_score >= high:
        decision = "AUTOMATIC FLAG"
        explanation = "At least one detection meets the upper threshold."
    elif image_score >= low:
        decision = "HUMAN REVIEW"
        explanation = "The strongest detection falls inside the review band."
    else:
        decision = "NO FLAG"
        explanation = "No detection reaches the lower threshold."

    message = (
        f"**Decision: {decision}**\n\n"
        f"Maximum detection confidence: {image_score:.3f}\n\n"
        f"Lower threshold: {low:.2f} | Upper threshold: {high:.2f}\n\n"
        f"{explanation}\n\n"
        "This is an image-level demonstration. The model's confidence "
        "is not a calibrated probability that the road is damaged."
    )

    return annotated, message, pd.DataFrame(
        detections,
        columns=["Class", "Confidence", "Bounding box"],
    )


# Load saved artifacts. This app never trains the model.
for required_path in (WEIGHTS_PATH, METRICS_PATH, VALIDATION_PATH):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {required_path}"
        )

METRICS = read_json(METRICS_PATH)
VALIDATION_RECORDS = load_records(VALIDATION_PATH)
MODEL = YOLO(str(WEIGHTS_PATH))

costs = METRICS.get("costs_eur", {})
FN_COST = float(costs.get("false_negative", FN_COST))
FP_COST = float(costs.get("false_positive", FP_COST))
REVIEW_COST = float(costs.get("human_review", REVIEW_COST))

saved_policy = (
    METRICS.get("image_level_test_metrics", {})
    .get("fine_tuned_at_validation_selected_band", {})
)
DEFAULT_LOW = float(saved_policy.get("low_threshold", 0.05))
DEFAULT_HIGH = float(saved_policy.get("high_threshold", 0.35))

test_metrics = METRICS.get("object_detection_test_metrics", {})
per_class = test_metrics.get("per_class", {})

class_rows = []
for class_name in CLASSES:
    values = per_class.get(class_name, {})
    class_rows.append(
        {
            "Class": class_name,
            "Precision": values.get("precision"),
            "Recall": values.get("recall"),
            "AP50": values.get("ap50"),
            "mAP50-95": values.get("map50_95"),
        }
    )

class_table = pd.DataFrame(class_rows)

overall_markdown = (
    f"**Held-out test-set object-detection metrics**\n\n"
    f"- Precision: {test_metrics.get('precision', float('nan')):.3f}\n"
    f"- Recall: {test_metrics.get('recall', float('nan')):.3f}\n"
    f"- mAP@50: {test_metrics.get('map50', float('nan')):.3f}\n"
    f"- mAP@50–95: {test_metrics.get('map50_95', float('nan')):.3f}\n\n"
    f"Dataset: {METRICS.get('dataset', {}).get('test_images', 'Unknown')} "
    "test images. The image-level cost policy is evaluated separately "
    "using validation predictions."
)

with gr.Blocks(title="Road Damage Detection and Decision Support") as demo:
    gr.Markdown(
        "# Road Damage Detection and Decision Support\n"
        "Upload a road image, inspect detections, and explore the trade-off "
        "between automatic decisions and human review."
    )

    with gr.Tab("Image inspection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload road image",
                )
                image_low = gr.Slider(
                    0.0, 1.0, value=DEFAULT_LOW, step=0.01,
                    label="Lower confidence threshold",
                )
                image_high = gr.Slider(
                    0.0, 1.0, value=DEFAULT_HIGH, step=0.01,
                    label="Upper confidence threshold",
                )
                predict_button = gr.Button(
                    "Detect road damage",
                    variant="primary",
                )

            with gr.Column():
                image_output = gr.Image(
                    type="pil",
                    label="Predictions",
                )
                decision_output = gr.Markdown()
                detections_output = gr.Dataframe(
                    label="Detected objects",
                    interactive=False,
                )

        predict_button.click(
            fn=predict_image,
            inputs=[image_input, image_low, image_high],
            outputs=[image_output, decision_output, detections_output],
        )

    with gr.Tab("Decision policy and costs"):
        gr.Markdown(
            "## Adjust the escalation band\n"
            "Images below the lower threshold are not flagged; images "
            "between the thresholds go to human review; images at or "
            "above the upper threshold are automatically flagged.\n\n"
            "The cost estimates below use the saved **validation predictions** "
            "and the cost assumptions specified in the project."
        )

        with gr.Row():
            low_slider = gr.Slider(
                0.0, 1.0, value=DEFAULT_LOW, step=0.01,
                label="Lower threshold — start human review",
            )
            high_slider = gr.Slider(
                0.0, 1.0, value=DEFAULT_HIGH, step=0.01,
                label="Upper threshold — automatic flag",
            )

        cost_summary = gr.Markdown()
        confusion_table = gr.Dataframe(
            label="Automatic-decision confusion matrix (reviewed cases excluded)",
            interactive=False,
        )
        cost_plot = gr.Plot(label="Cost curve")

        low_slider.change(
            fn=dashboard_update,
            inputs=[low_slider, high_slider],
            outputs=[cost_summary, confusion_table, cost_plot],
        )
        high_slider.change(
            fn=dashboard_update,
            inputs=[low_slider, high_slider],
            outputs=[cost_summary, confusion_table, cost_plot],
        )

        demo.load(
            fn=dashboard_update,
            inputs=[low_slider, high_slider],
            outputs=[cost_summary, confusion_table, cost_plot],
        )

    with gr.Tab("Model evaluation"):
        gr.Markdown("## Overall test-set performance")
        gr.Markdown(overall_markdown)
        gr.Markdown("## Per-class performance")
        gr.Dataframe(value=class_table, interactive=False)
        gr.Markdown(
            "AP and mAP are object-detection metrics. The cost dashboard "
            "uses a separate image-level rule: whether an image contains "
            "at least one target and the maximum target detection confidence. "
            "These metrics describe different evaluation tasks."
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
