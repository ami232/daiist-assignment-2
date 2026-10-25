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

import json
import math
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

with (ARTIFACTS / "business_policy.json").open(encoding="utf-8") as f:
    POLICY = json.load(f)

MODEL_PATH = ROOT / POLICY["model_path"]
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model missing: {MODEL_PATH}")

MODEL = YOLO(str(MODEL_PATH))
ACTIVE_CLASSES = set(int(x) for x in POLICY["active_classes"])
FLOOR = float(POLICY["inference_confidence_floor"])

VALIDATION = pd.read_csv(ARTIFACTS / "validation_image_scores.csv")
TEST_METRICS = pd.read_csv(ARTIFACTS / "test_detection_metrics.csv")
TEST_COMPARISON = pd.read_csv(ARTIFACTS / "test_business_comparison.csv")

if not {"actual_damage", "damage_score"}.issubset(VALIDATION.columns):
    raise ValueError("validation_image_scores.csv is missing required columns")

ACTUAL = VALIDATION["actual_damage"].to_numpy(dtype=int)
SCORES = VALIDATION["damage_score"].to_numpy(dtype=float)

# Model-comparison information from actual training histories, when available.
def read_final_history(filename):
    filepath = ARTIFACTS / filename
    if not filepath.is_file():
        return None
    history = pd.read_csv(filepath)
    history.columns = history.columns.str.strip()
    return history


def model_comparison_table():
    rows = []
    for label, filename in [
        ("Fine-tuned YOLO11n (end of training)", "asphalt_yolo11n_results.csv"),
        ("YOLO11n trained from scratch (end of training)", "scratch_results.csv"),
    ]:
        history = read_final_history(filename)
        if history is not None and len(history):
            last = history.iloc[-1]
            rows.append({
                "Model": label,
                "Validation mAP50": round(float(last["metrics/mAP50(B)"]), 3),
                "Validation mAP50-95": round(float(last["metrics/mAP50-95(B)"]), 3),
            })
    rows.append({"Model": "Predict nothing", "Validation mAP50": 0.0, "Validation mAP50-95": 0.0})
    return pd.DataFrame(rows)


def image_inspector(image_array, low, high):
    if image_array is None:
        return None, "Upload a road image to run the trained YOLO11n model."
    low, high = float(low), float(high)
    if low > high:
        return image_array, "The Pass threshold must be less than or equal to the Flag threshold."

    image = Image.fromarray(np.asarray(image_array).astype(np.uint8)).convert("RGB")
    result = MODEL.predict(
        source=np.asarray(image), conf=FLOOR, imgsz=640,
        device="cpu", verbose=False,
    )[0]

    max_active_score = 0.0
    detections = []
    for box in result.boxes:
        cls = int(box.cls.item())
        score = float(box.conf.item())
        if cls in ACTIVE_CLASSES:
            max_active_score = max(max_active_score, score)
        detections.append((cls, score, [float(x) for x in box.xyxy[0].tolist()]))

    if max_active_score < low:
        decision = "PASS"
        meaning = "No actionable defect detected above the lower threshold. Not a road-safety guarantee."
    elif max_active_score < high:
        decision = "HUMAN REVIEW"
        meaning = "Possible actionable damage needs a person to inspect the image."
    else:
        decision = "FLAG"
        meaning = "High-confidence actionable damage; prioritize inspection."

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    # Keep visible boxes readable; the decision score still uses all predictions down to FLOOR.
    visible_threshold = max(FLOOR, 0.10)
    visible_count = 0
    for cls, score, xyxy in detections:
        if score < visible_threshold:
            continue
        name = str(MODEL.names[cls])
        color = "#ea4335" if cls in ACTIVE_CLASSES else "#3776b6"
        draw.rectangle(xyxy, outline=color, width=3)
        label = f"{name} {score:.2f}"
        draw.text((max(0, xyxy[0]), max(0, xyxy[1] - 14)), label, fill=color)
        visible_count += 1

    summary = (
        f"### {decision}\n"
        f"**Actionable damage score:** {max_active_score:.3f}  |  "
        f"**Pass threshold:** {low:.2f}  |  **Flag threshold:** {high:.2f}\n\n"
        f"{meaning}\n\n"
        f"Displayed **{visible_count}** boxes at confidence ≥{visible_threshold:.2f}; "
        f"the decision uses active-class predictions down to {FLOOR:.2f}. "
        "The `Repair` class is excluded from the actionable-damage score."
    )
    return annotated, summary


def performance_chart(metric):
    data = TEST_METRICS.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(data["class"], data[metric])
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_title(f"Final test detection performance — {metric}")
    fig.tight_layout()
    return fig


def decision_counts(low, high):
    low, high = float(low), float(high)
    if low > high:
        raise ValueError("The Pass threshold cannot exceed the Flag threshold.")
    passed = SCORES < low
    reviewed = (SCORES >= low) & (SCORES < high)
    flagged = SCORES >= high
    fn = int((passed & (ACTUAL == 1)).sum())
    fp = int((flagged & (ACTUAL == 0)).sum())
    return int(passed.sum()), int(reviewed.sum()), int(flagged.sum()), fn, fp


def policy_cost(low, high, cost_fn, cost_fp, cost_review):
    passed, reviewed, flagged, fn, fp = decision_counts(low, high)
    return fn * cost_fn + fp * cost_fp + reviewed * cost_review


def cost_simulator(low, high, fn_cost, fp_cost, review_cost):
    low, high = float(low), float(high)
    if low > high:
        return "Lower threshold must be ≤ upper threshold.", None

    passed, reviewed, flagged, fn, fp = decision_counts(low, high)
    total_cost = policy_cost(low, high, fn_cost, fp_cost, review_cost)
    summary = (
        f"### Estimated validation cost: €{total_cost:,.0f}\n"
        f"**Pass:** {passed} · **Review:** {reviewed} · **Flag:** {flagged} "
        f"(out of {len(SCORES)} images)\n\n"
        f"**Missed actionable images:** {fn} · **Unnecessary flags:** {fp} · "
        f"**Review rate:** {reviewed / len(SCORES):.1%}\n\n"
        "Costs are hypothetical. Human reviewers are assumed to make correct decisions."
    )

    highs = np.round(np.arange(max(low, 0.01), 1.011, 0.02), 2)
    costs = [policy_cost(low, h, fn_cost, fp_cost, review_cost) for h in highs]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(highs, costs, marker=".")
    ax.axvline(high, linestyle="--", label=f"Selected flag threshold: {high:.2f}")
    ax.set_xlabel("Flag threshold (Pass threshold held fixed)")
    ax.set_ylabel("Estimated cost (€)")
    ax.set_title("Cost versus flag threshold — validation set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return summary, fig


def review_staffing(low, high, daily_images, reviewers, hours_per_reviewer, minutes_per_review):
    if float(low) > float(high):
        return "Lower threshold must be ≤ upper threshold."
    _, reviewed, _, _, _ = decision_counts(low, high)
    review_rate = reviewed / len(SCORES)
    volume = max(0, float(daily_images))
    current_staff = max(0, int(reviewers))
    hours = max(0.01, float(hours_per_reviewer))
    minutes = max(0.01, float(minutes_per_review))
    reviews_per_day = volume * review_rate
    hours_required = reviews_per_day * minutes / 60
    hours_available = current_staff * hours
    staff_required = math.ceil(hours_required / hours)
    status = "Within assumed capacity" if hours_required <= hours_available else "Exceeds assumed capacity"
    return (
        f"### {status}\n"
        f"**Historical validation review rate:** {review_rate:.1%}\n\n"
        f"**Expected reviews per day:** {reviews_per_day:,.0f}\n\n"
        f"**Reviewer-hours required:** {hours_required:.1f} · "
        f"**Available:** {hours_available:.1f}\n\n"
        f"**Staff required:** {staff_required} · **Currently available:** {current_staff}\n\n"
        "This projection assumes the validation-set case mix and constant review time."
    )


DEFAULT_LOW = float(POLICY["lower_threshold"])
DEFAULT_HIGH = float(POLICY["upper_threshold"])
COSTS = POLICY["costs_eur"]
OPS = POLICY["operations"]

with gr.Blocks(title="Asphalt Inspection — YOLO11") as demo:
    gr.Markdown(
        "# Asphalt Road Inspection — YOLO11n\n"
        "Detect road damage, explore model errors, and simulate human review. "
        "Trained model and saved evaluation artifacts only — no retraining."
    )

    with gr.Tab("Image Inspection"):
        gr.Markdown("Upload a road photograph to visualize YOLO detections and the three-way decision.")
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload road photograph", sources=["upload"])
            image_output = gr.Image(type="pil", label="Detected road conditions", interactive=False)
        with gr.Row():
            inspect_low = gr.Slider(0.0, 1.0, value=DEFAULT_LOW, step=0.01, label="Pass threshold")
            inspect_high = gr.Slider(0.0, 1.0, value=DEFAULT_HIGH, step=0.01, label="Flag threshold")
        inspect_button = gr.Button("Inspect image", variant="primary")
        inspect_result = gr.Markdown("Upload an image, then click Inspect image.")
        inspect_button.click(
            fn=image_inspector,
            inputs=[image_input, inspect_low, inspect_high],
            outputs=[image_output, inspect_result],
        )

    with gr.Tab("Model Performance"):
        gr.Markdown(
            "Final **test-set** object-detection metrics. These are different from "
            "the **image-level** Pass / Review / Flag business calculations."
        )
        metric_menu = gr.Dropdown(
            choices=["precision", "recall", "mAP50", "mAP50-95"],
            value="mAP50", label="Metric",
        )
        metric_plot = gr.Plot(value=performance_chart("mAP50"), label="Per-class metrics")
        metric_menu.change(performance_chart, inputs=[metric_menu], outputs=[metric_plot])
        gr.Dataframe(value=TEST_METRICS, interactive=False, label="Test metrics per class")
        gr.Markdown(
            "### Baseline comparison\n"
            "History values below are measured on the validation split at the end of training; "
            "the scratch model used the same 30-epoch training budget. "
            "The original COCO model is only a qualitative zero-shot baseline because its "
            "classes do not match asphalt categories."
        )
        gr.Dataframe(value=model_comparison_table(), interactive=False, label="Validation training-history comparison")

    with gr.Tab("Cost Simulator"):
        gr.Markdown(
            "Adjust thresholds and error costs on the **628 validation images**. "
            "No new predictions or training are performed."
        )
        with gr.Row():
            cost_low = gr.Slider(0.0, 1.0, value=DEFAULT_LOW, step=0.01, label="Pass threshold")
            cost_high = gr.Slider(0.0, 1.0, value=DEFAULT_HIGH, step=0.01, label="Flag threshold")
        with gr.Row():
            fn_cost = gr.Number(value=COSTS["false_negative"], label="Missed damage (€)", minimum=0)
            fp_cost = gr.Number(value=COSTS["false_positive"], label="False flag (€)", minimum=0)
            review_cost = gr.Number(value=COSTS["human_review"], label="Human review (€)", minimum=0)
        cost_button = gr.Button("Recalculate costs", variant="primary")
        cost_summary = gr.Markdown()
        cost_plot = gr.Plot(label="Validation cost curve")
        cost_inputs = [cost_low, cost_high, fn_cost, fp_cost, review_cost]
        cost_button.click(cost_simulator, inputs=cost_inputs, outputs=[cost_summary, cost_plot])
        for control in cost_inputs:
            control.change(cost_simulator, inputs=cost_inputs, outputs=[cost_summary, cost_plot])
        demo.load(cost_simulator, inputs=cost_inputs, outputs=[cost_summary, cost_plot])
        sensitivity_path = ARTIFACTS / "cost_sensitivity.csv"
        if sensitivity_path.is_file():
            gr.Dataframe(value=pd.read_csv(sensitivity_path), interactive=False, label="Saved sensitivity analysis")

    with gr.Tab("Human Review Capacity"):
        gr.Markdown("Estimate staffing using validation review rates and your operating assumptions.")
        with gr.Row():
            staff_low = gr.Slider(0.0, 1.0, value=DEFAULT_LOW, step=0.01, label="Pass threshold")
            staff_high = gr.Slider(0.0, 1.0, value=DEFAULT_HIGH, step=0.01, label="Flag threshold")
        with gr.Row():
            daily_images = gr.Number(value=OPS["daily_images"], label="Images per day", minimum=0)
            reviewers = gr.Number(value=OPS["reviewers"], label="Available reviewers", minimum=0, precision=0)
        with gr.Row():
            hours = gr.Number(value=OPS["hours_per_reviewer"], label="Productive hours per reviewer", minimum=0.1)
            minutes = gr.Number(value=OPS["minutes_per_review"], label="Minutes per reviewed image", minimum=0.1)
        staffing_button = gr.Button("Calculate staffing", variant="primary")
        staffing_summary = gr.Markdown()
        staffing_inputs = [staff_low, staff_high, daily_images, reviewers, hours, minutes]
        staffing_button.click(review_staffing, inputs=staffing_inputs, outputs=[staffing_summary])
        for control in staffing_inputs:
            control.change(review_staffing, inputs=staffing_inputs, outputs=[staffing_summary])
        demo.load(review_staffing, inputs=staffing_inputs, outputs=[staffing_summary])
        gr.Markdown(
            "### Final test business comparison\n"
            "Test results below use the policy chosen on validation data. "
            "The cost figures are hypothetical; they assume correct human review and do not "
            "fully capture downstream costs of flagged roads."
        )
        gr.Dataframe(value=TEST_COMPARISON, interactive=False, label="Test cost comparison")

# Running through main.py app executes this file; training is never invoked here.
demo.launch(server_name="0.0.0.0")
