import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

from operating_policy import evaluate, COST_REVIEW, COST_INSPECTION, COST_MISSED_DEFECT


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "best.pt"

CRITICAL = {
    "fishplate_boltmissing",
    "track_boltmissing",
    "track_crack",
}


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


if not WEIGHTS.exists():
    raise FileNotFoundError("Missing weights/best.pt. Run training first.")

model = YOLO(str(WEIGHTS))

valid_data = load_json("validation_predictions.json")
test_data = load_json("test_predictions.json")
detection = load_json("detection_metrics.json")
policy = load_json("policy_results.json")

DEFAULT_LOW = policy["validation"]["low_threshold"]
DEFAULT_HIGH = policy["validation"]["high_threshold"]


def decision(score, low, high):
    if score < low:
        return "PASS"
    if score < high:
        return "HUMAN REVIEW"
    return "FAIL"


def inspection(image_path, low, high):
    if image_path is None:
        return None, "Upload a railway image to inspect it."

    result = model.predict(
        source=image_path,
        conf=0.001,
        imgsz=416,
        device="cpu",
        verbose=False,
    )[0]

    boxes = result.boxes
    defect_scores = []
    detections = []

    for cls_id, confidence in zip(
        boxes.cls.tolist(), boxes.conf.tolist()
    ):
        name = model.names[int(cls_id)]
        confidence = float(confidence)

        if confidence >= low:
            detections.append(f"{name}: {confidence:.1%}")

        if name in CRITICAL:
            defect_scores.append(confidence)

    score = max(defect_scores, default=0.0)
    outcome = decision(score, low, high)

    # Display boxes above the lower threshold.
    shown = model.predict(
        source=image_path,
        conf=max(low, 0.001),
        imgsz=416,
        device="cpu",
        verbose=False,
    )[0]

    annotated = Image.fromarray(shown.plot()[:, :, ::-1])

    details = (
        f"### Decision: {outcome}\n\n"
        f"**Maximum critical-defect confidence:** {score:.1%}\n\n"
        f"**Detected objects:**\n"
    )

    details += "\n".join(
        f"- {item}" for item in detections
    ) if detections else "\n- No objects above the display threshold."

    details += (
        "\n\n*Decision-support only. A PASS does not certify "
        "that the track is safe.*"
    )
    return annotated, details


def outcome_table(records, low, high):
    counts = {
        "PASS": [0, 0],
        "HUMAN REVIEW": [0, 0],
        "FAIL": [0, 0],
    }

    for record in records:
        outcome = decision(record["score"], low, high)
        column = 1 if record["positive"] else 0
        counts[outcome][column] += 1

    return [
        [name, values[0], values[1], sum(values)]
        for name, values in counts.items()
    ]


def confusion_matrix_table(records, low, high):
    """
    Binary confusion matrix for automatic decisions.

    REVIEW cases are shown separately, because they
    are resolved by a human rather than automatically.
    """
    tp = fp = fn = tn = 0
    review_positive = review_negative = 0

    for record in records:
        score = record["score"]
        actual = record["positive"]

        if low <= score < high:
            if actual:
                review_positive += 1
            else:
                review_negative += 1
            continue

        predicted_positive = score >= high

        if predicted_positive and actual:
            tp += 1
        elif predicted_positive and not actual:
            fp += 1
        elif not predicted_positive and actual:
            fn += 1
        else:
            tn += 1

    matrix = [
        ["Actual defect", tp, fn],
        ["Actual no defect", fp, tn],
    ]

    explanation = (
        f"**Automatic decisions:** TP={tp}, FP={fp}, "
        f"FN={fn}, TN={tn}. "
        f"**Human review:** {review_positive} actual defects "
        f"and {review_negative} non-defective images. "
        "Reviewed cases are excluded from the binary matrix."
    )

    return matrix, explanation

def cost_chart(low, high):
    # Sweep lower threshold while keeping upper threshold fixed.
    grid = sorted(set(
        [0.001, 0.01]
        + [round(i / 100, 2) for i in range(5, 100, 5)]
        + [low, DEFAULT_LOW]
    ))

    xs = [x for x in grid if x <= high]
    ys = [
        evaluate(valid_data, x, high)["cost_per_1000_eur"]
        for x in xs
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(xs, ys, marker="o", markersize=3, label="Validation cost")

    current = evaluate(valid_data, low, high)
    ax.scatter(
        [low], [current["cost_per_1000_eur"]],
        marker="*", s=170, zorder=5, label="Current settings"
    )


    # Show the selected operating point on a separate reference line.
    original = evaluate(
        valid_data, DEFAULT_LOW, DEFAULT_HIGH
    )

    ax.axhline(
        y=original["cost_per_1000_eur"],
        linestyle="--",
        alpha=0.7,
        label=(
            f"Selected policy: low={DEFAULT_LOW:.2f}, "
            f"high={DEFAULT_HIGH:.2f}"
        ),
    )

    if abs(high - DEFAULT_HIGH) < 0.00001:
        ax.scatter(
            [DEFAULT_LOW],
            [original["cost_per_1000_eur"]],
            marker="D",
            s=65,
            label="Selected operating point",
        )

    ax.set_xlabel("Lower confidence threshold")
    ax.set_ylabel("Expected cost (€ per 1,000 images)")
    ax.set_title(
        f"Cost vs. lower threshold (upper threshold = {high:.2f})"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def update_dashboard(low, high):
    if low >= high:
        return (
            "The lower threshold must be below the upper threshold.",
            [],
            None,
            [],
            "",
        )

    stats = evaluate(test_data, low, high)
    n = len(test_data)
    counts = stats["decisions"]

    summary = f"""
### Test-set operating results

**PASS:** {counts['PASS']} / {n} images

**HUMAN REVIEW:** {counts['REVIEW']} / {n} images

**FAIL:** {counts['FAIL']} / {n} images

**Expected cost:** €{stats['cost_per_1000_eur']:,.2f} per 1,000 images

**Human review share:** {stats['review_share']:.1%}

**Estimated reviews per day:** {stats['daily_reviews']}

**Review staffing:** {stats['review_staff_fte']:.2f} FTE

**Missed defects sent to PASS:** {stats['errors']['missed_defects']}

**Unnecessary FAIL inspections:** {stats['errors']['unnecessary_inspections']}

*The table below shows actual defect status versus the
three decisions. Human review is assumed perfect in the
cost model; that assumption is not a measured result.*
"""

    cm_table, cm_explanation = confusion_matrix_table(
        test_data, low, high
    )

    return (
        summary,
        outcome_table(test_data, low, high),
        cost_chart(low, high),
        cm_table,
        cm_explanation,
    )

per_class = [
    [
        name,
        round(values["precision"], 4),
        round(values["recall"], 4),
        round(values["AP50"], 4),
    ]
    for name, values in detection["per_class"].items()
]


with gr.Blocks(title="Railway AI Inspection Dashboard") as demo:
    gr.Markdown(
        "# Railway Track Inspection — AI Decision Support\n"
        "YOLO11n trained on Railway Track Fault Detection v1. "
        "Results are experimental, not railway safety certification."
    )

    with gr.Row():
        low_slider = gr.Slider(
            minimum=0.001,
            maximum=0.95,
            value=DEFAULT_LOW,
            step=0.001,
            label="Lower threshold: PASS / HUMAN REVIEW",
        )
        high_slider = gr.Slider(
            minimum=0.01,
            maximum=1.0,
            value=DEFAULT_HIGH,
            step=0.01,
            label="Upper threshold: HUMAN REVIEW / FAIL",
        )

    with gr.Tab("Business dashboard"):
        summary = gr.Markdown()
        matrix = gr.Dataframe(
            headers=[
                "Decision",
                "No defect (actual)",
                "Defect (actual)",
                "Total",
            ],
            datatype=["str", "number", "number", "number"],
            interactive=False,
            label="Decision matrix — 105 test images",
        )
        plot = gr.Plot(label="Cost against threshold")

        gr.Markdown("### Binary confusion matrix")

        confusion_matrix = gr.Dataframe(
            headers=["Actual / Predicted", "FAIL", "PASS"],
            datatype=["str", "number", "number"],
            interactive=False,
            label="Automatic decisions only",
        )

        confusion_explanation = gr.Markdown()

        gr.Markdown(
            "### Model performance by class\n"
            f"Overall detection mAP50: "
            f"**{detection['mAP50']:.1%}**"
        )
        gr.Dataframe(
            value=per_class,
            headers=["Class", "Precision", "Recall", "AP50"],
            datatype=["str", "number", "number", "number"],
            interactive=False,
        )

        gr.Markdown(
            f"**Cost assumptions:** €{COST_REVIEW} desk review; "
            f"€{COST_INSPECTION} physical inspection; "
            f"€{COST_MISSED_DEFECT:,} expected missed-defect cost. "
            "Assumed: 1,000 images/day, 3 minutes per desk review, "
            "6 productive hours per inspector."
        )

    with gr.Tab("Live image inspection"):
        input_image = gr.Image(
            type="filepath",
            label="Upload a railway track image",
        )
        inspect_button = gr.Button("Inspect image")
        annotated_image = gr.Image(label="YOLO detections")
        decision_text = gr.Markdown()

        inspect_button.click(
            fn=inspection,
            inputs=[input_image, low_slider, high_slider],
            outputs=[annotated_image, decision_text],
        )

    low_slider.change(
        fn=update_dashboard,
        inputs=[low_slider, high_slider],
        outputs=[
            summary,
            matrix,
            plot,
            confusion_matrix,
            confusion_explanation,
        ],
    )

    high_slider.change(
        fn=update_dashboard,
        inputs=[low_slider, high_slider],
        outputs=[
            summary,
            matrix,
            plot,
            confusion_matrix,
            confusion_explanation,
        ],
    )

    demo.load(
        fn=update_dashboard,
        inputs=[low_slider, high_slider],
        outputs=[
            summary,
            matrix,
            plot,
            confusion_matrix,
            confusion_explanation,
        ],
    )


if __name__ == "__main__":
    demo.launch()
