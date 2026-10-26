
from pathlib import Path
import json

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from ultralytics import YOLO
from decision import BASE_RATE, FN_EUR, REVIEW_EUR, outcome, triage

ROOT = Path(__file__).resolve().parent


def load(filename):
    path = ROOT / "results" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run main.py train first."
        )
    return json.loads(path.read_text())


weights = ROOT / "weights" / "helmet_best.pt"
if not weights.exists():
    raise FileNotFoundError(
        "Train your model first: weights/helmet_best.pt is missing."
    )

# The app loads trained weights. It never trains a model.
model = YOLO(str(weights))
name_map = {int(i): str(n) for i, n in model.names.items()}

nohat = next(
    i for i, n in name_map.items()
    if "".join(c.lower() for c in n if c.isalnum()) == "nohardhat"
)

predictions = load("test_predictions.json")
metrics = load("per_class_metrics.json")
curve = load("cost_curve.json")
policy = load("policy.json")
analysis = load("analysis.json")

scores = np.array([p["score"] for p in predictions])
truth = np.array([bool(p["positive"]) for p in predictions])


def dashboard(low, high):
    low, high = sorted((float(low), float(high)))
    result = outcome(scores, truth, low, high, BASE_RATE)
    actions = triage(scores, low, high)

    counts = [
        [
            int(np.sum((truth == actual) & (actions == action)))
            for action in (0, 1, 2)
        ]
        for actual in (False, True)
    ]

    cm = go.Figure(go.Heatmap(
        z=counts,
        text=counts,
        texttemplate="%{text}",
        colorscale="Blues",
        x=["PASS", "REVIEW", "FAIL"],
        y=["Actually safe", "Actual violation"],
        showscale=False
    ))
    cm.update_layout(
        title="Test set: actual vs decision",
        height=340
    )
    cm.update_yaxes(autorange="reversed")

    chart = go.Figure(go.Scatter(
        x=[v["threshold"] for v in curve],
        y=[v["eur_per_1000"] for v in curve],
        mode="lines",
        name="Model alone"
    ))

    chart.add_vline(
        x=float(
            analysis["best_single_threshold_validation"]["threshold"]
        ),
        line_dash="dash",
        annotation_text="Best single threshold"
    )
    chart.add_vline(
        x=low, line_dash="dot", annotation_text="Lower"
    )
    chart.add_vline(
        x=high, line_dash="dot", annotation_text="Upper"
    )

    chart.update_layout(
        title="Cost vs threshold (validation)",
        xaxis_title="Confidence",
        yaxis_title="Euros per 1,000 cases",
        height=340
    )

    explanation = (
        f"**PASS:** below {low:.2f} · "
        f"**REVIEW:** {low:.2f}–{high:.2f} · "
        f"**FAIL:** at least {high:.2f}  \n"
        f"**Expected cost:** "
        f"€{result['eur_per_1000']:,.2f} per 1,000 cases  \n"
        f"**Human review:** {result['review_share']:.1%} / "
        f"{result['review_images_per_day']:.1f} images per day  \n"
        f"**Workload:** {result['review_minutes_per_day']:.1f} "
        f"minutes per day; "
        f"{result['review_staff_fte']:.3f} employee equivalents  \n"
        f"**Counts:** TP={result['TP']}, FP={result['FP']}, "
        f"TN={result['TN']}, FN={result['FN']}, "
        f"reviewed positives={result['review_positive']}, "
        f"reviewed negatives={result['review_negative']}  \n"
        f"**Alternatives per 1,000:** "
        f"Do nothing €{1000 * BASE_RATE * FN_EUR:,.0f}; "
        f"review everyone €{1000 * REVIEW_EUR:,.0f}; "
        f"model alone (validation optimum) "
        f"€{analysis['best_single_threshold_validation']['eur_per_1000']:,.2f}.  \n"
        "*Costs assume 10% violations and perfect human reviewers. "
        "The matrix shows unweighted test image counts.*"
    )

    return explanation, cm, chart


def predict(image, low, high):
    if image is None:
        return None, "Please upload an image."

    low, high = sorted((float(low), float(high)))

    result = model.predict(
        image,
        conf=max(low, 0.001),
        imgsz=640,
        verbose=False
    )[0]

    violation_scores = [
        float(b.conf.item())
        for b in result.boxes
        if int(b.cls.item()) == nohat
    ]

    score = max(violation_scores, default=0.0)

    if score < low:
        action = "PASS"
    elif score < high:
        action = "HUMAN REVIEW"
    else:
        action = "FAIL"

    marked = Image.fromarray(result.plot()[..., ::-1])

    return marked, (
        f"**{action}** — highest NO-Hardhat "
        f"confidence: {score:.3f}"
    )


rows = []

for name in name_map.values():
    values = metrics["per_class"].get(name, {})
    rows.append(
        [name] + [
            round(values[key], 3) if key in values else "N/A"
            for key in (
                "precision", "recall", "AP50", "mAP50_95"
            )
        ]
    )

low0 = float(policy["lower"])
high0 = float(policy["upper"])

summary0, matrix0, chart0 = dashboard(low0, high0)

with gr.Blocks(title="Construction Helmet Safety") as demo:
    gr.Markdown(
        "# Construction Helmet Safety Inspector\n"
        "YOLO11n · Real test data · No startup training"
    )

    gr.Markdown(
        f"**Overall AP50:** {metrics['AP50']:.3f}; "
        f"**mAP50-95:** {metrics['mAP50_95']:.3f}. "
        "Positive class: **NO-Hardhat**."
    )

    with gr.Row():
        lower = gr.Slider(
            0, 1, value=low0, step=0.01,
            label="Lower threshold (PASS / REVIEW)"
        )
        upper = gr.Slider(
            0, 1, value=high0, step=0.01,
            label="Upper threshold (REVIEW / FAIL)"
        )

    summary = gr.Markdown(summary0)

    with gr.Row():
        matrix = gr.Plot(value=matrix0)
        chart = gr.Plot(value=chart0)

    gr.Markdown("## Per-class test metrics")

    gr.Dataframe(
        value=rows,
        headers=[
            "Class", "Precision", "Recall",
            "AP50", "mAP50-95"
        ],
        interactive=False
    )

    gr.Markdown("## Upload an image")

    with gr.Row():
        image = gr.Image(type="pil", label="Input")
        preview = gr.Image(type="pil", label="Predictions")

    verdict = gr.Markdown()

    gr.Button("Analyze image").click(
        predict,
        [image, lower, upper],
        [preview, verdict]
    )

    for slider in (lower, upper):
        slider.change(
            dashboard,
            [lower, upper],
            [summary, matrix, chart]
        )
        slider.change(
            predict,
            [image, lower, upper],
            [preview, verdict]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
