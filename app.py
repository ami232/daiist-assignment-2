"""
Assignment 2: dashboard. Never trains anything; it only reads what train.py saved.

Run:  uv run python main.py app
"""

import json

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

# The same costs and cost maths as train.py, so the report and the dashboard match
from train import COSTS, HIGH, LOW, POSITIVE_CLASS, RESULTS, WEIGHTS, cost_per_1000, sweep

model = YOLO(str(WEIGHTS))
sick_id = [k for k, v in model.names.items() if v == POSITIVE_CLASS][0]
scores = pd.read_csv(RESULTS / "test_scores.csv")
S, Y = scores["finetuned"], scores["has_sick_bird"]
CHOSEN = json.loads((RESULTS / "summary.json").read_text())["chosen_threshold"]


# ------------------------------------------------ upload an image
def predict(image, low, high):
    if image is None:
        return None, "Upload an image."
    result = model.predict(image[..., ::-1], conf=low, imgsz=640, verbose=False)[0]
    conf = result.boxes.conf[result.boxes.cls == sick_id]
    score = float(conf.max()) if len(conf) else 0.0
    decision = "FAIL: inspect the bird" if score >= high else "REVIEW: check the image" if score >= low else "PASS"
    return result.plot()[..., ::-1], f"**Sick score:** {score:.2f} → **{decision}**"


# ------------------------------------------------ thresholds -> cost
def update(low, high, images_per_day, minutes_per_review, review_hours):
    low, high = min(low, high), max(low, high)
    r = cost_per_1000(S, Y, low, high)

    # Confusion matrix on the real test images
    sick, healthy = S[Y == 1], S[Y == 0]
    matrix = pd.DataFrame({
        "": ["Actually sick", "Actually healthy"],
        "PASS": [(sick < low).sum(), (healthy < low).sum()],
        "REVIEW": [((sick >= low) & (sick < high)).sum(), ((healthy >= low) & (healthy < high)).sum()],
        "FAIL": [(sick >= high).sum(), (healthy >= high).sum()],
    })

    # Staffing
    reviews = r["review_share"] * images_per_day
    hours = reviews * minutes_per_review / 60
    people = hours / review_hours
    text = (f"### EUR {r['cost_per_1000']:.2f} per 1,000 images\n"
            f"- Sent to a person: **{r['review_share']:.1%}** → {reviews:.0f} images a day, "
            f"{hours:.1f} hours, **{people:.1f} people**\n"
            f"- Sick birds: caught {r['sick_caught']:.0%}, reviewed {r['sick_reviewed']:.0%}, "
            f"**missed {r['sick_missed']:.0%}**\n"
            f"- Healthy birds: false alarms {r['healthy_false_alarm']:.0%}, reviewed {r['healthy_reviewed']:.0%}")

    # The alternatives the brief asks for
    p = COSTS["base_rate"]
    alternatives = pd.DataFrame([
        ["Do nothing", 1000 * p * COSTS["miss"]],
        ["A person checks every image", 1000 * (COSTS["review"] + p * COSTS["inspection"])],
        [f"Model alone (threshold {CHOSEN:.3f})", cost_per_1000(S, Y, CHOSEN, CHOSEN)["cost_per_1000"]],
        [f"Model + review ({low:.3f} to {high:.3f})", r["cost_per_1000"]],
    ], columns=["option", "EUR per 1,000 images"]).round(2)

    # Cost curve
    curve = sweep(S, Y)
    fig = go.Figure(go.Scatter(x=curve["threshold"], y=curve["cost_per_1000"], name="model alone"))
    fig.add_vline(x=CHOSEN, line_color="red", annotation_text=f"chosen {CHOSEN:.3f}")
    fig.add_vrect(x0=low, x1=high, opacity=0.15, annotation_text="review band")
    fig.update_layout(xaxis_title="threshold", yaxis_title="EUR per 1,000 images", height=400)

    return matrix, text, alternatives, fig


# ------------------------------------------------ layout
with gr.Blocks(title="Sick chicken detector") as demo:
    gr.Markdown("# Sick chicken detector\nOne case = one camera image. Positive class = **diseased-chicken**.")
    with gr.Row():
        low = gr.Slider(0, 1, LOW, step=0.001, label="Low threshold (below = PASS)")
        high = gr.Slider(0, 1, HIGH, step=0.001, label="High threshold (above = FAIL, between = REVIEW)")

    with gr.Tab("Cost and review"):
        with gr.Row():
            per_day = gr.Number(2000, label="Images per day")
            minutes = gr.Number(0.33, label="Minutes per review")
            hours = gr.Number(2, label="Review hours per person per day")
        summary = gr.Markdown()
        with gr.Row():
            matrix = gr.Dataframe(label="Confusion matrix (test images)")
            alternatives = gr.Dataframe(label="Compared with the alternatives")
        curve = gr.Plot()

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy")
            image_out = gr.Image()
        verdict = gr.Markdown()
        gr.Button("Run").click(predict, [image_in, low, high], [image_out, verdict])

    with gr.Tab("Per-class results"):
        gr.Dataframe(pd.read_csv(RESULTS / "per_class.csv"), label="Per class (test split)")
        gr.Dataframe(pd.read_csv(RESULTS / "baselines.csv"), label="Against the baselines (threshold 0.5)")
        gr.Dataframe(pd.read_csv(RESULTS / "dataset.csv"), label="Dataset")
        if (RESULTS / "confusion_matrix.png").exists():
            gr.Image(str(RESULTS / "confusion_matrix.png"), label="Detection confusion matrix")

    inputs, outputs = [low, high, per_day, minutes, hours], [matrix, summary, alternatives, curve]
    for control in inputs:
        control.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)

demo.launch()