"""
Assignment 2: Gradio dashboard for brain MRI triage.

Loads the committed weights and the outputs saved by train.ipynb.
Nothing is trained here.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costs

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

model = YOLO(ROOT / "weights" / "best.pt")
metrics = json.loads((OUT_DIR / "metrics.json").read_text())
test_df = pd.read_csv(OUT_DIR / "test_scores.csv")
sweep = pd.read_csv(OUT_DIR / "cost_sweep.csv")


def predict(image, t_low, t_high):
    """Class probabilities for one uploaded scan, and where it would be routed."""
    if image is None:
        return {}, ""
    probs = model.predict(image, verbose=False)[0].probs.data.tolist()
    probs = {model.names[i]: p for i, p in enumerate(probs)}
    score = 1 - probs[costs.NEGATIVE]
    decision = costs.route(score, t_low, t_high).item()
    return probs, f"Tumour score {score:.3f} → **{decision.upper()}**"


def confusion(t_low, t_high):
    """Test scans by true label and outcome, at the current thresholds."""
    out = costs.route(test_df["score"].to_numpy(), t_low, t_high)
    truth = (test_df["true"] != costs.NEGATIVE).map({True: "tumour", False: "no tumour"})
    table = pd.crosstab(truth, out).reindex(index=["tumour", "no tumour"],
                                            columns=["pass", "review", "fail"], fill_value=0)
    return table.reset_index(names="true \\ outcome")


def summary(t_low, t_high):
    """Cost and review load at the current thresholds."""
    if t_low > t_high:
        return "t_low must be at or below t_high."
    cost = costs.cost_per_1000(test_df, t_low, t_high)
    share = costs.review_share(test_df, t_low, t_high)
    per_day = share * costs.SCANS_PER_DAY
    readers = per_day * costs.MINUTES_PER_READ / 60 / costs.READER_HOURS_PER_DAY
    return (f"**Expected cost:** €{cost:,.0f} per 1,000 scans (base rate {costs.BASE_RATE:.0%})  \n"
            f"**Sent for review:** {share:.1%} of scans → {per_day:.1f} a day "
            f"→ {readers:.2f} neuroradiologists")


def cost_curve(t_low, t_high):
    """Cost against threshold, with the chosen and the current thresholds marked."""
    fig = go.Figure(go.Scatter(x=sweep["threshold"], y=sweep["test_cost"], name="model alone (test)"))
    fig.add_vline(x=metrics["t_single"], line_dash="dash", annotation_text="chosen single")
    fig.add_vrect(x0=metrics["t_low"], x1=metrics["t_high"], fillcolor="orange", opacity=0.2,
                  annotation_text="chosen review band")
    fig.add_vrect(x0=t_low, x1=t_high, line_color="red", fillcolor="rgba(0,0,0,0)",
                  annotation_text="current", annotation_position="bottom left")
    fig.update_layout(xaxis_title="threshold on tumour score", yaxis_title="EUR per 1,000 scans",
                      yaxis_type="log", height=420)
    return fig


def update(t_low, t_high):
    return confusion(t_low, t_high), summary(t_low, t_high), cost_curve(t_low, t_high)


with gr.Blocks(title="Brain MRI triage") as demo:
    gr.Markdown("# Brain MRI triage\npass → normal report · review → remote neuroradiologist · fail → urgent referral")
    with gr.Row():
        t_low = gr.Slider(0, 1, value=metrics["t_low"], step=0.01, label="t_low (pass below)")
        t_high = gr.Slider(0, 1, value=metrics["t_high"], step=0.01, label="t_high (fail at or above)")

    with gr.Tab("Scan an MRI"):
        with gr.Row():
            image = gr.Image(type="pil", label="Brain MRI")
            with gr.Column():
                probs = gr.Label(label="Class probabilities")
                decision = gr.Markdown()
        image.change(predict, [image, t_low, t_high], [probs, decision])

    with gr.Tab("Thresholds and cost"):
        table = gr.Dataframe(label="Test scans: true label × outcome")
        text = gr.Markdown()
        plot = gr.Plot()
        gr.Dataframe(pd.read_csv(OUT_DIR / "options.csv"), label="The four options, on test")

    with gr.Tab("Per-class results"):
        gr.Markdown(f"4-class accuracy: **{metrics['accuracy']:.3f}** · "
                    f"tumour vs no tumour AUC: **{metrics['tumour_auc']:.3f}**")
        gr.Dataframe(pd.read_csv(OUT_DIR / "per_class.csv"), label="Precision and recall per class")
        gr.Dataframe(pd.read_csv(OUT_DIR / "confusion_matrix.csv"), label="Confusion matrix (rows = true)")
        gr.Dataframe(pd.read_csv(OUT_DIR / "baselines.csv"), label="Against the baselines")
        gr.Dataframe(pd.read_csv(OUT_DIR / "dataset_counts.csv"), label="Images per class and split")

    for slider in (t_low, t_high):
        slider.change(update, [t_low, t_high], [table, text, plot])
        slider.change(predict, [image, t_low, t_high], [probs, decision])
    demo.load(update, [t_low, t_high], [table, text, plot])

demo.launch()
