import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

from costing import load_costs, operating_summary, reference_policies, sweep_single_threshold

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
OUT_DIR = ROOT / "outputs"

model = YOLO(str(WEIGHTS))
costs = load_costs()
positive = costs["positive_class"]
metrics = json.loads((OUT_DIR / "metrics.json").read_text())
operating_point = json.loads((OUT_DIR / "operating_point.json").read_text())
dataset_stats = json.loads((OUT_DIR / "dataset_stats.json").read_text())
image_scores = pd.read_csv(OUT_DIR / "image_scores.csv")
scores = image_scores["score"].to_numpy()
is_positive = image_scores["is_positive"].to_numpy().astype(bool)
sweep = sweep_single_threshold(scores, is_positive, costs)
chosen_threshold = float(sweep.loc[sweep["cost_per_1000"].idxmin(), "threshold"])
default_low = operating_point["band_low"]
default_high = operating_point["band_high"]


def decide(score: float, low: float, high: float) -> str:
    if score >= max(low, high):
        return "FAIL: pothole, decelerate"
    if score < low:
        return "PASS: no pothole, continue"
    return "REVIEW: unsure, warn the driver"


def predict(image, conf, low, high):
    if image is None:
        return None, pd.DataFrame(), ""
    image = np.ascontiguousarray(image[:, :, ::-1])  # Gradio gives RGB, YOLO expects BGR arrays
    result = model.predict(image, conf=conf, imgsz=512, verbose=False)[0]
    rows = [
        {"class": model.names[int(c)], "confidence": round(float(p), 3)}
        for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    full = model.predict(image, conf=0.001, imgsz=512, verbose=False)[0]
    pothole_scores = [
        float(p) for c, p in zip(full.boxes.cls.tolist(), full.boxes.conf.tolist()) if model.names[int(c)] == positive
    ]
    score = max(pothole_scores, default=0.0)
    annotated = result.plot()[:, :, ::-1]
    return annotated, pd.DataFrame(rows), f"{positive} score {score:.3f} → {decide(score, low, high)}"


def confusion_table(rates: dict) -> pd.DataFrame:
    n_pos, n_neg = int(is_positive.sum()), int((~is_positive).sum())
    pi = costs["base_rate"]
    rows = []
    for label, key, n, share in ((f"Actual {positive}", "pos", n_pos, pi), (f"Actual no {positive}", "neg", n_neg, 1 - pi)):
        r = rates[key]
        rows.append({
            "actual": label,
            "pass (test count)": round(r["pass"] * n),
            "review (test count)": round(r["review"] * n),
            "fail (test count)": round(r["fail"] * n),
            "pass per 1,000": round(r["pass"] * share * 1000, 1),
            "review per 1,000": round(r["review"] * share * 1000, 1),
            "fail per 1,000": round(r["fail"] * share * 1000, 1),
        })
    return pd.DataFrame(rows)


def cost_plot(low: float, high: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["cost_per_1000"], mode="lines", name="model alone"))
    chosen_cost = float(sweep.loc[sweep["cost_per_1000"].idxmin(), "cost_per_1000"])
    fig.add_trace(go.Scatter(
        x=[chosen_threshold], y=[chosen_cost], mode="markers+text", name="chosen threshold",
        text=[f"t={chosen_threshold:.2f}, €{chosen_cost:.0f}"], textposition="top center", marker={"size": 12},
    ))
    fig.add_vrect(x0=low, x1=max(low, high), fillcolor="orange", opacity=0.2, line_width=0, annotation_text="warning band")
    fig.update_layout(
        xaxis_title=f"{positive} confidence threshold", yaxis_title="Expected cost per 1,000 cases (€)",
        margin={"l": 40, "r": 20, "t": 30, "b": 40}, height=420,
    )
    return fig


def update_band(low, high):
    s = operating_summary(scores, is_positive, low, high, costs)
    summary = (
        f"**Expected cost:** €{s['cost_per_1000']:.2f} per 1,000 cases  \n"
        f"**Driver warnings:** {100 * s['review_share']:.1f}% of frames "
        f"→ {s['reviews_per_day']:.0f} cases/day at {costs['cases_per_day']:,} cases/day  \n"
        f"**Share of the driver's shift spent on warnings:** {100 * s['reviewers_needed']:.0f}% "
        f"({costs['review_seconds']} s per warning, {costs['reviewer_hours_per_day']} h shift)  \n"
        f"**Recall on {positive} (fail + review):** {100 * (s['rates']['pos']['fail'] + s['rates']['pos']['review']):.1f}%  \n"
        f"**{positive} passed unseen:** {100 * s['rates']['pos']['pass']:.1f}%"
    )
    return confusion_table(s["rates"]), summary, cost_plot(low, high)


def per_class_table() -> pd.DataFrame:
    test = dataset_stats["splits"]["test"]["per_class"]
    df = pd.DataFrame(metrics["detection"]["per_class"])
    df["test instances"] = df["class"].map(lambda c: test[c]["instances"])
    return df


def detection_confusion() -> go.Figure:
    cm = metrics["detection"]["confusion_matrix"]
    fig = go.Figure(go.Heatmap(
        z=cm["values"], x=cm["cols_actual"], y=cm["rows_predicted"], text=cm["values"], texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig.update_layout(
        xaxis_title="Actual", yaxis_title="Predicted", yaxis_autorange="reversed",
        title=f"Test split, conf {cm['conf']}", height=450, margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def baselines_table() -> pd.DataFrame:
    rows = []
    for name, b in metrics["baselines"].items():
        det = b.get("detection", {})
        img = b["image_level"]
        rows.append({
            "model": name,
            "mAP50": det.get("mAP50"),
            "mAP50-95": det.get("mAP50-95"),
            f"image {positive} recall": img["recall"],
            f"image {positive} precision": img["precision"],
            "image accuracy": img["accuracy"],
        })
    return pd.DataFrame(rows)


def dataset_table() -> pd.DataFrame:
    rows = []
    for split, info in dataset_stats["splits"].items():
        for name, c in info["per_class"].items():
            rows.append({"split": split, "images in split": info["images"], "class": name, **c})
    return pd.DataFrame(rows)


with gr.Blocks(title="Pothole warning dashboard") as demo:
    gr.Markdown(f"# Pothole warning dashboard\nPositive class: **{positive}**. Base rate assumed: **{costs['base_rate']:.0%}**. "
        "Costs are euro-equivalents of total harm (damage, safety, comfort, distraction), not only money.")
    with gr.Row():
        low_slider = gr.Slider(0, 1, value=default_low, step=0.01, label="Lower threshold (below → pass: continue)")
        high_slider = gr.Slider(0, 1, value=default_high, step=0.01, label="Upper threshold (at or above → fail: decelerate)")

    with gr.Tab("Predict"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Road image")
            image_out = gr.Image(label="Predictions")
        conf_slider = gr.Slider(0.01, 1, value=chosen_threshold, step=0.01, label="Box display confidence")
        decision = gr.Markdown()
        boxes = gr.Dataframe(label="Detections")
        predict_inputs = [image_in, conf_slider, low_slider, high_slider]
        for trigger in (image_in.change, conf_slider.release, low_slider.release, high_slider.release):
            trigger(predict, predict_inputs, [image_out, boxes, decision])

    with gr.Tab("Operating point"):
        summary_md = gr.Markdown()
        confusion = gr.Dataframe(label="Decision matrix at current band")
        curve = gr.Plot(label="Cost against threshold")
        gr.Dataframe(
            reference_policies(scores, is_positive, chosen_threshold, costs),
            label="Alternatives priced with the same costs",
        )

    with gr.Tab("Per-class results"):
        gr.Dataframe(pd.DataFrame([metrics["detection"]["overall"]]), label="Overall (test split)")
        gr.Dataframe(per_class_table(), label="Per class (test split)")
        gr.Plot(detection_confusion())
        gr.Dataframe(baselines_table(), label="Finetuned vs baselines (same test split)")

    with gr.Tab("Dataset"):
        gr.Markdown(f"Exact duplicate images across splits: **{dataset_stats['duplicates_across_splits']}**")
        gr.Dataframe(dataset_table())

    band_outputs = [confusion, summary_md, curve]
    for slider in (low_slider, high_slider):
        slider.change(update_band, [low_slider, high_slider], band_outputs)
    demo.load(update_band, [low_slider, high_slider], band_outputs)


if __name__ == "__main__":
    demo.launch()
