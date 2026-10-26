"""
Assignment 2: Gradio dashboard for the out-of-stock detector.

    uv run python main.py app

Loads weights/best.pt and the artifacts/ written by train.py. Trains nothing.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

from costs import (
    BASE_RATE, COSTS, POSITIVE_CLASS, alternatives, expected_cost, outcome_rates,
    photos_per_day, reviewers_needed,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
METRICS = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
SCORES = pd.read_csv(ARTIFACTS / "image_scores.csv")
TEST = SCORES[SCORES["split"] == "test"]
SWEEP = pd.read_csv(ARTIFACTS / "cost_sweep.csv")
OP = METRICS["operating_point"]
CHOSEN_T, BAND_LOW, BAND_HIGH = OP["single_threshold"], OP["band"]["low"], OP["band"]["high"]
POS_ID = [i for i, n in MODEL.names.items() if n == POSITIVE_CLASS][0]


def ordered(low: float, high: float) -> tuple[float, float]:
    return min(low, high), max(low, high)


def decide(score: float, low: float, high: float) -> str:
    if score >= high:
        return "FAIL: send an associate to refill"
    if score >= low:
        return "REVIEW: a person checks the photo"
    return "PASS: shelf looks stocked"


# --------------------------------------------------------------------- predict

def predict_image(image: np.ndarray | None, low: float, high: float):
    if image is None:
        return None, "Upload a shelf photo."
    low, high = ordered(low, high)
    result = MODEL.predict(image, conf=max(low, 0.01), verbose=False)[0]
    boxes = result.boxes
    gaps = boxes.conf[boxes.cls == POS_ID]
    score = float(gaps.max()) if len(gaps) else 0.0
    counts = {name: int((boxes.cls == i).sum()) for i, name in MODEL.names.items()}
    text = (f"### {decide(score, low, high)}\n"
            f"Photo score (highest '{POSITIVE_CLASS}' confidence): **{score:.2f}**  \n"
            f"Band: pass < {low:.2f} ≤ review < {high:.2f} ≤ fail  \n"
            f"Boxes shown (conf ≥ {max(low, 0.01):.2f}): "
            + ", ".join(f"{n}: {c}" for n, c in counts.items()))
    return result.plot()[:, :, ::-1], text  # BGR -> RGB


# ---------------------------------------------------------------------- triage

def triage(low: float, high: float):
    low, high = ordered(low, high)
    y, s = TEST["has_gap"].to_numpy().astype(bool), TEST["score"].to_numpy()
    fail, review = s >= high, (s >= low) & (s < high)
    passed = ~fail & ~review
    matrix = pd.DataFrame(
        [[int((passed & t).sum()), int((review & t).sum()), int((fail & t).sum())]
         for t in (y, ~y)],
        columns=["pass", "review", "fail"], index=["has gap (positive)", "no gap"],
    ).reset_index(names="truth \\ decision")

    rates = outcome_rates(y, s, low, high)
    cost = expected_cost(rates)
    staff = reviewers_needed(cost["review_share"])
    summary = (
        f"### Expected cost: **€{cost['cost_per_1000']:.2f} per 1,000 photos**\n"
        f"- Sent to review: **{100 * cost['review_share']:.1f}%** of photos "
        f"→ {staff['reviews_per_day']:,.0f} reviews/day of {photos_per_day():,} photos "
        f"→ **{staff['people']:.2f} people** full-time\n"
        f"- Gaps caught without a person: {100 * rates['tp']:.0f}% · "
        f"gaps missed: {100 * rates['fn']:.0f}% · false alarms: {100 * rates['fp']:.0f}% of full shelves\n"
        f"- Base rate assumed: {100 * BASE_RATE:.0f}% of photos have a gap. "
        f"Costs: FN €{COSTS['fn']}, FP €{COSTS['fp']}, review €{COSTS['review']}.\n"
        f"- Counts below are the {len(TEST)} test photos (test set gap rate "
        f"{100 * y.mean():.0f}%); euros are re-weighted to the assumed base rate."
    )
    single = expected_cost(outcome_rates(y, s, CHOSEN_T, CHOSEN_T))["cost_per_1000"]
    compare = pd.DataFrame(
        [*alternatives().items(),
         (f"model alone (t = {CHOSEN_T:.2f})", single),
         (f"model + review band (current {low:.2f}–{high:.2f})", cost["cost_per_1000"])],
        columns=["policy", "€ per 1,000 photos"],
    ).round(2)
    return matrix, summary, compare, cost_curve(low, high)


def cost_curve(low: float = BAND_LOW, high: float = BAND_HIGH) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=SWEEP["threshold"], y=SWEEP["cost_per_1000_val"], name="validation (used to choose)",
                    line=dict(dash="dot"))
    fig.add_scatter(x=SWEEP["threshold"], y=SWEEP["cost_per_1000"], name="test (reported)")
    chosen = SWEEP.loc[(SWEEP["threshold"] - CHOSEN_T).abs().idxmin()]
    fig.add_scatter(x=[CHOSEN_T], y=[chosen["cost_per_1000"]], mode="markers+text",
                    marker=dict(size=12, symbol="star"), text=[f"chosen t = {CHOSEN_T:.2f}"],
                    textposition="top center", name="chosen threshold")
    fig.add_vrect(x0=low, x1=high, opacity=0.15, line_width=0, annotation_text="review band")
    fig.update_layout(xaxis_title="confidence threshold (single threshold: pass below, fail above)",
                      yaxis_title="expected € per 1,000 photos", height=420,
                      margin=dict(l=40, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.25))
    return fig


# -------------------------------------------------------------------- per class

def per_class_table() -> pd.DataFrame:
    rows = []
    for model, res in METRICS["object_level_test"].items():
        for name, m in res["per_class"].items():
            rows.append({"model": model, "class": name, "instances": m["instances"],
                         "precision@0.25": m["precision"], "recall@0.25": m["recall"],
                         "AP50": m["ap50"], "mAP50-95": m["map50_95"]})
        rows.append({"model": model, "class": "ALL (mean)", "AP50": res["ap50"],
                     "mAP50-95": res["map50_95"]})
    return pd.DataFrame(rows).round(3)


def image_level_table() -> pd.DataFrame:
    rows = [{"model": m, **{k: v[k] for k in ("threshold", "recall", "precision",
                                              "false_alarm_rate", "accuracy", "cost_per_1000")}}
            for m, v in METRICS["image_level_test"].items()]
    return pd.DataFrame(rows).round(3)


def object_matrix() -> pd.DataFrame:
    cm = METRICS["object_confusion_matrix_test"]
    df = pd.DataFrame(cm["matrix"], columns=[f"true {l}" for l in cm["labels"]])
    df.insert(0, "predicted", cm["labels"])
    return df


# ------------------------------------------------------------------------ layout

with gr.Blocks(title="Out-of-stock detector") as demo:
    gr.Markdown(
        "# Out-of-stock detector: shelf photo → pass / review / fail\n"
        f"Finetuned yolo11n on `{METRICS['dataset']['workspace']}/{METRICS['dataset']['project']}"
        f"` v{METRICS['dataset']['version']}. Positive class: **{POSITIVE_CLASS}** "
        f"(a photo with at least one empty gap). Chosen on validation: single threshold "
        f"**{CHOSEN_T:.2f}**, review band **{BAND_LOW:.2f}–{BAND_HIGH:.2f}**."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=BAND_LOW, step=0.01, label="Lower threshold (below → pass)")
        high = gr.Slider(0, 1, value=BAND_HIGH, step=0.01, label="Upper threshold (at or above → fail)")

    with gr.Tab("Predict"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Shelf photo")
            image_out = gr.Image(label="Predictions")
        decision = gr.Markdown()
        gr.Button("Run").click(predict_image, [image_in, low, high], [image_out, decision])

    with gr.Tab("Triage and cost"):
        summary = gr.Markdown()
        with gr.Row():
            matrix = gr.Dataframe(label="Test photos: truth × decision")
            compare = gr.Dataframe(label="Priced against the alternatives")
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Per-class results"):
        gr.Markdown("Object level, test split, same evaluator for all three models.")
        gr.Dataframe(per_class_table(), label="Per class")
        gr.Markdown("Photo level (is there any gap?), test split.")
        gr.Dataframe(image_level_table(), label="Per photo")
        gr.Dataframe(object_matrix(), label="Object confusion matrix (finetuned, conf 0.25)")
        if (ARTIFACTS / "confusion_matrix.png").exists():
            gr.Image(str(ARTIFACTS / "confusion_matrix.png"), label="Ultralytics confusion matrix")

    outputs = [matrix, summary, compare, curve]
    for slider in (low, high):
        slider.change(triage, [low, high], outputs)
    demo.load(triage, [low, high], outputs)


if __name__ == "__main__":
    demo.launch()
