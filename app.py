"""
Assignment 2: Gradio dashboard for the hard-hat check at the site entrance.

    uv run python main.py app

Never trains anything. It only reads what train.py already produced:

- weights/hardhat_yolo11n.pt   the finetuned model, for uploaded images
- outputs/metrics.json         per-class results, baselines, chosen thresholds
- outputs/case_scores.csv      one row per test person with a "no helmet" score,
                               so moving the thresholds recomputes everything live
- outputs/cost_sweep_*.csv     cost against threshold for each model

The cost formulas come from costing.py, the same module train.py uses.
"""

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "hardhat_yolo11n.pt"
OUTPUTS = ROOT / "outputs"

METRICS = json.loads((OUTPUTS / "metrics.json").read_text())
CASES = pd.read_csv(OUTPUTS / "case_scores.csv")
SCORES, POSITIVE = CASES["score"].to_numpy(), CASES["positive"].to_numpy()
CURVES = {name: pd.read_csv(OUTPUTS / f"cost_sweep_{name}.csv")
          for name in ["finetuned", "coco_zero_shot", "predict_nothing"]}
BAND = METRICS["review_band"]
CHOSEN = METRICS["operating_point"]
IMGSZ = METRICS["training"]["imgsz"]
MODEL = YOLO(str(WEIGHTS))

HELMET_SHOW_CONF = 0.5  # helmets are only drawn for context; they never trigger a stop
COLOURS = {"stop": (220, 38, 38), "review": (245, 158, 11), "helmet": (22, 163, 74)}  # RGB


# =============================================================================
# Live numbers for the two thresholds
# =============================================================================
def ordered(low: float, high: float) -> tuple[float, float]:
    return (low, high) if low <= high else (high, low)


def confusion_table(low: float, high: float) -> pd.DataFrame:
    """Expected outcomes per 1,000 entrances at the production base rate."""
    low, high = ordered(low, high)
    c = costing.expected_counts(costing.three_way_rates(SCORES, POSITIVE, low, high))
    return pd.DataFrame({
        "": ["Really no helmet", "Really has helmet"],
        "Stopped (auto or by guard)": [round(c["tp"], 1), round(c["fp"], 1)],
        "Let in": [round(c["fn"], 1), round(c["tn"], 1)],
    })


def summary_md(low: float, high: float) -> str:
    low, high = ordered(low, high)
    r = costing.three_way_rates(SCORES, POSITIVE, low, high)
    cost = 1000 * costing.cost_per_case(r)
    share = costing.review_share(r)
    per_day = share * costing.CASES_PER_DAY
    peak_min = share * costing.PEAK_ARRIVALS * costing.REVIEW_SECONDS / 60
    fits = "fits" if share <= costing.MAX_REVIEW_SHARE else "**does not fit**"
    alone = CHOSEN["cost_per_1000"]
    return (
        f"### {cost:,.0f} EUR per 1,000 entrances\n"
        f"- Model alone at t={CHOSEN['threshold']:.3f}: {alone:,.0f} EUR · do nothing: "
        f"{1000 * costing.BASE_RATE * costing.COSTS['fn']:,.0f} EUR\n"
        f"- **{share:.1%}** of entrances sent to the guard → {per_day:.0f} reviews a day, "
        f"{per_day * costing.REVIEW_SECONDS / 60:.0f} min a day\n"
        f"- Morning peak ({costing.PEAK_ARRIVALS} people in {costing.PEAK_WINDOW_MIN} min): "
        f"{peak_min:.1f} min of reviewing, which {fits} in the window\n"
        f"- Base rate {costing.BASE_RATE:.0%}, FN {costing.COSTS['fn']:.2f} EUR, "
        f"FP/TP {costing.COSTS['fp']:.2f} EUR, review {costing.REVIEW_COST:.2f} EUR"
    )


def cost_curve(low: float, high: float) -> go.Figure:
    low, high = ordered(low, high)
    labels = {"finetuned": "finetuned yolo11n", "coco_zero_shot": "COCO zero-shot",
              "predict_nothing": "predict nothing (do nothing)"}
    fig = go.Figure()
    for name, curve in CURVES.items():
        fig.add_scatter(x=curve["threshold"], y=curve["cost_per_1000"], mode="lines", name=labels[name])
    fig.add_scatter(x=[CHOSEN["threshold"]], y=[CHOSEN["cost_per_1000"]], mode="markers",
                    marker={"size": 11, "color": "black"},
                    name=f"chosen single threshold {CHOSEN['threshold']:.3f}")
    fig.add_vrect(x0=low, x1=high, fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(
        xaxis_title="confidence threshold for 'no helmet' (model alone)",
        yaxis_title="EUR per 1,000 entrances", height=420, margin={"t": 30},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def update(low: float, high: float):
    return summary_md(low, high), confusion_table(low, high), cost_curve(low, high)


# =============================================================================
# Uploaded image
# =============================================================================
def predict(image: np.ndarray | None, low: float, high: float):
    if image is None:
        return None, "Upload an image."
    low, high = ordered(low, high)
    result = MODEL.predict(image, imgsz=IMGSZ, conf=0.001, device="cpu", verbose=False)[0]
    canvas = image.copy()
    counts = {"stop": 0, "review": 0}
    for box, cls, conf in zip(result.boxes.xyxy.numpy().astype(int),
                              result.boxes.cls.numpy().astype(int),
                              result.boxes.conf.numpy()):
        if cls == 0 and conf >= low:  # head = no helmet
            outcome = "stop" if conf >= high else "review"
            counts[outcome] += 1
        elif cls == 1 and conf >= HELMET_SHOW_CONF:
            outcome = "helmet"
        else:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOURS[outcome], 2)
        cv2.putText(canvas, f"{outcome} {conf:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[outcome], 1, cv2.LINE_AA)
    if counts["stop"]:
        verdict = f"**STOP**: {counts['stop']} head(s) without helmet above {high:.3f}"
    elif counts["review"]:
        verdict = f"**SEND TO GUARD**: {counts['review']} uncertain head(s) between {low:.3f} and {high:.3f}"
    else:
        verdict = "**LET IN**: no head without helmet above the lower threshold"
    return canvas, verdict


# =============================================================================
# Static tables
# =============================================================================
def per_class_table() -> pd.DataFrame:
    names = {"finetuned": "finetuned yolo11n", "coco_zero_shot": "COCO zero-shot",
             "predict_nothing": "predict nothing"}
    rows = []
    for key, label in names.items():
        d = METRICS["detection"][key]
        for cls, m in d["per_class"].items():
            rows.append({"model": label, "class": cls, "precision": round(m["precision"], 3),
                         "recall": round(m["recall"], 3), "AP50": round(m["ap50"], 3),
                         "AP50-95": round(m["ap50_95"], 3)})
        rows.append({"model": label, "class": "ALL", "precision": None, "recall": None,
                     "AP50": round(d["map50"], 3), "AP50-95": round(d["map50_95"], 3)})
    return pd.DataFrame(rows)


def alternatives_table() -> pd.DataFrame:
    df = pd.DataFrame(METRICS["alternatives"])
    return df.round({"cost_per_1000": 0, "missed_per_1000": 1, "review_share": 3,
                     "reviews_per_day": 0, "review_minutes_per_day": 0, "peak_review_minutes": 1})


# =============================================================================
# Layout
# =============================================================================
with gr.Blocks(title="Hard-hat gate check") as demo:
    gr.Markdown(
        "# Hard-hat check at the site entrance\n"
        "One case = one person walking through the gate. Positive class = **head without helmet**. "
        "Scores at or above the upper threshold are stopped automatically, scores between the two "
        "thresholds go to the gate guard, and the rest are let in."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=BAND["low"], step=0.001, label="Lower threshold (below = let in)")
        high = gr.Slider(0, 1, value=BAND["high"], step=0.001, label="Upper threshold (above = stop)")

    with gr.Tab("Thresholds and cost"):
        with gr.Row():
            with gr.Column():
                summary = gr.Markdown(summary_md(BAND["low"], BAND["high"]))
                table = gr.Dataframe(confusion_table(BAND["low"], BAND["high"]),
                                     label="Expected outcomes per 1,000 entrances", interactive=False)
            plot = gr.Plot(cost_curve(BAND["low"], BAND["high"]), label="Cost against threshold")
        gr.Markdown("### Our design against the alternatives (at the chosen thresholds)")
        gr.Dataframe(alternatives_table(), interactive=False)

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload an image")
            image_out = gr.Image(type="numpy", label="Predictions at the current thresholds")
        verdict = gr.Markdown()
        image_in.change(predict, [image_in, low, high], [image_out, verdict])

    with gr.Tab("Per-class results"):
        gr.Markdown(
            f"Test split: {METRICS['test_cases']['people']} people, "
            f"{METRICS['test_cases']['without_helmet']} without helmet. Detection metrics from "
            "Ultralytics on the same test split (P and R at the confidence that maximises F1)."
        )
        gr.Dataframe(per_class_table(), interactive=False)
        gr.Image(str(OUTPUTS / "confusion_matrix.png"), label="Detection confusion matrix (finetuned)")

    for slider in (low, high):
        slider.change(update, [low, high], [summary, table, plot])
        slider.change(predict, [image_in, low, high], [image_out, verdict])


if __name__ == "__main__":
    demo.launch()
