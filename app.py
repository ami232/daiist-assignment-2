"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Loads weights/best.pt and the files train.py wrote into artifacts/. Nothing is
trained here. Every number on screen is recomputed from the saved test-split
snapshot scores with the same functions in costing.py that picked the
thresholds.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
METRICS = json.loads((ARTIFACTS / "metrics.json").read_text())
SCORES = pd.read_csv(ARTIFACTS / "scores_test.csv")
SWEEP_VALID = pd.read_csv(ARTIFACTS / "cost_sweep_valid.csv")
OP = METRICS["operating_point"]
POSITIVE = METRICS["positive_class"]


# --------------------------------------------------------------------------- #
# Prediction tab
# --------------------------------------------------------------------------- #

def predict_image(image, low, high):
    if image is None:
        return None, "Upload an image."
    high = max(high, low)
    result = MODEL.predict(image, conf=low, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    names = result.names
    heads = conf[[names[c] == POSITIVE for c in cls]] if len(cls) else np.array([])
    score = float(heads.max()) if len(heads) else 0.0
    if score >= high:
        verdict = f"FAIL: send the foreman. Worst `{POSITIVE}` box at {score:.2f} >= {high:.2f}."
    elif score >= low:
        verdict = f"REVIEW: a person checks this image. Worst `{POSITIVE}` box at {score:.2f}."
    else:
        verdict = f"PASS: no `{POSITIVE}` box at or above {low:.2f}."
    counts = ", ".join(f"{names[c]}: {(cls == c).sum()}" for c in sorted(set(cls))) or "no boxes"
    return annotated, f"**{verdict}**\n\nBoxes at conf >= {low:.2f}: {counts}"


# --------------------------------------------------------------------------- #
# Operating point tab
# --------------------------------------------------------------------------- #

def operating_point(low, high, fn, fp, tp, review_cost, base_rate):
    high = max(high, low)
    costs = {**costing.COSTS, "fn": fn, "fp": fp, "tp": tp, "review": review_cost}
    res = costing.evaluate(SCORES["score"], SCORES["label"], low, high, costs, base_rate)

    # Confusion matrix as raw counts on the test split, and re-weighted to
    # 1,000 production snapshots at the chosen base rate.
    pos, neg = SCORES["label"] == 1, SCORES["label"] == 0
    rows = []
    for name, mask, share, n_prod in (
        (f"violation ({POSITIVE} present)", pos, res["shares"]["pos"], 1000 * base_rate),
        ("no violation", neg, res["shares"]["neg"], 1000 * (1 - base_rate)),
    ):
        s = SCORES["score"][mask]
        rows.append({
            "actual": name,
            "pass (test)": int((s < low).sum()),
            "review (test)": int(((s >= low) & (s < high)).sum()),
            "fail (test)": int((s >= high).sum()),
            "pass /1000 prod": round(share["pass"] * n_prod, 1),
            "review /1000 prod": round(share["review"] * n_prod, 1),
            "fail /1000 prod": round(share["fail"] * n_prod, 1),
        })
    matrix = pd.DataFrame(rows)

    alone = costing.best_single_threshold(SCORES["score"], SCORES["label"], costs, base_rate)
    alts = costing.alternatives(costs, base_rate)
    policies = pd.DataFrame([
        {"policy": "do nothing", "EUR / 1000 snapshots": alts["do_nothing"]["cost_per_1000"],
         "share reviewed": 0.0, "reviewers needed": 0.0},
        {"policy": "a person checks everything", "EUR / 1000 snapshots": alts["human_checks_all"]["cost_per_1000"],
         "share reviewed": 1.0, "reviewers needed": alts["human_checks_all"]["reviewers_needed"]},
        {"policy": f"model alone (t={alone['threshold']:.2f}, best on test)", "EUR / 1000 snapshots": alone["cost_per_1000"],
         "share reviewed": 0.0, "reviewers needed": 0.0},
        {"policy": f"model + review band [{low:.2f}, {high:.2f})", "EUR / 1000 snapshots": res["cost_per_1000"],
         "share reviewed": res["review_share"], "reviewers needed": res["reviewers_needed"]},
    ]).round(3)

    summary = (
        f"### Expected cost: **{res['cost_per_1000']:.0f} EUR per 1,000 snapshots**\n"
        f"Sent to review: **{100 * res['review_share']:.1f}%** of snapshots, "
        f"{res['reviews_per_day']:.0f} per day at {costing.DAILY_SNAPSHOTS} snapshots/day, "
        f"{res['reviewer_hours_per_day']:.1f} reviewer-hours, "
        f"**{res['reviewers_needed']:.2f} people** at {costing.REVIEWER_PRODUCTIVE_HOURS} productive h/day.\n\n"
        f"Violations caught: {100 * (1 - res['shares']['pos']['pass']):.1f}% before human error; "
        f"false alarms: {100 * res['shares']['neg']['fail']:.1f}% of clean snapshots."
    )
    return summary, matrix, policies, cost_curve(low, high, costs, base_rate)


def cost_curve(low, high, costs, base_rate):
    test = pd.DataFrame(costing.single_threshold_sweep(SCORES["score"], SCORES["label"], costs, base_rate))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["threshold"], y=test["cost_per_1000"], name="test split, current costs",
                             line=dict(color="#2563eb", width=2.5)))
    fig.add_trace(go.Scatter(x=SWEEP_VALID["threshold"], y=SWEEP_VALID["cost_per_1000"],
                             name="validation, report costs", line=dict(color="#9ca3af", dash="dot")))
    chosen = OP["single_threshold"]
    y_chosen = test.loc[(test["threshold"] - chosen).abs().idxmin(), "cost_per_1000"]
    fig.add_trace(go.Scatter(x=[chosen], y=[y_chosen], mode="markers+text", name="chosen (on validation)",
                             marker=dict(size=12, color="#dc2626"), text=[f"t={chosen:.2f}"],
                             textposition="top center"))
    fig.add_vrect(x0=low, x1=max(high, low), fillcolor="#f59e0b", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(xaxis_title="confidence threshold (model decides alone)",
                      yaxis_title="EUR per 1,000 snapshots", height=420, margin=dict(l=40, r=20, t=30, b=40),
                      legend=dict(orientation="h", y=-0.2))
    return fig


# --------------------------------------------------------------------------- #
# Per-class tab (static: comes straight from train.py)
# --------------------------------------------------------------------------- #

def per_class_table():
    rows = []
    for model, m in METRICS["models"].items():
        for cls, v in m["per_class"].items():
            rows.append({"model": model, "class": cls, "instances": v["instances"],
                         "AP50": v["AP50"], "AP50-95": v["AP50_95"],
                         f"precision @ {OP['single_threshold']:.2f}": v["precision"],
                         f"recall @ {OP['single_threshold']:.2f}": v["recall"]})
        rows.append({"model": model, "class": "ALL (mean)", "instances": None,
                     "AP50": m["mAP50"], "AP50-95": m["mAP50_95"]})
    return pd.DataFrame(rows).round(3)


def detection_confusion():
    labels = list(METRICS["class_names"].values()) + ["background"]
    m = np.array(METRICS["confusion_matrix_test"])
    df = pd.DataFrame(m, columns=[f"pred {l}" for l in labels])
    df.insert(0, "true", labels)
    return df


def examples():
    out = []
    for kind, items in METRICS["examples"].items():
        for item in items:
            out.append((str(ARTIFACTS / "examples" / item["file"]), f"{kind}: score {item['score']:.2f}"))
    return out


with gr.Blocks(title="Hard hat compliance: operating point") as demo:
    gr.Markdown(
        "# Hard hat compliance on site cameras\n"
        f"Positive class: **`{POSITIVE}`** (a worker without a hard hat). One decision per camera snapshot: "
        "pass, send to a remote reviewer, or send the foreman. "
        f"Model: finetuned yolo11s on [{METRICS['dataset']['workspace']}/{METRICS['dataset']['project']} "
        f"v{METRICS['dataset']['version']}]({METRICS['dataset']['url']}). "
        f"Chosen on validation: band **[{OP['low']:.2f}, {OP['high']:.2f})**, "
        f"single threshold **{OP['single_threshold']:.2f}**."
    )
    with gr.Row():
        low = gr.Slider(0.01, 0.99, value=OP["low"], step=0.01, label="low threshold: below = pass")
        high = gr.Slider(0.01, 0.99, value=OP["high"], step=0.01, label="high threshold: at or above = fail")

    with gr.Tab("Operating point"):
        with gr.Accordion("Cost assumptions (defaults from REPORT.md)", open=False):
            with gr.Row():
                fn = gr.Number(costing.COSTS["fn"], label="EUR per missed violation (FN)")
                fp = gr.Number(costing.COSTS["fp"], label="EUR per false alarm (FP)")
                tp = gr.Number(costing.COSTS["tp"], label="EUR per caught violation (TP)")
                rc = gr.Number(costing.COSTS["review"], label="EUR per human review")
                br = gr.Slider(0.005, 0.5, value=costing.BASE_RATE, step=0.005, label="base rate of violations")
        summary = gr.Markdown()
        matrix = gr.Dataframe(label="Confusion matrix: actual x decision")
        plot = gr.Plot(label="Cost against threshold")
        policies = gr.Dataframe(label="Against the alternatives (test split)")
        inputs = [low, high, fn, fp, tp, rc, br]
        for comp in inputs:
            comp.change(operating_point, inputs, [summary, matrix, policies, plot])
        demo.load(operating_point, inputs, [summary, matrix, policies, plot])

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload a site photo")
            image_out = gr.Image(label="Predictions at the low threshold")
        verdict = gr.Markdown()
        for comp in (image_in, low, high):
            comp.change(predict_image, [image_in, low, high], [image_out, verdict])

    with gr.Tab("Per-class results"):
        gr.Markdown("Test split. Finetuned model against COCO yolo11s zero-shot (its `person` boxes scored as "
                    "each class) and a model that predicts nothing.")
        gr.Dataframe(per_class_table(), label="Per-class detection metrics")
        gr.Dataframe(detection_confusion(),
                     label=f"Detection confusion matrix, finetuned, conf >= {OP['single_threshold']:.2f}, IoU 0.5")
        gr.Dataframe(pd.DataFrame(METRICS["head_recall_by_size_test"]).round(3),
                     label=f"`{POSITIVE}` recall by box size")
        gr.Gallery(examples(), label="Worst test failures at the chosen band", columns=6, height=260)


if __name__ == "__main__":
    demo.launch()
