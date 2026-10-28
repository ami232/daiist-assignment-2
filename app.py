"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Reads only what train.py produced: weights/best.pt and artifacts/. Nothing is
trained here. Moving the two thresholds recomputes the decision table, cost
and review load from the saved test-half blister scores, using the same
costing module that chose the thresholds.
"""

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import config
import costing

RESULTS = json.loads((config.ARTIFACTS / "results.json").read_text())
SCORES = pd.read_csv(config.ARTIFACTS / "blister_scores.csv")
SWEEP = pd.read_csv(config.ARTIFACTS / "cost_sweep.csv")
TEST = SCORES[SCORES.split == "test"]
POINT = RESULTS["operating_point"]
MODEL = YOLO(config.WEIGHTS)
EXAMPLES = sorted(str(p) for p in (config.ARTIFACTS / "examples").glob("*.jpg"))


def ordered(t_low, t_high):
    return min(t_low, t_high), max(t_low, t_high)


def kpis(t_low, t_high):
    s = costing.summarise(TEST.score.values, TEST.defective.values, t_low, t_high)
    return (
        f"### Expected cost: **€{s['cost_per_1000']:.2f} per 1,000 blisters**\n"
        f"- Sent to review: **{s['review_share']:.2%}** of production, "
        f"about **{s['reviews_per_day']:,.0f} blisters/day**, "
        f"**{s['operators_per_shift']:.1f} operators per shift**\n"
        f"- Defective blisters passed without a check: **{s['missed_defect_rate']:.1%}**\n"
        f"- Good blisters auto-rejected: **{s['good_rejected_rate']:.1%}**\n\n"
        f"*Rates measured on the {len(TEST)}-image test half, reweighted to a "
        f"{config.BASE_RATE:.1%} defect rate and a line of "
        f"{config.LINE_RATE_PER_MIN}/min for {config.HOURS_PER_DAY} h/day.*"
    )


def decision_tables(t_low, t_high):
    counts = costing.outcome_counts(TEST.score.values, TEST.defective.values, t_low, t_high)
    per_1000 = costing.per_1000(counts).round(2)
    return counts.reset_index(names="actual"), per_1000.reset_index(names="actual")


def alternatives_table(t_low, t_high):
    options = costing.alternatives(
        TEST.score.values, TEST.defective.values, t_low, t_high, POINT["t_single"]
    )
    return pd.DataFrame([
        {
            "design": name,
            "cost per 1,000 (EUR)": round(s["cost_per_1000"], 2),
            "cost per day (EUR)": round(s["cost_per_1000"] * s["blisters_per_day"] / 1000),
            "review share": f"{s['review_share']:.2%}",
            "operators per shift": round(s["operators_per_shift"], 1),
            "defects passed unchecked": f"{s['missed_defect_rate']:.1%}",
        }
        for name, s in options.items()
    ])


def cost_curve(t_low, t_high):
    fig = go.Figure()
    for split, dash in [("tune", "solid"), ("test", "dash")]:
        part = SWEEP[SWEEP.split == split]
        fig.add_trace(go.Scatter(x=part.threshold, y=part.cost_per_1000, name=f"{split} half",
                                 line={"dash": dash}))
    chosen = SWEEP[(SWEEP.split == "tune") & np.isclose(SWEEP.threshold, POINT["t_single"])]
    fig.add_trace(go.Scatter(x=chosen.threshold, y=chosen.cost_per_1000, mode="markers",
                             marker={"size": 12, "color": "red"},
                             name=f"chosen single threshold {POINT['t_single']:.2f}"))
    band = costing.cost_per_1000(costing.outcome_counts(TEST.score.values, TEST.defective.values, t_low, t_high))
    fig.add_vrect(x0=t_low, x1=max(t_high, t_low + 0.002), fillcolor="orange", opacity=0.2, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.add_hline(y=band, line_dash="dot", line_color="orange",
                  annotation_text=f"current band: €{band:.2f}", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Threshold on blister score", yaxis_title="EUR per 1,000 blisters (log)",
                      yaxis_type="log", height=420, margin={"t": 30}, legend={"orientation": "h", "y": -0.2})
    return fig


def update(t_low, t_high):
    t_low, t_high = ordered(t_low, t_high)
    counts, per_1000 = decision_tables(t_low, t_high)
    return kpis(t_low, t_high), counts, per_1000, alternatives_table(t_low, t_high), cost_curve(t_low, t_high)


def inspect(image, t_low, t_high):
    if image is None:
        return None, "Upload an image of a blister."
    t_low, t_high = ordered(t_low, t_high)
    result = MODEL.predict(image, conf=0.001, imgsz=config.IMGSZ, verbose=False)[0]
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    empty = np.isin(classes, config.DEFECT_CLASSES)
    score = float(confs[empty].max()) if empty.any() else 0.0

    # Empty pockets are drawn from t_low up, since they drive the decision;
    # full pockets only when the model is fairly sure, to keep the image legible.
    keep = (empty & (confs >= t_low)) | (~empty & (confs >= 0.5))
    drawn = result[np.flatnonzero(keep)].plot()[:, :, ::-1]
    decision = costing.decide([score], t_low, t_high)[0]
    verdict = {"pass": "PASS: ship it", "review": "REVIEW: send to an operator",
               "reject": "REJECT: remove from the line"}[decision]
    return drawn, f"## {verdict}\nBlister score (highest empty-pocket confidence): **{score:.3f}**"


def per_class_tables():
    boxes = RESULTS["box_metrics_test"]
    per_class = pd.DataFrame(boxes["per_class"]).round(3)
    overall = pd.DataFrame([{"class": "all (mean)", **boxes["overall"]}]).round(3)
    blister = pd.DataFrame(RESULTS["blister_results_test"])
    blister["recall_ci95"] = blister.recall_ci95.apply(lambda ci: f"{ci[0]:.2f}–{ci[1]:.2f}")
    blister = blister.round(3)
    side = pd.DataFrame(RESULTS["recall_by_side_test"]).T.reset_index(names="side")
    side["recall_ci95"] = side.recall_ci95.apply(lambda ci: f"{ci[0]:.2f}–{ci[1]:.2f}")
    matrix = boxes["box_confusion_matrix"]
    confusion = pd.DataFrame(matrix["matrix"], index=[f"pred {l}" for l in matrix["labels"]],
                             columns=[f"true {l}" for l in matrix["labels"]]).astype(int)
    return pd.concat([per_class, overall]), blister, side, confusion.reset_index(names="")


def cost_assumptions():
    lines = ["### False-negative cost, built as an expected value",
             "| consequence | probability per escape | cost if it happens | expected |", "|---|---|---|---|"]
    for name, (p, cost) in config.FN_COMPONENTS.items():
        lines.append(f"| {name} | {p:.1%} | €{cost:,.0f} | €{p * cost:,.0f} |")
    lines.append(f"| **total (COST_FN)** | | | **€{config.COST_FN:,.0f}** |")
    ratio = config.BASE_RATE * config.COST_FN / ((1 - config.BASE_RATE) * config.COST_FP)
    lines += [
        "",
        f"False positive €{config.COST_FP:.2f}, review €{config.COST_REVIEW:.3f}, base rate {config.BASE_RATE:.2%}. "
        f"Per blister produced, missing defects weighs **{ratio:.0f}×** more than rejecting good ones "
        f"(base rate × FN cost against (1 − base rate) × FP cost).",
    ]
    return "\n".join(lines)


def sensitivity_table():
    table = pd.read_csv(config.ARTIFACTS / "sensitivity.csv")
    for col in ["base_rate", "missed_defect_rate", "good_rejected_rate", "review_share"]:
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    return table.round(2)


with gr.Blocks(title="Blister inspection") as demo:
    gr.Markdown(
        "# Blister inspection: pass, review or reject\n"
        "Positive class: **defective blister** (at least one empty pocket). "
        "Score = highest confidence of any `nopill_front`/`nopill_back` box. "
        "Below the lower threshold the blister passes, at or above the upper one it is rejected, "
        "in between it goes to an operator."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=POINT["t_low"], step=0.01, label="Lower threshold (pass below)")
        t_high = gr.Slider(0, 1, value=POINT["t_high"], step=0.01, label="Upper threshold (reject at or above)")
    gr.Markdown(
        f"Chosen on the tuning half: band **{POINT['t_low']:.2f}–{POINT['t_high']:.2f}**; "
        f"model-alone threshold **{POINT['t_single']:.2f}**."
    )

    with gr.Tab("Operating point"):
        summary = gr.Markdown()
        with gr.Row():
            counts = gr.Dataframe(label="Test half: actual × decision (image counts)")
            per_1000 = gr.Dataframe(label="Per 1,000 production blisters at the base rate")
        alternatives = gr.Dataframe(label="Priced against the alternatives (test half)")
        curve = gr.Plot(label="Cost against threshold (model decides alone)")

    with gr.Tab("Inspect an image"):
        with gr.Row():
            upload = gr.Image(type="numpy", label="Blister photo")
            output = gr.Image(label="Predictions at the current thresholds")
        verdict = gr.Markdown()
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=upload)

    with gr.Tab("Per-class results"):
        per_class, blister, side, confusion = per_class_tables()
        gr.Markdown("### Box level, test half (finetuned yolo11s)")
        gr.Dataframe(per_class)
        gr.Markdown("Box confusion matrix at confidence 0.25 (rows: predicted, columns: true)")
        gr.Dataframe(confusion)
        gr.Markdown("### Blister level, test half: finetuned model against baselines")
        gr.Dataframe(blister)
        gr.Markdown("### Defects caught by the review band, by side of the pack")
        gr.Dataframe(side)

    with gr.Tab("Cost assumptions"):
        gr.Markdown(cost_assumptions())
        gr.Markdown(
            "### Sensitivity: thresholds re-chosen on the tuning half for each FN cost and base rate, "
            "priced on the test half"
        )
        gr.Dataframe(sensitivity_table())

    outputs = [summary, counts, per_1000, alternatives, curve]
    for slider in (t_low, t_high):
        slider.change(update, [t_low, t_high], outputs)
        slider.change(inspect, [upload, t_low, t_high], [output, verdict])
    upload.change(inspect, [upload, t_low, t_high], [output, verdict])
    demo.load(update, [t_low, t_high], outputs)


if __name__ == "__main__":
    demo.launch()
