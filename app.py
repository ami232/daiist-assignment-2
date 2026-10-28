"""
Assignment 2: Gradio dashboard for weld inspection.

Loads the committed weights in weights/ and the files train.ipynb saved in
artifacts/, and never trains anything. The test-set views re-price the weld
scores the training stage already computed, so moving a slider changes which
decision each weld gets, never the model's scores.
"""

import json
import os

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO

import pipeline

MODEL = YOLO(pipeline.WEIGHTS_PATH)
SEAMS = pd.read_csv(pipeline.SEAMS_CSV)
VALID, TEST = SEAMS[SEAMS["split"] == "valid"], SEAMS[SEAMS["split"] == "test"]
METRICS = pd.read_csv(pipeline.METRICS_CSV)
DETECTION_CM = pd.read_csv(pipeline.CONFUSION_CSV, index_col=0)
OPERATING_POINT = json.loads(pipeline.OPERATING_POINT_JSON.read_text())
EXAMPLES = sorted(str(p) for p in (pipeline.ARTIFACTS_DIR / "examples").glob("*.jpg"))

DECISION_COLORS = {"pass": "#1baf7a", "review": "#eda100", "fail": "#e34948"}
GOOD_WELD_COLOR = "#2a78d6"
GOOD_WELD_MIN_CONF = 0.25  # Good-weld boxes are drawn for context only, above this confidence
INK = "#898781"
GRID = "rgba(137, 135, 129, 0.25)"
TRUTHS = [("bad", "Bad-weld"), ("good", "Good-weld")]


def style(fig, title, x_title, y_title, height=420):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height, margin=dict(l=60, r=20, t=85, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK),
        legend=dict(orientation="h", y=-0.2, x=0),
        hoverlabel=dict(font_family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def euro(x):
    return f"€{x:,.0f}"


def ordered(low, high):
    """A low threshold above the high one closes the review band rather than inverting it."""
    return min(low, high), high


# ---------------------------------------------------------------- tab: inspect a photo

def detect(image):
    """Every box the model finds in the photo, kept so the sliders can redraw without re-running it."""
    if image is None:
        return None
    result = MODEL.predict(image, conf=0.01, verbose=False)[0]
    return [{"class": result.names[int(c)], "conf": float(s), "box": b}
            for b, s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())]


def draw(image, found, low, high):
    if image is None or found is None:
        return None, "Upload a photo of a weld, or pick one of the test images below.", pd.DataFrame()
    low, high = ordered(low, high)
    picture = image.convert("RGB")
    canvas = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=max(12, picture.width // 45))
    rows, worst = [], 0.0
    for box in sorted(found, key=lambda b: b["conf"]):
        if box["class"] in pipeline.SCORE_CLASSES:
            worst = max(worst, box["conf"])
            decision = str(pipeline.decide([box["conf"]], low, high)[0])
            if decision == "pass":
                continue
            color = DECISION_COLORS[decision]
        elif box["conf"] >= GOOD_WELD_MIN_CONF:
            decision, color = "shown for context", GOOD_WELD_COLOR
        else:
            continue
        x1, y1, x2, y2 = box["box"]
        canvas.rectangle((x1, y1, x2, y2), outline=color, width=3)
        canvas.text((x1 + 4, y1 + 2), f"{box['class']} {box['conf']:.2f}", fill=color, font=font)
        rows.append({"class": box["class"], "confidence": round(box["conf"], 3), "decision": decision})

    verdict = str(pipeline.decide([worst], low, high)[0])
    summary = (
        f"### Photo verdict: {verdict}\n"
        f"Highest Bad-weld or Defect confidence: **{worst:.2f}**. Fail at {high:.2f} or above, "
        f"send to review from {low:.2f}, pass below {low:.2f}.\n\n"
        f"Red boxes fail, amber boxes go to review, blue boxes are welds the model calls good. "
        f"Bad-weld and Defect boxes below the review threshold are not drawn."
    )
    table = pd.DataFrame(rows[::-1], columns=["class", "confidence", "decision"])
    return picture, summary, table


# ---------------------------------------------------------------- tab: thresholds and cost

def policy_costs(base_rate, escape, per_shift=False):
    """The three alternatives the review band is priced against."""
    t = OPERATING_POINT["threshold"]
    alone = pipeline.decision_rates(TEST["score"], TEST["is_bad"], t, t)
    price = lambda rates: pipeline.cost_per_1000(rates, base_rate, paid_per_shift=per_shift, escape=escape)
    return {
        "do nothing": price(pipeline.fixed_rates("pass")),
        "a person checks every weld": price(pipeline.fixed_rates("review")),
        f"model alone at {t:.2f}": price(alone),
    }


def operating_view(low, high, base_pct, escape, per_shift):
    low, high = ordered(low, high)
    base_rate = base_pct / 100
    rates = pipeline.decision_rates(TEST["score"], TEST["is_bad"], low, high)
    cost = pipeline.cost_per_1000(rates, base_rate, paid_per_shift=per_shift, escape=escape)
    share = pipeline.review_share(rates, base_rate)
    staff = pipeline.staffing(share)
    capacity = pipeline.review_capacity()
    caught = rates["bad"]["fail"] + rates["bad"]["review"] * (1 - pipeline.HUMAN_MISS_RATE)
    fits = ("fits" if share <= capacity else "**does not fit**")
    alternatives = "\n".join(f"- {name}: {euro(value)}"
                             for name, value in policy_costs(base_rate, escape, per_shift).items())
    pay = ("reviewers paid for whole shifts" if per_shift else "reviews paid by the minute")

    summary = (
        f"### {euro(cost)} per 1,000 welds\n"
        f"At a {base_pct:g}% share of bad welds, {euro(escape)} per bad weld that gets through, {pay}.\n\n"
        f"- Sent to review: **{share:.1%}** of welds, {staff['reviews per day']:.0f} of "
        f"{pipeline.WELDS_PER_DAY} a day, {staff['inspector hours per day']:.1f} inspector hours, "
        f"**{staff['inspectors needed']:.2f}** full-time inspectors ({pipeline.reviewers_paid(share)} on the payroll). That {fits} the "
        f"{pipeline.INSPECTORS} available, who can check {capacity:.0%}.\n"
        f"- Bad welds caught, by the model or the reviewer: **{caught:.0%}**\n"
        f"- Good welds sent to repair for nothing: **{rates['good']['fail']:.1%}**\n\n"
        f"Compared with:\n{alternatives}"
    )
    return summary, matrix_figure(low, high, base_rate, escape), cost_figure(low, high, base_rate, escape, per_shift)


def matrix_figure(low, high, base_rate, escape):
    decisions = pipeline.decide(TEST["score"], low, high)
    rates = pipeline.decision_rates(TEST["score"], TEST["is_bad"], low, high)
    unit = pipeline.case_costs(escape=escape)
    weight = {"bad": base_rate, "good": 1 - base_rate}
    z, text = [], []
    for truth, name in TRUTHS:
        is_class = (TEST["class"] == name).to_numpy()
        row_z, row_text = [], []
        for d in pipeline.DECISIONS:
            count = int((decisions[is_class] == d).sum())
            per_1000 = 1000 * weight[truth] * rates[truth][d]
            row_z.append(rates[truth][d])
            row_text.append(f"{count} test welds<br>{per_1000:.1f} per 1,000<br>{euro(per_1000 * unit[(truth, d)])}")
        z.append(row_z)
        text.append(row_text)
    fig = go.Figure(go.Heatmap(
        z=z, x=pipeline.DECISIONS, y=[name for _, name in TRUTHS], text=text, texttemplate="%{text}",
        textfont=dict(size=14), showscale=False, zmin=0, zmax=1, xgap=2, ygap=2,
        colorscale=[[0, "#eef4fc"], [1, "#2a78d6"]],
        hovertemplate="%{y}, %{x}: %{z:.0%} of the class<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return style(fig, f"Test welds by decision (pass < {low:.2f} ≤ review < {high:.2f} ≤ fail)<br>"
                      f"<sup>shading: share of the class; € per 1,000 welds at the chosen base rate</sup>",
                 "decision", "true class", height=380)


def cost_figure(low, high, base_rate, escape, per_shift=False):
    fig = go.Figure()
    for df, dash, label in [(TEST, "solid", "test"), (VALID, "dash", "valid")]:
        curve = pipeline.sweep(df["score"], df["is_bad"], base_rate, escape=escape)
        fig.add_scatter(x=curve["threshold"], y=curve["cost per 1000"], mode="lines", name=f"model alone, {label}",
                        line=dict(color="#2a78d6", width=2, dash=dash),
                        hovertemplate="threshold %{x:.2f}<br>€%{y:,.0f} per 1,000 welds<extra></extra>")
    references = policy_costs(base_rate, escape, per_shift)
    for (name, value), color in zip(list(references.items())[:2], ["#52514e", "#1baf7a"]):
        fig.add_hline(y=value, line=dict(color=color, dash="dot", width=1.5),
                      annotation_text=f"{name} {euro(value)}", annotation_position="top right")
    rates = pipeline.decision_rates(TEST["score"], TEST["is_bad"], low, high)
    band_cost = pipeline.cost_per_1000(rates, base_rate, paid_per_shift=per_shift, escape=escape)
    fig.add_vrect(x0=low, x1=max(high, low + 0.002), fillcolor="#eda100", opacity=0.18, line_width=0)
    fig.add_hline(y=band_cost, line=dict(color="#eda100", width=2),
                  annotation_text=f"your band {euro(band_cost)}", annotation_position="bottom right")
    t = OPERATING_POINT["threshold"]
    fig.add_vline(x=t, line=dict(color="#e34948", width=1.5),
                  annotation_text=f"chosen threshold {t:.2f}", annotation_position="top left")
    fig.update_yaxes(range=[0, 1.3 * references["do nothing"]])
    return style(fig, "Cost against threshold, model deciding alone<br>"
                      "<sup>amber: your review band and what it costs; red: the threshold chosen on the valid split</sup>",
                 "confidence threshold (fail at or above)", "€ per 1,000 welds", height=460)


# ---------------------------------------------------------------- tab: per-class results

def score_histogram():
    fig = go.Figure()
    for name, color in [("Good-weld", "#1baf7a"), ("Bad-weld", "#e34948")]:
        fig.add_histogram(x=TEST.loc[TEST["class"] == name, "score"], name=name, marker_color=color,
                          xbins=dict(start=0, end=1, size=0.05), opacity=0.75)
    fig.update_layout(barmode="overlay")
    return style(fig, "How the model scores good and bad test welds", "weld score", "test welds", height=360)


def training_note():
    return (f"Kept **{OPERATING_POINT['model']}**, the higher validation mAP50-95, trained on "
            f"`{OPERATING_POINT['device']}` with {OPERATING_POINT['train args']}.")


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Weld inspection") as demo:
    gr.Markdown(
        "# Weld inspection\n"
        "A YOLO model finetuned on photos of welds flags the ones that need repair. Each weld is passed, "
        "sent to a person for review, or failed and sent to repair. Everything shown comes from the "
        "weights and results saved by `train.ipynb`; nothing is retrained here."
    )
    with gr.Row():
        low_in = gr.Slider(0, 1, value=OPERATING_POINT["low"], step=0.01, label="Review from (low threshold)")
        high_in = gr.Slider(0, 1, value=OPERATING_POINT["high"], step=0.01, label="Fail from (high threshold)")

    with gr.Tab("Inspect a photo"):
        found = gr.State()
        with gr.Row():
            image_in = gr.Image(type="pil", label="Weld photo", sources=["upload", "clipboard"])
            image_out = gr.Image(type="pil", label="Predictions at the current thresholds", interactive=False)
        verdict_out = gr.Markdown()
        boxes_out = gr.Dataframe(label="Boxes drawn", interactive=False)
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=image_in, label="Test images")
        image_in.change(detect, image_in, found).then(draw, [image_in, found, low_in, high_in],
                                                      [image_out, verdict_out, boxes_out])
        for control in (low_in, high_in):
            control.change(draw, [image_in, found, low_in, high_in], [image_out, verdict_out, boxes_out])

    with gr.Tab("Thresholds and cost"):
        with gr.Row():
            base_in = gr.Number(value=pipeline.BASE_RATE * 100, minimum=0.1, maximum=50, step=0.5,
                                label="Bad welds, % of all welds")
            escape_in = gr.Number(value=round(pipeline.ESCAPE_EUR, 2), minimum=0, step=50,
                                  label="Cost of a bad weld that gets through, €")
            shift_in = gr.Checkbox(value=False, label="Reviewers paid for whole shifts, not by the minute")
        if "low per shift" in OPERATING_POINT:
            gr.Markdown(f"Band chosen on the valid split: **{OPERATING_POINT['low']:.2f} to {OPERATING_POINT['high']:.2f}** "
                        f"with reviews paid by the minute, **{OPERATING_POINT['low per shift']:.2f} to "
                        f"{OPERATING_POINT['high per shift']:.2f}** with reviewers paid for whole shifts.")
        with gr.Row():
            with gr.Column(scale=2):
                summary_out = gr.Markdown()
            with gr.Column(scale=3):
                matrix_out = gr.Plot()
        curve_out = gr.Plot()
        operating_inputs = [low_in, high_in, base_in, escape_in, shift_in]
        operating_outputs = [summary_out, matrix_out, curve_out]
        for control in operating_inputs:
            control.change(operating_view, operating_inputs, operating_outputs)

    with gr.Tab("Per-class results"):
        gr.Dataframe(METRICS.round(3), interactive=False,
                     label="Test split, per class: finetuned model against the two baselines")
        gr.Dataframe(DETECTION_CM.reset_index(), interactive=False,
                     label="Test boxes, finetuned model: rows are the true class, columns the prediction "
                           "(confidence 0.25, IoU 0.45)")
        gr.Plot(score_histogram())
        failures = pipeline.ARTIFACTS_DIR / "failures.png"
        if failures.exists():
            gr.Image(str(failures), label="Test welds the chosen band gets wrong", interactive=False)
        gr.Markdown(training_note())
        gr.Dataframe(pd.DataFrame(OPERATING_POINT["candidates"]).round(3), interactive=False,
                     label="Models trained, scored on the valid split")

    demo.load(operating_view, operating_inputs, operating_outputs)


if __name__ == "__main__":
    # Open a browser tab locally; CI (which sets CI=true) only needs the server to respond
    demo.launch(inbrowser=not os.environ.get("CI"))
