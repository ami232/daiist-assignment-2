"""
Assignment 2: Gradio dashboard (run with `uv run python main.py app`).

Loads the committed weights in weights/ and the artifacts train.py saved in
artifacts/. Never trains anything.
"""

from __future__ import annotations

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import config as C
import costing

# ----------------------------------------------------------------------------- load
MODEL = YOLO(str(C.BEST_WEIGHTS))
SCORES = pd.read_csv(C.ARTIFACTS_DIR / "test_scores.csv")
METRICS = json.loads((C.ARTIFACTS_DIR / "metrics.json").read_text())
OP = json.loads((C.ARTIFACTS_DIR / "operating_point.json").read_text())
STATS = json.loads((C.ARTIFACTS_DIR / "dataset_stats.json").read_text())

S = SCORES.score_finetuned.to_numpy()
Y = SCORES.is_violation.to_numpy().astype(bool)
T_BIN = OP["binary_threshold"]
T_LOW0, T_HIGH0 = OP["band"]["t_low"], OP["band"]["t_high"]


def params(base_rate, c_fn, c_fp, c_review):
    p = C.cost_params()
    p.update(base_rate=base_rate, c_fn=c_fn, c_fp=c_fp, c_tp=c_fp, c_review=c_review)
    return p


# ----------------------------------------------------------------------------- image
def predict(image, t_low, t_high):
    if image is None:
        return None, "Upload an image."
    t_high = max(t_high, t_low)
    res = MODEL.predict(image, conf=min(t_low, 0.99), imgsz=C.IMGSZ, verbose=False)[0]
    score = max(
        (c for k, c in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist())
         if MODEL.names[int(k)] == C.POSITIVE_CLASS),
        default=0.0,
    )
    if score >= t_high:
        decision = f"## 🔴 FAIL: alert supervisor\nHighest no-helmet score {score:.2f} ≥ {t_high:.2f}"
    elif score >= t_low:
        decision = f"## 🟡 REVIEW: send to a person\nHighest no-helmet score {score:.2f} is between {t_low:.2f} and {t_high:.2f}"
    else:
        decision = f"## 🟢 PASS\nHighest no-helmet score {score:.2f} < {t_low:.2f}"
    return res.plot()[:, :, ::-1], decision


# ----------------------------------------------------------------------------- costs
def cost_curve(p, t_low, t_high):
    sweep = costing.sweep_binary(S, Y, p)
    t_best = float(sweep.loc[sweep.cost_per_1000.idxmin(), "threshold"])
    fig = go.Figure(go.Scatter(x=sweep.threshold, y=sweep.cost_per_1000, mode="lines",
                               name="Model decides alone"))
    fig.add_vline(x=T_BIN, line_dash="dash", line_color="crimson",
                  annotation_text=f"chosen t={T_BIN:.2f}", annotation_position="top")
    if abs(t_best - T_BIN) > 1e-9:
        fig.add_vline(x=t_best, line_dash="dot", line_color="green",
                      annotation_text=f"min under current sliders t={t_best:.2f}",
                      annotation_position="bottom")
    fig.add_vrect(x0=t_low, x1=max(t_high, t_low), fillcolor="gold", opacity=0.2,
                  line_width=0, annotation_text="review band")
    fig.update_layout(xaxis_title="Threshold on 'head' (no helmet) confidence",
                      yaxis_title="Expected € per 1,000 snapshots", height=380,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def update(t_low, t_high, base_rate, c_fn, c_fp, c_review):
    t_high = max(t_high, t_low)
    p = params(base_rate, c_fn, c_fp, c_review)
    e = costing.evaluate(S, Y, t_low, t_high, p)
    matrix = costing.expected_matrix_per_1000(e, base_rate).reset_index(names="")
    reviews_day = e["review_share"] * C.CASES_PER_DAY
    hours = reviews_day * C.SECONDS_PER_REVIEW / 3600
    people = hours / C.REVIEWER_PRODUCTIVE_HOURS
    fits = "fits" if reviews_day <= C.REVIEW_CAPACITY_PER_DAY else "**does NOT fit**"
    headline = (
        f"### Expected cost: **€{1000 * e['cost_per_case']:,.0f} per 1,000 snapshots**\n"
        f"- Sent to review: **{100 * e['review_share']:.1f}%** → {reviews_day:,.0f} snapshots/day "
        f"of {C.CASES_PER_DAY:,} → {hours:.1f} h/day → {people:.2f} reviewers "
        f"({fits} the {C.REVIEWERS_AVAILABLE} available)\n"
        f"- Violations caught (alert or review): **{100 * (e['pos_fail'] + e['pos_review']):.1f}%**; "
        f"missed: {100 * e['pos_pass']:.1f}%"
    )
    binary = costing.sweep_binary(S, Y, p)
    alts = {k: 1000 * v for k, v in costing.alternatives(p).items()}
    alts["Model decides alone (cheapest t)"] = float(binary.cost_per_1000.min())
    alts["Model + review band (current sliders)"] = 1000 * e["cost_per_case"]
    alt_df = pd.DataFrame({"Option": list(alts), "€ per 1,000": [round(v, 0) for v in alts.values()]})
    return matrix, headline, cost_curve(p, t_low, t_high), alt_df


# ----------------------------------------------------------------------------- tables
per_class = pd.DataFrame(METRICS["detection"]["per_class"])
overall = METRICS["detection"]["overall"]
img_cmp = pd.DataFrame(METRICS["image_level_comparison"])
def cm_frame(cm: dict) -> pd.DataFrame:
    return pd.DataFrame(cm["matrix"], index=[f"pred {l}" for l in cm["labels"]],
                        columns=[f"true {l}" for l in cm["labels"]]).reset_index(names="")


cm = METRICS["detection"]["confusion_matrix"]
cm_df = cm_frame(cm)
CM_LOW_CONF = cm.get("conf", 0.001)
# Present once train.py has been re-run with confusion_at_threshold().
cm_t = METRICS["detection"].get("confusion_matrix_at_threshold")
cm_t_df = cm_frame(cm_t) if cm_t else None
test_counts = pd.DataFrame(
    [[int(((S >= T_BIN) & Y).sum()), int(((S < T_BIN) & Y).sum())],
     [int(((S >= T_BIN) & ~Y).sum()), int(((S < T_BIN) & ~Y).sum())]],
    index=["Actual: violation", "Actual: compliant"], columns=["Flagged", "Passed"],
).reset_index(names="")
split_df = pd.DataFrame([
    {"split": k, "images": v["images"], **{f"{c} instances": n for c, n in v["instances"].items()}}
    for k, v in STATS["splits"].items()
])

with gr.Blocks(title="Hard-hat compliance: costed thresholds") as demo:
    gr.Markdown(
        "# Hard-hat compliance from site cameras\n"
        "Positive class = **head** (a worker without a helmet). Each snapshot is scored by "
        "its most confident *head* box, then **passed**, **sent to a person**, or **alerted**."
    )
    with gr.Row():
        t_low = gr.Slider(0.01, 0.99, value=T_LOW0, step=0.01, label="t_low: below this → PASS")
        t_high = gr.Slider(0.01, 0.99, value=T_HIGH0, step=0.01, label="t_high: at/above this → FAIL (alert)")
    with gr.Accordion("Assumptions (change to test sensitivity)", open=False):
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.5, value=C.BASE_RATE, step=0.005, label="Base rate of violations")
            c_fn = gr.Number(value=C.COST_FN, label="€ missed violation (FN)")
            c_fp = gr.Number(value=C.COST_FP, label="€ supervisor walk-over (FP / TP)")
            c_review = gr.Number(value=C.COST_REVIEW, label="€ per human review")

    with gr.Tab("Costs & review load"):
        headline = gr.Markdown()
        with gr.Row():
            matrix = gr.Dataframe(label="Expected outcomes per 1,000 snapshots (production base rate)")
            alt_table = gr.Dataframe(label="Compared with the alternatives")
        curve = gr.Plot(label="Cost vs threshold")

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload a site photo")
            img_out = gr.Image(label="Boxes at confidence ≥ t_low")
        decision = gr.Markdown()
        gr.Button("Run").click(predict, [img_in, t_low, t_high], [img_out, decision])

    with gr.Tab("Per-class results"):
        gr.Markdown(f"**Overall (test split):** AP50 = {overall['AP50']:.3f}, mAP50-95 = {overall['mAP50-95']:.3f}")
        gr.Dataframe(per_class, label="Per-class detection metrics (test split)")
        if cm_t_df is not None:
            gr.Dataframe(cm_t_df, label=f"Box-level confusion matrix at the chosen threshold, "
                                        f"conf ≥ {cm_t['conf']:.2f} ({cm_t['orientation']})")
        gr.Dataframe(cm_df, label=f"Box-level confusion matrix at Ultralytics' validation confidence "
                                  f"{CM_LOW_CONF:g}: counts every faint box, so the background "
                                  f"column is inflated ({cm['orientation']})")
        gr.Dataframe(test_counts, label=f"Image-level confusion on the raw test split at chosen t={T_BIN:.2f}")
        gr.Dataframe(img_cmp, label="Image-level comparison vs baselines (same test split)")
        gr.Dataframe(split_df, label="Dataset splits")

    inputs = [t_low, t_high, base_rate, c_fn, c_fp, c_review]
    outputs = [matrix, headline, curve, alt_table]
    for comp in inputs:
        comp.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)

if __name__ == "__main__":
    demo.launch()
