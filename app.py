"""
Assignment 2: Gradio dashboard.

Serves the committed weights and the artifacts train.py saved. Nothing is
trained here. Tabs:

- Image: upload a frame, see boxes and the pass / review / escalate decision.
- Thresholds: move the two thresholds and watch the outcome matrix, expected
  cost per 1,000 cases and review share update, on the test-split scores.
- Cost curve: cost against a single threshold, with the chosen one marked.
- Per class: per-class precision / recall / AP, the confusion matrix, and the
  baselines.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "fire_smoke_yolo11n.pt"
ARTIFACTS_DIR = ROOT / "artifacts"

POSITIVE_NAMES = {"Fire", "Smoke"}

model = YOLO(WEIGHTS)
metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
scores = pd.read_csv(ARTIFACTS_DIR / "test_scores.csv")
cost_sweep = pd.read_csv(ARTIFACTS_DIR / "cost_sweep.csv")

CHOSEN = metrics.get("operating_point", {}).get("best_threshold", 0.5)
# the review band train.py chose; the sliders start here
BAND = metrics.get("operating_point", {}).get("band", {"low": CHOSEN, "high": CHOSEN})


def decide(score, low, high):
    if score >= high:
        return "ESCALATE"
    if score >= low:
        return "SEND TO OPERATOR"
    return "PASS"


def predict(image, low, high):
    if image is None:
        return None, "Upload an image.", None
    low, high = min(low, high), max(low, high)
    # Draw everything at or above the review threshold: those are the boxes
    # that can change the decision.
    r = model.predict(image, conf=max(low, 0.01), verbose=False)[0]
    boxes = pd.DataFrame({
        "class": [model.names[int(c)] for c in r.boxes.cls.tolist()],
        "confidence": [round(c, 3) for c in r.boxes.conf.tolist()],
    })
    score = boxes.loc[boxes["class"].isin(POSITIVE_NAMES), "confidence"].max()
    score = 0.0 if pd.isna(score) else float(score)
    decision = f"**{decide(score, low, high)}**: highest Fire/Smoke confidence {score:.2f}"
    return r.plot()[:, :, ::-1], decision, boxes


def outcome_table(low, high):
    """Expected outcomes per 1,000 cases at the assumed base rate."""
    low, high = min(low, high), max(low, high)
    s, y = scores["score_positive"], scores["gt_positive"]
    r = costing.outcome_rates(s, y, low, high)
    n_pos = 1000 * costing.BASE_RATE
    n_neg = 1000 - n_pos
    expected = pd.DataFrame({
        "Actual": ["Fire / smoke", "No fire"],
        "Pass": [n_pos * r["fnr"], n_neg * r["tnr"]],
        "Send to operator": [n_pos * r["pos_review"], n_neg * r["neg_review"]],
        "Escalate": [n_pos * r["tpr"], n_neg * r["fpr"]],
    }).round(2)

    c = costing.cost_per_1000(s, y, low, high)
    summary = (
        f"### €{c['cost']:,.0f} expected cost per 1,000 cases\n"
        f"- Sent to an operator: **{c['review_share']:.1%}** of cases "
        f"({c['review']:.0f} per 1,000)\n"
        f"- Missed fires (pass on a real fire): **{c['fn']:.2f}** per 1,000\n"
        f"- False escalations: **{c['fp']:.1f}** per 1,000\n\n"
        f"Base rate {costing.BASE_RATE:.1%}; costs FN €{costing.COSTS['fn']:,}, "
        f"FP €{costing.COSTS['fp']:,}, review €{costing.REVIEW_COST:,.2f}. "
        f"Operators are assumed to decide reviewed cases correctly."
    )
    return summary, expected


def cost_curve(low, high):
    fig = go.Figure()
    fig.add_scatter(x=cost_sweep["threshold"], y=cost_sweep["cost"], mode="lines",
                    line={"width": 2, "color": "#2a6fdb"}, name="Model decides alone",
                    hovertemplate="threshold %{x:.2f}<br>€%{y:,.0f} per 1,000<extra></extra>")
    chosen_cost = cost_sweep.loc[(cost_sweep["threshold"] - CHOSEN).abs().idxmin(), "cost"]
    fig.add_scatter(x=[CHOSEN], y=[chosen_cost], mode="markers+text",
                    marker={"size": 10, "color": "#222222"},
                    text=[f"chosen {CHOSEN:.2f}"], textposition="top right",
                    name="Chosen threshold")
    for value, label in ((min(low, high), "review from"), (max(low, high), "escalate from")):
        fig.add_vline(x=value, line_dash="dot", line_color="#888888",
                      annotation_text=label, annotation_position="top")
    fig.update_layout(
        xaxis_title="Confidence threshold (Fire or Smoke)",
        yaxis_title="Expected cost per 1,000 cases (€)",
        yaxis_tickprefix="€", yaxis_tickformat=",.0f",
        plot_bgcolor="white", showlegend=False, margin={"t": 40},
    )
    fig.update_yaxes(gridcolor="#e5e5e5")
    return fig


def per_class_table():
    rows = [{"class": name, **{k: round(v, 3) for k, v in m.items()}}
            for name, m in metrics["finetuned"]["per_class"].items()]
    rows.append({"class": "all", "precision": round(metrics["finetuned"]["precision"], 3),
                 "recall": round(metrics["finetuned"]["recall"], 3),
                 "ap50": round(metrics["finetuned"]["map50"], 3),
                 "ap50_95": round(metrics["finetuned"]["map50_95"], 3)})
    return pd.DataFrame(rows)


def baseline_table():
    rows = {
        "Finetuned (conf 0.25)":
            metrics["finetuned"]["frame_level_at_default_conf"],
        "Finetuned (cheapest single threshold)":
            metrics["finetuned"].get("frame_level_at_best_threshold",
                                     metrics["finetuned"]["frame_level_at_default_conf"]),
        "COCO zero-shot (any detection)": metrics["coco_zero_shot"]["frame_level_any_detection"],
        "Predict nothing": metrics["predict_nothing"]["frame_level"],
        "Always escalate": metrics["always_escalate"]["frame_level"],
    }
    return pd.DataFrame([{"model": k, **{m: round(v[m], 3) for m in
                          ("precision", "recall", "accuracy", "tp", "fp", "fn", "tn")}}
                         for k, v in rows.items()])


def confusions_table():
    return pd.DataFrame(metrics["finetuned"].get("top_confusions", [])[:8])


with gr.Blocks(title="Fire & smoke escalation") as demo:
    gr.Markdown(
        "# Fire & smoke escalation\n"
        "Fire or Smoke escalates; Lights never does. Two thresholds split each "
        "case into pass, send to an operator, or escalate."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=BAND["low"], step=0.01, label="Review threshold (below: pass)")
        high = gr.Slider(0, 1, value=BAND["high"], step=0.01, label="Escalate threshold (at or above: escalate)")

    with gr.Tab("Image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Camera frame")
            image_out = gr.Image(label="Detections at or above the review threshold")
        decision = gr.Markdown()
        boxes = gr.Dataframe(label="Detections")

    with gr.Tab("Thresholds"):
        summary = gr.Markdown()
        expected = gr.Dataframe(label=f"Expected outcomes per 1,000 cases (base rate {costing.BASE_RATE:.1%})")

    with gr.Tab("Cost curve"):
        curve = gr.Plot()

    with gr.Tab("Per class"):
        gr.Markdown("Box-level results on the test split. Positive for the decision = Fire or Smoke.")
        gr.Dataframe(per_class_table(), label="Per-class precision, recall, AP")
        gr.Dataframe(confusions_table(), label="Most frequent confusions (true → predicted)")
        matrix_png = ARTIFACTS_DIR / "confusion_matrix_normalized.png"
        if matrix_png.exists():
            gr.Image(str(matrix_png), label="Confusion matrix (normalised)")
        gr.Dataframe(baseline_table(), label="Frame-level comparison with baselines")

    def update_all(img, lo, hi):
        return (*predict(img, lo, hi), *outcome_table(lo, hi), cost_curve(lo, hi))

    outputs = [image_out, decision, boxes, summary, expected, curve]
    for trigger in (image_in.change, low.change, high.change):
        trigger(update_all, [image_in, low, high], outputs)
    demo.load(update_all, [image_in, low, high], outputs)


if __name__ == "__main__":
    demo.launch()
