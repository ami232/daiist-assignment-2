"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Loads the committed weights in weights/ and the outputs train.py saved in
artifacts/. Never trains anything. Costs are re-read from costs.json, so the
cost curve and the review band reflect the current assumptions.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import decision

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "avocado_yolo11n.pt"

model = YOLO(str(WEIGHTS))
metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
per_class = pd.read_csv(ARTIFACTS / "per_class.csv")
confusion = json.loads((ARTIFACTS / "confusion_matrix.json").read_text())
test_scores = pd.read_csv(ARTIFACTS / "test_scores.csv")
val_scores = pd.read_csv(ARTIFACTS / "valid_scores.csv")
costs = decision.load_costs()

# Thresholds are chosen on validation; everything shown is measured on test.
SCORES = test_scores.finetuned_score.to_numpy()
LABELS = test_scores.diseased.to_numpy()
VAL = (val_scores.finetuned_score.to_numpy(), val_scores.diseased.to_numpy())
DISEASE_IDS = [i for i, n in model.names.items() if n in decision.DISEASE_CLASSES]
VAL_SWEEP = pd.DataFrame(decision.single_threshold_sweep(*VAL, costs))
SWEEP = pd.DataFrame(decision.single_threshold_sweep(SCORES, LABELS, costs))
CHOSEN_T = decision.best_single_threshold(*VAL, costs)
CHOSEN = SWEEP.loc[SWEEP.threshold == CHOSEN_T].iloc[0]
BAND = decision.best_review_band(*VAL, costs)


def ordered(lo: float, hi: float) -> tuple[float, float]:
    return (lo, hi) if lo <= hi else (hi, lo)


# --- image tab --------------------------------------------------------------

def predict(image, lo, hi):
    if image is None:
        return None, "Upload an image."
    lo, hi = ordered(lo, hi)
    result = model.predict(image, conf=max(lo, 0.01), verbose=False)[0]
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    disease = conf[np.isin(cls, DISEASE_IDS)]
    score = float(disease.max()) if len(disease) else 0.0
    if score >= hi:
        verdict = "REJECT (diseased)"
    elif score >= lo:
        verdict = "SEND TO HUMAN REVIEW"
    else:
        verdict = "PASS (export)"
    found = ", ".join(f"{model.names[c]} {p:.2f}" for c, p in zip(cls, conf)) or "nothing above the lower threshold"
    text = f"**{verdict}**\n\nDisease score: {score:.2f}  (review band {lo:.2f} to {hi:.2f})\n\nBoxes: {found}"
    return result.plot()[:, :, ::-1], text


# --- operating point tab ----------------------------------------------------

def cost_curve(lo, hi):
    fig = go.Figure()
    fig.add_scatter(x=VAL_SWEEP.threshold, y=VAL_SWEEP.cost_per_1000, mode="lines",
                    name="validation (used to choose)")
    fig.add_scatter(x=SWEEP.threshold, y=SWEEP.cost_per_1000, mode="lines",
                    line=dict(dash="dash"), name="test (reported)")
    fig.add_scatter(x=[CHOSEN.threshold], y=[CHOSEN.cost_per_1000], mode="markers",
                    marker=dict(size=12, symbol="star"),
                    name=f"chosen t={CHOSEN.threshold:.2f} (test EUR {CHOSEN.cost_per_1000:.0f})")
    fig.add_vrect(x0=lo, x1=hi, opacity=0.15, line_width=0, annotation_text="review band")
    fig.update_layout(xaxis_title="Reject threshold (disease score)",
                      yaxis_title="Expected cost per 1,000 avocados (EUR)",
                      margin=dict(l=20, r=20, t=30, b=20), height=380)
    return fig


def operating_point(lo, hi):
    lo, hi = ordered(lo, hi)
    counts = decision.confusion_3way(SCORES, LABELS, lo, hi)
    table = pd.DataFrame(counts).T.rename_axis("actual").reset_index()
    cost = decision.expected_cost(decision.outcome_rates(SCORES, LABELS, lo, hi), costs)
    staff = decision.staffing(cost["review_share"], costs)
    summary = (
        f"### EUR {1000 * cost['cost_per_case']:.2f} per 1,000 avocados\n"
        f"- Sent to human review: **{100 * cost['review_share']:.1f}%** "
        f"({staff['reviews_per_day']:.0f} per day at {costs['cases_per_day']:,} avocados/day)\n"
        f"- Review time: {staff['review_hours_per_day']:.1f} h/day, "
        f"about **{staff['reviewers_needed']:.1f} people**\n"
        f"- Base rate assumed: {100 * costs['base_rate']:.0f}% diseased "
        f"(costs re-weighted to this, not to the test split's mix)"
    )
    return table, summary, cost_curve(lo, hi)


def alternatives_table():
    alts = decision.alternatives(SCORES, LABELS, costs, CHOSEN_T, BAND["lo"], BAND["hi"])
    return pd.DataFrame({"policy": list(alts), "EUR per 1,000": [round(v, 2) for v in alts.values()]})


# --- per-class tab ----------------------------------------------------------

def class_confusion():
    labels = confusion["labels"]
    fig = go.Figure(go.Heatmap(z=confusion["matrix"], x=labels, y=labels, text=confusion["matrix"],
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(xaxis_title="True class", yaxis_title="Predicted class",
                      yaxis_autorange="reversed", height=450, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def baselines_table():
    rows = []
    for name, m in metrics["baselines"].items():
        rows.append({"model": name, "mAP50": m.get("map50"), "mAP50-95": m.get("map50_95"),
                     "image-level AUC": m.get("auc"),
                     "threshold (chosen on val)": m.get("threshold_chosen_on_val"),
                     "test EUR per 1,000": m.get("test_cost_per_1000")})
    return pd.DataFrame(rows)


with gr.Blocks(title="Avocado disease sorting") as demo:
    gr.Markdown(
        "# Avocado disease sorting\n"
        f"Finetuned yolo11n on `{metrics['dataset']['workspace']}/{metrics['dataset']['project']}` "
        f"v{metrics['dataset']['version']}. Positive class: **diseased** (reject). "
        f"Test split: {metrics['dataset']['test_images']} images, {metrics['dataset']['test_diseased']} diseased. "
        "Thresholds are chosen on the validation split; every number shown is measured on test."
    )
    with gr.Row():
        lo = gr.Slider(0, 1, value=BAND["lo"], step=0.01, label="Lower threshold (below: pass)")
        hi = gr.Slider(0, 1, value=BAND["hi"], step=0.01, label="Upper threshold (at or above: reject)")

    with gr.Tab("Operating point"):
        with gr.Row():
            with gr.Column():
                matrix = gr.Dataframe(label="Test split: actual x decision")
                summary = gr.Markdown()
            curve = gr.Plot(label="Cost against threshold")
        gr.Dataframe(alternatives_table(), label="Priced against the alternatives (EUR per 1,000)")

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Avocado image")
            image_out = gr.Image(label="Predictions")
        verdict = gr.Markdown()
        image_in.change(predict, [image_in, lo, hi], [image_out, verdict])

    with gr.Tab("Per-class results"):
        gr.Markdown(f"Overall on test: mAP50 **{metrics['detection']['map50']}**, "
                    f"mAP50-95 **{metrics['detection']['map50_95']}**")
        gr.Dataframe(per_class, label="Per class (test split)")
        gr.Plot(class_confusion(), label="Confusion matrix (5 classes + background)")
        gr.Dataframe(baselines_table(), label="Against the baselines (same test split)")

    for slider in (lo, hi):
        slider.change(operating_point, [lo, hi], [matrix, summary, curve])
        slider.change(predict, [image_in, lo, hi], [image_out, verdict])
    demo.load(operating_point, [lo, hi], [matrix, summary, curve])


if __name__ == "__main__":
    demo.launch()
