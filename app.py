"""
Assignment 2: Gradio dashboard for helmet compliance detection.

    uv run python main.py app

Reads only what train.py already produced: the committed weights in weights/
and the evaluation outputs in artifacts/. It never trains or re-evaluates.

The two sliders at the top set the review band. Every view below follows
them: the predictions on an uploaded image, the outcome table, the expected
cost per 1,000 items, the share sent to a person and the cost curve.
Costs and volumes come from decisions.py, the same file train.py used.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # draw to images, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

import costing
import decisions

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"
OUTCOMES = ["pass", "human review", "fail"]


def load_json(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


model = YOLO(str(WEIGHTS_PATH))
run = load_json("run.json") or {}
box_metrics = load_json("box_metrics.json")
baselines = load_json("baselines.json")
dataset_stats = load_json("dataset_stats.json")
scores_path = ARTIFACTS / "test_scores.csv"
test_scores = pd.read_csv(scores_path) if scores_path.exists() else None

# Item-level labels and scores, if training has run and positives are chosen.
have_scores = test_scores is not None and decisions.POSITIVE_CLASSES is not None
if have_scores:
    y_true, y_score = costing.item_level(test_scores, decisions.POSITIVE_CLASSES)
missing = costing.missing_decisions()
can_cost = have_scores and not missing

if can_cost:
    sweep = costing.sweep(y_true, y_score)
    best_t = costing.best_threshold(sweep)
    start_band = decisions.REVIEW_BAND or costing.cheapest_band(y_true, y_score)
else:
    sweep, best_t = None, None
    start_band = decisions.REVIEW_BAND or (0.25, 0.25)


def status_note() -> str:
    notes = [f"**Dataset:** `{run.get('dataset', 'not trained yet')}` · **Model:** `{WEIGHTS_PATH.relative_to(ROOT)}`"]
    if run.get("smoke_test"):
        notes.append("⚠️ These weights come from a SMOKE_TEST run (1 epoch, 10% of data). Retrain before submitting.")
    if decisions.POSITIVE_CLASSES is None:
        notes.append("⚠️ Set `POSITIVE_CLASSES` in decisions.py to see item-level results.")
    if missing:
        notes.append(f"⚠️ Cost views are off until these are set in decisions.py: {', '.join(missing)}.")
    return "\n\n".join(notes)


# --- Inspect an item --------------------------------------------------------

def verdict(score: float, low: float, high: float) -> str:
    low, high = min(low, high), max(low, high)
    if score < low:
        return "PASS"
    if score < high:
        return "SEND TO HUMAN REVIEW"
    return "FAIL (no helmet)"


def inspect(image_path: str | None, low: float, high: float):
    if image_path is None:
        return None, "Upload an image of a rider."
    # Draw every box at or above the band's lower edge: those are the ones
    # that move an item out of "pass".
    result = model.predict(image_path, conf=min(low, high), verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    positive = decisions.POSITIVE_CLASSES or list(model.names.values())
    confs = [
        float(conf) for conf, cls in zip(result.boxes.conf, result.boxes.cls)
        if model.names[int(cls)] in positive
    ]
    score = max(confs, default=0.0)
    return annotated, f"### {verdict(score, low, high)}\nItem score: **{score:.2f}** (highest confidence among {', '.join(positive)})"


# --- Operating point ---------------------------------------------------------

def cost_curve(low: float, high: float):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if sweep is None:
        ax.text(0.5, 0.5, "Fill in decisions.py to see the cost curve", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ax.plot([r["threshold"] for r in sweep], [r["cost_per_1000"] for r in sweep], color="#1f4e79")
    ax.axvline(best_t, color="#c0392b", linestyle="--", label=f"cheapest single threshold: {best_t:.2f}")
    ax.axvspan(min(low, high), max(low, high), color="#f39c12", alpha=0.25, label="current review band")
    if decisions.REVIEW_BAND:
        for edge in decisions.REVIEW_BAND:
            ax.axvline(edge, color="#2e7d32", linestyle=":", linewidth=2)
        ax.plot([], [], color="#2e7d32", linestyle=":", linewidth=2,
                label=f"chosen band: {decisions.REVIEW_BAND[0]:.2f} to {decisions.REVIEW_BAND[1]:.2f}")
    ax.set_xlabel("confidence threshold (model decides alone)")
    ax.set_ylabel("expected cost per 1,000 items (€)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def operating_point(low: float, high: float):
    if not have_scores:
        return None, "Run training and set `POSITIVE_CLASSES` first.", cost_curve(low, high), None
    counts = costing.outcome_counts(y_true, y_score, min(low, high), max(low, high))
    table = pd.DataFrame(counts, columns=OUTCOMES, index=["truly wearing a helmet", "truly without a helmet"]).reset_index(names="test items")
    if not can_cost:
        return table, f"Cost needs: {', '.join(missing)}", cost_curve(low, high), None
    result = costing.evaluate_band(y_true, y_score, low, high)
    summary = (
        f"**Expected cost:** €{result.cost_per_1000:,.2f} per 1,000 items  \n"
        f"**Sent to review:** {result.review_share:.1%} of items "
        f"→ {result.reviews_per_day:,.0f} a day → {result.reviewers_needed:.2f} reviewers  \n"
        f"_Rates come from the test split; costs are mixed at a base rate of {decisions.BASE_RATE:.2%}._"
    )
    policies = pd.DataFrame(costing.policy_comparison(y_true, y_score, (min(low, high), max(low, high))))
    return table, summary, cost_curve(low, high), policies


def update(image_path, low, high):
    return (*inspect(image_path, low, high), *operating_point(low, high))


# --- Static tables -----------------------------------------------------------

def per_class_table():
    if box_metrics is None:
        return None
    return pd.DataFrame(box_metrics["per_class"]).round(3)


def overall_table():
    if box_metrics is None:
        return None
    return pd.DataFrame([box_metrics["overall"]]).round(3)


def box_confusion_table():
    if box_metrics is None:
        return None
    cm = box_metrics["confusion_matrix"]
    frame = pd.DataFrame(cm["matrix"], columns=[f"true {n}" for n in cm["labels"]])
    frame.insert(0, "predicted", cm["labels"])
    return frame


def dataset_table():
    if dataset_stats is None:
        return None
    rows = []
    for split, info in dataset_stats.items():
        for name, c in info["classes"].items():
            rows.append({
                "split": split, "images in split": info["images"],
                "images without any box": info["images_without_boxes"],
                "class": name, "instances": c["instances"], "images containing": c["images_containing"],
                "median box area (% of image)": None if c["box_area_share_median"] is None else round(100 * c["box_area_share_median"], 2),
            })
    return pd.DataFrame(rows)


with gr.Blocks(title="Helmet compliance check") as demo:
    gr.Markdown("# Helmet compliance check")
    gr.Markdown(status_note())
    with gr.Row():
        low = gr.Slider(0, 1, value=start_band[0], step=0.01, label="Below this score: pass")
        high = gr.Slider(0, 1, value=start_band[1], step=0.01, label="At or above this score: fail (in between: human review)")

    with gr.Tab("Inspect an item"):
        with gr.Row():
            image_in = gr.Image(type="filepath", label="Rider image")
            image_out = gr.Image(label="Detections at the lower edge of the band")
        verdict_md = gr.Markdown()

    with gr.Tab("Operating point"):
        outcome_df = gr.Dataframe(label="Test items by true label and outcome")
        cost_md = gr.Markdown()
        curve = gr.Plot(label="Cost against threshold")
        policy_df = gr.Dataframe(label="Policies compared (per 1,000 items, at your base rate)")

    with gr.Tab("Model results"):
        gr.Markdown("Box-level results on the test split. Per-class first: the average hides which case the model gets wrong.")
        gr.Dataframe(per_class_table(), label="Per class")
        gr.Dataframe(overall_table(), label="Overall")
        gr.Dataframe(pd.DataFrame(baselines).round(3) if baselines else None,
                     label="Item-level comparison with both baselines (threshold 0.25)")
        gr.Dataframe(box_confusion_table(), label="Box-level confusion matrix (conf 0.25)")
        if (ARTIFACTS / "confusion_matrix.png").exists():
            gr.Image(str(ARTIFACTS / "confusion_matrix.png"), label="Confusion matrix")

    with gr.Tab("Dataset"):
        gr.Dataframe(dataset_table(), label="Instances and box size per class and split")

    inputs = [image_in, low, high]
    outputs = [image_out, verdict_md, outcome_df, cost_md, curve, policy_df]
    for control in inputs:
        control.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)


if __name__ == "__main__":
    demo.launch()
