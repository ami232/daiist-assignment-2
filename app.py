"""
Assignment 2 - Gradio dashboard.

Run with:   uv run python main.py app

Never trains anything. It only loads:
  - weights/best.pt          (the finetuned model)
  - outputs/                 (everything train.py saved: scores, metrics, sweep)
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

# ---- Load what the training stage produced (no training here) ------------
model = YOLO(str(ROOT / "weights" / "best.pt"))
summary = json.loads((OUT_DIR / "summary.json").read_text())
test = pd.read_csv(OUT_DIR / "test_scores.csv")
sweep = pd.read_csv(OUT_DIR / "cost_sweep.csv")
per_class = pd.read_csv(OUT_DIR / "per_class_finetuned.csv")
confusion_6 = pd.read_csv(OUT_DIR / "confusion_matrix_finetuned.csv", index_col=0)
models = pd.read_csv(OUT_DIR / "model_comparison.csv")

test_scores = test["score"].values
test_is_rotten = test["true_class"].str.lower().str.contains("rotten").values
T_CHOSEN = summary["chosen_threshold"]
BAND = summary["chosen_band"]


# ---- 1. Predict one uploaded image ----------------------------------------
def decide(score, t_low, t_high):
    if score >= t_high:
        return "FAIL - discard"
    if score >= t_low:
        return "REVIEW - send to a person"
    return "PASS - goes into the pack"


def predict_image(image, t_low, t_high):
    if image is None:
        return {}, "Upload an image."
    result = model.predict(image, imgsz=summary["training"]["imgsz"], verbose=False)[0]
    probs = result.probs.data.cpu().numpy()
    names = result.names
    class_probs = {names[i]: float(p) for i, p in enumerate(probs)}
    score = sum(p for n, p in class_probs.items() if "rotten" in n.lower())
    text = f"**Rotten score:** {score:.3f}  \n**Decision:** {decide(score, t_low, t_high)}"
    return class_probs, text


# ---- 2. Everything that depends on the two thresholds ----------------------
def threshold_views(t_low, t_high):
    if t_low > t_high:
        t_low, t_high = t_high, t_low

    # Confusion matrix: real class x decision, counted on the test images
    rot, fresh = costing.outcome_rates(test_scores, test_is_rotten, t_low, t_high)
    n_rot, n_fresh = int(test_is_rotten.sum()), int((~test_is_rotten).sum())
    matrix = pd.DataFrame({
        "real class": ["rotten", "fresh"],
        "PASS": [round(rot["pass"] * n_rot), round(fresh["pass"] * n_fresh)],
        "REVIEW": [round(rot["review"] * n_rot), round(fresh["review"] * n_fresh)],
        "FAIL": [round(rot["fail"] * n_rot), round(fresh["fail"] * n_fresh)],
    })

    # Cost and review load at the production base rate
    cost = costing.expected_cost(test_scores, test_is_rotten, t_low, t_high)
    share = costing.review_share(test_scores, test_is_rotten, t_low, t_high)
    reviews = share * costing.DAILY_VOLUME
    text = (
        f"### Expected cost: **{1000 * cost:.2f} EUR per 1,000 pieces**\n"
        f"- Sent to human review: **{100 * share:.1f}%** of pieces "
        f"(~{reviews:,.0f} per day out of {costing.DAILY_VOLUME:,})\n"
        f"- Reviewers needed: **{costing.staff_needed(share):.2f}** "
        f"(available: {costing.REVIEWERS}, capacity {costing.REVIEW_CAPACITY_PER_DAY:,}/day)\n"
        f"- Base rate assumed: {100 * costing.BASE_RATE:.0f}% rotten. "
        f"Costs: FN {costing.COST_FN:.2f} EUR, FP {costing.COST_FP:.2f} EUR, "
        f"review {costing.COST_REVIEW:.2f} EUR."
    )

    # Cost curve with the chosen threshold and the current sliders marked
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], 1000 * sweep["cost_per_piece_test"], label="cost, model alone (test)")
    ax.axvline(T_CHOSEN, color="red", linestyle=":", label=f"chosen threshold {T_CHOSEN:.2f}")
    ax.axvspan(t_low, t_high, color="orange", alpha=0.2, label="current review band")
    ax.axhline(1000 * cost, color="green", linestyle="--", label="cost with current band")
    ax.set_xlabel("Rotten threshold")
    ax.set_ylabel("EUR per 1,000 pieces")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return matrix, text, fig


# ---- 3. Layout ---------------------------------------------------------------
with gr.Blocks(title="Rotten fruit sorting") as demo:
    gr.Markdown(
        "# Rotten fruit sorting - packing line\n"
        "Positive class = **rotten**. Score below the lower threshold -> PASS, "
        "above the upper threshold -> FAIL, in between -> a person checks it."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=BAND["t_low"], step=0.01, label="Lower threshold (below = PASS)")
        t_high = gr.Slider(0, 1, value=BAND["t_high"], step=0.01, label="Upper threshold (above = FAIL)")

    with gr.Tab("Classify an image"):
        with gr.Row():
            image = gr.Image(type="pil", label="Upload a fruit photo")
            with gr.Column():
                label = gr.Label(num_top_classes=6, label="Class probabilities")
                decision = gr.Markdown()
        image.change(predict_image, [image, t_low, t_high], [label, decision])

    with gr.Tab("Cost and human review"):
        cost_text = gr.Markdown()
        matrix = gr.Dataframe(label="Test images: real class x decision")
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Per-class results"):
        gr.Markdown("### Model comparison on the test split")
        gr.Dataframe(models)
        gr.Markdown("### Finetuned model, per class (test split)")
        gr.Dataframe(per_class)
        gr.Markdown("### Confusion matrix (rows = real class, columns = predicted)")
        gr.Dataframe(confusion_6.reset_index().rename(columns={"true_class": "real \\ predicted"}))

    for slider in (t_low, t_high):
        slider.change(threshold_views, [t_low, t_high], [matrix, cost_text, curve])
        slider.change(predict_image, [image, t_low, t_high], [label, decision])
    demo.load(threshold_views, [t_low, t_high], [matrix, cost_text, curve])


if __name__ == "__main__":
    demo.launch()
