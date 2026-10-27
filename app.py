"""
Assignment 2: Gradio dashboard.

Loads the committed model (weights/best.pt) and the precomputed evaluation
results (outputs/eval_results.json) -- never retrains, never re-runs the
full evaluation. Everything here is either live single-image inference or
cheap recomputation over already-stored per-image probabilities.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["healthy", "powdery mildew", "rust"]

model = YOLO(str(ROOT / "weights" / "best.pt"))

with open(ROOT / "outputs" / "eval_results.json") as f:
    eval_results = json.load(f)

finetuned_records = eval_results["finetuned"]["records"]
sweep = eval_results["threshold_sweep"]["sweep"]
cost_assumptions = eval_results["cost_assumptions"]


def disease_probability(probs: dict) -> float:
    return 1.0 - probs.get("healthy", 0.0)


def predict_image(image):
    """Run the committed model on an uploaded image, return a label dict
    Gradio can render as a bar chart of class probabilities."""
    if image is None:
        return None
    result = model.predict(image, verbose=False)[0]
    names = result.names
    probs = result.probs.data.tolist()
    return {names[i]: float(probs[i]) for i in range(len(probs))}


def compute_dashboard(pass_threshold: float, fail_threshold: float):
    """Recompute everything the dashboard shows, from stored per-image
    probabilities, given the two band edges the user has set.

    pass_threshold: P(disease) below this -> PASS (no action)
    fail_threshold: P(disease) at/above this -> FAIL (treat automatically)
    in between -> REVIEW (send to a person)
    """
    if pass_threshold > fail_threshold:
        pass_threshold, fail_threshold = fail_threshold, pass_threshold

    counts = {"pass": 0, "review": 0, "fail": 0}
    confusion = {t: {p: 0 for p in CLASS_NAMES} for t in CLASS_NAMES}
    outcome_cost = 0.0
    n = len(finetuned_records)

    for r in finetuned_records:
        p_disease = disease_probability(r["probs"])
        true_label = r["true_label"]
        predicted = max(r["probs"], key=r["probs"].get)
        confusion[true_label][predicted] += 1

        if p_disease < pass_threshold:
            band = "pass"
            acted_as_disease = False
        elif p_disease >= fail_threshold:
            band = "fail"
            acted_as_disease = True
        else:
            band = "review"
            # In review, assume a person judges correctly (reviewed cases
            # cost only the review time, not a mistake cost). We still
            # count which band each case landed in for the escalation
            # rate. Cost is attributed below, action-based.
            acted_as_disease = true_label != "healthy"

        counts[band] += 1

        if band != "review":
            is_actually_disease = true_label != "healthy"
            if acted_as_disease and is_actually_disease:
                outcome_cost += cost_assumptions["true_positive"]
            elif acted_as_disease and not is_actually_disease:
                outcome_cost += cost_assumptions["false_positive"]
            elif not acted_as_disease and is_actually_disease:
                outcome_cost += cost_assumptions["false_negative"]
            else:
                outcome_cost += cost_assumptions["true_negative"]

    cost_per_1000 = (outcome_cost / n) * 1000 if n else 0.0
    review_share = counts["review"] / n if n else 0.0

    confusion_table = [[confusion[t][p] for p in CLASS_NAMES] for t in CLASS_NAMES]

    per_class_lines = []
    for c in CLASS_NAMES:
        tp = confusion[c][c]
        fn = sum(confusion[c][p] for p in CLASS_NAMES if p != c)
        fp = sum(confusion[t][c] for t in CLASS_NAMES if t != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        per_class_lines.append(f"{c}: precision={precision:.2f}, recall={recall:.2f}, support={tp+fn}")
    per_class_text = "\n".join(per_class_lines)

    summary_text = (
        f"Pass: {counts['pass']} | Review: {counts['review']} | Fail: {counts['fail']}\n"
        f"Share sent to human review: {review_share:.1%}\n"
        f"Expected cost per 1000 cases: €{cost_per_1000:,.2f}"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(confusion_table, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, confusion_table[i][j], ha="center", va="center")
    fig.tight_layout()

    return summary_text, per_class_text, fig


def cost_curve_plot(pass_threshold: float, fail_threshold: float):
    thresholds = [p["threshold"] for p in sweep]
    costs = [p["cost_per_1000"] for p in sweep]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, costs, label="Expected cost per 1000 cases")
    best = eval_results["threshold_sweep"]["best_threshold"]
    ax.scatter([best["threshold"]], [best["cost_per_1000"]], color="green", zorder=5,
               label=f"Chosen operating point ({best['threshold']:.2f})")
    ax.axvline(pass_threshold, color="orange", linestyle="--", label=f"Pass edge ({pass_threshold:.2f})")
    ax.axvline(fail_threshold, color="red", linestyle="--", label=f"Fail edge ({fail_threshold:.2f})")
    ax.set_xlabel("Confidence threshold on P(disease)")
    ax.set_ylabel("Expected cost per 1000 cases (€)")
    ax.set_title("Cost vs. threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def update_dashboard(pass_threshold, fail_threshold):
    summary_text, per_class_text, confusion_fig = compute_dashboard(pass_threshold, fail_threshold)
    curve_fig = cost_curve_plot(pass_threshold, fail_threshold)
    return summary_text, per_class_text, confusion_fig, curve_fig


with gr.Blocks(title="Leaf Disease Triage Dashboard") as demo:
    gr.Markdown("# Leaf Disease Triage Dashboard")
    gr.Markdown(
        "Upload a leaf photo to see the model's prediction. Move the two "
        "sliders below to set the escalation band: below the pass edge is "
        "left alone, above the fail edge is treated automatically, and "
        "anything in between is sent to a person to check."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload a leaf photo")
            predict_button = gr.Button("Predict")
            prediction_output = gr.Label(label="Model prediction")
        with gr.Column():
            pass_slider = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Pass edge (below = no action)")
            fail_slider = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="Fail edge (at/above = treat)")
            summary_output = gr.Textbox(label="Outcome summary", lines=3)
            per_class_output = gr.Textbox(label="Per-class precision/recall", lines=4)

    with gr.Row():
        confusion_plot = gr.Plot(label="Confusion matrix")
        curve_plot = gr.Plot(label="Cost vs. threshold")

    predict_button.click(fn=predict_image, inputs=image_input, outputs=prediction_output)

    pass_slider.change(fn=update_dashboard, inputs=[pass_slider, fail_slider],
                        outputs=[summary_output, per_class_output, confusion_plot, curve_plot])
    fail_slider.change(fn=update_dashboard, inputs=[pass_slider, fail_slider],
                        outputs=[summary_output, per_class_output, confusion_plot, curve_plot])

    demo.load(fn=update_dashboard, inputs=[pass_slider, fail_slider],
              outputs=[summary_output, per_class_output, confusion_plot, curve_plot])


if __name__ == "__main__":
    demo.launch()