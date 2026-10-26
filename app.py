import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from ultralytics import YOLO

# Load everything once at startup
MODEL_PATH = "weights/banana_ripeness_cls.pt"
CSV_PATH = "artifacts/test_predictions.csv"

model = YOLO(MODEL_PATH)
class_names = model.names
rotten_idx = [i for i, n in class_names.items() if n == "rotten"][0]

df = pd.read_csv(CSV_PATH)
df["is_rotten"] = (df["true_label"] == "rotten").astype(int)
positives = df[df["is_rotten"] == 1]
negatives = df[df["is_rotten"] == 0]

FP_COST = 0.20
FN_COST = 3.00
REVIEW_COST = (10 / 3600) * 9.0
ASSUMED_PREVALENCE = 0.10

# Precompute the single-threshold (no review) cost curve once
_thresholds = np.arange(0.0, 1.01, 0.01)
_costs = []
for t in _thresholds:
    recall = (positives["prob_rotten"] >= t).mean()
    fpr = (negatives["prob_rotten"] >= t).mean()
    exp_fn = 1000 * ASSUMED_PREVALENCE * (1 - recall)
    exp_fp = 1000 * (1 - ASSUMED_PREVALENCE) * fpr
    _costs.append(exp_fn * FN_COST + exp_fp * FP_COST)
_costs = np.array(_costs)
BEST_SINGLE_THRESHOLD = _thresholds[np.argmin(_costs)]

# Per-class report: table has only the 3 classes; accuracy kept separate, macro/weighted dropped
_report_dict = classification_report(df["true_label"], df["pred_label"], digits=3, output_dict=True)
OVERALL_ACCURACY = _report_dict.pop("accuracy")
_report_dict.pop("macro avg", None)
_report_dict.pop("weighted avg", None)

PER_CLASS_DF = pd.DataFrame(_report_dict).T.round(3)
PER_CLASS_DF.index.name = "class"
PER_CLASS_DF = PER_CLASS_DF.reset_index()


def cost_curve_plot(single_threshold):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(_thresholds, _costs)
    ax.axvline(BEST_SINGLE_THRESHOLD, color="gray", linestyle=":",
               label=f"cost-minimizing t={BEST_SINGLE_THRESHOLD:.2f}")
    ax.axvline(single_threshold, color="red", linestyle="--",
               label=f"your threshold={single_threshold:.2f}")
    ax.set_xlabel("Confidence threshold (prob_rotten)")
    ax.set_ylabel("Expected cost per 1,000 cases (€)")
    ax.set_title("Cost vs. threshold — single cutoff, NO human review")
    ax.legend()
    return fig


def update_dashboard(low, high):
    if high < low:
        high = low  # keep bands valid if someone drags them past each other

    fn_rate = (positives["prob_rotten"] < low).mean()
    fp_rate = (negatives["prob_rotten"] >= high).mean()
    review_rate_pos = ((positives["prob_rotten"] >= low) & (positives["prob_rotten"] < high)).mean()
    review_rate_neg = ((negatives["prob_rotten"] >= low) & (negatives["prob_rotten"] < high)).mean()

    n_pos = 1000 * ASSUMED_PREVALENCE
    n_neg = 1000 * (1 - ASSUMED_PREVALENCE)

    tp = n_pos * (1 - fn_rate - review_rate_pos)
    fn = n_pos * fn_rate
    review_pos = n_pos * review_rate_pos
    fp = n_neg * fp_rate
    tn = n_neg * (1 - fp_rate - review_rate_neg)
    review_neg = n_neg * review_rate_neg

    cost = fn * FN_COST + fp * FP_COST + (review_pos + review_neg) * REVIEW_COST
    review_share = (review_pos + review_neg) / 1000

    confusion = pd.DataFrame(
        [["Throw", f"{tp:.1f}", f"{fp:.1f}"],
         ["Review", f"{review_pos:.1f}", f"{review_neg:.1f}"],
         ["Sell", f"{fn:.1f}", f"{tn:.1f}"]],
        columns=["Decision", "True rotten", "True sellable"],
    )

    summary = (
        f"**Cost per 1,000 cases: €{cost:.2f}**\n\n"
        f"**Share sent to human review: {review_share:.1%}**"
    )

    return confusion, summary


def predict_image(image, low, high):
    if image is None:
        return "Upload an image first.", None
    result = model.predict(image, verbose=False)[0]
    prob_rotten = float(result.probs.data[rotten_idx])

    if prob_rotten < low:
        decision = "SELL"
    elif prob_rotten >= high:
        decision = "THROW"
    else:
        decision = "SEND FOR HUMAN REVIEW"

    all_probs = {class_names[i]: float(result.probs.data[i]) for i in class_names}
    text = f"**Decision: {decision}**\n\nprob(rotten) = {prob_rotten:.3f}\n\n" + "\n".join(
        f"- {k}: {v:.3f}" for k, v in all_probs.items()
    )
    return text, all_probs


with gr.Blocks(title="Banana Ripeness Sorter") as demo:
    gr.Markdown("# Banana Ripeness Sorter — Sell / Review / Throw")

    with gr.Tab("Upload & Predict"):
        with gr.Row():
            img_input = gr.Image(type="filepath", label="Upload a banana photo")
            pred_output = gr.Markdown()
        predict_btn = gr.Button("Predict")

    with gr.Tab("Threshold Tuning"):
        gr.Markdown(
            "Move the two thresholds to set the sell / review / throw bands. "
            "Everything below **low** is sold automatically; everything at or above **high** "
            "is thrown automatically; everything between goes to a person."
        )
        low_slider = gr.Slider(0.0, 1.0, value=0.04, step=0.01, label="Low threshold (sell below this)")
        high_slider = gr.Slider(0.0, 1.0, value=0.76, step=0.01, label="High threshold (throw at/above this)")
        confusion_table = gr.Dataframe(label="Expected outcomes per 1,000 cases")
        cost_summary = gr.Markdown()

        for slider in [low_slider, high_slider]:
            slider.change(
                update_dashboard,
                inputs=[low_slider, high_slider],
                outputs=[confusion_table, cost_summary],
            )

        gr.Markdown("---")
        gr.Markdown(
            "### Single-threshold comparison (no human review)\n"
            "This separate slider shows what cost we would get with **one** cutoff and no review "
            "option. This adds useful context for seeing what cost reduction adding the review band above actually brings."
        )
        single_slider = gr.Slider(0.0, 1.0, value=float(BEST_SINGLE_THRESHOLD), step=0.01,
                                   label="Single threshold (no review band)")
        cost_plot = gr.Plot(label="Cost vs. threshold — single cutoff, no review")

        single_slider.change(cost_curve_plot, inputs=[single_slider], outputs=[cost_plot])

    with gr.Tab("Per-Class Results"):
        gr.Markdown("### Precision / recall / F1 by class, on the held-out test set")
        gr.Dataframe(value=PER_CLASS_DF, interactive=False)
        gr.Markdown(f"**Overall accuracy (all classes combined): {OVERALL_ACCURACY:.3f}**")

    predict_btn.click(
        predict_image,
        inputs=[img_input, low_slider, high_slider],
        outputs=[pred_output, gr.JSON(visible=False)],
    )
    demo.load(update_dashboard, inputs=[low_slider, high_slider],
              outputs=[confusion_table, cost_summary])
    demo.load(cost_curve_plot, inputs=[single_slider], outputs=[cost_plot])

if __name__ == "__main__":
    demo.launch()
