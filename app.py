import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import config
import costing

# ---------------------------------------------------------------------------
# Load what training produced (nothing is trained here)
# ---------------------------------------------------------------------------
MODEL = YOLO(str(config.WEIGHTS_PATH))
NAMES = [MODEL.names[i] for i in range(len(MODEL.names))]
RES = config.RESULTS_DIR
SCORES = pd.read_csv(RES / "scores.csv")
TEST = SCORES[SCORES["split"] == "test"]
SWEEP = pd.read_csv(RES / "cost_sweep.csv")
METRICS = json.loads((RES / "metrics.json").read_text())
OP = json.loads((RES / "operating_point.json").read_text())

CHOSEN_T = OP["single_threshold"]
# Shipped band: break-even maths corrected to the 5% base rate (costing.py)
BAND_LOW, BAND_HIGH = OP["shipped_band"]["t_low"], OP["shipped_band"]["t_high"]


def _order(t_low, t_high):
    """If the sliders cross, treat them as one threshold."""
    return (t_low, t_high) if t_low <= t_high else (t_high, t_high)


# ---------------------------------------------------------------------------
# 1. Upload an image
# ---------------------------------------------------------------------------
def predict(image, t_low, t_high):
    if image is None:
        return "Upload a photo of one apple, banana or orange.", None
    t_low, t_high = _order(t_low, t_high)
    probs = MODEL.predict(image, imgsz=config.IMGSZ, verbose=False)[0].probs.data
    probs = probs.cpu().numpy()
    p_rotten = float(sum(probs[NAMES.index(c)] for c in config.POSITIVE_CLASSES))

    if p_rotten >= t_high:
        decision = "FAIL: the air jet removes it"
    elif p_rotten >= t_low:
        decision = "REVIEW: diverted to a person"
    else:
        decision = "PASS: goes into the box"

    text = (f"### {decision}\n\nP(rotten) = **{p_rotten:.3f}**  "
            f"(pass < {t_low:.2f} ≤ review < {t_high:.2f} ≤ fail)\n\n"
            f"Most likely class: **{NAMES[int(probs.argmax())]}**")
    table = pd.DataFrame({"class": NAMES, "probability": np.round(probs, 4)})
    return text, table


# ---------------------------------------------------------------------------
# 2. Move the thresholds: confusion table, cost, review share
# ---------------------------------------------------------------------------
def operating_view(t_low, t_high):
    t_low, t_high = _order(t_low, t_high)
    counts = costing.confusion_counts(TEST["p_rotten"], TEST["is_rotten"], t_low, t_high)
    table = pd.DataFrame(
        [["Actually rotten", counts["rotten"]["pass"], counts["rotten"]["review"],
          counts["rotten"]["fail"]],
         ["Actually fresh", counts["fresh"]["pass"], counts["fresh"]["review"],
          counts["fresh"]["fail"]]],
        columns=["Test images", "Pass", "Review", "Fail"])

    shares = costing.outcome_shares(TEST["p_rotten"], TEST["is_rotten"], t_low, t_high)
    cost = 1000 * costing.expected_cost(shares)
    share = costing.review_share(shares)
    people = costing.reviewers_needed(share)
    per_day = config.FRUIT_PER_DAY * share

    alt = OP["strategies_on_test"]
    warning = ("\n\n⚠️ **Placeholder assumptions in config.py, not final figures.**"
               if config.ASSUMPTIONS_ARE_PLACEHOLDERS else "")
    summary = (
        f"### Expected cost: €{cost:,.2f} per 1,000 fruit\n\n"
        f"Sent to review: **{100 * share:.1f}%** of fruit, about **{per_day:,.0f} "
        f"a day**, needing **{people:.2f} people** "
        f"({config.REVIEWERS_AVAILABLE} available)\n\n"
        f"Rotten fruit caught (fail or review): "
        f"{100 * (shares['rotten']['fail'] + shares['rotten']['review']):.1f}% · "
        f"Fresh fruit wrongly blasted off: {100 * shares['fresh']['fail']:.1f}%\n\n"
        f"**Compare, per 1,000 fruit:** do nothing €{alt['do_nothing']['cost_per_1000']:,.2f}"
        f" · person checks everything €{alt['person_checks_everything']['cost_per_1000']:,.2f}"
        f" · model alone (t={CHOSEN_T:.2f}) €{alt['model_alone']['cost_per_1000']:,.2f}"
        f" · shipped band ({BAND_LOW:.2f}–{BAND_HIGH:.2f}) "
        f"€{alt['model_plus_breakeven_band']['cost_per_1000']:,.2f}"
        f"\n\nBase rate assumed: {100 * config.BASE_RATE:.0f}% rotten. Shares are "
        f"measured on the {len(TEST)} test images, then reweighted to that base rate."
        + warning)
    return table, summary, cost_curve(t_low, t_high)


# ---------------------------------------------------------------------------
# 3. Cost against threshold, chosen threshold marked
# ---------------------------------------------------------------------------
def cost_curve(t_low=None, t_high=None):
    fig = go.Figure()
    styles = {("finetuned", "test"): ("Finetuned (test)", "solid"),
              ("finetuned", "val"): ("Finetuned (val, used to choose)", "dash"),
              ("zero_shot", "test"): ("Zero-shot pretrained (test)", "dot")}
    for (model, split), (label, dash) in styles.items():
        d = SWEEP[(SWEEP.model == model) & (SWEEP.split == split)]
        fig.add_trace(go.Scatter(x=d.threshold, y=d.cost_per_1000, name=label,
                                 mode="lines", line={"dash": dash}))
    fig.add_hline(y=OP["strategies_on_test"]["do_nothing"]["cost_per_1000"],
                  line_color="grey", annotation_text="do nothing")
    fig.add_vline(x=CHOSEN_T, line_dash="dash", line_color="black",
                  annotation_text=f"chosen {CHOSEN_T:.2f}")
    if t_low is not None and t_high is not None and t_high > t_low:
        fig.add_vrect(x0=t_low, x1=t_high, fillcolor="orange", opacity=0.15,
                      line_width=0, annotation_text="review band")
    fig.update_layout(xaxis_title="Threshold on P(rotten)",
                      yaxis_title="Expected cost per 1,000 fruit (€)",
                      title="Cost against threshold (model deciding alone)",
                      height=420, margin={"t": 50})
    return fig


# ---------------------------------------------------------------------------
# 4. Per-class results and baselines
# ---------------------------------------------------------------------------
def per_class_table():
    rows = [{"class": c, "precision": round(m["precision"], 3),
             "recall": round(m["recall"], 3), "test images": m["support"]}
            for c, m in METRICS["finetuned"]["per_class"].items()]
    return pd.DataFrame(rows)


def confusion_4class_table():
    cm = METRICS["finetuned"]["confusion_matrix"]
    df = pd.DataFrame(cm["rows_true_cols_pred"], columns=cm["labels"])
    df.insert(0, "true \\ predicted", cm["labels"])
    return df


def baselines_table():
    ft, zs, tr = METRICS["finetuned"], METRICS["zero_shot"], METRICS["trivial"]
    alt = OP["strategies_on_test"]

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    return pd.DataFrame([
        {"model": "Finetuned yolo11n-cls", "accuracy (all classes)": pct(ft["accuracy_4class"]),
         "rotten recall @0.5": pct(ft["binary_at_0.5"]["recall_rotten"]),
         "rotten precision @0.5": pct(ft["binary_at_0.5"]["precision_rotten"]),
         "AUC": round(ft["binary_at_0.5"]["auc"], 3),
         "€ / 1,000 at best threshold": round(alt["model_alone"]["cost_per_1000"], 2)},
        {"model": "Pretrained, zero-shot", "accuracy (all classes)": "n/a (no rotten class)",
         "rotten recall @0.5": pct(zs["binary_at_0.5"]["recall_rotten"]),
         "rotten precision @0.5": pct(zs["binary_at_0.5"]["precision_rotten"]),
         "AUC": round(zs["binary_at_0.5"]["auc"], 3),
         "€ / 1,000 at best threshold": round(alt["zero_shot_model_alone"]["cost_per_1000"], 2)},
        {"model": f"Trivial (majority class: {tr['majority_class']}; pass everything)",
         "accuracy (all classes)": pct(tr["accuracy_4class"]),
         "rotten recall @0.5": "0.0%", "rotten precision @0.5": "n/a", "AUC": 0.5,
         "€ / 1,000 at best threshold": round(alt["do_nothing"]["cost_per_1000"], 2)},
    ])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Rotten fruit sorter") as demo:
    gr.Markdown("# Rotten fruit sorter: apples, bananas and oranges\n"
                "Positive class = **rotten**. Below the low threshold the fruit "
                "passes, above the high one the air jet removes it, in between a "
                "person checks it.")
    with gr.Row():
        t_low = gr.Slider(0, 1, value=BAND_LOW, step=0.01, label="Low threshold (pass below)")
        t_high = gr.Slider(0, 1, value=BAND_HIGH, step=0.01, label="High threshold (fail above)")

    with gr.Tab("Operating point"):
        cm_out = gr.Dataframe(label="Test split at these thresholds (counts)")
        summary_out = gr.Markdown()
        curve_out = gr.Plot()

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Photo of one fruit")
            with gr.Column():
                pred_text = gr.Markdown()
                pred_table = gr.Dataframe(label="Class probabilities")

    with gr.Tab("Per-class results"):
        gr.Markdown(f"Finetuned model on the test split, {len(NAMES)} classes, argmax prediction.")
        gr.Dataframe(value=per_class_table(), label="Precision and recall per class")
        gr.Dataframe(value=confusion_4class_table(), label="Confusion matrix")
        gr.Markdown("### Against the baselines (same test split)")
        gr.Dataframe(value=baselines_table())

    for slider in (t_low, t_high):
        slider.change(operating_view, [t_low, t_high], [cm_out, summary_out, curve_out])
        slider.change(predict, [image_in, t_low, t_high], [pred_text, pred_table])
    image_in.change(predict, [image_in, t_low, t_high], [pred_text, pred_table])
    demo.load(operating_view, [t_low, t_high], [cm_out, summary_out, curve_out])

if __name__ == "__main__":
    demo.launch()