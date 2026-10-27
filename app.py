"""
Assignment 2 dashboard: road condition -> pass / operator review / send a truck.

    uv run python main.py app

Loads the committed weights and the files train.py saved in results/. It
never trains anything. The two sliders at the top are shared by every tab.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import decision

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "road_conditions_yolo11n_cls.pt"

ASSUMPTIONS = decision.load_assumptions()
POSITIVE = ASSUMPTIONS["positive_classes"]
MODEL = YOLO(str(WEIGHTS))
TEST = pd.read_csv(RESULTS / "scores_test.csv")
SWEEP = pd.read_csv(RESULTS / "cost_sweep.csv")
CHOSEN = json.loads((RESULTS / "operating_point.json").read_text())
METRICS = json.loads((RESULTS / "metrics.json").read_text())
STATS = json.loads((RESULTS / "dataset_stats.json").read_text())
PER_CLASS = (
    pd.read_csv(RESULTS / "per_class.csv")
    .round(3)
    .rename(columns={"class": "Class", "test_images": "Test images", "precision": "Precision", "recall": "Recall"})
)
POLICIES = (
    pd.read_csv(RESULTS / "policies.csv")
    .replace(float("inf"), "never")
    .assign(review_share=lambda d: (100 * d["review_share"]).round(1))
    .round({"cost_per_1000": 0, "reviews_per_day": 0, "operators_needed": 1,
            "missed_snowy_per_1000": 1, "needless_trucks_per_1000": 1})
    .rename(columns={
        "policy": "Policy", "low": "Low", "high": "High", "cost_per_1000": "€ per 1,000",
        "review_share": "% reviewed", "reviews_per_day": "Reviews/night", "operators_needed": "Operators",
        "missed_snowy_per_1000": "Missed snowy /1,000", "needless_trucks_per_1000": "Needless trucks /1,000",
    })
    .drop(columns="snowy_caught_share")
)
SENSITIVITY = (
    pd.read_csv(RESULTS / "sensitivity.csv")
    .assign(value=lambda d: [f"{v:g}" + (" (current)" if c else "") for v, c in zip(d["value"], d["is_current"])])
    .round({"best_threshold": 2, "cost_per_1000": 0, "do_nothing_cost_per_1000": 0,
            "truck_everywhere_cost_per_1000": 0})
    .rename(columns={
        "assumption": "Assumption changed", "value": "Value", "best_threshold": "Best threshold",
        "cost_per_1000": "€ per 1,000 (model)", "do_nothing_cost_per_1000": "€ per 1,000 (do nothing)",
        "truck_everywhere_cost_per_1000": "€ per 1,000 (truck everywhere)",
    })
    [["Assumption changed", "Value", "Best threshold", "€ per 1,000 (model)",
      "€ per 1,000 (do nothing)", "€ per 1,000 (truck everywhere)"]]
)
EXAMPLES = sorted(str(p) for p in (RESULTS / "examples").glob("*.jpg"))

OUTCOMES = ["Pass", "Operator review", "Send a truck"]


def decide(score: float, low: float, high: float) -> str:
    if score >= high:
        return OUTCOMES[2]
    if score >= low:
        return OUTCOMES[1]
    return OUTCOMES[0]


# Predict tab -----------------------------------------------------------------------------


def predict(image, low, high):
    if image is None:
        return None, "Upload an image or pick an example."
    low = min(low, high)
    probs = MODEL.predict(image, imgsz=224, verbose=False)[0].probs.data.cpu().numpy()
    by_class = {MODEL.names[i]: float(p) for i, p in enumerate(probs)}
    score = sum(by_class[c] for c in POSITIVE)
    text = (
        f"### {decide(score, low, high)}\n"
        f"Probability the road needs treatment ({' + '.join(POSITIVE)}): **{score:.2f}**  \n"
        f"Pass below {low:.2f}, send a truck from {high:.2f}, operator review in between."
    )
    return by_class, text


# Thresholds tab ----------------------------------------------------------------------------


def three_way_matrix(low, high) -> pd.DataFrame:
    outcome = [decide(s, low, high) for s in TEST["p_treat"]]
    actual = TEST["needs_treatment"].map({1: f"Snowy ({'/'.join(POSITIVE)})", 0: "Fine (Clear/Wet)"})
    matrix = pd.crosstab(actual, pd.Series(outcome, name="outcome")).reindex(columns=OUTCOMES, fill_value=0)
    return matrix.reset_index().rename(columns={"needs_treatment": "Actual road"})


def threshold_view(low, high):
    low = min(low, high)
    e = decision.evaluate(TEST["p_treat"], TEST["needs_treatment"], low, high, ASSUMPTIONS)
    capacity = decision.review_capacity_per_day(ASSUMPTIONS)
    fits = "fits" if e["reviews_per_day"] <= capacity else "**does not fit**"
    summary = (
        f"### Expected cost: €{e['cost_per_1000']:,.0f} per 1,000 images\n"
        f"- Sent to an operator: **{e['review_share']:.1%}** of images "
        f"= {e['reviews_per_day']:,.0f} a night = {e['operators_needed']:.1f} operators "
        f"({fits} the {ASSUMPTIONS['operators']} available, capacity {capacity:,.0f}/night)\n"
        f"- Per 1,000 real images: {e['missed_snowy_per_1000']:.1f} snowy roads missed, "
        f"{e['needless_trucks_per_1000']:.1f} trucks sent for nothing\n"
        f"- Snowy roads caught (truck or operator): {e['snowy_caught_share']:.1%}\n\n"
        f"The table counts the {len(TEST)} test images. The cost reweights them to a "
        f"{ASSUMPTIONS['base_rate']:.0%} base rate of snowy roads."
    )
    return three_way_matrix(low, high), summary, cost_curve(low, high)


def cost_curve(low=None, high=None):
    fig = go.Figure()
    for split, name in [("valid", "Validation (used to choose)"), ("test", "Test (reported)")]:
        part = SWEEP[SWEEP["split"] == split]
        fig.add_trace(go.Scatter(x=part["threshold"], y=part["cost_per_1000"], mode="lines", name=name))
    fig.add_vline(
        x=CHOSEN["threshold"], line_dash="dash", line_color="red",
        annotation_text=f"chosen {CHOSEN['threshold']:.2f}", annotation_position="top",
    )
    if low is not None:
        for x in sorted({low, high}):
            fig.add_vline(x=x, line_dash="dot", line_color="grey")
    fig.update_layout(
        xaxis_title="Threshold on the snow/slush probability (model decides alone)",
        yaxis_title="Expected cost per 1,000 images (€)",
        height=420, margin=dict(t=40, b=40), legend=dict(x=0.01, y=0.99),
    )
    return fig


# Per-class tab -----------------------------------------------------------------------------


def accuracy_table() -> pd.DataFrame:
    finetuned = METRICS["finetuned"]
    rows = [
        ("Finetuned, all test images", finetuned["test_all"]),
        ("Finetuned, test images with no near-twin in train", finetuned["test_without_train_twin"]),
        ("Finetuned, daytime test images", finetuned["test_day"]),
        ("Finetuned, night test images", finetuned["test_night"]),
    ]
    table = [{"Model / subset": name, "Images": m["images"], "Accuracy": round(m["accuracy"], 3)} for name, m in rows]
    baseline = METRICS["majority_class_baseline"]
    table.append(
        {
            "Model / subset": f"Majority-class baseline (always '{baseline['always_predicts']}')",
            "Images": finetuned["test_all"]["images"],
            "Accuracy": round(baseline["test_accuracy"], 3),
        }
    )
    return pd.DataFrame(table)


def needs_treatment_table() -> pd.DataFrame:
    labels = {
        "finetuned": "Finetuned model",
        "zero_shot": "Pretrained model, zero-shot",
        "always_treat": "Always send a truck",
        "never_treat": "Never send a truck",
    }
    rows = []
    for key, name in labels.items():
        m = METRICS["needs_treatment_on_test"][key]
        rows.append(
            {
                "Approach": name,
                "Threshold": m.get("threshold", ""),
                "Precision": round(m["precision"], 3),
                "Recall": round(m["recall"], 3),
            }
        )
    return pd.DataFrame(rows)


def dataset_table() -> pd.DataFrame:
    rows = [
        {"Class": c, **counts, "Night share": STATS["night_share_per_class"][c]}
        for c, counts in STATS["images_per_class_and_split"].items()
    ]
    return pd.DataFrame(rows)


# Layout ------------------------------------------------------------------------------------

with gr.Blocks(title="Winter road treatment") as demo:
    gr.Markdown(
        "# Winter road treatment: pass, operator review, or send a truck\n"
        f"Finetuned `yolo11n-cls` on UDOT Winter Road Conditions v1. "
        f"The road **needs treatment** when it shows {' or '.join(POSITIVE)}. "
        f"Thresholds were chosen on the validation split; everything shown is the test split."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=CHOSEN["low"], step=0.01, label="Low threshold: below this, do nothing")
        high = gr.Slider(0, 1, value=CHOSEN["high"], step=0.01, label="High threshold: from this up, send a truck")
    gr.Markdown(
        f"Chosen on validation: model alone at **{CHOSEN['threshold']:.2f}**; with review, "
        f"pass below **{CHOSEN['low']:.2f}** and truck from **{CHOSEN['high']:.2f}**. "
        f"Set both sliders to the same value to let the model decide alone."
    )

    with gr.Tab("Predict"):
        with gr.Row():
            image = gr.Image(type="pil", label="Road image")
            with gr.Column():
                decision_text = gr.Markdown()
                probabilities = gr.Label(label="Class probabilities")
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=image)

    with gr.Tab("Thresholds and cost"):
        with gr.Row():
            matrix = gr.Dataframe(label="Test images by actual road and outcome")
            summary = gr.Markdown()
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Per-class results"):
        gr.Dataframe(PER_CLASS, label="Per class on the test split (4-class prediction)")
        with gr.Row():
            gr.Image(str(RESULTS / "confusion_matrix.png"), label="Confusion matrix (test)")
            gr.Dataframe(accuracy_table(), label="Accuracy")
        gr.Dataframe(needs_treatment_table(), label="'Needs treatment' on test, each at its own validation-chosen threshold")

    with gr.Tab("Policies and dataset"):
        gr.Dataframe(POLICIES, label="Every way of running this, priced on the test split")
        gr.Markdown(
            f"Assumptions (assumptions.json): truck run €{ASSUMPTIONS['cost_truck_run_eur']}, "
            f"missed snowy road €{ASSUMPTIONS['cost_missed_snow_eur']}, review "
            f"{ASSUMPTIONS['review_seconds']} s at €{ASSUMPTIONS['operator_eur_per_hour']}/h, base rate "
            f"{ASSUMPTIONS['base_rate']:.0%}, {ASSUMPTIONS['images_per_day']:,} images a night, "
            f"{ASSUMPTIONS['operators']} operators x {ASSUMPTIONS['operator_hours_per_day']} h."
        )
        gr.Dataframe(
            SENSITIVITY,
            label="Sensitivity: one assumption changed at a time, threshold re-chosen on validation, priced on test",
        )
        gr.Dataframe(dataset_table(), label=f"Images per class and split ({STATS['source']})")

    for control in (low, high):
        control.change(threshold_view, [low, high], [matrix, summary, curve])
        control.change(predict, [image, low, high], [probabilities, decision_text])
    image.change(predict, [image, low, high], [probabilities, decision_text])
    demo.load(threshold_view, [low, high], [matrix, summary, curve])


if __name__ == "__main__":
    demo.launch()
