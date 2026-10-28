"""
app.py - the Gradio dashboard. Run it with:

    uv run python main.py app

It NEVER trains anything. It only loads:
  - weights/best.pt   (the model train.py saved)
  - results/*         (the test-set scores and tables train.py saved)

Then it lets you:
  - upload a photo of fruit and see what the model says
  - move the two cut-offs (pass / review / bin) and watch the outcome table,
    the cost per 1,000 fruit and the share sent to a person update
  - see the cost-against-threshold curve with the chosen threshold marked
  - see per-class results and the baselines
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import costing  # noqa: E402
import settings as S  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Load what train.py produced (no training here)
# ---------------------------------------------------------------------------
model = YOLO(str(ROOT / "weights" / "best.pt"))
POSITIVE_IDS = [i for i, n in model.names.items() if S.POSITIVE_WORD.lower() in n.lower()]

predictions = pd.read_csv(RESULTS / "test_predictions.csv")
SCORES = predictions["rotten_score"].to_numpy()
IS_ROTTEN = predictions["is_rotten"].to_numpy().astype(bool)

per_class = pd.read_csv(RESULTS / "per_class.csv")
confusion = pd.read_csv(RESULTS / "confusion_matrix.csv")
baselines = pd.read_csv(RESULTS / "baselines.csv")
dataset_counts = pd.read_csv(RESULTS / "dataset_counts.csv")

# Cost numbers are recomputed from settings.py, so they always match it.
SWEEP = costing.sweep_single_threshold(SCORES, IS_ROTTEN)
BEST_T = float(SWEEP.loc[SWEEP["cost_per_1000"].idxmin(), "low"])
BAND = costing.best_band(SCORES, IS_ROTTEN)


def decide(score: float, low: float, high: float) -> str:
    if score < low:
        return "PASS - send to the shelf"
    if score >= high:
        return "FAIL - throw in the bin"
    return "REVIEW - a worker checks it"


def fix_order(low: float, high: float):
    """The pass cut-off can't be above the bin cut-off."""
    return (high, low) if low > high else (low, high)


# ---------------------------------------------------------------------------
# Tab 1: try a photo
# ---------------------------------------------------------------------------
def predict(image_path, low, high):
    if image_path is None:
        return None, "Upload a photo of a piece of fruit to see the model's answer."
    low, high = fix_order(low, high)
    probs = model(image_path, verbose=False)[0].probs.data.cpu().numpy()
    label = {model.names[i]: float(p) for i, p in enumerate(probs)}
    score = float(probs[POSITIVE_IDS].sum())
    text = (f"### Rotten score: {score:.2f}\n"
            f"Pass below **{low:.2f}**, bin from **{high:.2f}**.\n\n"
            f"## Decision: {decide(score, low, high)}")
    return label, text


# ---------------------------------------------------------------------------
# Tab 2: thresholds and cost
# ---------------------------------------------------------------------------
def outcome_table(low, high) -> pd.DataFrame:
    """Confusion matrix with three outcomes, counted on the test split."""
    rows = []
    for name, mask in [("Actually rotten", IS_ROTTEN), ("Actually fresh", ~IS_ROTTEN)]:
        s = SCORES[mask]
        rows.append({
            "": name,
            "Pass (shelf)": int((s < low).sum()),
            "Review (worker)": int(((s >= low) & (s < high)).sum()),
            "Fail (bin)": int((s >= high).sum()),
        })
    return pd.DataFrame(rows)


def alternatives_table(current: dict) -> pd.DataFrame:
    table = costing.alternatives(SCORES, IS_ROTTEN, SWEEP, BAND)
    mine = {"option": "Your current sliders", **current}
    table = pd.concat([table, pd.DataFrame([mine])[table.columns]], ignore_index=True)
    return pd.DataFrame({
        "Option": table["option"],
        "Cost per 1,000 fruit (EUR)": table["cost_per_1000"].round(2),
        "Sent to a person": (table["review_share"] * 100).round(1).astype(str) + "%",
        "Reviews per day": table["reviews_per_day"].round(0).astype(int),
        "Workers needed": table["people_needed"].astype(int),
        "Rotten missed": (table["rotten_missed"] * 100).round(1).astype(str) + "%",
        "Fresh binned": (table["fresh_binned"] * 100).round(1).astype(str) + "%",
    })


def update_costs(low, high):
    low, high = fix_order(low, high)
    r = costing.evaluate(SCORES, IS_ROTTEN, low, high)
    fits = "fits" if r["people_needed"] <= S.REVIEWERS_AVAILABLE else "does NOT fit"
    text = (
        f"## Expected cost: EUR {r['cost_per_1000']:,.2f} per 1,000 fruit\n"
        f"**Sent to a person:** {r['review_share']:.1%} of fruit "
        f"= {r['reviews_per_day']:,.0f} checks per day "
        f"= {r['review_hours_per_day']:.1f} hours "
        f"= **{r['people_needed']} worker(s)** "
        f"(you have {S.REVIEWERS_AVAILABLE}, so this {fits}).\n\n"
        f"Rotten fruit reaching the shelf: {r['rotten_missed']:.1%} - "
        f"fresh fruit thrown away: {r['fresh_binned']:.1%}.\n\n"
        f"*Costs use a real-life base rate of {S.BASE_RATE:.0%} rotten fruit, "
        f"not the mix in the test set. Table below counts test images.*"
    )
    fig = costing.plot_cost_curve(SWEEP, BEST_T, low=low, high=high)
    table = outcome_table(low, high)
    plt.close("all")
    return text, table, fig, alternatives_table(r)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Fruit sorting: pass, review or bin") as demo:
    gr.Markdown(
        "# Fruit sorting: pass, review or bin\n"
        "The model gives each piece of fruit a **rotten score** from 0 to 1. "
        "Below the first cut-off it goes to the shelf, from the second cut-off it goes "
        "in the bin, and anything in between is checked by a worker."
    )
    with gr.Row():
        low_slider = gr.Slider(0, 1, value=BAND["low"], step=0.01,
                               label="Pass cut-off: below this, send to the shelf")
        high_slider = gr.Slider(0, 1, value=BAND["high"], step=0.01,
                                label="Bin cut-off: from this up, throw away")
    gr.Markdown(
        f"Cheapest setting found in training: pass below **{BAND['low']:.2f}**, "
        f"bin from **{BAND['high']:.2f}** (EUR {BAND['cost_per_1000']:,.2f} per 1,000 fruit). "
        f"Cheapest single threshold with no worker: **{BEST_T:.2f}**."
    )

    with gr.Tab("Try a photo"):
        with gr.Row():
            image_in = gr.Image(type="filepath", label="Photo of a piece of fruit")
            with gr.Column():
                label_out = gr.Label(num_top_classes=6, label="Model's answer")
                decision_out = gr.Markdown()

    with gr.Tab("Cost and review"):
        cost_text = gr.Markdown()
        outcome_out = gr.Dataframe(label="Outcomes on the test split (number of images)")
        curve_out = gr.Plot(label="Cost against threshold")
        alt_out = gr.Dataframe(label="Compared with the alternatives")

    with gr.Tab("Per-class results"):
        gr.Markdown(f"Positive class: any class containing **'{S.POSITIVE_WORD}'**.")
        gr.Dataframe(value=per_class.round(3), label="Precision and recall per class (test split)")
        gr.Dataframe(value=confusion, label="Confusion matrix: rows = actual, columns = predicted")
        gr.Dataframe(value=baselines.round(3), label="Finetuned model vs the two baselines")
        gr.Dataframe(value=dataset_counts, label="Images per class in each split")

    # Wiring: anything that changes -> recompute
    for slider in (low_slider, high_slider):
        slider.change(update_costs, [low_slider, high_slider],
                      [cost_text, outcome_out, curve_out, alt_out])
        slider.change(predict, [image_in, low_slider, high_slider], [label_out, decision_out])
    image_in.change(predict, [image_in, low_slider, high_slider], [label_out, decision_out])
    demo.load(update_costs, [low_slider, high_slider], [cost_text, outcome_out, curve_out, alt_out])


if __name__ == "__main__":
    demo.launch()
