"""
Assignment 2: Gradio dashboard.   Run:  uv run python main.py app

Uses only what train.py already produced: weights/best.pt and results/.
It never trains anything.

Decision rule (same as costs.py), per fruit image:
    highest "rotten" confidence >= upper edge  -> DISCARD
    between lower and upper edge               -> HUMAN REVIEW
    below lower edge                           -> SHIP
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import costs  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# ---------------------------------------------------------------- load once
MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
NAMES = MODEL.names
ROTTEN_IDS = {i for i, n in NAMES.items() if costs.POSITIVE in n}

SCORES = pd.read_csv(RESULTS / "test_scores.csv")
S = SCORES["rotten_score"].to_numpy(dtype=float)
Y = SCORES["is_rotten"].astype(str).str.lower().eq("true").to_numpy()

POINT = json.loads((RESULTS / "operating_point.json").read_text())
METRICS = json.loads((RESULTS / "metrics.json").read_text())
PER_CLASS = pd.read_csv(RESULTS / "per_class.csv")
CM = np.loadtxt(RESULTS / "confusion_matrix.csv", delimiter=",")
CM_LABELS = json.loads((RESULTS / "confusion_matrix_labels.json").read_text())

DEFAULT_LOW = POINT["model_plus_review"]["t_low"]
DEFAULT_HIGH = POINT["model_plus_review"]["t_high"]
CHOSEN_SINGLE = POINT["model_alone"]["threshold"]
GRID = np.round(np.arange(0, 1.0001, 0.01), 2)


# ---------------------------------------------------------------- cost maths
def expected_cost(rates: dict, c_fp: float, c_fn: float, c_rev: float, p: float) -> float:
    """EUR per 1,000 fruits at the real base rate p (same formula as costs.py)."""
    rotten = p * (rates["fn"] * c_fn + rates["review_pos"] * c_rev)
    fresh = (1 - p) * (rates["fp"] * c_fp + rates["review_neg"] * c_rev)
    return 1000 * (rotten + fresh)


def decision_view(lo, hi, c_fp, c_fn, wage, secs, base_pct, per_day):
    hi = max(hi, lo)  # the upper edge can never sit below the lower edge
    p = base_pct / 100
    c_rev = wage * secs / 3600

    fail = S >= hi
    review = (S >= lo) & ~fail
    ship = ~fail & ~review
    outcomes = pd.DataFrame(
        {
            "Discard": [int((fail & Y).sum()), int((fail & ~Y).sum())],
            "Human review": [int((review & Y).sum()), int((review & ~Y).sum())],
            "Ship": [int((ship & Y).sum()), int((ship & ~Y).sum())],
        },
        index=["Actually rotten", "Actually fresh"],
    ).reset_index(names="Test images")

    rates = costs.outcome_rates(S, Y, lo, hi)
    cost = expected_cost(rates, c_fp, c_fn, c_rev, p)
    share = p * rates["review_pos"] + (1 - p) * rates["review_neg"]
    reviews_day = share * per_day
    hours_day = reviews_day * secs / 3600

    single = np.array([expected_cost(costs.outcome_rates(S, Y, t, t), c_fp, c_fn, c_rev, p) for t in GRID])
    best_t = float(GRID[single.argmin()])
    alternatives = pd.DataFrame(
        [
            ["Your design: model + human review band", cost],
            [f"Model decides alone (best threshold {best_t:.2f})", single.min()],
            ["Do nothing: ship everything", 1000 * p * c_fn],
            ["A person checks every fruit", 1000 * c_rev],
            ["Discard everything", 1000 * (1 - p) * c_fp],
        ],
        columns=["Design", "EUR per 1,000 fruits"],
    ).round(2)

    summary = (
        f"### Expected cost: **{cost:.2f} EUR per 1,000 fruits** "
        f"(about {cost * per_day / 1000:,.0f} EUR per day)\n"
        f"- Sent to a person: **{share:.1%}** of fruits at a {base_pct:.1f}% base rate "
        f"-> **{reviews_day:,.0f} checks/day**, **{hours_day:.1f} reviewer hours/day** "
        f"(~{hours_day / 8:.1f} full-time people)\n"
        f"- On the test set: {rates['fn']:.1%} of rotten fruit would be shipped, "
        f"{rates['fp']:.1%} of fresh fruit would be discarded.\n"
        f"- Assumes the reviewer is always right."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(GRID, single, label="Model decides alone (one threshold)")
    chosen_cost = expected_cost(costs.outcome_rates(S, Y, CHOSEN_SINGLE, CHOSEN_SINGLE), c_fp, c_fn, c_rev, p)
    ax.scatter([CHOSEN_SINGLE], [chosen_cost], zorder=3, label=f"Chosen operating point ({CHOSEN_SINGLE:.2f})")
    ax.axvspan(lo, hi, alpha=0.15, label=f"Current review band {lo:.2f}-{hi:.2f}")
    ax.axhline(cost, linestyle="--", label=f"Current design: {cost:.2f} EUR")
    ax.set_yscale("log")
    ax.set_xlabel("Confidence threshold for 'rotten'")
    ax.set_ylabel("Expected EUR per 1,000 fruits (log scale)")
    ax.set_title("Cost against threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return summary, outcomes, alternatives, fig


def predict_image(path, lo, hi):
    if path is None:
        return None, "Upload a photo to see the model's decision.", pd.DataFrame()
    hi = max(hi, lo)
    res = MODEL.predict(path, conf=max(lo, 0.001), verbose=False)[0]
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    rotten = [c for k, c in zip(cls, conf) if k in ROTTEN_IDS]
    score = float(max(rotten)) if rotten else 0.0
    if score >= hi:
        decision = "DISCARD (model is confident it is rotten)"
    elif score >= lo:
        decision = "HUMAN REVIEW (model is unsure)"
    else:
        decision = "SHIP (no rotten fruit above the lower edge)"
    text = f"### Decision: **{decision}**\nHighest rotten confidence: {score:.2f} (review band {lo:.2f}-{hi:.2f})"
    boxes = pd.DataFrame({"Class": [NAMES[k] for k in cls], "Confidence": conf.round(3)})
    return res.plot()[:, :, ::-1], text, boxes


# ---------------------------------------------------------------- static tables
fin, coco = METRICS["finetuned"], METRICS["coco_zero_shot"]
BASELINES = pd.DataFrame(
    [
        ["Finetuned YOLO11n", "6 classes: fruit type x fresh/rotten", fin["map50"], fin["map50_95"]],
        ["COCO YOLO11n, zero-shot", "fruit type only (cannot see rot)", coco["fruit_type_map50"], None],
        ["Predict nothing", "6 classes", 0.0, 0.0],
    ],
    columns=["Model", "What it was scored on", "mAP50", "mAP50-95"],
).round(3)
CM_TABLE = pd.DataFrame(
    CM.astype(int),
    index=[f"predicted: {n}" for n in CM_LABELS],
    columns=[f"true: {n}" for n in CM_LABELS],
).reset_index(names="")


# ---------------------------------------------------------------- layout
with gr.Blocks(title="Rotten fruit sorting") as demo:
    gr.Markdown(
        "# Rotten fruit sorting: what to ship, discard, or send to a person\n"
        "Each fruit gets the model's highest *rotten* confidence. Move the two edges of the "
        "review band and everything below updates together."
    )
    with gr.Row():
        lo = gr.Slider(0, 1, value=DEFAULT_LOW, step=0.01, label="Lower edge: below this, ship")
        hi = gr.Slider(0, 1, value=DEFAULT_HIGH, step=0.01, label="Upper edge: above this, discard")
    with gr.Accordion("Cost assumptions (defaults from costs.py)", open=False):
        with gr.Row():
            c_fp = gr.Number(costs.COST_FALSE_POSITIVE, label="False positive: good fruit discarded (EUR)")
            c_fn = gr.Number(costs.COST_FALSE_NEGATIVE, label="False negative: rotten fruit shipped (EUR)")
            wage = gr.Number(costs.HOURLY_WAGE, label="Reviewer cost per hour (EUR)")
        with gr.Row():
            secs = gr.Number(costs.REVIEW_SECONDS, label="Seconds per human check")
            base_pct = gr.Number(costs.BASE_RATE * 100, label="Base rate: % of fruit really rotten")
            per_day = gr.Number(costs.CASES_PER_DAY or 100_000, label="Fruits per day")

    with gr.Tabs():
        with gr.Tab("Decision and cost"):
            summary = gr.Markdown()
            with gr.Row():
                outcomes = gr.Dataframe(label="Confusion matrix on the test set (images)")
                alternatives = gr.Dataframe(label="Your design against the alternatives")
            curve = gr.Plot(label="Cost against threshold")
        with gr.Tab("Try an image"):
            with gr.Row():
                img_in = gr.Image(type="filepath", label="Upload a fruit photo")
                img_out = gr.Image(label="Boxes at or above the lower edge")
            decision = gr.Markdown()
            boxes = gr.Dataframe(label="Detections")
        with gr.Tab("Model performance"):
            gr.Markdown("### Finetuned model against the baselines (test split)")
            gr.Dataframe(BASELINES)
            gr.Markdown("### Per-class performance (test split)")
            gr.Dataframe(PER_CLASS)
            gr.Markdown("### Detection confusion matrix (test split)")
            gr.Dataframe(CM_TABLE)

    cost_inputs = [lo, hi, c_fp, c_fn, wage, secs, base_pct, per_day]
    cost_outputs = [summary, outcomes, alternatives, curve]
    for component in cost_inputs:
        component.change(decision_view, cost_inputs, cost_outputs)
    for component in (img_in, lo, hi):
        component.change(predict_image, [img_in, lo, hi], [img_out, decision, boxes])
    demo.load(decision_view, cost_inputs, cost_outputs)


if __name__ == "__main__":
    demo.launch()