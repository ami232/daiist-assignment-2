"""
Assignment 2 - Gradio dashboard.
Run with: uv run python main.py app

Serves the committed weights in weights/ and the artefacts written by the
training stage into results/. Trains nothing at startup: every number on
screen is recomputed from results/test_scores.csv, which is one row per
test image (y_true, score).
"""

import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from train import (  # single source of truth for the cost model
    BASE_RATE, COST_FN, COST_FP, COST_REVIEW, COST_TN, COST_TP, DEFECT_KEYWORDS,
)

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
RESULTS = ROOT / "results"


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"Missing {path.relative_to(ROOT)}. Run `uv run python main.py train` first."
        )
    return path


MODEL = YOLO(_require(WEIGHTS))
SCORES = pd.read_csv(_require(RESULTS / "test_scores.csv"))
PER_CLASS = pd.read_csv(_require(RESULTS / "per_class.csv")).round(3)
METRICS = json.loads(_require(RESULTS / "metrics.json").read_text())
DEFECT_IDS = {i for i, n in MODEL.names.items()
              if any(k in str(n).lower() for k in DEFECT_KEYWORDS)}


def band_rates(t_low: float, t_high: float) -> dict:
    """Split the test set three ways, conditional on the true label.

    score <  t_low            -> pass automatically
    t_low <= score < t_high   -> send to a human
    score >= t_high           -> fail automatically

    Rates are conditional on y_true so they can be reweighted to any
    base rate: the test split's prevalence is not the line's prevalence.
    """
    if t_high < t_low:
        t_low, t_high = t_high, t_low
    out = {}
    for label, key in ((1, "pos"), (0, "neg")):
        s = SCORES.loc[SCORES["y_true"] == label, "score"]
        n = max(len(s), 1)
        out[f"{key}_fail"] = float((s >= t_high).sum()) / n
        out[f"{key}_review"] = float(((s >= t_low) & (s < t_high)).sum()) / n
        out[f"{key}_pass"] = float((s < t_low).sum()) / n
    return out


def cost_per_1000(t_low, t_high, base_rate, c_fp, c_fn, c_review) -> float:
    """Expected euro cost per 1,000 pieces at a given operating point.

    A reviewed case costs c_review and is assumed to be judged correctly,
    which is the optimistic assumption and is stated as such in REPORT.md.
    """
    r = band_rates(t_low, t_high)
    p = base_rate
    per_case = (
        p * (r["pos_fail"] * COST_TP + r["pos_review"] * c_review + r["pos_pass"] * c_fn)
        + (1 - p) * (r["neg_pass"] * COST_TN + r["neg_review"] * c_review
                     + r["neg_fail"] * c_fp)
    )
    return 1000 * per_case


def review_share(t_low, t_high, base_rate) -> float:
    r = band_rates(t_low, t_high)
    return base_rate * r["pos_review"] + (1 - base_rate) * r["neg_review"]


def single_threshold_curve(base_rate, c_fp, c_fn):
    """Cost against a single threshold, model deciding alone (no review band)."""
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = [cost_per_1000(t, t, base_rate, c_fp, c_fn, 0.0) for t in grid]
    return grid, np.array(costs)


def confusion_table(t_low, t_high, base_rate):
    """Counts per 1,000 pieces, at the assumed base rate, in three columns."""
    r = band_rates(t_low, t_high)
    pos, neg = 1000 * base_rate, 1000 * (1 - base_rate)
    return pd.DataFrame({
        "": ["Truly defective", "Truly good"],
        "Auto-pass": [round(pos * r["pos_pass"], 1), round(neg * r["neg_pass"], 1)],
        "Human review": [round(pos * r["pos_review"], 1), round(neg * r["neg_review"], 1)],
        "Auto-fail": [round(pos * r["pos_fail"], 1), round(neg * r["neg_fail"], 1)],
    })


def cost_plot(t_low, t_high, base_rate, c_fp, c_fn):
    grid, costs = single_threshold_curve(base_rate, c_fp, c_fn)
    best = float(grid[int(np.argmin(costs))])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, costs, linewidth=2)
    ax.axvspan(min(t_low, t_high), max(t_low, t_high), alpha=0.15,
               label="human review band")
    ax.axvline(best, linestyle="--", color="red",
               label=f"cost-minimising threshold = {best:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("expected cost per 1,000 pieces (EUR)")
    ax.legend()
    fig.tight_layout()
    return fig


def predict(image, threshold):
    """Run the committed model on an uploaded image at the current threshold."""
    if image is None:
        return None, "Upload an image."
    res = MODEL.predict(image, conf=float(threshold), verbose=False)[0]
    annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
    lines = []
    worst = 0.0
    for conf, cls in zip(res.boxes.conf, res.boxes.cls):
        name = MODEL.names[int(cls)]
        lines.append(f"{name}: {float(conf):.2f}")
        if int(cls) in DEFECT_IDS:
            worst = max(worst, float(conf))
    verdict = f"Highest defect confidence: {worst:.2f}"
    return annotated, verdict + ("\n" + "\n".join(lines) if lines else "\nNo detections.")


def refresh(t_low, t_high, base_rate, c_fp, c_fn, c_review, throughput):
    cost = cost_per_1000(t_low, t_high, base_rate, c_fp, c_fn, c_review)
    share = review_share(t_low, t_high, base_rate)
    model_alone = cost_per_1000(t_high, t_high, base_rate, c_fp, c_fn, 0.0)
    do_nothing = 1000 * base_rate * c_fn
    check_all = 1000 * c_review

    daily_reviews = share * throughput
    minutes = daily_reviews * 10 / 60  # 10 s per check
    people = minutes / (60 * 7.5)      # 7.5 productive hours per shift

    headline = (
        f"## {cost:,.0f} EUR per 1,000 pieces\n"
        f"- Sent to a human: **{share * 100:.1f}%**\n"
        f"- At {throughput:,.0f} pieces/day that is **{daily_reviews:,.0f} checks/day**, "
        f"about **{minutes:,.0f} min** of work, roughly **{people:.2f} FTE**\n\n"
        f"### Priced against the alternatives\n"
        f"- Do nothing: **{do_nothing:,.0f} EUR**\n"
        f"- Human checks everything: **{check_all:,.0f} EUR**\n"
        f"- Model decides alone at {max(t_low, t_high):.2f}: **{model_alone:,.0f} EUR**\n"
    )
    return (headline, confusion_table(t_low, t_high, base_rate),
            cost_plot(t_low, t_high, base_rate, c_fp, c_fn))


with gr.Blocks(title="Fruit quality inspection - operating point") as demo:
    gr.Markdown(
        "# Fruit quality inspection\n"
        f"Finetuned {METRICS['training']['base_model']} on "
        f"`{METRICS['dataset']['workspace']}/{METRICS['dataset']['project']}` "
        f"v{METRICS['dataset']['version']}. Positive class = defective fruit."
    )

    with gr.Tab("Operating point"):
        with gr.Row():
            with gr.Column(scale=1):
                t_low = gr.Slider(0.0, 1.0, 0.25, step=0.01, label="Pass below this")
                t_high = gr.Slider(0.0, 1.0, 0.50, step=0.01, label="Fail above this")
                base_rate = gr.Slider(0.0, 0.5, BASE_RATE, step=0.005, label="Base rate")
                c_fp = gr.Number(COST_FP, label="Cost of a false positive (EUR)")
                c_fn = gr.Number(COST_FN, label="Cost of a false negative (EUR)")
                c_review = gr.Number(COST_REVIEW, label="Cost of one human review (EUR)")
                throughput = gr.Number(50000, label="Pieces inspected per day")
            with gr.Column(scale=2):
                headline = gr.Markdown()
                matrix = gr.Dataframe(label="Outcomes per 1,000 pieces", interactive=False)
                plot = gr.Plot(label="Cost against threshold")

        inputs = [t_low, t_high, base_rate, c_fp, c_fn, c_review, throughput]
        outputs = [headline, matrix, plot]
        for component in inputs:
            component.change(refresh, inputs, outputs)
        demo.load(refresh, inputs, outputs)

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload an image")
            img_out = gr.Image(label="Predictions")
        conf = gr.Slider(0.01, 1.0, 0.25, step=0.01, label="Confidence threshold")
        detections = gr.Textbox(label="Detections", lines=8)
        for component in (img_in, conf):
            component.change(predict, [img_in, conf], [img_out, detections])

    with gr.Tab("Per-class results"):
        gr.Dataframe(PER_CLASS, label="Test split, per class", interactive=False)
        gr.JSON(METRICS, label="Headline metrics and baselines")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("APP_PORT", 7860)))
