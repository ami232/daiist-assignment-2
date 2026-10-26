"""Assignment 2 dashboard. Loads the trained weights and the files saved by
train.py (outputs/). It never trains anything."""
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
M = json.loads((OUT / "metrics.json").read_text())
SWEEP = pd.read_csv(OUT / "cost_sweep.csv")
SCORES = pd.read_csv(OUT / "test_scores.csv")
MODEL = YOLO(str(ROOT / "weights" / "best.pt"))

C = M["costs"]
POS = M["positive_class"]
CHOSEN = M["chosen_threshold"]
Y = SCORES["y_true"].values
S = SCORES["score"].values


def evaluate(t_low, t_high):
    """Three outcomes: score < t_low -> pass, score >= t_high -> fail (alert),
    in between -> human review. A person is assumed to decide correctly."""
    lo, hi = sorted((float(t_low), float(t_high)))
    passed, failed = S < lo, S >= hi
    review = ~passed & ~failed
    pos, neg = Y == 1, Y == 0

    def rates(mask):
        n = max(int(mask.sum()), 1)
        return passed[mask].sum() / n, review[mask].sum() / n, failed[mask].sum() / n

    p_pass, p_rev, p_fail = rates(pos)
    n_pass, n_rev, n_fail = rates(neg)
    b = C["BASE_RATE"]
    cost = 1000 * (
        b * (p_pass * C["C_FN"] + p_rev * (C["C_REVIEW"] + C["C_TP"]) + p_fail * C["C_TP"])
        + (1 - b) * (n_pass * C["C_TN"] + n_rev * (C["C_REVIEW"] + C["C_TN"]) + n_fail * C["C_FP"])
    )
    share = b * p_rev + (1 - b) * n_rev  # share of cases sent to a person at the assumed base rate
    cm = pd.DataFrame(
        {
            "Actual": ["Bare head", "No bare head"],
            "Pass": [int(passed[pos].sum()), int(passed[neg].sum())],
            "Human review": [int(review[pos].sum()), int(review[neg].sum())],
            "Fail (alert)": [int(failed[pos].sum()), int(failed[neg].sum())],
        }
    )
    return cm, cost, share, lo, hi


def cost_plot(lo, hi):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(SWEEP["threshold"], SWEEP["cost_per_1000"], color="tab:blue", label="Cost, single threshold")
    ax.axvline(CHOSEN, color="green", lw=2, label=f"Chosen threshold = {CHOSEN:.2f}")
    ax.axvline(lo, color="gray", ls="--", label="Current pass/review")
    ax.axvline(hi, color="red", ls="--", label="Current review/fail")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Expected cost per 1,000 cases (EUR)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def update(t_low, t_high, cases_day, minutes_per_check, hours_per_person):
    cm, cost, share, lo, hi = evaluate(t_low, t_high)
    reviews_day = share * cases_day
    people = reviews_day * minutes_per_check / 60 / hours_per_person
    summary = (
        f"### Expected cost per 1,000 cases: **{cost:,.0f} EUR**\n"
        f"### Share sent to human review: **{share:.1%}** (at the assumed base rate of {C['BASE_RATE']:.0%})\n\n"
        f"Reviews per day: **{reviews_day:,.0f}**. "
        f"People needed: **{people:.2f}** full-time reviewers.\n\n"
        f"Reference cost per 1,000 cases: predict nothing = {M['baselines']['predict_nothing']['cost_per_1000']:,.0f} EUR, "
        f"person checks everything = {M['baselines']['human_checks_everything_per_1000']:,.0f} EUR."
    )
    return cm, summary, cost_plot(lo, hi)


def predict(img, t_low, t_high):
    if img is None:
        return None, "Upload an image first."
    lo, hi = sorted((float(t_low), float(t_high)))
    r = MODEL.predict(img[:, :, ::-1], conf=max(lo, 0.001), imgsz=M["training"]["imgsz"], verbose=False)[0]
    score = max(
        [c for k, c in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()) if M["classes"][int(k)] == POS],
        default=0.0,
    )
    if score >= hi:
        decision = "FAIL: bare head found, alert the supervisor"
    elif score >= lo:
        decision = "HUMAN REVIEW: a person must check this image"
    else:
        decision = "PASS: no bare head found"
    return r.plot()[:, :, ::-1], f"**{decision}**  \nHighest `{POS}` confidence: {score:.2f}"


def per_class_table():
    rows = []
    for name, v in M["finetuned"]["per_class"].items():
        rows.append([name, v["precision"], v["recall"], v["ap50"], v["ap50_95"]])
    rows.append(["ALL (overall)", None, None, M["finetuned"]["map50"], M["finetuned"]["map50_95"]])
    return pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "AP50", "mAP50-95"]).round(3)


with gr.Blocks(title="Hard hat check: cost and review dashboard") as demo:
    gr.Markdown(
        f"# Hard hat check\nPositive class: **{POS}** (a bare head). "
        "Move the two thresholds to see cost and human-review load change."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=0.01, step=0.005, label="Pass / review threshold")
        t_high = gr.Slider(0, 1, value=0.5, step=0.005, label="Review / fail threshold")
    with gr.Row():
        cases_day = gr.Number(value=2000, label="Images per day (my assumption)")
        minutes = gr.Number(value=0.5, label="Minutes per review (my assumption)")
        hours = gr.Number(value=7, label="Productive hours per person per day")

    with gr.Tabs():
        with gr.Tab("Try an image"):
            img_in = gr.Image(type="numpy", label="Upload an image")
            btn = gr.Button("Predict")
            img_out = gr.Image(label="Predictions")
            verdict = gr.Markdown()
            btn.click(predict, [img_in, t_low, t_high], [img_out, verdict])
        with gr.Tab("Cost and review"):
            summary = gr.Markdown()
            cm_out = gr.Dataframe(label="Confusion matrix on the test split (counts)")
            plot_out = gr.Plot(label="Cost against threshold")
        with gr.Tab("Per-class results"):
            gr.Dataframe(per_class_table(), label="Finetuned model on the test split")
            gr.Markdown(
                f"Test images: {M['test_images']} ({M['test_positive_images']} with a bare head). "
                f"Instances per class: {json.dumps(M['instances'])}"
            )
            cm_img = OUT / "confusion_matrix_normalized.png"
            if cm_img.exists():
                gr.Image(str(cm_img), label="Detector confusion matrix (normalized)")

    controls = [t_low, t_high, cases_day, minutes, hours]
    for c in controls:
        c.change(update, controls, [cm_out, summary, plot_out])
    demo.load(update, controls, [cm_out, summary, plot_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("APP_PORT", 7860)))