"""Gradio dashboard. Loads committed artefacts only and trains nothing.

Run via:  uv run python main.py app
Needs: weights/best.pt, costs.json, outputs/costing_results.json, outputs/eval_results.json
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import gradio as gr
import numpy as np
from matplotlib.figure import Figure
from ultralytics import YOLO

from costing import THR, C, params, three_action  # reads costs.json on import

OUT = ROOT / "outputs"
RES = json.loads((OUT / "costing_results.json").read_text())
EVAL = json.loads((OUT / "eval_results.json").read_text())
N = len(json.loads((OUT / "predictions_test.json").read_text())["images"])
FP, FN = np.array(RES["F_pos"]), np.array(RES["F_neg"])
P = params()
CHOSEN = RES["chosen"]
MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
CM_IMAGE = OUT / "confusion_matrix.png"
RATES = sorted({1e-5, 1e-4, 1e-3, 1e-2, float(C["base_rate"])})


def idx(t):
    return int(np.argmin(np.abs(THR - t)))


def update(t_review, t_alert, base_rate):
    t_alert = max(t_alert, t_review)
    i, j = idx(t_review), idx(t_alert)
    cost, share = three_action(FP, FN, base_rate, P)

    pos = [1 - FP[i], FP[i] - FP[j], FP[j]]
    neg = [1 - FN[i], FN[i] - FN[j], FN[j]]
    rows = [[f"Frames with fire/smoke (test, n={N})"] + [int(round(N * v)) for v in pos],
            [f"Empty-frame proxy (n={N})"] + [int(round(N * v)) for v in neg]]

    reviews = float(share[i, j] * P["frames_day"])
    needed = reviews * C["minutes_per_review"] / 60 / C["effective_hours_per_operator"]
    alone = base_rate * (1 - FP[j]) * P["c_fn"] + (1 - base_rate) * FN[j] * C["false_dispatch_eur"]
    md = (
        f"**Thresholds:** review >= {THR[i]:.2f}, alert >= {THR[j]:.2f}\n\n"
        f"**Expected cost per 1,000 cases:** {cost[i, j] * 1000:.2f} EUR\n\n"
        f"**Share sent to review:** {share[i, j]:.2%} "
        f"({reviews:.0f} reviews/day, {needed:.2f} operators needed, capacity {P['capacity']:.0f}/day)\n\n"
        f"Comparison per 1,000 cases: do nothing {base_rate * P['c_fn'] * 1000:.2f} EUR | "
        f"human checks everything {P['c_rev'] * 1000:.2f} EUR | "
        f"model decides alone (at the alert threshold) {alone * 1000:.2f} EUR\n\n"
        f"Assumed base rate: {base_rate:g}. The empty-frame row is a proxy: the test set has no empty frames, "
        f"so it uses the top score of boxes matching no label."
    )
    if reviews > P["capacity"]:
        md += "\n\n**Warning: review load exceeds operator capacity.**"
    if t_alert > t_review or True:
        pass

    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.plot(THR, np.diag(cost) * 1000, label="model + operator (single threshold)")
    ax.axhline(base_rate * P["c_fn"] * 1000, ls="--", c="gray", label="do nothing")
    ax.axhline(P["c_rev"] * 1000, ls="--", c="k", label="human checks everything")
    ax.axvline(CHOSEN["review_threshold"], c="g", ls="-", label="chosen review threshold (costs.json base rate)")
    ax.axvline(CHOSEN["alert_threshold"], c="r", ls="-", label="chosen alert threshold (costs.json base rate)")
    ax.axvline(THR[i], c="g", ls=":", label="current review")
    ax.axvline(THR[j], c="r", ls=":", label="current alert")
    ax.set_yscale("log")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("expected cost per 1,000 frames (EUR)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return rows, md, fig


def predict(image, t_review, t_alert):
    if image is None:
        return None, "Upload an image."
    t_alert = max(t_alert, t_review)
    bgr = np.ascontiguousarray(image[..., ::-1])
    res = MODEL.predict(bgr, conf=0.001, imgsz=640, verbose=False)[0]
    top = float(res.boxes.conf.max()) if len(res.boxes) else 0.0
    decision = "ALERT" if top >= t_alert else "REVIEW" if top >= t_review else "PASS"
    shown = MODEL.predict(bgr, conf=max(t_review, 0.001), imgsz=640, verbose=False)[0]
    return shown.plot()[..., ::-1], f"Top confidence {top:.2f} -> **{decision}**"


def per_class_rows():
    t = EVAL["test"]
    rows = [[k, round(v["precision"], 3), round(v["recall"], 3), round(v["ap50"], 3), round(v["map50_95"], 3)]
            for k, v in t["per_class"].items()]
    o = t["overall"]
    rows.append(["all", round(o["precision"], 3), round(o["recall"], 3), round(o["ap50"], 3), round(o["map50_95"], 3)])
    return rows


with gr.Blocks(title="Wildfire early-warning decision dashboard") as demo:
    gr.Markdown("# Wildfire early-warning: pass / human review / alert\n"
                "Model: yolo11s finetuned on fire-smoke-lwjte v1. Nothing is trained here.")
    with gr.Row():
        t_review = gr.Slider(0.01, 1.0, value=CHOSEN["review_threshold"], step=0.01,
                             label="Review threshold (send to operator queue at or above)")
        t_alert = gr.Slider(0.01, 1.0, value=CHOSEN["alert_threshold"], step=0.01,
                            label="Alert threshold (alert at or above)")
        base_rate = gr.Dropdown(choices=[(f"{r:g}", r) for r in RATES], value=float(C["base_rate"]),
                                label="Assumed base rate (share of frames with real fire/smoke)")
    with gr.Tab("Decision policy"):
        conf_table = gr.Dataframe(headers=["True state", "Pass", "Review", "Alert"], interactive=False)
        summary = gr.Markdown()
        plot = gr.Plot()
    with gr.Tab("Try an image"):
        image = gr.Image(type="numpy", label="Upload an image")
        annotated = gr.Image(label="Predictions at the current review threshold")
        verdict = gr.Markdown()
    with gr.Tab("Per-class results (test split)"):
        gr.Dataframe(headers=["Class", "Precision", "Recall", "AP50", "mAP50-95"],
                     value=per_class_rows(), interactive=False)
        if CM_IMAGE.exists():
            gr.Image(value=str(CM_IMAGE), label="Box-level confusion matrix (test)")

    inputs = [t_review, t_alert, base_rate]
    for comp in inputs:
        comp.change(update, inputs, [conf_table, summary, plot])
    demo.load(update, inputs, [conf_table, summary, plot])
    for comp in (image, t_review, t_alert):
        comp.change(predict, [image, t_review, t_alert], [annotated, verdict])

if __name__ == "__main__":
    demo.launch()