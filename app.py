"""
Assignment 2: Gradio dashboard.

Run via `uv run python main.py app`. Loads the committed weights in weights/
and the outputs train.py wrote to outputs/. Never trains anything.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

import train as pipeline

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

model = YOLO(str(ROOT / "weights" / "best.pt"))
metrics = json.loads((OUTPUTS / "test_metrics.json").read_text())
sweep = json.loads((OUTPUTS / "cost_sweep.json").read_text())
baselines = json.loads((OUTPUTS / "baselines.json").read_text())
rows = json.loads((OUTPUTS / "image_scores.json").read_text())
names = [model.names[i] for i in range(len(model.names))]

assumptions = sweep["assumptions"]
chosen_low = sweep["chosen_pair"]["t_low"]
chosen_high = sweep["chosen_pair"]["t_high"]
chosen_single = sweep["chosen_single"]["t_low"]


def outcome(score: float, t_low: float, t_high: float) -> str:
    if score >= t_high:
        return "FAIL (flag as diseased)"
    if score < t_low:
        return "PASS (healthy)"
    return "HUMAN REVIEW"


def predict(image, conf, t_low, t_high):
    if image is None:
        return None, pd.DataFrame(), ""
    result = model.predict(image, conf=conf, imgsz=pipeline.IMGSZ, verbose=False)[0]
    table = pd.DataFrame(
        [(names[int(c)], round(float(s), 3)) for c, s in zip(result.boxes.cls, result.boxes.conf)],
        columns=["class", "confidence"],
    )
    full = model.predict(image, conf=pipeline.PRED_CONF_FLOOR, imgsz=pipeline.IMGSZ, verbose=False)[0]
    score = max(
        (float(s) for c, s in zip(full.boxes.cls, full.boxes.conf) if names[int(c)] in pipeline.DISEASE_CLASSES),
        default=0.0,
    )
    decision = f"Disease score {score:.3f} -> **{outcome(score, t_low, t_high)}**"
    return result.plot()[:, :, ::-1], table, decision


def cost_curve(costs, t_low, t_high):
    xs = pipeline.THRESHOLD_GRID
    ys = [pipeline.expected_cost(rows, t, t, costs)["cost_per_case"] * 1000 for t in xs]
    best = min(range(len(xs)), key=lambda i: ys[i])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(xs, ys, color="#1f5aa6", label="model alone (single threshold)")
    ax.axvline(xs[best], color="#c0392b", ls="--", label=f"minimum cost at {xs[best]:.2f}")
    ax.axvspan(t_low, t_high, color="#f39c12", alpha=0.2, label="current review band")
    ax.set_xlabel("disease-score threshold")
    ax.set_ylabel("expected cost per 1,000 cases (EUR)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def update(t_low, t_high, cost_fn, cost_fp, base_rate):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    costs = {"fn": cost_fn, "fp": cost_fp, "tp": cost_fp, "base_rate": base_rate}
    r = pipeline.expected_cost(rows, t_low, t_high, costs)
    c = r["counts"]
    three_way = pd.DataFrame(
        {"pass": [c["pos_pass"], c["neg_pass"]],
         "human review": [c["pos_review"], c["neg_review"]],
         "fail": [c["pos_fail"], c["neg_fail"]]},
        index=["truly diseased", "truly healthy"],
    ).reset_index(names="test images")
    cm = pipeline.class_confusion(rows, t_high, names)
    four_class = pd.DataFrame(cm["matrix"], columns=cm["cols_pred"], index=cm["rows_true"]).reset_index(
        names="true \\ predicted")
    per_day = r["review_share"] * assumptions["cases_per_day"]
    people = per_day * assumptions["review_minutes"] / assumptions["work_minutes_per_day"]
    summary = (
        f"### Expected cost: EUR {r['cost_per_case'] * 1000:,.0f} per 1,000 cases\n"
        f"- Sent to human review: **{r['review_share']:.1%}** "
        f"(~{per_day:.0f} of {assumptions['cases_per_day']} samples/day, ~{people:.2f} people)\n"
        f"- Disease recall (fail + review): {r['recall']:.1%} | healthy wrongly failed: {r['fpr']:.1%}\n"
        f"- Base rate {base_rate:.1%}; counts are raw test images, cost is reweighted to the base rate."
    )
    per_class = []
    for i, n in enumerate(names):
        row_total = sum(cm["matrix"][i])
        col_total = sum(cm["matrix"][j][i] for j in range(len(names)))
        hit = cm["matrix"][i][i]
        per_class.append({"class": n, "image recall @ t_high": round(hit / row_total, 3) if row_total else None,
                          "image precision @ t_high": round(hit / col_total, 3) if col_total else None})
    return summary, three_way, four_class, cost_curve(costs, t_low, t_high), pd.DataFrame(per_class)


det_table = pd.DataFrame(
    [{"class": k, **{m: round(v, 3) for m, v in vals.items()}} for k, vals in metrics["per_class"].items()]
)
baseline_table = pd.DataFrame([
    {"model": "finetuned yolo11s", "mAP50": round(metrics["map50"], 3),
     "cost per 1,000 (EUR)": round(sweep["chosen_pair"]["cost_per_case"] * 1000)},
    {"model": "COCO yolo11s zero-shot", "mAP50": 0.0,
     "cost per 1,000 (EUR)": round(baselines["coco_zero_shot"]["cost_per_case"] * 1000)},
    {"model": "predict nothing", "mAP50": 0.0,
     "cost per 1,000 (EUR)": round(baselines["predict_nothing"]["cost_per_case"] * 1000)},
    {"model": f"majority class ({baselines['majority_class']['class']})", "mAP50": None,
     "cost per 1,000 (EUR)": round(baselines["majority_class"]["cost_per_case"] * 1000)},
])

with gr.Blocks(title="Poultry droppings triage") as demo:
    gr.Markdown(
        "# Poultry droppings disease triage\n"
        f"Finetuned yolo11s. Test mAP50 **{metrics['map50']:.3f}**, mAP50-95 **{metrics['map50_95']:.3f}**. "
        f"Chosen review band: pass < {chosen_low:.2f} <= review < {chosen_high:.2f} <= fail."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=chosen_low, step=0.01, label="pass / review edge (t_low)")
        t_high = gr.Slider(0, 1, value=chosen_high, step=0.01, label="review / fail edge (t_high)")
    with gr.Row():
        cost_fn = gr.Number(value=assumptions["cost_fn"], label="cost of a missed diseased sample (EUR)")
        cost_fp = gr.Number(value=assumptions["cost_fp"], label="cost of failing a sample (EUR)")
        base_rate = gr.Slider(0.001, 0.5, value=assumptions["base_rate"], step=0.001, label="base rate")

    with gr.Tab("Operating point"):
        summary = gr.Markdown()
        three_way = gr.Dataframe(label="Test images by outcome")
        curve = gr.Plot(label="Cost against threshold")
    with gr.Tab("Per-class results"):
        gr.Dataframe(det_table, label="Detection metrics on test (Ultralytics, max-F1 confidence)")
        per_class = gr.Dataframe(label="Image-level per-class precision / recall at t_high")
        four_class = gr.Dataframe(label="Image-level confusion matrix at t_high (rows true, columns predicted)")
        gr.Dataframe(baseline_table, label="Finetuned model against baselines (same test split)")
    with gr.Tab("Try an image"):
        with gr.Row():
            image = gr.Image(type="filepath", label="dropping photo")
            annotated = gr.Image(label="predictions")
        conf = gr.Slider(0.01, 1, value=0.25, step=0.01, label="display confidence threshold")
        decision = gr.Markdown()
        detections = gr.Dataframe(label="detections")
        inputs = [image, conf, t_low, t_high]
        for comp in inputs:
            comp.change(predict, inputs, [annotated, detections, decision])

    controls = [t_low, t_high, cost_fn, cost_fp, base_rate]
    outputs = [summary, three_way, four_class, curve, per_class]
    for comp in controls:
        comp.change(update, controls, outputs)
    demo.load(update, controls, outputs)

if __name__ == "__main__":
    demo.launch()
