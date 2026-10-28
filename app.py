"""
Assignment 2: Gradio dashboard.

    uv run python main.py app
    QUICK=1 uv run python app.py   # against the smoke test outputs in runs/quick

Loads the committed weights and what train.py saved in artifacts/. It never
trains, and it does not need the dataset or the Roboflow key. All euro figures
and the base rate come from costs.py, the same module train.py used.
"""

import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costs

ROOT = Path(__file__).resolve().parent

QUICK = os.environ.get("QUICK") == "1"
if QUICK:
    WEIGHTS = ROOT / "runs" / "quick" / "best.pt"
    ARTIFACTS = ROOT / "runs" / "quick" / "artifacts"
else:
    WEIGHTS = ROOT / "weights" / "best.pt"
    ARTIFACTS = ROOT / "artifacts"
EXAMPLES = ROOT / "examples"

POSITIVE_CLASS = "head"
MIN_DRAW_CONF = 0.05
BOX_COLORS = {"head": (220, 40, 40), "helmet": (40, 170, 70)}  # RGB

model = YOLO(str(WEIGHTS))
POSITIVE_ID = {v: k for k, v in model.names.items()}[POSITIVE_CLASS]

metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
IMGSZ = metrics["training"].get("imgsz", 640)  # predict at the size it was trained at
operating_point = json.loads((ARTIFACTS / "operating_point.json").read_text())
sweep = pd.read_csv(ARTIFACTS / "cost_sweep.csv")
test_scores = pd.read_csv(ARTIFACTS / "test_scores.csv")
SCORES = test_scores["score"].to_numpy(dtype=float)
LABELS = test_scores["has_no_hard_hat"].to_numpy(dtype=bool)

CHOSEN = float(operating_point["threshold"])
band = operating_point.get("review_band", {})
DEFAULT_LOW = float(band.get("low", CHOSEN))
DEFAULT_HIGH = float(band.get("high", CHOSEN))


def decide(score: float, low: float, high: float) -> str:
    if score < low:
        return "PASS"
    if score >= max(high, low):
        return "FAIL"
    return "REVIEW"


def check_image(image: np.ndarray | None, low: float, high: float):
    """Predict on one uploaded image and turn its top `head` score into a decision."""
    if image is None:
        return None, "Upload an image or pick an example."
    bgr = np.ascontiguousarray(image[..., ::-1])
    result = model.predict(bgr, imgsz=IMGSZ, conf=0.001, verbose=False)[0]
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    head_confs = confs[classes == POSITIVE_ID]
    score = float(head_confs.max()) if head_confs.size else 0.0
    decision = decide(score, low, high)

    canvas = image.copy()
    draw_at = max(min(low, high), MIN_DRAW_CONF)
    for (x1, y1, x2, y2), conf, cls in zip(xyxy.astype(int), confs, classes):
        if conf < draw_at:
            continue
        name = model.names[cls]
        color = BOX_COLORS.get(name, (60, 60, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, f"{name} {conf:.2f}", (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    text = (
        f"## {decision}\n"
        f"Highest 'no hard hat' score: **{score:.2f}**. "
        f"Pass below {low:.2f}, fail at or above {max(high, low):.2f}, review in between. "
        f"Boxes drawn from confidence {draw_at:.2f} (red: no hard hat, green: hard hat)."
    )
    return canvas, text


def cost_curve(low: float, high: float) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=sweep["threshold"], y=sweep["cost_per_1000"], mode="lines",
        name="Model alone", line={"color": "#2a6fdb", "width": 2},
    ))
    chosen_cost = float(operating_point["cost_per_1000"])
    figure.add_trace(go.Scatter(
        x=[CHOSEN], y=[chosen_cost], mode="markers+text", name="Chosen threshold",
        marker={"color": "#d62728", "size": 11},
        text=[f"{CHOSEN:.2f}: {chosen_cost:.0f} EUR"], textposition="top right",
    ))
    figure.add_vline(x=low, line_dash="dot", line_color="#555", annotation_text="pass below")
    if high > low:
        figure.add_vline(x=high, line_dash="dot", line_color="#555", annotation_text="fail at")
    figure.update_layout(
        xaxis_title="Confidence threshold (fail if the top 'no hard hat' score is at or above it)",
        yaxis_title="Expected cost per 1,000 gate passes (EUR)",
        margin={"l": 60, "r": 20, "t": 30, "b": 60},
        height=430,
    )
    return figure


def policy_row(name: str, shares: dict[str, float]) -> dict:
    review = costs.review_share(shares)
    return {
        "Policy": name,
        "Cost per 1,000 (EUR)": round(1000 * costs.cost_per_case(shares), 1),
        "Checked by a person": f"{review:.1%}",
        "Person min/day": round(review * costs.CASES_PER_DAY * costs.REVIEW_SECONDS / 60, 1),
    }


def guard_row() -> dict:
    """A guard stands at the gate all shift and looks at every worker."""
    cost = costs.cost_per_case(costs.EVERYONE_CHECKED, review_cost=costs.GUARD_COST_PER_CASE)
    return {
        "Policy": "A person checks everyone (gate guard)",
        "Cost per 1,000 (EUR)": round(1000 * cost, 1),
        "Checked by a person": "100.0%",
        "Person min/day": costs.SHIFT_HOURS * 60,
    }


def operating_view(low: float, high: float):
    """Everything in the operating point tab, for one pair of thresholds."""
    high = max(high, low)
    counts = costs.outcome_counts(SCORES, LABELS, low, high)
    shares = costs.outcome_shares(SCORES, LABELS, low, high)
    cost = 1000 * costs.cost_per_case(shares)
    review = costs.review_share(shares)
    reviews_per_day = review * costs.CASES_PER_DAY
    minutes_per_day = reviews_per_day * costs.REVIEW_SECONDS / 60

    test_matrix = counts.reset_index().rename(columns={"true": "True class (test set)"})
    pos, neg = 1000 * costs.BASE_RATE, 1000 * (1 - costs.BASE_RATE)
    gate_matrix = pd.DataFrame({
        "True class (per 1,000 at the gate)": ["no hard hat", "hard hat"],
        "pass": [pos * shares["pos_pass"], neg * shares["neg_pass"]],
        "review": [pos * shares["pos_review"], neg * shares["neg_review"]],
        "fail": [pos * shares["pos_fail"], neg * shares["neg_fail"]],
    }).round(1)

    summary = (
        f"### {cost:.0f} EUR per 1,000 gate passes\n"
        f"- Sent to review: **{review:.1%}** of passes, about **{reviews_per_day:.0f} a day** "
        f"out of {costs.CASES_PER_DAY}, or **{minutes_per_day:.0f} supervisor minutes a day**\n"
        f"- Workers without a hard hat caught (fail or review): "
        f"**{shares['pos_fail'] + shares['pos_review']:.1%}**, let through: "
        f"**{shares['pos_pass']:.1%}**\n"
        f"- Workers with a hard hat stopped by the model: **{shares['neg_fail']:.1%}**"
    )

    alone = costs.outcome_shares(SCORES, LABELS, CHOSEN, CHOSEN)
    comparison = pd.DataFrame([
        policy_row("Do nothing: everyone walks in", costs.NOBODY_CHECKED),
        guard_row(),
        policy_row(f"Model alone at {CHOSEN:.2f}", alone),
        policy_row(f"This setting ({low:.2f} / {high:.2f})", shares),
    ])
    return test_matrix, gate_matrix, summary, comparison, cost_curve(low, high)


def per_class_table() -> pd.DataFrame:
    detection = metrics["detection"]["finetuned"]
    rows = [
        {
            "Class": name + (" (positive)" if name == POSITIVE_CLASS else ""),
            "Instances": row["instances"],
            "Precision": round(row["precision"], 3),
            "Recall": round(row["recall"], 3),
            "AP50": round(row["ap50"], 3),
            "mAP50-95": round(row["ap50_95"], 3),
        }
        for name, row in detection["per_class"].items()
    ]
    rows.append({
        "Class": "all", "Instances": None, "Precision": None, "Recall": None,
        "AP50": round(detection["map50"], 3), "mAP50-95": round(detection["map50_95"], 3),
    })
    return pd.DataFrame(rows)


def baseline_table() -> pd.DataFrame:
    detection, image_level = metrics["detection"], metrics["image_level"]
    models = [
        ("Finetuned model", "finetuned", "finetuned_at_chosen_threshold"),
        ("COCO zero-shot (fail if person)", "coco_zero_shot", "coco_zero_shot_fail_if_person"),
        ("Trivial: everyone passes", "trivial_predict_nothing", "trivial_everyone_passes"),
    ]
    rows = []
    for name, det_key, img_key in models:
        row = image_level[img_key]
        rows.append({
            "Model": name,
            "mAP50": round(detection[det_key]["map50"], 3),
            "mAP50-95": round(detection[det_key]["map50_95"], 3),
            "Image threshold": row["threshold"],
            "Recall (no hard hat)": round(row["recall"], 3),
            "False alarm rate": round(row["false_positive_rate"], 3),
            "Accuracy (test)": round(row["accuracy_test"], 3),
            "Cost per 1,000 (EUR)": round(row["cost_per_1000"], 1),
        })
    return pd.DataFrame(rows)


HEADER = f"""
# Hard hat check at the site gate
Dataset: `{metrics['dataset']['workspace']}/{metrics['dataset']['project']}` v{metrics['dataset']['version']}
· Model: finetuned `{metrics['training'].get('base_model', 'yolo11n.pt')}`
· Positive class: **{POSITIVE_CLASS}** (worker without a hard hat) · One case = one gate pass

Costs per case: false alarm **{costs.COST_FP:.2f} EUR**, missed worker **{costs.COST_FN:.2f} EUR**,
worker caught **{costs.COST_TP:.2f} EUR**, review **{costs.REVIEW_COST:.2f} EUR**
· Base rate at the gate: **{costs.BASE_RATE:.0%}** · {costs.CASES_PER_DAY} gate passes a day
"""

with gr.Blocks(title="Hard hat gate check") as demo:
    gr.Markdown(HEADER)
    with gr.Tabs():
        with gr.Tab("Operating point"):
            with gr.Row():
                low = gr.Slider(0, 1, value=DEFAULT_LOW, step=0.01, label="Pass below")
                high = gr.Slider(0, 1, value=DEFAULT_HIGH, step=0.01, label="Fail at or above")
            gr.Markdown("Scores between the two thresholds go to the supervisor for review.")
            summary = gr.Markdown()
            with gr.Row():
                test_matrix = gr.Dataframe(label="Confusion matrix, 706 test images", interactive=False)
                gate_matrix = gr.Dataframe(label="Expected per 1,000 gate passes at the base rate", interactive=False)
            comparison = gr.Dataframe(label="Compared with the alternatives", interactive=False)
        with gr.Tab("Cost curve"):
            curve = gr.Plot()
        with gr.Tab("Check a worker"):
            with gr.Row():
                upload = gr.Image(type="numpy", label="Gate photo")
                annotated = gr.Image(type="numpy", label="Predictions")
            decision = gr.Markdown()
            examples = sorted(str(p) for p in EXAMPLES.glob("*.jpg"))
            if examples:
                gr.Examples(examples=examples, inputs=upload)
        with gr.Tab("Per-class results"):
            gr.Dataframe(per_class_table(), label="Finetuned model on the test split", interactive=False)
            gr.Dataframe(baseline_table(), label="Against the baselines, same test split", interactive=False)
            with gr.Row():
                for plot in ["confusion_matrix_normalized.png", "BoxPR_curve.png"]:
                    if (ARTIFACTS / plot).exists():
                        gr.Image(str(ARTIFACTS / plot), label=plot, interactive=False)

    view_outputs = [test_matrix, gate_matrix, summary, comparison, curve]
    for slider in (low, high):
        slider.change(operating_view, [low, high], view_outputs)
        slider.change(check_image, [upload, low, high], [annotated, decision])
    upload.change(check_image, [upload, low, high], [annotated, decision])
    demo.load(operating_view, [low, high], view_outputs)

if __name__ == "__main__":
    demo.launch()
