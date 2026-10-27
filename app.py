"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Loads weights/helmet.pt and weights/vest.pt and the files train.py wrote into
artifacts/<task>/. Nothing is trained here. Every number on screen is
recomputed from the saved test-split snapshot scores with the same functions in
costing.py that picked the thresholds.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
LABELS = {"helmet": "Hard hat", "vest": "Safety vest"}


def load_task(key):
    folder = ROOT / "artifacts" / key
    metrics = json.loads((folder / "metrics.json").read_text())
    return {
        "model": YOLO(str(ROOT / "weights" / f"{key}.pt")),
        "metrics": metrics,
        "scores": pd.read_csv(folder / "scores_test.csv"),
        "sweep_valid": pd.read_csv(folder / "cost_sweep_valid.csv"),
        "op": metrics["operating_point"],
        "positive": metrics["positive_class"],
        "folder": folder,
    }


TASKS = {key: load_task(key) for key in LABELS}


def key_of(label):
    return next(k for k, v in LABELS.items() if v == label)


# --------------------------------------------------------------------------- #
# Prediction tab
# --------------------------------------------------------------------------- #

def decide(score, low, high):
    if score >= high:
        return "FAIL"
    return "REVIEW" if score >= low else "PASS"


def worst_positive(result, positive):
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    hits = conf[[result.names[c] == positive for c in cls]] if len(cls) else np.array([])
    return float(hits.max()) if len(hits) else 0.0


def predict_image(image, label, low, high):
    if image is None:
        return None, "Upload an image."
    key = key_of(label)
    task = TASKS[key]
    high = max(high, low)
    result = task["model"].predict(image, conf=low, verbose=False)[0]
    score = worst_positive(result, task["positive"])
    verdict = decide(score, low, high)
    lines = [f"**{LABELS[key]} model: {verdict}.** Worst `{task['positive']}` box at {score:.2f} "
             f"(review band [{low:.2f}, {high:.2f}))."]

    # The site needs both checks, so also run the other model at its own chosen band.
    other = next(k for k in TASKS if k != key)
    o = TASKS[other]
    o_score = worst_positive(o["model"].predict(image, conf=o["op"]["low"], verbose=False)[0], o["positive"])
    o_verdict = decide(o_score, o["op"]["low"], o["op"]["high"])
    lines.append(f"{LABELS[other]} model at its chosen band: {o_verdict} (score {o_score:.2f}).")
    combined = "FAIL" if "FAIL" in (verdict, o_verdict) else "REVIEW" if "REVIEW" in (verdict, o_verdict) else "PASS"
    lines.append(f"**Combined decision for the snapshot: {combined}.**")
    return result.plot()[:, :, ::-1], "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Operating point tab
# --------------------------------------------------------------------------- #

def operating_point(label, low, high, fn, fp, tp, review_cost, base_rate):
    task = TASKS[key_of(label)]
    scores, op, positive = task["scores"], task["op"], task["positive"]
    high = max(high, low)
    costs = {**task["metrics"]["costs"], "fn": fn, "fp": fp, "tp": tp, "review": review_cost}
    res = costing.evaluate(scores["score"], scores["label"], low, high, costs, base_rate)

    # Confusion matrix as raw counts on the test split, and re-weighted to
    # 1,000 production snapshots at the chosen base rate.
    pos, neg = scores["label"] == 1, scores["label"] == 0
    rows = []
    for name, mask, share, n_prod in (
        (f"violation ({positive} present)", pos, res["shares"]["pos"], 1000 * base_rate),
        ("no violation", neg, res["shares"]["neg"], 1000 * (1 - base_rate)),
    ):
        s = scores["score"][mask]
        rows.append({
            "actual": name,
            "pass (test)": int((s < low).sum()),
            "review (test)": int(((s >= low) & (s < high)).sum()),
            "fail (test)": int((s >= high).sum()),
            "pass /1000 prod": round(share["pass"] * n_prod, 1),
            "review /1000 prod": round(share["review"] * n_prod, 1),
            "fail /1000 prod": round(share["fail"] * n_prod, 1),
        })
    matrix = pd.DataFrame(rows)

    # Same threshold as REPORT.md (chosen on validation), so the two agree.
    t = op["single_threshold"]
    alone = costing.evaluate(scores["score"], scores["label"], t, t, costs, base_rate)
    alts = costing.alternatives(costs, base_rate)
    policies = pd.DataFrame([
        {"policy": "do nothing", "EUR / 1000 snapshots": alts["do_nothing"]["cost_per_1000"],
         "share reviewed": 0.0, "reviewers needed": 0.0},
        {"policy": "a person checks everything", "EUR / 1000 snapshots": alts["human_checks_all"]["cost_per_1000"],
         "share reviewed": 1.0, "reviewers needed": alts["human_checks_all"]["reviewers_needed"]},
        {"policy": f"model alone (t={t:.2f}, chosen on validation)", "EUR / 1000 snapshots": alone["cost_per_1000"],
         "share reviewed": 0.0, "reviewers needed": 0.0},
        {"policy": f"model + review band [{low:.2f}, {high:.2f})", "EUR / 1000 snapshots": res["cost_per_1000"],
         "share reviewed": res["review_share"], "reviewers needed": res["reviewers_needed"]},
    ]).round(3)

    summary = (
        f"### {label}: expected cost **{res['cost_per_1000']:.0f} EUR per 1,000 snapshots**\n"
        f"Sent to review: **{100 * res['review_share']:.1f}%** of snapshots, "
        f"{res['reviews_per_day']:.0f} per day at {costing.DAILY_SNAPSHOTS} snapshots/day, "
        f"{res['reviewer_hours_per_day']:.1f} reviewer-hours, "
        f"**{res['reviewers_needed']:.2f} people** at {costing.REVIEWER_PRODUCTIVE_HOURS} productive h/day.\n\n"
        f"Violations caught: {100 * (1 - res['shares']['pos']['pass']):.1f}% before human error; "
        f"false alarms: {100 * res['shares']['neg']['fail']:.1f}% of clean snapshots."
    )
    return summary, matrix, policies, cost_curve(task, low, high, costs, base_rate)


def cost_curve(task, low, high, costs, base_rate):
    scores, op = task["scores"], task["op"]
    test = pd.DataFrame(costing.single_threshold_sweep(scores["score"], scores["label"], costs, base_rate))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["threshold"], y=test["cost_per_1000"], name="test split, current costs",
                             line=dict(color="#2563eb", width=2.5)))
    fig.add_trace(go.Scatter(x=task["sweep_valid"]["threshold"], y=task["sweep_valid"]["cost_per_1000"],
                             name="validation, report costs", line=dict(color="#9ca3af", dash="dot")))
    chosen = op["single_threshold"]
    y_chosen = test.loc[(test["threshold"] - chosen).abs().idxmin(), "cost_per_1000"]
    fig.add_trace(go.Scatter(x=[chosen], y=[y_chosen], mode="markers+text", name="chosen (on validation)",
                             marker=dict(size=12, color="#dc2626"), text=[f"t={chosen:.2f}"],
                             textposition="top center"))
    fig.add_vrect(x0=low, x1=max(high, low), fillcolor="#f59e0b", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(xaxis_title="confidence threshold (model decides alone)",
                      yaxis_title="EUR per 1,000 snapshots", height=420, margin=dict(l=40, r=20, t=30, b=40),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def switch_model(label):
    """Reset sliders and cost inputs to the selected model's report values."""
    m = TASKS[key_of(label)]["metrics"]
    op, costs = m["operating_point"], m["costs"]
    return op["low"], op["high"], costs["fn"], costs["fp"], costs["tp"], costs["review"], m["base_rate"]


# --------------------------------------------------------------------------- #
# Per-class tab (static: comes straight from train.py)
# --------------------------------------------------------------------------- #

def per_class_table(task):
    op = task["op"]["single_threshold"]
    rows = []
    for model, m in task["metrics"]["models"].items():
        for cls, v in m["per_class"].items():
            rows.append({"model": model, "class": cls, "instances": v["instances"],
                         "AP50": v["AP50"], "AP50-95": v["AP50_95"],
                         f"precision @ {op:.2f}": v["precision"], f"recall @ {op:.2f}": v["recall"]})
        rows.append({"model": model, "class": "ALL (mean)", "instances": None,
                     "AP50": m["mAP50"], "AP50-95": m["mAP50_95"]})
    return pd.DataFrame(rows).round(3)


def detection_confusion(task):
    labels = list(task["metrics"]["class_names"].values()) + ["background"]
    df = pd.DataFrame(np.array(task["metrics"]["confusion_matrix_test"]), columns=[f"pred {l}" for l in labels])
    df.insert(0, "true", labels)
    return df


def examples(task):
    return [(str(task["folder"] / "examples" / item["file"]), f"{kind}: score {item['score']:.2f}")
            for kind, items in task["metrics"]["examples"].items() for item in items]


def header():
    parts = []
    for key, task in TASKS.items():
        d, op = task["metrics"]["dataset"], task["op"]
        parts.append(f"**{LABELS[key]}**: positive class `{task['positive']}`, yolo11s finetuned on "
                     f"[{d['workspace']}/{d['project']} v{d['version']}]({d['url']}), "
                     f"band [{op['low']:.2f}, {op['high']:.2f}), single threshold {op['single_threshold']:.2f}")
    return "\n\n".join(parts)


first = TASKS["helmet"]
with gr.Blocks(title="PPE compliance: operating point") as demo:
    gr.Markdown("# PPE compliance on site cameras\nOne decision per camera snapshot: pass, send to a remote "
                "reviewer, or send the foreman. Thresholds below were chosen on validation.\n\n" + header())
    task_choice = gr.Radio(list(LABELS.values()), value=LABELS["helmet"], label="Model")
    with gr.Row():
        low = gr.Slider(0.01, 0.99, value=first["op"]["low"], step=0.01, label="low threshold: below = pass")
        high = gr.Slider(0.01, 0.99, value=first["op"]["high"], step=0.01, label="high threshold: at or above = fail")

    with gr.Tab("Operating point"):
        with gr.Accordion("Cost assumptions (defaults from REPORT.md)", open=False):
            with gr.Row():
                c = first["metrics"]["costs"]
                fn = gr.Number(c["fn"], label="EUR per missed violation (FN)")
                fp = gr.Number(c["fp"], label="EUR per false alarm (FP)")
                tp = gr.Number(c["tp"], label="EUR per caught violation (TP)")
                rc = gr.Number(c["review"], label="EUR per human review")
                br = gr.Slider(0.005, 0.5, value=first["metrics"]["base_rate"], step=0.005,
                               label="base rate of violations")
        summary = gr.Markdown()
        matrix = gr.Dataframe(label="Confusion matrix: actual x decision")
        plot = gr.Plot(label="Cost against threshold")
        policies = gr.Dataframe(label="Against the alternatives (test split)")
        inputs = [task_choice, low, high, fn, fp, tp, rc, br]
        outputs = [summary, matrix, policies, plot]
        task_choice.change(switch_model, task_choice, [low, high, fn, fp, tp, rc, br]).then(
            operating_point, inputs, outputs)
        for comp in inputs[1:]:
            comp.change(operating_point, inputs, outputs)
        demo.load(operating_point, inputs, outputs)

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload a site photo")
            image_out = gr.Image(label="Predictions of the selected model at the low threshold")
        verdict = gr.Markdown()
        for comp in (image_in, task_choice, low, high):
            comp.change(predict_image, [image_in, task_choice, low, high], [image_out, verdict])

    with gr.Tab("Per-class results"):
        gr.Markdown("Test split of each dataset. Finetuned model against COCO yolo11s zero-shot (its `person` "
                    "boxes scored as each class) and a model that predicts nothing.")
        for key, task in TASKS.items():
            op = task["op"]["single_threshold"]
            with gr.Accordion(f"{LABELS[key]} model", open=key == "helmet"):
                gr.Dataframe(per_class_table(task), label="Per-class detection metrics")
                gr.Dataframe(detection_confusion(task),
                             label=f"Detection confusion matrix, finetuned, conf >= {op:.2f}, IoU 0.5")
                gr.Dataframe(pd.DataFrame(task["metrics"]["positive_recall_by_size_test"]).round(3),
                             label=f"`{task['positive']}` recall by box size")
                gr.Gallery(examples(task), label="Worst test failures at the chosen band", columns=6, height=260)


if __name__ == "__main__":
    demo.launch()
