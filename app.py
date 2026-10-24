"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Loads the committed weights in weights/ and what train.py saved in
artifacts/. It never trains: moving a slider re-costs the saved test-set
predictions, it does not re-run the model over the test set.
"""

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from ultralytics import YOLO

import costing
import settings as S

METRICS = json.loads((S.ARTIFACTS_DIR / "metrics.json").read_text())
PREDICTIONS = json.loads((S.ARTIFACTS_DIR / "predictions.json").read_text())
NAMES: list[str] = PREDICTIONS["class_names"]
POSITIVE = NAMES.index(PREDICTIONS["positive_class"])
OP = METRICS["operating_point"]
T_POINT = OP["point_estimate_choice"]["t_high"]
MODEL = YOLO(str(S.WEIGHTS_PATH))

DECISION_COLOURS = {"publish": "#1e8449", "review": "#d68910", "context": "#2e86c1"}


def frames(split: str, labels: str) -> list[dict]:
    """A split's frames, with the mislabelled no-scoreboard frames left out in the audited view."""
    if labels == "published":
        return PREDICTIONS[split]
    drop = set(S.MISLABELLED_NULL_FRAMES.get(split, []))
    return [r for r in PREDICTIONS[split] if r["file"] not in drop]


CASES = {(split, labels): costing.build_cases(frames(split, labels), POSITIVE, S.IOU_MATCH)
         for split in ("val", "test") for labels in ("audited", "published")}


def make_costs(c_fp, c_fn, c_review, base_rate) -> costing.Costs:
    return costing.Costs(fp=float(c_fp), fn=float(c_fn), review=float(c_review), tp=S.COST_TP,
                         base_rate=float(base_rate), p_change=S.P_SCORE_CHANGE)


# --- Operating point tab ------------------------------------------------------

def decision_table(r: dict) -> pd.DataFrame:
    """The confusion matrix with three outcomes instead of two, as raw counts of Team Points readings."""
    return pd.DataFrame(
        {
            "": ["Real Team Points box", "Not a Team Points box"],
            "Published automatically": [r["tp_auto"], r["fp_auto"]],
            "In the review band": [r["tp_review"], r["fp_review"]],
            "Dropped / not read": [r["fn"], "-"],
        }
    )


def alternatives_table(cases, costs, t_low, t_high) -> pd.DataFrame:
    rows = [
        ("Do nothing (a scout per game)", costing.do_nothing(S.SCOUT_EUR_PER_FRAME)),
        ("A person checks every frame", costing.human_checks_everything(cases, costs, S.MIN_CONF)),
        (f"Model alone at {OP['t_single']:.2f} (chosen)", costing.evaluate(cases, costs, OP["t_single"], OP["t_single"])),
        (f"Model alone at {T_POINT:.2f} (point-estimate sweep)", costing.evaluate(cases, costs, T_POINT, T_POINT)),
        (f"Model + review band {t_low:.2f}-{t_high:.2f} (sliders)", costing.evaluate(cases, costs, t_low, t_high)),
    ]
    return pd.DataFrame(
        {
            "Policy": [name for name, _ in rows],
            "EUR / 1,000 frames": [round(r["cost_per_1000"], 2) for _, r in rows],
            "at 95% upper FP rate": [round(r.get("cost_per_1000_upper", r["cost_per_1000"]), 2) for _, r in rows],
            "Wrong scores published": [r["fp_auto"] for _, r in rows],
            "Frames to a person": [f"{r['review_share']:.1%}" for _, r in rows],
        }
    )


def cost_curve(cases, costs, t_low, t_high) -> go.Figure:
    sweep = costing.sweep_single(cases, costs)
    t = [r["t_high"] for r in sweep]
    fig = go.Figure()
    fig.add_scatter(x=t, y=[r["cost_per_1000"] for r in sweep], mode="lines", name="model alone",
                    line=dict(color="#1f5fa8", width=2))
    fig.add_scatter(x=t, y=[r["cost_per_1000_upper"] for r in sweep], mode="lines",
                    name="model alone, FP rate at 95% upper bound", line=dict(color="#c0392b", width=1))
    fig.add_hline(y=1000 * S.SCOUT_EUR_PER_FRAME, line_dash="dash", line_color="#27ae60",
                  annotation_text="do nothing (scouts)", annotation_position="bottom left")
    for x, label, colour in ((OP["t_single"], f"chosen {OP['t_single']:.2f}", "#c0392b"),
                             (T_POINT, f"point estimate {T_POINT:.2f}", "#7f8c8d")):
        r = costing.evaluate(cases, costs, x, x)
        fig.add_scatter(x=[x], y=[r["cost_per_1000_upper"]], mode="markers", name=label,
                        marker=dict(color=colour, size=12, symbol="star"))
    fig.add_vrect(x0=t_low, x1=max(t_high, t_low + 0.002), fillcolor="#f5b041", opacity=0.25, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    values = [v for r in sweep for v in (r["cost_per_1000"], r["cost_per_1000_upper"]) if v > 0]
    values.append(1000 * S.SCOUT_EUR_PER_FRAME)
    fig.update_layout(
        xaxis_title="confidence threshold for publishing", yaxis_title="EUR per 1,000 frames",
        yaxis_type="log", yaxis_range=[np.log10(min(values)) - 0.3, np.log10(max(values)) + 0.3], height=400, margin=dict(l=40, r=20, t=30, b=40), legend=dict(x=0.02, y=0.98),
    )
    return fig


def confusion_figure(images: list[dict], conf: float) -> go.Figure:
    cm = costing.confusion_matrix(images, len(NAMES), conf, S.IOU_MATCH)
    labels = NAMES + ["background"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", text=cm, texttemplate="%{text}",
                               showscale=False))
    fig.update_layout(
        xaxis_title="predicted", yaxis_title="actual", yaxis_autorange="reversed", height=420,
        margin=dict(l=40, r=20, t=40, b=40), title=f"All classes at the lower threshold, {conf:.2f}",
    )
    return fig


def summary_markdown(r: dict, costs, split: str, labels: str) -> str:
    share = r["review_share"]
    reviews_day = share * S.FRAMES_PER_DAY
    peak_busy = share * S.PEAK_CONCURRENT_GAMES / S.FRAME_INTERVAL_S * S.REVIEW_SECONDS
    scouts = 1000 * S.SCOUT_EUR_PER_FRAME
    return (
        f"### EUR {r['cost_per_1000']:,.2f} per 1,000 frames "
        f"(EUR {r['cost_per_1000_upper']:,.2f} at the 95% upper FP rate) · {share:.1%} of frames to a person\n"
        f"Doing nothing costs EUR {scouts:.2f}. **{split}** split, **{labels}** labels, reweighted to "
        f"{costs.base_rate:.0%} of frames showing a scoreboard. At {S.FRAMES_PER_DAY:,} frames a day: "
        f"**{reviews_day:,.0f} reviews/day** ({reviews_day * S.REVIEW_SECONDS / 3600:,.1f} operator-hours, "
        f"against {S.GAMES_PER_DAY * S.BROADCAST_MINUTES / 60:,.0f} scout-hours today) and "
        f"**{peak_busy:.1f} operators busy at peak**, against {S.PEAK_CONCURRENT_GAMES} scouts."
    )


def update_operating_point(t_low, t_high, c_fp, c_fn, c_review, base_rate, split, labels):
    t_low = min(t_low, t_high)
    costs = make_costs(c_fp, c_fn, c_review, base_rate)
    cases = CASES[(split, labels)]
    r = costing.evaluate(cases, costs, t_low, t_high)
    return (
        summary_markdown(r, costs, split, labels),
        decision_table(r),
        alternatives_table(cases, costs, t_low, t_high),
        cost_curve(cases, costs, t_low, t_high),
        confusion_figure(frames(split, labels), t_low),
    )


def reset_to_chosen():
    return OP["t_low"], OP["t_high"], S.COST_FP, S.COST_FN, S.COST_REVIEW, S.BASE_RATE


# --- Per-class tab ------------------------------------------------------------

def per_class_table(conf: float) -> pd.DataFrame:
    rows = costing.per_class_pr(PREDICTIONS["test"], len(NAMES), conf, S.IOU_MATCH)
    ap = METRICS["finetuned_test"]["per_class"]
    return pd.DataFrame(
        {
            "Class": NAMES,
            "Test instances": [r["instances"] for r in rows],
            "AP50": [round(ap[n]["ap50"], 3) for n in NAMES],
            "mAP50-95": [round(ap[n]["ap50_95"], 3) for n in NAMES],
            f"Precision @ {conf:.2f}": [round(r["precision"], 3) for r in rows],
            f"Recall @ {conf:.2f}": [round(r["recall"], 3) for r in rows],
            "FP": [r["fp"] for r in rows],
            "FN": [r["fn"] for r in rows],
        }
    )


def baselines_table() -> pd.DataFrame:
    b = METRICS["baselines"]
    keys = ("finetuned", "coco_zero_shot", "predict_nothing")
    return pd.DataFrame(
        {
            "Model (same test split)": ["Finetuned yolo11n", "COCO yolo11n zero-shot", "Predict nothing"],
            "mAP50": [round(b[k]["map50"], 3) for k in keys],
            "mAP50-95": [round(b[k]["map50_95"], 3) for k in keys],
        }
    )


def dataset_table() -> pd.DataFrame:
    classes = METRICS["dataset"]["classes"]
    return pd.DataFrame(
        {
            "Class": list(classes),
            "Train": [c["instances"]["train"] for c in classes.values()],
            "Val": [c["instances"]["val"] for c in classes.values()],
            "Test": [c["instances"]["test"] for c in classes.values()],
            "Median box area (% of image)": [round(c["median_rel_area_pct"] or 0, 2) for c in classes.values()],
        }
    )


# --- Single image tab --------------------------------------------------------

def predict_image(image: Image.Image, t_low: float, t_high: float):
    if image is None:
        return None, pd.DataFrame()
    t_low = min(t_low, t_high)
    image = image.convert("RGB")
    result = MODEL.predict(image, conf=max(t_low, 0.05), imgsz=S.TRAIN_ARGS["imgsz"], verbose=False)[0]
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 400)
    rows = []
    for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
        cls = int(cls)
        if cls == POSITIVE:
            decision = "publish" if conf >= t_high else "review"
        else:
            decision = "context"
        colour = DECISION_COLOURS[decision]
        draw.rectangle(box, outline=colour, width=width)
        draw.text((box[0] + 2, max(box[1] - 12, 0)), f"{NAMES[cls]} {conf:.2f}", fill=colour)
        rows.append({"class": NAMES[cls], "confidence": round(conf, 3),
                     "decision": decision if cls == POSITIVE else "other field (not published)"})
    table = pd.DataFrame(rows or [{"class": "-", "confidence": None, "decision": "nothing above the lower threshold"}])
    return image, table


# --- Layout -------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    examples = sorted(str(p) for p in (S.ARTIFACTS_DIR / "examples").glob("*.jpg"))
    with gr.Blocks(title="Scoreboard reader: operating point") as demo:
        gr.Markdown(
            "# Scoreboard reader for a live basketball data feed\n"
            f"A **{PREDICTIONS['positive_class']}** box is cropped, read and published as the score. At or above "
            "the upper threshold it goes out automatically. Between the two, an operator confirms it whenever it "
            "would change the published score. Below the lower one it is dropped. Moving a slider re-costs the "
            "saved test predictions; nothing is retrained."
        )
        with gr.Row():
            t_low = gr.Slider(0.01, 1.0, value=OP["t_low"], step=0.01, label="Lower threshold (review from here)")
            t_high = gr.Slider(0.01, 1.0, value=OP["t_high"], step=0.01,
                               label="Upper threshold (publish from here; 1.00 = never)")

        with gr.Tab("Operating point"):
            with gr.Row():
                c_fp = gr.Number(S.COST_FP, label="EUR per wrong score published (FP)")
                c_fn = gr.Number(round(S.COST_FN, 4), label="EUR per score not read (FN)")
                c_review = gr.Number(round(S.COST_REVIEW, 4), label="EUR per frame reviewed")
                base_rate = gr.Slider(0.05, 1.0, value=S.BASE_RATE, step=0.05, label="Base rate: frames with a scoreboard")
            with gr.Row():
                split = gr.Radio(["test", "val"], value="test", label="Split")
                labels = gr.Radio(["audited", "published"], value="audited",
                                  label="Labels (audited leaves out frames mislabelled as having no scoreboard)")
                reset = gr.Button("Reset to the chosen operating point", size="sm")
            summary = gr.Markdown()
            with gr.Row():
                decisions = gr.Dataframe(label="Team Points readings by outcome (raw counts)", interactive=False)
                alternatives = gr.Dataframe(label="Against the alternatives", interactive=False)
            with gr.Row():
                curve = gr.Plot(label="Cost against threshold")
                confusion = gr.Plot(label="Confusion matrix")

        with gr.Tab("Per-class results"):
            per_class = gr.Dataframe(label="Per class on the test split, at the lower threshold", interactive=False)
            gr.Dataframe(baselines_table(), label="Finetuned model against both baselines (test split)",
                         interactive=False)
            gr.Dataframe(dataset_table(), label="Instances per class and split", interactive=False)
            with gr.Row():
                for name, label in (("test_confusion_matrix.png", "Ultralytics confusion matrix (test)"),
                                    ("train_results.png", "Training curves")):
                    if (S.ARTIFACTS_DIR / name).is_file():
                        gr.Image(str(S.ARTIFACTS_DIR / name), label=label, interactive=False)

        with gr.Tab("Try an image"):
            with gr.Row():
                image_in = gr.Image(type="pil", label="Broadcast frame")
                image_out = gr.Image(type="pil", label="Green: publish · orange: review · blue: other fields")
            detections = gr.Dataframe(label="Detections at or above the lower threshold", interactive=False)
            if examples:
                gr.Examples(examples, inputs=image_in)

        controls = [t_low, t_high, c_fp, c_fn, c_review, base_rate, split, labels]
        outputs = [summary, decisions, alternatives, curve, confusion]
        for control in controls:
            control.change(update_operating_point, controls, outputs)
        t_low.change(per_class_table, t_low, per_class)
        reset.click(reset_to_chosen, None, [t_low, t_high, c_fp, c_fn, c_review, base_rate])
        for trigger in (image_in.change, t_low.change, t_high.change):
            trigger(predict_image, [image_in, t_low, t_high], [image_out, detections])
        demo.load(update_operating_point, controls, outputs)
        demo.load(per_class_table, t_low, per_class)
    return demo


if __name__ == "__main__":
    build_demo().launch()
