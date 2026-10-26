"""
Assignment 2: Gradio dashboard. Run via:

    uv run python main.py app

Loads the committed weights and the files train.py saved in weights/, and
never trains anything. Two sliders set the review band for the ripe decision:
a tomato whose ripe score is at or above the upper edge is called ripe, below
the lower edge it is called not ripe, and in between a person checks it. The
cost, the decision table, the cost curve and any uploaded image all follow the
sliders.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
from ultralytics import YOLO

import costs
from evaluation import BACKGROUND, IOU_MATCH, box_iou, confusion_matrix

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"

# Boxes below this confidence are not drawn on an uploaded image.
DETECTION_CONF = 0.25
COLOURS = dict(zip(costs.OUTCOMES, ["#2e7d32", "#ef6c00", "#1565c0"]))

model = YOLO(WEIGHTS_DIR / "best.pt")
metrics = json.loads((WEIGHTS_DIR / "metrics.json").read_text())
cost_summary = json.loads((WEIGHTS_DIR / "costs.json").read_text())
names = metrics["dataset"]["names"]
ripe_id = names.index(costs.POSITIVE_CLASS)
chosen = cost_summary["model_with_review"]

tomatoes = pd.read_csv(WEIGHTS_DIR / "tomato_scores.csv")
sweep = pd.read_csv(WEIGHTS_DIR / "cost_sweep.csv")
predictions = pd.read_csv(WEIGHTS_DIR / "test_predictions.csv")
finetuned_preds = predictions[predictions["model"] == "finetuned"]
truths = pd.read_csv(WEIGHTS_DIR / "test_labels.csv")


def ordered(lower: float, upper: float) -> tuple[float, float]:
    """The two slider values as (lower, upper), whichever way round they are set."""
    return min(lower, upper), max(lower, upper)


def outcome(score: float, lower: float, upper: float) -> str:
    if score >= upper:
        return costs.OUTCOMES[0]
    if score < lower:
        return costs.OUTCOMES[2]
    return costs.OUTCOMES[1]


def cost_table(lower: float, upper: float) -> pd.DataFrame:
    """Expected cost of the slider band next to the chosen band and the alternatives."""
    r = costs.rates(tomatoes, lower, upper)
    share = float(costs.review_share(r))
    options = [
        ("This band", 1000 * float(costs.expected_cost(r)), share),
        ("Chosen band", chosen["cost_per_1000"], chosen["review_share"]),
        ("Model alone", cost_summary["model_alone"]["cost_per_1000"], 0.0),
        ("Person checks every tomato", cost_summary["person_checks_everything"]["cost_per_1000"], 1.0),
        ("Do nothing", cost_summary["do_nothing"]["cost_per_1000"], 0.0),
    ]
    per_day = costs.TOMATOES_PER_DAY / 1000
    return pd.DataFrame(
        [
            [name, f"{cost:.2f}", f"{cost * per_day:,.0f}", f"{share:.1%}", f"{costs.reviewers_needed(share):.1f}"]
            for name, cost, share in options
        ],
        columns=[
            "Option", "EUR per 1,000 tomatoes", "EUR per day", "Sent to a person",
            f"People ({costs.SHIFT_HOURS}-hour shifts)",
        ],
    )


def cost_curve(lower: float, upper: float) -> Figure:
    """Cost against threshold, with the chosen operating point and the slider band."""
    fig = Figure(figsize=(7, 4), layout="tight")
    ax = fig.subplots()
    ax.plot(sweep["threshold"], sweep["cost_per_1000"], label="Model decides alone")
    ax.axhline(cost_summary["do_nothing"]["cost_per_1000"], ls="--", color="grey", label="Do nothing")
    ax.axhline(
        cost_summary["person_checks_everything"]["cost_per_1000"], ls=":", color="grey",
        label="Person checks everything",
    )
    ax.axvspan(lower, upper, color=COLOURS[costs.OUTCOMES[1]], alpha=0.15, label="Review band (sliders)")
    ax.axvline(chosen["lower"], color="black", lw=1, ls="-.", label="Chosen band edges")
    ax.axvline(chosen["upper"], color="black", lw=1, ls="-.")
    alone = cost_summary["model_alone"]
    ax.scatter(
        [alone["threshold"]], [alone["cost_per_1000"]], color="black", zorder=3,
        label=f"Cheapest single threshold ({alone['threshold']:.2f})",
    )
    ax.set_xlabel("Confidence threshold for calling a tomato ripe")
    ax.set_ylabel("Expected cost per 1,000 tomatoes (EUR)")
    ax.legend(fontsize=8)
    return fig


def decision_view(lower: float, upper: float):
    lower, upper = ordered(lower, upper)
    decisions = costs.decision_table(tomatoes, lower, upper).reset_index()
    return cost_table(lower, upper), decisions, cost_curve(lower, upper)


def predict_image(image: Image.Image | None, lower: float, upper: float):
    """Draw each tomato the model finds, coloured by what the band decides for it.

    Each tomato is one box from class-agnostic NMS. Its ripe score is the
    highest ripe confidence on a box overlapping it, as in costs.tomato_scores.
    """
    if image is None:
        return None, None
    lower, upper = ordered(lower, upper)
    image = image.convert("RGB")
    found = model.predict(image, conf=DETECTION_CONF, agnostic_nms=True, verbose=False)[0].boxes
    ripe = model.predict(image, conf=0.001, classes=[ripe_id], verbose=False)[0].boxes

    boxes = np.array(found.xyxy.tolist()).reshape(-1, 4)
    ripe_boxes = np.array(ripe.xyxy.tolist()).reshape(-1, 4)
    ripe_conf = np.array(ripe.conf.tolist())
    scores = np.zeros(len(boxes))
    if len(boxes) and len(ripe_boxes):
        iou = box_iou(boxes, ripe_boxes)
        scores = np.where(iou >= IOU_MATCH, ripe_conf[None, :], 0.0).max(axis=1)

    draw = ImageDraw.Draw(image)
    rows = []
    for box, cls, conf, score in zip(boxes, found.cls.tolist(), found.conf.tolist(), scores):
        decision = outcome(score, lower, upper)
        draw.rectangle(box.tolist(), outline=COLOURS[decision], width=3)
        draw.text((box[0] + 3, box[1] + 2), f"{names[int(cls)]} {conf:.2f}", fill=COLOURS[decision])
        rows.append([names[int(cls)], round(conf, 2), round(float(score), 2), decision])
    table = pd.DataFrame(rows, columns=["Model's class", "Confidence", "Ripe score", "Decision"])
    return image, table


def per_class_view(conf: float):
    """Per-class precision and recall at `conf`, the confusion matrix, and the fixed AP50."""
    matrix = confusion_matrix(finetuned_preds, truths, names, conf)
    ap50 = {row["class"]: row["ap50"] for row in metrics["models"]["finetuned"]["per_class"]}
    rows = []
    for c, name in enumerate(names):
        found = int(matrix.iloc[c, c])
        predicted = int(matrix.iloc[:, c].sum())
        actual = int(matrix.iloc[c].sum())
        rows.append([name, actual, ap50[name], found / predicted if predicted else None, found / actual])
    table = pd.DataFrame(rows, columns=["Class", "Tomatoes", "AP50", "Precision", "Recall"]).round(3)
    matrix = matrix.rename(index={BACKGROUND: "no labelled tomato"}, columns={BACKGROUND: "missed"})
    return table, matrix.reset_index()


def baseline_table() -> pd.DataFrame:
    rows = [
        [name, s["map50"]] + [row["ap50"] for row in s["per_class"]]
        for name, s in metrics["models"].items()
    ]
    return pd.DataFrame(rows, columns=["Model", "mAP50"] + [f"AP50 {n}" for n in names]).round(3)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Tomato ripeness") as demo:
        gr.Markdown(
            "# Tomato ripeness: when should a person check?\n"
            f"A tomato whose **ripe score** is at or above the upper edge is called ripe, below the lower "
            f"edge it is called not ripe, and in between a person checks it. Costs: a missed ripe tomato "
            f"EUR {costs.FN_COST:.2f}, a tomato wrongly called ripe EUR {costs.FP_COST:.2f}, a check "
            f"EUR {costs.REVIEW_COST:.4f}. Base rate {costs.BASE_RATE:.0%} ripe, "
            f"{costs.TOMATOES_PER_DAY:,} tomatoes a day. All numbers are from the test split."
        )
        with gr.Row():
            lower = gr.Slider(0, 1, value=chosen["lower"], step=0.01, label="Lower edge: below this, called not ripe")
            upper = gr.Slider(0, 1, value=chosen["upper"], step=0.01, label="Upper edge: at or above this, called ripe")

        with gr.Tab("Cost and review"):
            cost_out = gr.Dataframe(label="Expected cost", interactive=False)
            decisions_out = gr.Dataframe(label="Test tomatoes by true class and decision", interactive=False)
            curve_out = gr.Plot(label="Cost against threshold")
        with gr.Tab("Try an image"):
            with gr.Row():
                upload = gr.Image(type="pil", label="Upload a photo of tomatoes")
                annotated = gr.Image(type="pil", label="Decisions at the current band")
            gr.Markdown("Green: called ripe. Orange: sent to a person. Blue: called not ripe.")
            boxes_out = gr.Dataframe(label="Each tomato found", interactive=False)
        with gr.Tab("Per-class results"):
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Detection confidence")
            per_class_out = gr.Dataframe(label="Finetuned model per class (AP50 does not depend on the confidence)", interactive=False)
            matrix_out = gr.Dataframe(label="Confusion matrix: rows are the true class, columns the prediction", interactive=False)
            gr.Dataframe(value=baseline_table(), label="Against the baselines", interactive=False)

        for slider in (lower, upper):
            slider.change(decision_view, [lower, upper], [cost_out, decisions_out, curve_out])
            slider.release(predict_image, [upload, lower, upper], [annotated, boxes_out])
        upload.change(predict_image, [upload, lower, upper], [annotated, boxes_out])
        conf.release(per_class_view, conf, [per_class_out, matrix_out])
        demo.load(decision_view, [lower, upper], [cost_out, decisions_out, curve_out])
        demo.load(per_class_view, conf, [per_class_out, matrix_out])
    return demo


if __name__ == "__main__":
    build_demo().launch()
