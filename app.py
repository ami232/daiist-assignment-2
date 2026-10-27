"""
Assignment 2: Gradio dashboard. Run with `uv run python main.py app`.

Built only from what `main.py train` already produced: the committed
weights in weights/best.pt and the files in artifacts/. Nothing is trained
here.

Two sliders set the review band on the Poisonous confidence:

    conf <  lower threshold   -> PASS   (treated as edible)
    in between                -> REVIEW (a person checks it)
    conf >= upper threshold   -> FAIL   (treated as poisonous)

Tabs:
- Check a mushroom: upload an image, see each mushroom's decision.
- Cost and review: confusion matrix, cost per 1,000 mushrooms, share sent
  to review, reviewers needed, and the cost-against-threshold curve. All of
  it on the test split, and all of it follows the sliders.
- Per-class results: precision, recall and AP per class, the baselines, and
  Ultralytics' confusion matrix.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw
from ultralytics import YOLO

import costing
from train import IMGSZ, MATCH_IOU, iou  # helpers only: importing train never trains

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

DETECT_CONF = 0.25  # a box at least this confident (any class) counts as a mushroom
COLOURS = {"PASS": "#1a7f37", "REVIEW": "#bf8700", "FAIL": "#cf222e"}


def load_json(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `uv run python main.py train` first.")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Everything the training run saved
# --------------------------------------------------------------------------- #

model = YOLO(str(WEIGHTS))
metrics = load_json("metrics.json")
operating = load_json("operating_point.json")
test = pd.read_csv(ARTIFACTS / "confidences_test.csv")
sweep_val = pd.read_csv(ARTIFACTS / "cost_sweep_val.csv")
sweep_test = pd.read_csv(ARTIFACTS / "cost_sweep_test.csv")

COSTS = operating["costs"]  # the figures the thresholds were chosen with
POSITIVE = operating["positive_class"]
NEGATIVE = next(n for n in metrics["dataset"]["classes"] if n != POSITIVE)
POISON_IDX = next(i for i, name in model.names.items() if name == POSITIVE)
T_SINGLE = operating["single_threshold"]
T_LOW = operating["review_band"]["t_low"]
T_HIGH = operating["review_band"]["t_high"]


def decide(conf: float, t_low: float, t_high: float) -> str:
    if conf < t_low:
        return "PASS"
    return "FAIL" if conf >= t_high else "REVIEW"


# --------------------------------------------------------------------------- #
# Tab 1: check a mushroom
# --------------------------------------------------------------------------- #

def check_image(image, t_low, t_high):
    """Draw every mushroom found in the image with its pass/review/fail decision.

    Same rule as the training costing: a mushroom's Poisonous confidence is
    the highest score of any Poisonous box overlapping it (0 if none).
    """
    if image is None:
        return None, pd.DataFrame(columns=["mushroom", "poisonous_confidence", "decision"])
    t_low, t_high = sorted((t_low, t_high))

    boxes = model.predict(image, conf=0.001, imgsz=IMGSZ, verbose=False)[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    # One mushroom per confident box, skipping boxes that overlap one already kept.
    mushrooms = []
    for i in np.argsort(-conf):
        if conf[i] < DETECT_CONF:
            break
        if mushrooms and iou(xyxy[i], np.array(mushrooms)).max() >= MATCH_IOU:
            continue
        mushrooms.append(xyxy[i])

    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    rows = []
    for n, box in enumerate(mushrooms, start=1):
        covers = (iou(box, xyxy) >= MATCH_IOU) & (cls == POISON_IDX)
        p_poison = float(conf[covers].max(initial=0))
        decision = decide(p_poison, t_low, t_high)
        draw.rectangle(box.tolist(), outline=COLOURS[decision], width=4)
        draw.text((box[0] + 4, box[1] + 4), f"#{n} {decision} {p_poison:.2f}",
                  fill=COLOURS[decision])
        rows.append({"mushroom": n, "poisonous_confidence": round(p_poison, 3),
                     "decision": decision})
    return annotated, pd.DataFrame(rows, columns=["mushroom", "poisonous_confidence", "decision"])


# --------------------------------------------------------------------------- #
# Tab 2: cost and review, on the test split
# --------------------------------------------------------------------------- #

def cost_view(t_low, t_high):
    t_low, t_high = sorted((t_low, t_high))
    result = costing.expected_cost(test.poisonous_conf, test.poisonous, t_low, t_high, COSTS)

    confusion = pd.DataFrame(
        {
            "true class": [POSITIVE, NEGATIVE],
            "PASS (eaten)": [result["fn"], result["tn"]],
            "REVIEW (person)": [result["review_poisonous"], result["review_edible"]],
            "FAIL (thrown away)": [result["tp"], result["fp"]],
        }
    )

    summary = (
        f"**Expected cost: €{result['cost_per_1000']:,.0f} per 1,000 mushrooms**  \n"
        f"Sent to review: **{result['review_share']:.1%}** "
        f"→ {result['reviews_per_day']:,.0f} reviews a day "
        f"→ **{result['reviewers_needed']:.2f} reviewers** "
        f"(at {COSTS['cases_per_day']:,.0f} mushrooms a day, "
        f"{COSTS['review_minutes']:g} min each, "
        f"{COSTS['reviewer_hours_per_day']:g} h per reviewer)  \n"
        f"Counts are the {len(test)} test-split mushrooms; euros are reweighted to a "
        f"{COSTS['base_rate']:.0%} base rate of {POSITIVE}."
    )

    policies = costing.policy_costs(test.poisonous_conf, test.poisonous, COSTS,
                                    T_SINGLE, t_low, t_high)
    labels = {
        "do_nothing": "Do nothing (everything eaten)",
        "person_checks_all": "A person checks everything",
        "model_alone": f"Model alone (threshold {T_SINGLE:.2f})",
        "model_plus_review": f"Model + review ({t_low:.2f} – {t_high:.2f})",
    }
    policy_table = pd.DataFrame(
        {
            "policy": [labels[k] for k in policies],
            "€ per 1,000": [round(p["cost_per_1000"]) for p in policies.values()],
            "share reviewed": [f"{p['review_share']:.1%}" for p in policies.values()],
            "reviewers needed": [round(p["reviewers_needed"], 2) for p in policies.values()],
        }
    )

    return confusion, summary, policy_table, cost_curve(t_low, t_high)


def cost_curve(t_low, t_high):
    fig = go.Figure()
    fig.add_scatter(x=sweep_val.threshold, y=sweep_val.cost_per_1000,
                    name="validation (threshold chosen here)")
    fig.add_scatter(x=sweep_test.threshold, y=sweep_test.cost_per_1000, name="test")
    fig.add_vline(x=T_SINGLE, line_dash="dash", line_color="black",
                  annotation_text=f"chosen {T_SINGLE:.2f}")
    for t, name in ((t_low, "lower"), (t_high, "upper")):
        fig.add_vline(x=t, line_dash="dot", line_color="grey",
                      annotation_text=name, annotation_position="bottom right")
    fig.update_layout(
        title="Cost against threshold (single threshold, no review)",
        xaxis_title="Poisonous confidence threshold",
        yaxis_title="€ per 1,000 mushrooms",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# --------------------------------------------------------------------------- #
# Tab 3: per-class results
# --------------------------------------------------------------------------- #

def per_class_table() -> pd.DataFrame:
    finetuned = metrics["finetuned"]
    rows = [
        {"class": name, "precision": v["precision"], "recall": v["recall"],
         "AP50": v["ap50"], "AP50-95": v["ap50_95"]}
        for name, v in finetuned["per_class"].items()
    ]
    rows.append({"class": "all (mean)", "precision": None, "recall": None,
                 "AP50": finetuned["map50"], "AP50-95": finetuned["map50_95"]})
    return pd.DataFrame(rows).round(3)


def baseline_table() -> pd.DataFrame:
    at_chosen = operating["test_policies"]["model_alone"]["cost_per_1000"]
    rows = [{"model": "Finetuned yolo11n", "mAP50": metrics["finetuned"]["map50"],
             "mAP50-95": metrics["finetuned"]["map50_95"], "€ per 1,000": at_chosen}]
    for name, b in metrics["baselines"].items():
        rows.append({"model": name, "mAP50": b["map50"], "mAP50-95": b["map50_95"],
                     "€ per 1,000": b["test_cost_at_chosen_threshold"]["cost_per_1000"]})
    return pd.DataFrame(rows).round(3)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

with gr.Blocks(title="Mushroom safety check") as demo:
    gr.Markdown(
        f"# Can I eat this mushroom?\n"
        f"Positive class: **{POSITIVE}**. Thresholds apply to the model's "
        f"{POSITIVE} confidence: below the lower one → PASS, between → REVIEW, "
        f"at or above the upper one → FAIL."
    )
    with gr.Row():
        lower = gr.Slider(0, 1, value=T_LOW, step=0.01, label="Lower threshold (PASS below)")
        upper = gr.Slider(0, 1, value=T_HIGH, step=0.01, label="Upper threshold (FAIL at or above)")

    with gr.Tab("Check a mushroom"):
        with gr.Row():
            upload = gr.Image(type="pil", label="Upload a photo")
            annotated = gr.Image(label="Decision per mushroom")
        decisions = gr.Dataframe(label="Mushrooms found", interactive=False)

    with gr.Tab("Cost and review"):
        summary = gr.Markdown()
        confusion = gr.Dataframe(label="Test split: true class × decision", interactive=False)
        policies = gr.Dataframe(label="Against the alternatives (test split)", interactive=False)
        curve = gr.Plot()

    with gr.Tab("Per-class results"):
        gr.Dataframe(per_class_table(), label="Finetuned model on the test split",
                     interactive=False)
        gr.Dataframe(baseline_table(), label="Against the baselines (same test split)",
                     interactive=False)
        if (ARTIFACTS / "confusion_matrix.png").exists():
            gr.Image(str(ARTIFACTS / "confusion_matrix.png"),
                     label="Ultralytics confusion matrix (test split, confidence 0.25)")

    cost_outputs = [confusion, summary, policies, curve]
    for slider in (lower, upper):
        slider.change(cost_view, [lower, upper], cost_outputs)
        slider.change(check_image, [upload, lower, upper], [annotated, decisions])
    upload.change(check_image, [upload, lower, upper], [annotated, decisions])
    demo.load(cost_view, [lower, upper], cost_outputs)


if __name__ == "__main__":
    demo.launch()
