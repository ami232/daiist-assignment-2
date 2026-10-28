"""
Assignment 2: Gradio dashboard (run with `uv run python main.py app`).

Loads only what the training pipeline committed: weights/best.pt and the tables in
artifacts/. It never trains, downloads or re-evaluates anything. The threshold
sliders recompute costs from the cached per-fruit scores of the test split.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from ultralytics import YOLO

import costs
from evaluate import FRUIT_IOU, iou_matrix

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "artifacts" / "eval"
COSTS_DIR = ROOT / "artifacts" / "costs"

MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
NAMES = MODEL.names
OPERATING = json.loads((COSTS_DIR / "operating_point.json").read_text())
TRAINING = json.loads((ROOT / "artifacts" / "training.json").read_text())
TEST_SCORES = pd.read_csv(EVAL_DIR / "fruit_scores_test.csv")
VAL_SCORES = pd.read_csv(EVAL_DIR / "fruit_scores_val.csv")
SWEEP_VAL = pd.read_csv(COSTS_DIR / "sweep_val.csv")
SWEEP_TEST = pd.read_csv(COSTS_DIR / "sweep_test.csv")
DEFAULT_REVIEW = OPERATING["review"]["review_from"]
DEFAULT_REJECT = min(OPERATING["review"]["reject_at"], 1.0)
MIN_FRUIT_CONF = 0.25  # a cluster of boxes counts as a fruit if its best box reaches this

BLUE, ORANGE, INK_2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
DECISION_COLORS = {"pass": "#0ca30c", "review": "#fab219", "reject": "#d03b3b"}


def decide(score: float, review_from: float, reject_at: float) -> str:
    if score >= reject_at:
        return "reject"
    return "review" if score >= review_from else "pass"


def inspect_image(image: Image.Image | None, review_from: float, reject_at: float):
    """Detect fruit, group overlapping fresh/rotten boxes into one fruit, and decide per fruit."""
    if image is None:
        return None, pd.DataFrame()
    review_from, reject_at = sorted((review_from, reject_at))
    # Same confidence floor as evaluate.predict, so the thresholds mean the same here as in the analysis
    result = MODEL.predict(image, conf=0.001, iou=0.7, imgsz=640, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.int().cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    rotten = np.array([NAMES[c].startswith("rotten") for c in cls], dtype=bool)

    fruits, used = [], np.zeros(len(boxes), dtype=bool)
    iou = iou_matrix(boxes, boxes) if len(boxes) else np.zeros((0, 0))
    for i in np.argsort(-conf):
        if used[i] or conf[i] < MIN_FRUIT_CONF:
            continue
        group = (iou[i] >= FRUIT_IOU) & ~used
        used |= group
        score = float(conf[group & rotten].max(initial=0.0))
        fruits.append({"box": boxes[i], "label": NAMES[cls[i]], "rotten_score": score,
                       "decision": decide(score, review_from, reject_at)})

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, canvas.width // 200)
    for k, f in enumerate(fruits, 1):
        x1, y1, x2, y2 = f["box"]
        color = DECISION_COLORS[f["decision"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1 + 4, y1 + 4), f"#{k} {f['decision'].upper()} {f['rotten_score']:.3f}", fill=color)
    table = pd.DataFrame([{"fruit": k, "top box": f["label"], "rotten score": round(f["rotten_score"], 3),
                           "decision": f["decision"]} for k, f in enumerate(fruits, 1)])
    return canvas, table


def operating_view(review_from: float, reject_at: float, base_rate_pct: float):
    """Everything that depends on the two thresholds, computed from the cached test scores."""
    review_from, reject_at = sorted((review_from, reject_at))
    base_rate = base_rate_pct / 100
    r = costs.rates(TEST_SCORES, reject_at, review_from)
    cost = costs.cost_per_1000(r, base_rate, review=costs.review_terms())
    share = costs.review_share(r, base_rate)
    people = costs.people_needed(share, costs.REVIEW["seconds"])
    capacity = costs.review_capacity(costs.REVIEW["people"])
    refs = costs.reference_policies(base_rate)

    per_1000 = pd.DataFrame(
        [[round(1000 * base_rate * r[f"rotten_{o}"], 1) for o in ("pass", "review", "reject")],
         [round(1000 * (1 - base_rate) * r[f"fresh_{o}"], 1) for o in ("pass", "review", "reject")]],
        index=["truly rotten", "truly fresh"], columns=["pass", "review", "reject"]).reset_index(names="")
    n_rot, n_fresh = int(TEST_SCORES.rotten.sum()), int((~TEST_SCORES.rotten).sum())
    counts = pd.DataFrame(
        [[round(n_rot * r[f"rotten_{o}"]) for o in ("pass", "review", "reject")],
         [round(n_fresh * r[f"fresh_{o}"]) for o in ("pass", "review", "reject")]],
        index=[f"rotten ({n_rot})", f"fresh ({n_fresh})"], columns=["pass", "review", "reject"]).reset_index(names="")

    status = "within" if share <= capacity + 1e-9 else "OVER"
    summary = (
        f"### €{cost:,.2f} expected cost per 1,000 fruit\n"
        f"- **Sent to review:** {share:.1%} of fruit, about {share * costs.FRUIT_PER_DAY:,.0f} a day, "
        f"which takes **{share * costs.FRUIT_PER_DAY * costs.REVIEW['seconds'] / 3600:.1f} h** of an operator's "
        f"time at {costs.REVIEW['seconds']:.0f} s each ({people} person, rounded up). "
        f"That is {status} the capacity of {costs.REVIEW['people']} person ({capacity:.1%}).\n"
        f"- **Rotten fruit caught:** {r['rotten_reject']:.1%} rejected automatically and "
        f"{r['rotten_review']:.1%} reviewed. **Missed:** {r['rotten_pass']:.1%}.\n"
        f"- **Fresh fruit lost to industry:** {r['fresh_reject']:.1%}.\n"
        f"- **For comparison:** passing everything costs €{refs['pass_all']:.0f}, and rejecting everything costs "
        f"€{refs['reject_all']:.0f}.\n\n"
        f"Rates come from the test split ({len(TEST_SCORES):,} labelled fruit), and the cost assumes a "
        f"{base_rate:.1%} base rate."
    )
    return summary, per_1000, counts, cost_curve(review_from, reject_at, base_rate)


def cost_curve(review_from: float, reject_at: float, base_rate: float) -> go.Figure:
    if np.isclose(base_rate, costs.BASE_RATE):
        val, test = SWEEP_VAL, SWEEP_TEST
    else:
        val, test = costs.sweep(VAL_SCORES, base_rate), costs.sweep(TEST_SCORES, base_rate)
    chosen = OPERATING["chosen_threshold"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val.threshold, y=val.cost_per_1000, name="validation (used to choose)",
                             line={"color": BLUE, "width": 2},
                             hovertemplate="t = %{x:.3f}<br>€%{y:.1f} per 1,000<extra>validation</extra>"))
    fig.add_trace(go.Scatter(x=test.threshold, y=test.cost_per_1000, name="test (reported)",
                             line={"color": ORANGE, "width": 2},
                             hovertemplate="t = %{x:.3f}<br>€%{y:.1f} per 1,000<extra>test</extra>"))
    chosen_cost = float(test.loc[np.isclose(test.threshold, chosen), "cost_per_1000"].iloc[0])
    fig.add_trace(go.Scatter(x=[chosen], y=[chosen_cost], mode="markers", name=f"chosen t = {chosen:g}",
                             marker={"size": 11, "color": ORANGE, "line": {"color": "white", "width": 2}}))
    fig.add_vrect(x0=review_from, x1=min(reject_at, 1.0), fillcolor="#fab219", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=380, margin={"l": 60, "r": 20, "t": 40, "b": 50},
                      title={"text": "Expected cost against threshold (model alone, no review)", "x": 0.01},
                      xaxis_title="rotten-score threshold t (reject if score ≥ t)",
                      yaxis_title="€ per 1,000 fruit", legend={"orientation": "h", "y": -0.25},
                      hovermode="x unified")
    return fig


def results_tables():
    overall = pd.read_csv(EVAL_DIR / "overall_test.csv").round(3)
    per_class = pd.read_csv(EVAL_DIR / "per_class_test.csv").round(3)
    by_state = pd.read_csv(EVAL_DIR / "by_state_test.csv").round(3)
    scenarios = pd.read_csv(COSTS_DIR / "scenarios.csv")
    scenarios = scenarios[["scenario", "people", "checked_per_day", "checking_hours_per_day", "rotten_shipped_per_day", "fresh_lost_per_day",
                           "rotten_share_of_packed", "labour_cost_per_day", "error_cost_per_day",
                           "total_per_day", "total_per_1000"]].round(3)
    return overall, per_class, by_state, scenarios


with gr.Blocks(title="Rotten fruit sorting") as demo:
    gr.Markdown(
        "# Rotten fruit sorting: cost-based operating point\n"
        f"Finetuned {TRAINING['chosen_model']} (`weights/best.pt`). "
        "Each fruit is **passed**, sent for **review**, or **rejected** to industry according to its rotten score. "
        f"Costs: false positive €{costs.COSTS['fp']:.2f}, false negative €{costs.COSTS['fn']:.2f}.")
    with gr.Row():
        review_slider = gr.Slider(0.001, 1.0, value=DEFAULT_REVIEW, step=0.001, label="Review from (lower threshold)")
        reject_slider = gr.Slider(0.001, 1.0, value=DEFAULT_REJECT, step=0.001, label="Reject at (upper threshold)")
        base_slider = gr.Slider(0.5, 20, value=costs.BASE_RATE * 100, step=0.5, label="Assumed base rate (% rotten)")

    with gr.Tab("Operating point"):
        summary = gr.Markdown()
        with gr.Row():
            per_1000 = gr.Dataframe(label="Expected outcomes per 1,000 fruit at the assumed base rate")
            counts = gr.Dataframe(label="Confusion matrix on the test split (fruit counts)")
        curve = gr.Plot()

    with gr.Tab("Inspect an image"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Upload an image")
            image_out = gr.Image(type="pil", label="Decisions (green pass, yellow review, red reject)")
        fruit_table = gr.Dataframe(label="Per fruit")

    with gr.Tab("Per-class results"):
        overall, per_class, by_state, scenarios = results_tables()
        gr.Markdown("Test split, same images for every model. Precision and recall are taken at the "
                    "max-F1 confidence; AP50 and mAP50-95 are threshold-free.")
        gr.Dataframe(overall, label="Overall: finetuned models against the baselines")
        gr.Dataframe(by_state, label="Fresh vs rotten (positive class: rotten)")
        gr.Dataframe(per_class, label="Per class")
        confusion_png = EVAL_DIR / f"confusion_{overall.model.iloc[0]}.png"
        if confusion_png.exists():
            gr.Image(str(confusion_png), label="Detection confusion matrix of the chosen model")

    with gr.Tab("Human review scenarios"):
        gr.Markdown(f"{costs.FRUIT_PER_DAY:,} fruit a day, with thresholds chosen on validation and rates "
                    "measured on test. Checking every fruit needs dedicated staff, paid by the shift; "
                    "the review band is a task for an operator already on the line, paid for the time it takes.")
        gr.Dataframe(scenarios, label="Cost of each way of running the line")

    view_inputs = [review_slider, reject_slider, base_slider]
    view_outputs = [summary, per_1000, counts, curve]
    for control in view_inputs:
        control.change(operating_view, view_inputs, view_outputs)
    for control in (review_slider, reject_slider):
        control.change(inspect_image, [image_in, review_slider, reject_slider], [image_out, fruit_table])
    image_in.change(inspect_image, [image_in, review_slider, reject_slider], [image_out, fruit_table])
    demo.load(operating_view, view_inputs, view_outputs)


if __name__ == "__main__":
    demo.launch()
