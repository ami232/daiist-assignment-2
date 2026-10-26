"""
Assignment 2: Gradio dashboard (fire and smoke detection).

Run with:  uv run python main.py app

It never trains anything. It only loads files that the pipeline already made:
  weights/best.pt                the finetuned model
  outputs/test_predictions.json  the model's predictions on the test images
  outputs/cost_results.json      the assumptions and the chosen operating point
  outputs/train_results.json     test metrics, per class, and the baselines
  outputs/evaluation.json        object-level confusion matrix and results by size
  outputs/confusion_matrix.png

Two thresholds split every case into three outcomes:
  below the lower threshold  -> PASS (nothing happens)
  between the two            -> REVIEW (a person checks the image)
  at or above the upper one  -> FAIL (the firemen are called directly)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "outputs"

import gradio as gr
import numpy as np
from matplotlib.figure import Figure
from PIL import ImageDraw
from ultralytics import YOLO

from costing import GRID, load_scores

PORT = int(os.environ.get("APP_PORT", "7860"))


# ----------------------------------------------------------------- loading --
def read_json(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the training pipeline (and costing.py and "
            f"evaluation.py) first: this app only reads what they produced."
        )
    return json.loads(path.read_text())


def find_weights():
    best = ROOT / "weights" / "best.pt"
    if best.exists():
        return best
    found = sorted((ROOT / "weights").rglob("*.pt"))
    if not found:
        raise FileNotFoundError("No .pt file in weights/. Commit your trained weights.")
    return found[0]


COST = read_json("cost_results.json")
TRAIN = read_json("train_results.json")
EVAL = read_json("evaluation.json")
A = COST["assumptions"]
CHOSEN = COST["chosen"]
POS, NEG = load_scores()  # best Fire score per test image: with fire / without fire
CAP_SHARE = A["review_capacity_per_day"] / A["cases_per_day"]

MODEL = YOLO(str(find_weights()))
NAMES = dict(MODEL.names)
FIRE_ID = [i for i, n in NAMES.items() if n.lower() == "fire"][0]
COLORS = {"fire": (220, 40, 40), "smoke": (30, 120, 220)}


# -------------------------------------------------------------------- cost --
def shares(scores, low, high):
    """Share of cases that pass, go to review, and fail."""
    passed = float((scores < low).mean())
    fail = float((scores >= high).mean())
    return passed, 1.0 - passed - fail, fail


def breakdown(low, high):
    """Expected cost per case, split into its four parts."""
    p = A["base_rate"]
    pos_pass, pos_rev, _ = shares(POS, low, high)
    _, neg_rev, neg_fail = shares(NEG, low, high)
    parts = {
        "missed": p * pos_pass * A["cost_missed_fire"],
        "delay": p * pos_rev * A["cost_review_delay"],
        "checks": (p * pos_rev + (1 - p) * neg_rev) * A["cost_review"],
        "callouts": (1 - p) * neg_fail * A["cost_false_callout"],
    }
    parts["total"] = sum(parts.values())
    parts["share"] = p * pos_rev + (1 - p) * neg_rev
    return parts


def fix(low, high):
    low, high = float(low), float(high)
    if low > high:
        return low, low, True
    return low, high, False


ALONE = np.array([breakdown(float(t), float(t))["total"] * 1000 for t in GRID])


def curve(high):
    lows = GRID[GRID <= high + 1e-9]
    costs = np.empty(len(lows))
    review = np.empty(len(lows))
    for n, low in enumerate(lows):
        b = breakdown(float(low), high)
        costs[n], review[n] = b["total"] * 1000, b["share"]
    return lows, costs, review


def make_curve(low, high, current):
    fig = Figure(figsize=(8.5, 4.8))
    ax = fig.subplots()
    ax.plot(GRID, ALONE, color="tab:blue", label="Model alone (one threshold, nobody checks)")
    lows, costs, review = curve(high)
    ok = review <= CAP_SHARE + 1e-9
    ax.plot(lows, costs, color="tab:orange", linestyle="--", alpha=0.6,
            label="With human review: more checks than the team can do")
    ax.plot(lows[ok], costs[ok], color="tab:orange",
            label=f"With human review: within {A['review_capacity_per_day']:,} checks a day "
                  f"(upper threshold {high:.2f})")
    ax.scatter([CHOSEN["lower_threshold"]], [CHOSEN["cost_per_1000"]], color="red", s=80,
               zorder=6, label=f"Chosen operating point (lower {CHOSEN['lower_threshold']:.2f}, "
                               f"upper {CHOSEN['upper_threshold']:.2f})")
    ax.scatter([low], [current], marker="D", color="black", s=45, zorder=7,
               label=f"Your current setting (lower {low:.2f}, upper {high:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("Confidence threshold (lower threshold for the orange line)")
    ax.set_ylabel("Expected cost per 1,000 cases (euros, log scale)")
    ax.set_title("Cost against threshold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def update_thresholds(low, high):
    low, high, moved = fix(low, high)
    b = breakdown(low, high)
    p = A["base_rate"]

    reviews = b["share"] * A["cases_per_day"]
    hours = reviews * A["seconds_per_review"] / 3600
    operators = hours / A["hours_per_operator_day"]
    fits = reviews <= A["review_capacity_per_day"] + 1e-9
    summary = (
        f"### Expected cost per 1,000 cases: EUR {b['total'] * 1000:,.0f}\n"
        f"- Sent to a person: **{b['share'] * 100:.1f}%** of cases, which is "
        f"**{reviews:,.0f} a day** (of {A['cases_per_day']:,} images), "
        f"{hours:.1f} hours of checking, **{operators:.2f} operators**.\n"
        f"- The team can do {A['review_capacity_per_day']:,} checks a day: "
        f"{'this fits.' if fits else '**this is TOO MANY for the team.**'}\n"
        f"- Chosen operating point from the report: lower {CHOSEN['lower_threshold']:.2f}, "
        f"upper {CHOSEN['upper_threshold']:.2f}, EUR {CHOSEN['cost_per_1000']:,.0f} per 1,000 cases."
        + ("\n- The upper threshold cannot be below the lower one, so it was moved up to match."
           if moved else "")
    )

    rows_test, rows_1000 = [], []
    for label, scores, n_per_1000, decimals in (
        ("Real fire", POS, 1000 * p, 3),
        ("No fire (smoke only in the test set)", NEG, 1000 * (1 - p), 1),
    ):
        n = len(scores)
        pa, re, fa = shares(scores, low, high)
        counts = [round(pa * n), round(re * n), round(fa * n)]
        rows_test.append([f"{label} ({n} test images)"] +
                         [f"{c} ({c / n:.0%})" for c in counts])
        rows_1000.append([f"{label} ({n_per_1000:g} per 1,000 cases)"] +
                         [f"{n_per_1000 * s:.{decimals}f}" for s in (pa, re, fa)])
    head = ["", "Pass", "Review (a person checks)", "Fail (firemen called)"]

    rows_cost = [
        ["Real fires that were passed (missed)", f"{b['missed'] * 1000:,.2f}"],
        ["Extra damage while a real fire waits for a person", f"{b['delay'] * 1000:,.2f}"],
        ["The checks themselves", f"{b['checks'] * 1000:,.2f}"],
        ["False callouts (firemen called for nothing)", f"{b['callouts'] * 1000:,.2f}"],
        ["Total", f"{b['total'] * 1000:,.2f}"],
    ]
    fig = make_curve(low, high, b["total"] * 1000)
    return (summary,
            gr.update(value=rows_test, headers=head),
            gr.update(value=rows_1000, headers=head),
            rows_cost, fig)


# ------------------------------------------------------------- predictions --
def predict_image(image, low, high):
    if image is None:
        return None, "Upload an image to see what the model finds.", []
    low, high, _ = fix(low, high)
    image = image.convert("RGB")
    result = MODEL.predict(image, conf=0.001, iou=0.6, imgsz=640, verbose=False)[0]
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)

    fire_conf = conf[cls == FIRE_ID]
    score = float(fire_conf.max()) if len(fire_conf) else 0.0
    if score < low:
        verdict = (f"## PASS\nThe highest fire confidence is {score:.2f}, below the lower "
                   f"threshold ({low:.2f}). Nothing happens.")
    elif score < high:
        verdict = (f"## SEND TO A PERSON\nThe highest fire confidence is {score:.2f}, between "
                   f"{low:.2f} and {high:.2f}. The operator looks at this image and decides.")
    else:
        verdict = (f"## FAIL: CALL THE FIREMEN\nThe highest fire confidence is {score:.2f}, "
                   f"at or above the upper threshold ({high:.2f}).")

    shown = image.copy()
    draw = ImageDraw.Draw(shown)
    rows = []
    for box, c, k in sorted(zip(xyxy, conf, cls), key=lambda t: -t[1]):
        if c < low:
            continue
        color = COLORS.get(NAMES[k].lower(), (255, 165, 0))
        draw.rectangle([float(v) for v in box], outline=color, width=max(2, shown.width // 200))
        draw.text((float(box[0]) + 4, float(box[1]) + 4), f"{NAMES[k]} {c:.2f}", fill=color)
        rows.append([NAMES[k], f"{c:.2f}"])
    if not rows:
        verdict += "\n\nNo boxes are drawn: nothing reaches the lower threshold."
    return shown, verdict, rows


# --------------------------------------------------------- static results ---
ft = TRAIN["finetuned"]
PER_CLASS_ROWS = [
    [n, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['AP50']:.3f}", f"{v['mAP50_95']:.3f}"]
    for n, v in ft["per_class"].items()
]
PER_CLASS_ROWS.append(["All classes", "-", "-", f"{ft['AP50']:.3f}", f"{ft['mAP50_95']:.3f}"])

BASELINE_ROWS = [
    ["My finetuned yolo11n", f"{ft['AP50']:.3f}", f"{ft['mAP50_95']:.3f}"],
    ["Pretrained COCO model, no finetuning", f"{TRAIN['coco_zero_shot']['AP50']:.3f}",
     f"{TRAIN['coco_zero_shot']['mAP50_95']:.3f}"],
    ["Predict nothing", f"{TRAIN['predict_nothing']['AP50']:.3f}",
     f"{TRAIN['predict_nothing']['mAP50_95']:.3f}"],
]

LABELS = ["Nothing" if l == "Background" else l for l in EVAL["labels"]]
MATRIX_ROWS = [[LABELS[r]] + [str(v) for v in row]
               for r, row in enumerate(EVAL["matrix_rows_actual_columns_predicted"])]
MATRIX_HEAD = ["Really there / model said"] + LABELS

SIZE_ROWS = []
for cname, row in EVAL["per_class"].items():
    for size, s in row["recall_by_size"].items():
        if s["objects"]:
            SIZE_ROWS.append([cname, size, f"{s['found']} of {s['objects']}", f"{s['recall']:.0%}"])

ALT = COST["alternatives_cost_per_1000"]
ALT_ROWS = [
    ["Do nothing", f"{ALT['do_nothing']:,.0f}", "0"],
    ["A person checks every image", f"{ALT['person_checks_everything']:,.0f}",
     f"{COST['everyone_checked_workload']['operators_needed']:.1f}"],
    ["The model decides alone (best single threshold)", f"{ALT['model_alone']:,.0f}", "0"],
    ["Model plus a person (chosen)", f"{ALT['model_plus_people']:,.0f}",
     f"{COST['chosen_workload']['operators_needed']:.2f}"],
]


# ---------------------------------------------------------------------- UI --
with gr.Blocks(title="Fire and smoke detection: cost dashboard") as demo:
    gr.Markdown(
        "# Fire and smoke detection: cost and human review\n"
        "Below the lower threshold an image passes. Between the two thresholds a person "
        "checks it. From the upper threshold up, the firemen are called directly. "
        "Move the sliders and every tab updates."
    )
    with gr.Row():
        low_slider = gr.Slider(0, 1, value=CHOSEN["lower_threshold"], step=0.01,
                               label="Lower threshold (below this: pass)")
        high_slider = gr.Slider(0, 1, value=CHOSEN["upper_threshold"], step=0.01,
                                label="Upper threshold (from here up: call the firemen)")
    sliders = [low_slider, high_slider]

    with gr.Tabs():
        with gr.Tab("Try an image"):
            image_in = gr.Image(type="pil", label="Upload an image")
            verdict_out = gr.Markdown()
            image_out = gr.Image(label="Predictions at the lower threshold", interactive=False)
            boxes_out = gr.Dataframe(headers=["Class", "Confidence"], value=[],
                                     label="Boxes found at or above the lower threshold",
                                     interactive=False)

        with gr.Tab("Cost and human review"):
            summary_out = gr.Markdown()
            gr.Markdown("**Confusion matrix** at these thresholds: what happens to the "
                        "test images, and what that means per 1,000 cases at my assumed base rate.")
            table_test_out = gr.Dataframe(interactive=False, label="Test images")
            table_1000_out = gr.Dataframe(interactive=False, label="Per 1,000 cases")
            cost_out = gr.Dataframe(headers=["Part of the cost", "EUR per 1,000 cases"],
                                    value=[], interactive=False, label="Where the cost comes from")
            curve_out = gr.Plot(label="Cost against threshold")
            gr.Dataframe(headers=["Plan", "EUR per 1,000 cases", "Operators needed"],
                         value=ALT_ROWS, interactive=False,
                         label="My design against the alternatives")

        with gr.Tab("Per-class results"):
            gr.Markdown("All numbers are on the 435 test images. Fire is the positive class.")
            gr.Dataframe(headers=["Class", "Precision", "Recall", "AP50", "mAP50-95"],
                         value=PER_CLASS_ROWS, interactive=False, label="Per class")
            gr.Dataframe(headers=["Model", "AP50", "mAP50-95"], value=BASELINE_ROWS,
                         interactive=False, label="Against the two baselines")
            gr.Markdown(f"**Confusion matrix by object** (model confidence at least "
                        f"{EVAL['confidence_used']:.2f}, boxes must overlap at least "
                        f"{EVAL['iou_match']:.0%}). 'Nothing' in the last column is a missed "
                        f"object, in the last row it is a box where nothing was there.")
            gr.Dataframe(headers=MATRIX_HEAD, value=MATRIX_ROWS, interactive=False)
            matrix_png = OUTPUT_DIR / "confusion_matrix.png"
            if matrix_png.exists():
                gr.Image(value=str(matrix_png), interactive=False, label="Confusion matrix")
            gr.Dataframe(headers=["Class", "Size of the object in the image", "Found", "Recall"],
                         value=SIZE_ROWS, interactive=False,
                         label="Where it breaks: recall by object size")

    threshold_outputs = [summary_out, table_test_out, table_1000_out, cost_out, curve_out]
    prediction_inputs = [image_in] + sliders
    prediction_outputs = [image_out, verdict_out, boxes_out]
    for slider in sliders:
        slider.change(update_thresholds, sliders, threshold_outputs)
        slider.change(predict_image, prediction_inputs, prediction_outputs)
    image_in.change(predict_image, prediction_inputs, prediction_outputs)
    demo.load(update_thresholds, sliders, threshold_outputs)

demo.launch(server_port=PORT)