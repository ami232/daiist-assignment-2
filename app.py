"""
Care-home fall alerts: Gradio dashboard.

Run with `uv run python main.py app`. It loads the committed model in weights/ and the results
train.ipynb saved in outputs/, and never trains anything. Every number on screen is recomputed
from the saved per-image fall scores, so moving a slider is instant.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

OUTPUTS = Path("outputs")
model = YOLO("weights/best.pt")
metrics = json.loads((OUTPUTS / "test_metrics.json").read_text())
scores = json.loads((OUTPUTS / "test_scores.json").read_text())
sweep = json.loads((OUTPUTS / "cost_sweep.json").read_text())
review = json.loads((OUTPUTS / "review_design.json").read_text())

COSTS, BASE_RATE = sweep["costs"], sweep["base_rate"]
A = review["assumptions"]
FALLING = next(i for i, name in model.names.items() if name == "Falling")
MODEL_ALONE_CUT = sweep["chosen"]["cut"]
CHOSEN_LOW, CHOSEN_HIGH = review["design"]["low"], review["design"]["high"]

# Same split as train.ipynb: thresholds were chosen on the tune half, costs are quoted on the
# report half. The internal split is offered only as the leaked upper bound.
external = {c["image"]: c for c in scores["external"]}
TEST_SETS = {
    "External, report half": [external[n] for n in sweep["split"]["report"]],
    "External, tune half": [external[n] for n in sweep["split"]["tune"]],
    "Internal test (leaked, upper bound)": scores["internal"],
}


def three_way(cases, low, high):
    # The same formula as train.ipynb Section 6: zone shares measured separately for falls and
    # non-falls, projected onto 1,000 cases at the real base rate.
    def zones(values):
        n = len(values)
        return (sum(x < low for x in values) / n,
                sum(low <= x < high for x in values) / n,
                sum(x >= high for x in values) / n)

    fall_pass, fall_review, fall_fail = zones([c["finetuned"] for c in cases if c["truth"]])
    other_pass, other_review, other_fail = zones([c["finetuned"] for c in cases if not c["truth"]])
    falls, others = 1000 * BASE_RATE, 1000 * (1 - BASE_RATE)

    tp = falls * (fall_fail + fall_review * A["reviewer_recall"])
    fn = falls - tp
    fp = others * (other_fail + other_review * (1 - A["reviewer_specificity"]))
    tn = others - fp
    reviews = falls * fall_review + others * other_review
    cost = (tp * COSTS["tp"] + fp * COSTS["fp"] + fn * COSTS["fn"] + tn * COSTS["tn"]
            + reviews * A["review_cost"])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": cost, "falls_caught": tp / falls,
            "review_share": reviews / 1000, "trip_share": (tp + fp) / 1000}


def ordered(low, high):
    # The review line can never sit above the nurse line; if the sliders cross, treat them as equal.
    return low, max(low, high)


# ---------- Tab 1: check an image ----------

def check_image(image, low, high):
    if image is None:
        return None, "Upload an image to see a prediction.", pd.DataFrame(columns=["class", "confidence"])
    low, high = ordered(low, high)
    result = model.predict(image, conf=0.01, imgsz=640, verbose=False)[0]
    boxes = result.boxes
    fall_score = max((float(c) for c, k in zip(boxes.conf, boxes.cls) if int(k) == FALLING),
                     default=0.0)

    if fall_score >= high:
        decision = f"## NURSE GOES NOW\nfall score {fall_score:.2f} is at or above the nurse line {high:.2f}"
    elif fall_score >= low:
        decision = f"## SEND TO SCREEN REVIEW\nfall score {fall_score:.2f} is between {low:.2f} and {high:.2f}"
    else:
        decision = f"## PASS\nfall score {fall_score:.2f} is below the review line {low:.2f}"

    # Draw only boxes a person would care to see; the decision above still uses every box.
    shown = result[boxes.conf >= 0.10]
    annotated = shown.plot()[:, :, ::-1]  # Ultralytics draws in BGR, Gradio expects RGB
    table = pd.DataFrame(
        [{"class": model.names[int(k)], "confidence": round(float(c), 2)}
         for c, k in zip(shown.boxes.conf, shown.boxes.cls)],
        columns=["class", "confidence"],
    )
    return annotated, decision, table


# ---------- Tab 2: thresholds and cost ----------

def thresholds_and_cost(low, high, test_set):
    low, high = ordered(low, high)
    cases = TEST_SETS[test_set]
    d = three_way(cases, low, high)
    reviews_night = d["review_share"] * A["images_per_night"]
    trips_night = d["trip_share"] * A["images_per_night"]

    kpis = (
        f"### EUR {d['cost']:,.0f} per 1,000 cases\n"
        f"**{d['review_share']:.0%}** sent to screen review · **{d['falls_caught']:.0%}** of falls caught · "
        f"**{reviews_night:.0f}** screen checks and **{trips_night:.1f}** nurse trips per night "
        f"(100-bed home, about {A['images_per_night']:.0f} images a night)"
    )

    def route(truth):
        values = [c["finetuned"] for c in cases if c["truth"] == truth]
        return [sum(x < low for x in values), sum(low <= x < high for x in values),
                sum(x >= high for x in values)]

    n_falls = sum(c["truth"] for c in cases)
    routing = pd.DataFrame(
        [[f"Fall ({n_falls})", *route(True)], [f"No fall ({len(cases) - n_falls})", *route(False)]],
        columns=["truth", "pass", "review", "nurse"],
    )
    confusion = pd.DataFrame(
        [["nurse sent", round(d["tp"], 1), round(d["fp"], 1)],
         ["nurse not sent", round(d["fn"], 1), round(d["tn"], 1)]],
        columns=["", "truth: fall", "truth: no fall"],
    )

    designs = {
        "Do nothing": {"cost": 1000 * BASE_RATE * COSTS["fn"], "falls_caught": 0.0},
        f"Model alone (cut {MODEL_ALONE_CUT})": three_way(cases, MODEL_ALONE_CUT, MODEL_ALONE_CUT),
        "Nurse walks to every image": three_way(cases, 0, 0),
        "Staff review every image": three_way(cases, 0, 1.01),
        f"Chosen design ({CHOSEN_LOW} / {CHOSEN_HIGH})": three_way(cases, CHOSEN_LOW, CHOSEN_HIGH),
        f"Current sliders ({low:.2f} / {high:.2f})": d,
    }
    alternatives = pd.DataFrame(
        [{"design": name, "EUR per 1,000": round(x["cost"]), "falls caught": f"{x['falls_caught']:.0%}"}
         for name, x in designs.items()]
    )
    return kpis, routing, confusion, alternatives, cost_curve(low, high, test_set)


# ---------- Tab 3: cost curve ----------

def cost_curve(low, high, test_set):
    low, high = ordered(low, high)
    cases = TEST_SETS[test_set]
    lines = [round(i / 100, 2) for i in range(0, 102)]
    curve = [three_way(cases, low, line) if line >= low else None for line in lines]

    fig = go.Figure()
    xs = [x for x, d in zip(lines, curve) if d]
    ys = [d["cost"] for d in curve if d]
    within_cap = [d["review_share"] <= A["review_cap"] for d in curve if d]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="cost, review line fixed", line={"width": 3}))
    fig.add_trace(go.Scatter(x=[x for x, ok in zip(xs, within_cap) if not ok],
                             y=[y for y, ok in zip(ys, within_cap) if not ok], mode="markers",
                             name=f"over the {A['review_cap']:.0%} review cap",
                             marker={"color": "lightgrey", "size": 5}))
    chosen = three_way(cases, CHOSEN_LOW, CHOSEN_HIGH)
    fig.add_trace(go.Scatter(x=[CHOSEN_HIGH], y=[chosen["cost"]], mode="markers+text",
                             name="chosen design", text=[f"chosen {CHOSEN_HIGH}"],
                             textposition="top center", marker={"size": 13, "color": "black"}))
    current = three_way(cases, low, high)
    fig.add_trace(go.Scatter(x=[high], y=[current["cost"]], mode="markers", name="current sliders",
                             marker={"size": 11, "symbol": "diamond", "color": "crimson"}))
    fig.add_hline(y=three_way(cases, 0, 0)["cost"], line_dash="dash",
                  annotation_text="nurse walks to every image")
    fig.update_layout(title=f"Cost against the nurse line ({test_set})",
                      xaxis_title="nurse line (go now at or above)",
                      yaxis_title="EUR per 1,000 cases", height=420, margin={"t": 50})
    return fig


def single_threshold_curve():
    # Section 5 of the notebook: one cut, no review, every system, on the report half.
    fig = go.Figure()
    for system, rows in sweep["sweeps"]["report"].items():
        fig.add_trace(go.Scatter(x=[r["cut"] for r in rows], y=[r["cost"] for r in rows],
                                 mode="lines", name=system))
    fig.add_hline(y=sweep["check_all"], line_dash="dash", annotation_text="nurse walks to every image")
    fig.update_layout(title="Model alone: cost against a single threshold (report half)",
                      xaxis_title="confidence threshold", yaxis_title="EUR per 1,000 cases",
                      height=380, margin={"t": 50})
    return fig


cap_table = pd.DataFrame(
    [{"review cap": f"{r['cap']:.0%}", "nurse line": r["high"], "EUR per 1,000": round(r["cost"]),
      "reviewed": f"{r['review_share']:.0%}", "falls caught": f"{r['falls_caught']:.0%}",
      "nurse trips / night": round(r["trip_share"] * A["images_per_night"], 1)}
     for r in review["cap_table"]]
)


# ---------- Tab 4: per-class results ----------

per_class = pd.DataFrame(
    [{"class": name + (" (positive)" if name == "Falling" else ""),
      "internal precision": round(metrics["internal"]["per_class"][name]["precision"], 2),
      "internal recall": round(metrics["internal"]["per_class"][name]["recall"], 2),
      "external precision": round(metrics["external"]["per_class"][name]["precision"], 2),
      "external recall": round(metrics["external"]["per_class"][name]["recall"], 2),
      "external AP50": round(metrics["external"]["per_class"][name]["ap50"], 3)}
     for name in metrics["internal"]["per_class"]]
    + [{"class": "overall mAP50", "internal precision": None,
        "internal recall": round(metrics["internal"]["overall"]["map50"], 3),
        "external precision": None,
        "external recall": round(metrics["external"]["overall"]["map50"], 3), "external AP50": None}]
)


def confusion_heatmap(which):
    labels = metrics["matrix_labels"]
    fig = go.Figure(go.Heatmap(z=metrics[which]["confusion_matrix"], x=labels, y=labels,
                               text=metrics[which]["confusion_matrix"], texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title=f"{which} test", xaxis_title="true class", yaxis_title="predicted class",
                      height=380, margin={"t": 50})
    return fig


nd = metrics["near_duplicates"]
leak_note = (
    f"**Read the internal numbers as an upper bound.** {nd['test']['within_10']} of {nd['test']['images']} "
    f"internal test images are near-copies of training images (median hash distance "
    f"{nd['test']['median']} of 256 bits). The external set has {nd['external']['within_10']} "
    f"(median {nd['external']['median']}), so it is the realistic number."
)


# ---------- Layout ----------

with gr.Blocks(title="Care-home fall alerts") as demo:
    gr.Markdown(
        "# Care-home fall alerts\n"
        "yolo11n finetuned on `falldetection-bwfdz/fall-detection-ehgga` v1. Loads `weights/best.pt` "
        "and `outputs/`; never trains. Below the review line: pass. Between the lines: staff check "
        "the image on screen. At or above the nurse line: the nurse goes at once."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=CHOSEN_LOW, step=0.01, label="Review line (below = pass)")
        high = gr.Slider(0, 1.01, value=CHOSEN_HIGH, step=0.01, label="Nurse line (at or above = nurse goes now)")

    with gr.Tab("1 · Check an image"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Upload a camera image")
            image_out = gr.Image(label="Prediction")
        decision = gr.Markdown()
        boxes_table = gr.Dataframe(headers=["class", "confidence"],
                                   label="Boxes at confidence 0.10 or above")

    with gr.Tab("2 · Thresholds & cost"):
        test_set = gr.Dropdown(list(TEST_SETS), value="External, report half", label="Test set")
        kpis = gr.Markdown()
        with gr.Row():
            routing = gr.Dataframe(label="Where the test images go (counts)")
            confusion = gr.Dataframe(label="Confusion matrix after review, per 1,000 at the real base rate")
        alternatives = gr.Dataframe(label="Against the alternatives")
        gr.Markdown(
            f"Costs: TP EUR {COSTS['tp']:.0f}, FP EUR {COSTS['fp']:.0f}, FN EUR {COSTS['fn']:,.0f}, "
            f"TN EUR {COSTS['tn']:.0f}, review EUR {A['review_cost']:.2f}. Base rate {BASE_RATE:.0%}. "
            f"Reviewer recall {A['reviewer_recall']:.0%}, specificity {A['reviewer_specificity']:.0%}."
        )

    with gr.Tab("3 · Cost curve"):
        curve = gr.Plot()
        gr.Dataframe(cap_table, label="Price of the review cap (fail line chosen on the tune half)")
        gr.Plot(single_threshold_curve())

    with gr.Tab("4 · Per-class results"):
        gr.Markdown(leak_note)
        gr.Dataframe(per_class, label="Per class, box level")
        with gr.Row():
            gr.Plot(confusion_heatmap("internal"))
            gr.Plot(confusion_heatmap("external"))

    image_inputs = [image_in, low, high]
    image_outputs = [image_out, decision, boxes_table]
    image_in.change(check_image, image_inputs, image_outputs)

    cost_inputs = [low, high, test_set]
    cost_outputs = [kpis, routing, confusion, alternatives, curve]
    for control in (low, high):
        control.change(check_image, image_inputs, image_outputs)
        control.change(thresholds_and_cost, cost_inputs, cost_outputs)
    test_set.change(thresholds_and_cost, cost_inputs, cost_outputs)
    demo.load(thresholds_and_cost, cost_inputs, cost_outputs)

if __name__ == "__main__":
    demo.launch()
