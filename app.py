"""
Assignment 2 dashboard: potato sorting on a conveyor.

    uv run python main.py app

Loads the committed weights in weights/ and the outputs train.py wrote to artifacts/.
Nothing is trained here.
"""

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "potato_yolo11s.pt"

MODEL = YOLO(str(WEIGHTS))
NAMES = MODEL.names
METRICS = json.loads((ART / "metrics.json").read_text())
STATS = json.loads((ART / "dataset_stats.json").read_text())
UNITS = pd.read_csv(ART / "test_units.csv")
OP = METRICS["operating_point"]
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

COLOURS = {"pass": (40, 170, 60), "review": (255, 170, 0), "reject": (220, 40, 40)}
POTATO_CONF = 0.25   # a potato box must be at least this confident to count as an object


# image tab --------------------------------------------------------------------------

def _iou(a, b):
    """IoU of one xyxy box against an (N, 4) array."""
    b = np.asarray(b).reshape(-1, 4)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def sort_image(image, t_low, t_high):
    if image is None:
        return None, pd.DataFrame()
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    # Gradio gives RGB; Ultralytics reads a numpy array as BGR (as OpenCV would)
    r = MODEL.predict(np.ascontiguousarray(image[:, :, ::-1]), imgsz=640, conf=0.001, device="cpu",
                      verbose=False, max_det=300)[0]
    cls = r.boxes.cls.cpu().numpy().astype(int); conf = r.boxes.conf.cpu().numpy(); xyxy = r.boxes.xyxy.cpu().numpy()
    name = np.array([NAMES[c] for c in cls])

    # Same rule as train.build_units: objects are the potatoes; a blemish raises the score of the
    # smallest potato it sits on, a stone box overlapping a potato (IoU >= 0.3) raises that one,
    # and reject boxes on no potato become objects of their own (merged so one thing = one object).
    pot = np.where((name == "potato") & (conf >= POTATO_CONF))[0]
    boxes = xyxy[pot]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    score = np.zeros(len(pot)); reason = ["potato"] * len(pot)
    strays = []
    for i in np.argsort(-conf):
        if name[i] not in ("bad", "cut", "stone"):
            continue
        cx, cy = (xyxy[i, 0] + xyxy[i, 2]) / 2, (xyxy[i, 1] + xyxy[i, 3]) / 2
        inside = (cx >= boxes[:, 0]) & (cx <= boxes[:, 2]) & (cy >= boxes[:, 1]) & (cy <= boxes[:, 3])
        if name[i] == "stone":
            inside &= _iou(xyxy[i], boxes) >= 0.3
        cand = np.where(inside)[0]
        if len(cand):
            k = cand[np.argmax(_iou(xyxy[i], boxes[cand]))] if name[i] == "stone" else cand[np.argmin(area[cand])]
            if conf[i] > score[k]:
                score[k], reason[k] = float(conf[i]), str(name[i])
        elif all(_iou(xyxy[i], s[0][None])[0] < 0.3 for s in strays):
            strays.append((xyxy[i], float(conf[i]), str(name[i])))
    objs = list(zip(boxes, score, reason)) + [s for s in strays if s[1] >= t_low]

    canvas = image.copy()
    rows = []
    for b, s, n in objs:
        verdict = "reject" if s >= t_high else ("review" if s >= t_low else "pass")
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOURS[verdict], 2)
        if verdict != "pass":
            cv2.putText(canvas, f"{n} {s:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, COLOURS[verdict], 1, cv2.LINE_AA)
        rows.append({"object": n, "score": s, "decision": verdict})
    df = pd.DataFrame(rows, columns=["object", "score", "decision"])
    summary = (df.groupby("decision").size().reindex(["pass", "review", "reject"], fill_value=0)
               .rename("objects").reset_index())
    return canvas, summary


# operating-point tab ------------------------------------------------------------------

def operating_view(t_low, t_high, c_def, c_stone, c_fp, c_rev, pi_def, pi_stone):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    costs = dict(costing.COSTS, fn_defect=c_def, fn_stone=c_stone, fp=c_fp, review=c_rev)
    base = {"defect": pi_def / 100, "stone": pi_stone / 100}
    res = costing.expected_cost(UNITS, t_low, t_high, costs=costs, base=base)
    staff = costing.staffing(res["review_share"])
    alts = costing.alternatives(UNITS, OP["threshold"], res, costs=costs, base=base)

    cm = costing.unit_confusion(UNITS, t_low, t_high).reset_index().rename(columns={"index": "truth"})
    cm["truth"] = cm["truth"].map({"good": "good potato (incl. stray detections)",
                                   "defect": "defective potato", "stone": "stone"})

    summary = pd.DataFrame([
        ["Expected cost per 1,000 objects", f"EUR {res['cost_per_1000']:.2f}"],
        ["Share sent to a person", f"{100 * res['review_share']:.2f}%"],
        ["Reviews per day", f"{staff['reviews_per_day']:,.0f}"],
        ["People on the review belt at once", f"{staff['people_on_line_at_once']:.2f} "
         f"(available: {costing.LINE['reviewers']})"
         + ("  <- more than available" if staff["people_on_line_at_once"] > costing.LINE["reviewers"] else "")],
        ["Defects removed (incl. review)", f"{100 * res['recall_defect']:.1f}%"],
        ["Stones removed (incl. review)", f"{100 * res['recall_stone']:.1f}%"],
        ["Good potatoes thrown out", f"{100 * res['false_reject_rate']:.2f}%"],
    ], columns=["at the current thresholds", "value"])

    comp = pd.DataFrame([
        ["Do nothing (no sorting)", alts["do_nothing"]],
        [f"Manual picking table ({costing.LINE['pickers']} people, today's practice)", alts["person_checks_everything"]],
        [f"Model alone at t = {OP['threshold']:.3f}", alts["model_alone"]],
        [f"This band ({t_low:.3f} to {t_high:.3f})", res["cost_per_1000"]],
    ], columns=["policy", "EUR per 1,000 objects"]).round(2)

    sweep = costing.sweep_single(UNITS, costs=costs, base=base)
    best = float(sweep.loc[sweep.cost_per_1000.idxmin(), "threshold"])
    fig = go.Figure()
    fig.add_scatter(x=sweep.threshold, y=sweep.cost_per_1000, name="total", line=dict(width=3))
    fig.add_scatter(x=sweep.threshold, y=sweep.cost_fn_defect, name="missed defects", line=dict(dash="dot"))
    fig.add_scatter(x=sweep.threshold, y=sweep.cost_fn_stone, name="missed stones", line=dict(dash="dot"))
    fig.add_scatter(x=sweep.threshold, y=sweep.cost_fp, name="good potatoes thrown out", line=dict(dash="dot"))
    # on a log axis Plotly places shapes in log10 units
    lg = np.log10
    fig.add_vline(x=lg(OP["threshold"]), line_color="black",
                  annotation_text=f"chosen t = {OP['threshold']:.3f}", annotation_position="top right")
    if abs(best - OP["threshold"]) > 1e-9:
        fig.add_vline(x=lg(best), line_dash="dash", line_color="grey",
                      annotation_text=f"minimum under these inputs = {best:.3f}",
                      annotation_position="bottom right")
    fig.add_vrect(x0=lg(t_low), x1=lg(max(t_high, t_low * 1.02)), fillcolor="orange", opacity=0.15,
                  line_width=0, annotation_text="review band", annotation_position="top left")
    fig.update_layout(xaxis_type="log", xaxis_title="reject threshold (model alone, log scale)", yaxis_title="EUR per 1,000 objects",
                      height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return cm, summary, comp, fig


# results tab --------------------------------------------------------------------------

def per_class_table():
    rows = []
    for key, label in [("finetuned", "finetuned yolo11s"), ("coco_zero_shot", "COCO yolo11s zero-shot"),
                       ("predict_nothing", "predict nothing")]:
        m = METRICS[key]
        for cls, v in m["per_class"].items():
            n = STATS["splits"]["test"]["per_class"][cls]["instances"]
            rows.append([label, cls, n, v["precision"], v["recall"], v["ap50"], v["ap50_95"]])
        rows.append([label, "all (mean)", None, m.get("precision", 0.0), m.get("recall", 0.0),
                     m["map50"], m["map50_95"]])
    return pd.DataFrame(rows, columns=["model", "class", "test instances", "precision", "recall",
                                       "AP50", "AP50-95"]).round(3)


def detection_confusion():
    m = METRICS["finetuned"]
    labels = m["confusion_labels"]
    df = pd.DataFrame(m["confusion_matrix"], index=[f"pred {x}" for x in labels],
                      columns=[f"true {x}" for x in labels])
    return df.reset_index().rename(columns={"index": ""})


def dataset_table():
    rows = []
    for split, s in STATS["splits"].items():
        for cls, c in s["per_class"].items():
            rows.append([split, s["images"], cls, c["instances"], c["images_with"], c["rel_size_median"]])
    return pd.DataFrame(rows, columns=["split", "images", "class", "instances", "images containing it",
                                       "median box side / image side"]).round(3)


# layout ---------------------------------------------------------------------------------

with gr.Blocks(title="Potato sorter") as demo:
    gr.Markdown(
        "## Potato sorting on a conveyor\n"
        "Every object on the belt gets a reject score: the model's confidence that it carries a rot or "
        "blemish spot, or is a stone. Below the lower threshold it passes, at or above the upper one the "
        "ejector removes it, in between it goes to a person. Finetuned yolo11s on "
        f"`{METRICS['dataset']}`.")
    with gr.Row():
        t_low = gr.Slider(0.001, 0.99, value=OP["t_low"], step=0.001, label="pass below (lower threshold)")
        t_high = gr.Slider(0.001, 0.99, value=OP["t_high"], step=0.001,
                           label="reject at or above (upper threshold)")

    with gr.Tab("Sort an image"):
        with gr.Row():
            inp = gr.Image(type="numpy", label="conveyor image")
            out = gr.Image(type="numpy", label="green = pass, amber = review, red = reject")
        counts = gr.Dataframe(label="decisions on this image")
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=inp)
        inp.change(sort_image, [inp, t_low, t_high], [out, counts])
        t_low.release(sort_image, [inp, t_low, t_high], [out, counts])
        t_high.release(sort_image, [inp, t_low, t_high], [out, counts])

    with gr.Tab("Operating point and cost"):
        with gr.Accordion("Cost and base-rate assumptions (defaults are the ones argued in REPORT.md)",
                          open=False):
            with gr.Row():
                c_def = gr.Number(costing.COSTS["fn_defect"], label="EUR per missed defective potato")
                c_stone = gr.Number(costing.COSTS["fn_stone"], label="EUR per missed stone")
                c_fp = gr.Number(costing.COSTS["fp"], label="EUR per good potato thrown out")
                c_rev = gr.Number(costing.COSTS["review"], label="EUR per human review")
            with gr.Row():
                pi_def = gr.Number(100 * costing.BASE_RATES["defect"], label="defective potatoes, % of objects")
                pi_stone = gr.Number(100 * costing.BASE_RATES["stone"], label="stones, % of objects")
        with gr.Row():
            summary = gr.Dataframe()
            cm = gr.Dataframe(label="test belt: truth x decision (object counts)")
        comp = gr.Dataframe(label="against the alternatives")
        curve = gr.Plot(label="cost against threshold")
        inputs = [t_low, t_high, c_def, c_stone, c_fp, c_rev, pi_def, pi_stone]
        outputs = [cm, summary, comp, curve]
        t_low.release(operating_view, inputs, outputs)
        t_high.release(operating_view, inputs, outputs)
        for box in [c_def, c_stone, c_fp, c_rev, pi_def, pi_stone]:
            box.change(operating_view, inputs, outputs)
        demo.load(operating_view, inputs, outputs)
        if METRICS.get("sensitivity"):
            gr.Dataframe(pd.DataFrame(METRICS["sensitivity"]).round(3),
                         label="sensitivity: best thresholds when one assumption moves (from train.py)")

    with gr.Tab("Per-class results"):
        gr.Markdown("Test split. For the sorting decision the positive classes are **bad** (a blemish "
                    "on a potato) and **stone**; **potato** is the negative.")
        gr.Dataframe(per_class_table(), label="precision / recall / AP per class")
        gr.Dataframe(detection_confusion(),
                     label=f"box-level confusion matrix, finetuned model (conf {METRICS['finetuned']['confusion_conf']}, IoU 0.45)")
        if (ART / "confusion_matrix_normalized.png").exists():
            gr.Image(str(ART / "confusion_matrix_normalized.png"), label="normalised confusion matrix")
        gr.Dataframe(dataset_table(), label="dataset composition")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
