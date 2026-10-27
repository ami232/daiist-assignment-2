"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

This file never trains. It loads
  - weights/best.pt           the committed model
  - results/*.json, *.csv     what train.py saved: metrics, per-weld scores,
                              the cost assumptions and the chosen thresholds

What you can do here:
  - upload a photo of a weld and see the boxes and the decision (pass / human review / fail)
  - move the two thresholds and watch the confusion matrix, the expected cost per
    1,000 welds and the share sent to a person change together
  - see the cost-against-threshold curve with the chosen threshold marked
  - see per-class results, the baselines, and the four options priced side by side
"""

from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

import costing
from weld_data import MIN_CONF, is_positive_name

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
RESULTS = ROOT / "results"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


METRICS = load_json(RESULTS / "metrics.json")
OPERATING = load_json(RESULTS / "operating_point.json")
READY = bool(WEIGHTS.is_file() and METRICS and OPERATING
             and (RESULTS / "cases_val.csv").is_file() and (RESULTS / "cases_test.csv").is_file())

if READY:
    CONFIG = METRICS["config"]
    IMGSZ = int(CONFIG.get("imgsz", 512))
    MODEL = YOLO(str(WEIGHTS))
    CASES = {s: pd.read_csv(RESULTS / f"cases_{s}.csv") for s in ("val", "test")}
    SCORES = {s: costing.split_scores(df) for s, df in CASES.items()}
    _known = {f.name for f in fields(costing.Assumptions)}
    A0 = costing.Assumptions(**{k: v for k, v in OPERATING["assumptions"].items() if k in _known})
    CHOSEN_T = OPERATING["model_alone"]["threshold"]
    BAND = OPERATING["model_plus_review"]
else:
    A0 = costing.Assumptions()
    CHOSEN_T, BAND, IMGSZ = 0.5, {"t_low": 0.4, "t_high": 0.6}, 512


# ---------------------------------------------------------------------------
# Tab: upload an image
# ---------------------------------------------------------------------------

DETECTION_COLUMNS = ["Class", "Confidence", "Decision for this box"]


def predict_image(image, t_low, t_high):
    if not READY:
        return None, "Training results are missing. Run `uv run python main.py train` first.", \
            pd.DataFrame(columns=DETECTION_COLUMNS)
    if image is None:
        return None, "Upload a photo of a weld to get a decision.", pd.DataFrame(columns=DETECTION_COLUMNS)

    t_low, t_high = sorted((float(t_low), float(t_high)))
    result = MODEL.predict(image, conf=max(t_low, MIN_CONF), iou=0.6, imgsz=IMGSZ, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]          # Ultralytics draws BGR; Gradio wants RGB

    rows, defect_scores = [], []
    for box in result.boxes:
        name, conf = result.names[int(box.cls)], float(box.conf)
        if is_positive_name(name):
            defect_scores.append(conf)
            decision = "FAIL" if conf >= t_high else "HUMAN REVIEW"
        else:
            decision = "none (good weld)"
        rows.append([name, round(conf, 3), decision])

    top = max(defect_scores, default=0.0)
    if top >= t_high:
        verdict = f"### FAIL\nThe highest defect score is **{top:.2f}**, at or above the upper threshold {t_high:.2f}. Send for rework."
    elif top >= t_low:
        verdict = (f"### SEND FOR HUMAN REVIEW\nThe highest defect score is **{top:.2f}**, between "
                   f"{t_low:.2f} and {t_high:.2f}. An inspector decides.")
    else:
        verdict = (f"### PASS\nNo defect box at or above the lower threshold {t_low:.2f}. "
                   f"The weld carries on.")
    return annotated, verdict, pd.DataFrame(rows, columns=DETECTION_COLUMNS)


# ---------------------------------------------------------------------------
# Tabs: confusion matrix, cost, curve, options
# ---------------------------------------------------------------------------

def money(x: float) -> str:
    return f"€{x:,.2f}" if abs(x) < 100 else f"€{x:,.0f}"


def cost_curve_figure(a: costing.Assumptions, t_low: float, t_high: float):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for split, style, label in (("val", "-", "Model alone, validation (threshold chosen here)"),
                                ("test", "--", "Model alone, test")):
        sweep = costing.sweep_single(*SCORES[split], a)
        ax.plot(sweep["threshold"], sweep["cost_per_1000"], style, label=label)
    ax.axhline(costing.cost_do_nothing(a)["cost_per_1000"], color="grey", ls=":", label="Do nothing")
    ax.axhline(costing.cost_human_everything(a)["cost_per_1000"], color="green", ls=":",
               label="A person checks everything")
    ax.axvspan(t_low, t_high, color="orange", alpha=0.2, label="Human-review band (sliders)")
    ax.axvline(CHOSEN_T, color="black", lw=1.2)
    val_sweep = costing.sweep_single(*SCORES["val"], a)
    chosen_cost = float(val_sweep.iloc[(val_sweep["threshold"] - CHOSEN_T).abs().argmin()]["cost_per_1000"])
    ax.scatter([CHOSEN_T], [chosen_cost], color="black", zorder=5,
               label=f"Chosen threshold {CHOSEN_T:.2f} (report assumptions)")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Expected cost per 1,000 welds (€)")
    ax.set_title(f"Cost against threshold  |  defect rate {a.base_rate:.1%}, "
                 f"cost of a missed defect €{a.fn_cost:.0f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def dashboard(t_low, t_high, split, base_rate_pct, fn_cost):
    if not READY:
        empty = pd.DataFrame()
        return empty, empty, "Training results are missing. Run `uv run python main.py train` first.", \
            empty, None

    t_low, t_high = sorted((float(t_low), float(t_high)))
    a = replace(A0, base_rate=float(base_rate_pct) / 100.0, fn_cost=float(fn_cost))
    pos, neg, spurious = SCORES[split]
    e = costing.evaluate(pos, neg, spurious, t_low, t_high, a)
    c = e["counts"]

    confusion = pd.DataFrame(
        [["Defective weld", c["pos_pass"], c["pos_review"], c["pos_fail"]],
         ["Good weld", c["neg_pass"], c["neg_review"], c["neg_fail"]]],
        columns=["Truth \\ Decision", "Pass", "Human review", "Fail"])

    p = e["per_1000"]
    final = pd.DataFrame([[round(p["TP"], 1), round(p["FN"], 1), round(p["FP"], 1), round(p["TN"], 1)]],
                         columns=["TP: defect caught", "FN: defect ships", "FP: good weld flagged",
                                  "TN: good weld passed"],
                         index=["Per 1,000 welds, after human review"]).reset_index(names="")

    summary = (
        f"### Expected cost: **{money(e['cost_per_1000'])} per 1,000 welds** "
        f"({money(e['cost_per_day'])} per day)\n"
        f"- **Sent to a person: {e['review_share']:.1%} of welds** = {e['reviews_per_day']:,.0f} reviews a day "
        f"at {a.review_seconds:.0f} s each = **{e['reviewers']:.2f} reviewers** "
        f"({a.productive_hours_per_reviewer:.0f} productive hours each).\n"
        f"- Evaluated on the **{split}** split: {c['n_pos']} defective and {c['n_neg']} good welds. "
        f"The counts in the matrix are real welds from that split; the cost re-weights them to a "
        f"defect rate of {a.base_rate:.1%}.\n"
        f"- Not in the matrix: {c['spurious_review']} spurious boxes sent to review and "
        f"{c['spurious_fail']} failed (false alarms on the background). They are included in the cost.\n"
        + ("\n> **SMOKE TEST MODEL.** These weights were trained for one short epoch. Run the real "
           "training before trusting anything here.\n" if CONFIG.get("smoke_test") else "")
    )

    alone_t = costing.best_single(*SCORES["val"], a)["threshold"]
    alone = costing.evaluate(pos, neg, spurious, alone_t, alone_t, a)
    options = []
    for label, d in (("Do nothing", costing.cost_do_nothing(a)),
                     ("A person checks everything", costing.cost_human_everything(a)),
                     (f"Model decides alone (threshold {alone_t:.2f})", alone),
                     (f"Model + human review ({t_low:.2f} to {t_high:.2f})", e)):
        options.append([label, round(d["cost_per_1000"], 1), round(d["cost_per_day"], 0),
                        f"{d['review_share']:.1%}", round(d["reviewers"], 2)])
    options_df = pd.DataFrame(options, columns=["Option", "Cost per 1,000 welds (€)", "Cost per day (€)",
                                                "Share sent to a person", "Reviewers needed"])

    return confusion, final, summary, options_df, cost_curve_figure(a, t_low, t_high)


# ---------------------------------------------------------------------------
# Tab: per-class results
# ---------------------------------------------------------------------------

def per_class_tables():
    if not READY:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cols = {"class": "Class", "instances": "Instances", "precision": "Precision", "recall": "Recall",
            "ap50": "AP50", "map50_95": "mAP50-95"}
    tables = []
    for split in ("test", "val"):
        df = pd.DataFrame(METRICS[split]["per_class"]).rename(columns=cols)
        o = METRICS[split]["overall"]
        df.loc[len(df)] = ["All", int(df["Instances"].sum()), o["precision"], o["recall"], o["ap50"],
                           o["map50_95"]]
        tables.append(df.round(3))
    b = METRICS["baselines"]
    o = METRICS["test"]["overall"]
    base = pd.DataFrame([
        [f"Fine-tuned {CONFIG.get('model', '')}", o["precision"], o["recall"], o["ap50"], o["map50_95"]],
        ["Pretrained COCO, zero-shot", b["coco_zero_shot"]["precision"], b["coco_zero_shot"]["recall"],
         b["coco_zero_shot"]["ap50"], b["coco_zero_shot"]["map50_95"]],
        ["Trivial: predict nothing", 0.0, 0.0, 0.0, 0.0],
    ], columns=["Model (test split)", "Precision", "Recall", "AP50", "mAP50-95"]).round(3)
    return tables[0], tables[1], base


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

TEST_TABLE, VAL_TABLE, BASELINE_TABLE = per_class_tables()
EXAMPLES = sorted((RESULTS / "examples").glob("*")) if (RESULTS / "examples").is_dir() else []
CM_IMAGE = RESULTS / "test_confusion_matrix_normalized.png"

with gr.Blocks(title="Weld inspection: pass, review or fail") as demo:
    gr.Markdown(
        "# Weld inspection: pass, human review or fail\n"
        "Two thresholds split every weld into **pass** (below the lower one), **human review** "
        "(between them) and **fail** (at or above the upper one). Move them and watch the cost change."
    )
    with gr.Row():
        t_low = gr.Slider(0.05, 1.0, value=BAND["t_low"], step=0.01,
                          label="Lower threshold: a defect score below this PASSES")
        t_high = gr.Slider(0.05, 1.0, value=BAND["t_high"], step=0.01,
                           label="Upper threshold: a defect score at or above this FAILS")
    with gr.Accordion("Assumptions and data split (defaults come from REPORT.md)", open=False):
        split = gr.Radio(["test", "val"], value="test", label="Evaluate the counts on this split")
        base_rate = gr.Slider(0.1, 10.0, value=round(A0.base_rate * 100, 2), step=0.1,
                              label="How often a weld is really defective on the line (%)")
        fn_cost = gr.Slider(20, 1500, value=A0.fn_cost, step=10,
                            label="Cost of a defective weld that ships (€)")

    with gr.Tabs():
        with gr.Tab("Upload an image"):
            with gr.Row():
                image = gr.Image(type="pil", label="Weld photo", sources=["upload", "clipboard"])
                annotated = gr.Image(label="Model predictions at the current thresholds")
            verdict = gr.Markdown()
            detections = gr.Dataframe(headers=DETECTION_COLUMNS, interactive=False, label="Boxes")
            if EXAMPLES:
                gr.Examples(examples=[[str(p)] for p in EXAMPLES], inputs=[image], label="Test-set examples",
                            cache_examples=False)

        with gr.Tab("Confusion matrix and cost"):
            summary = gr.Markdown()
            confusion = gr.Dataframe(interactive=False, label="Labelled welds by decision (counts)")
            final = gr.Dataframe(interactive=False, label="Final outcomes per 1,000 production welds")
            options = gr.Dataframe(interactive=False, label="The four options priced side by side")

        with gr.Tab("Cost against threshold"):
            curve = gr.Plot(label="Cost against threshold")

        with gr.Tab("Per-class results"):
            gr.Markdown("**Test split.** Precision and recall are at the confidence that maximises F1 "
                        "(Ultralytics). AP is threshold-free.")
            gr.Dataframe(value=TEST_TABLE, interactive=False, label="Per class, test")
            gr.Dataframe(value=BASELINE_TABLE, interactive=False, label="Against the two baselines, test")
            gr.Dataframe(value=VAL_TABLE, interactive=False, label="Per class, validation")
            if CM_IMAGE.is_file():
                gr.Image(value=str(CM_IMAGE), label="Confusion matrix (test, normalised, confidence 0.001)",
                         interactive=False)

    cost_inputs = [t_low, t_high, split, base_rate, fn_cost]
    cost_outputs = [confusion, final, summary, options, curve]
    for control in cost_inputs:
        control.change(dashboard, cost_inputs, cost_outputs)
    demo.load(dashboard, cost_inputs, cost_outputs)

    predict_inputs = [image, t_low, t_high]
    predict_outputs = [annotated, verdict, detections]
    for control in predict_inputs:
        control.change(predict_image, predict_inputs, predict_outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("APP_PORT", "7860")))
