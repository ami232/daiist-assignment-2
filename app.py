"""
Assignment 2: Gradio dashboard. Run with

    uv run python main.py app

Builds everything from weights/best.pt and results/, which train.py produced.
It never trains. Costs are recomputed from results/test_scores.csv and the
figures in config.py, so changing a cost there only needs an app restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import config as cfg
import decision

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
RESULTS = ROOT / "results"

# ----------------------------------------------------------------------------
# Load what training produced
# ----------------------------------------------------------------------------
MODEL = YOLO(str(WEIGHTS))
NAMES = [MODEL.names[i] for i in sorted(MODEL.names)]
POS_IDS = [i for i, n in enumerate(NAMES) if any(k in n.lower() for k in cfg.POSITIVE_CLASS_KEYWORDS)]


def _read_json(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


METRICS = _read_json("metrics.json")
STATS = _read_json("dataset_stats.json")
SCORES_PATH = RESULTS / "test_scores.csv"
HAVE_RESULTS = SCORES_PATH.exists() and bool(METRICS)

if HAVE_RESULTS:
    _scores = pd.read_csv(SCORES_PATH)
    SCORES = _scores["score_finetuned"].to_numpy(float)
    LABELS = _scores["is_bad"].to_numpy(int)
    POINT = decision.operating_point(SCORES, LABELS)
    SWEEP = decision.sweep(SCORES, LABELS)
else:
    SCORES, LABELS = np.array([]), np.array([], int)
    POINT = {"threshold": 0.5, "band": [0.3, 0.7], "chosen_by": "default (no results/ found)"}
    SWEEP = pd.DataFrame(columns=["threshold", "cost_per_1000"])

MISSING = (
    "No results found. Run `uv run python main.py train` and commit `results/` "
    "together with `weights/`."
)


def decide(score: float, low: float, high: float) -> str:
    low, high = min(low, high), max(low, high)
    if score < low:
        return "Pass"
    if score >= high:
        return "Fail"
    return "Send to an inspector"


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
def inspect(image, box_threshold, low, high):
    if image is None:
        return None, "Upload a weld photo to see the model's verdict.", pd.DataFrame()
    result = MODEL.predict(image, conf=0.001, imgsz=cfg.IMG_SIZE, verbose=False)[0]
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    defect = np.isin(cls, POS_IDS)
    score = float(conf[defect].max()) if defect.any() else 0.0

    shown = result[conf >= box_threshold]
    drawn = shown.plot()[..., ::-1]  # BGR -> RGB
    boxes = pd.DataFrame(
        {"class": [NAMES[c] for c in cls[conf >= box_threshold]],
         "confidence": np.round(conf[conf >= box_threshold], 3)}
    )
    verdict = (
        f"### {decide(score, low, high)}\n"
        f"Highest defect confidence in this image: **{score:.2f}**. "
        f"Pass below {min(low, high):.2f}, fail at or above {max(low, high):.2f}, "
        f"anything in between goes to an inspector."
    )
    return drawn, verdict, boxes


def band_view(low, high):
    if not HAVE_RESULTS:
        empty = pd.DataFrame()
        return MISSING, empty, empty, empty, cost_figure(low, high)
    low, high = min(low, high), max(low, high)
    counts, expected = decision.outcome_table(SCORES, LABELS, low, high)
    cost = 1000 * decision.cost_per_case(SCORES, LABELS, low, high)
    share = decision.review_share(SCORES, LABELS, low, high)
    staff = decision.staffing(share)
    summary = (
        f"**Expected cost: €{cost:,.2f} per 1,000 welds** "
        f"(€{cost * cfg.DAILY_VOLUME / 1000:,.2f} per day at {cfg.DAILY_VOLUME:,} welds).  \n"
        f"**Sent to an inspector: {share:.1%}** = {staff['checks_per_day']:,.0f} checks a day, "
        f"{staff['hours_per_day']:,.1f} hours, **{staff['inspectors_needed']:.2f} inspectors** "
        f"at {cfg.INSPECTOR_HOURS_PER_DAY:g} productive hours each."
    )
    alternatives = decision.alternatives(SCORES, LABELS, POINT["threshold"], (low, high))
    return summary, expected, counts, alternatives, cost_figure(low, high)


def cost_figure(low, high):
    fig = go.Figure()
    if HAVE_RESULTS:
        fig.add_trace(go.Scatter(
            x=SWEEP["threshold"], y=SWEEP["cost_per_1000"], mode="lines",
            name="Model alone", line={"width": 3},
        ))
        t = POINT["threshold"]
        t_cost = float(SWEEP.loc[np.isclose(SWEEP["threshold"], t), "cost_per_1000"].iloc[0])
        fig.add_trace(go.Scatter(
            x=[t], y=[t_cost], mode="markers+text", name="Chosen threshold",
            marker={"size": 14, "symbol": "diamond"},
            text=[f"  {t:.2f}: €{t_cost:,.0f}"], textposition="middle right",
        ))
        nothing = 1000 * decision.cost_per_case(SCORES, LABELS, 1.01, 1.01)
        everyone = 1000 * decision.cost_per_case(SCORES, LABELS, -0.01, 1.01)
        fig.add_hline(y=nothing, line_dash="dot", annotation_text="Do nothing")
        fig.add_hline(y=everyone, line_dash="dot", annotation_text="Inspector checks every weld")
        fig.add_vrect(x0=min(low, high), x1=max(low, high), opacity=0.12, line_width=0,
                      annotation_text="Review band", annotation_position="top left")
    fig.update_layout(
        xaxis_title="Confidence threshold (fail at or above)",
        yaxis_title="Expected cost per 1,000 welds (€)",
        margin={"l": 10, "r": 10, "t": 30, "b": 10}, height=430, legend={"orientation": "h"},
    )
    return fig


def performance_tables():
    det = METRICS.get("detection", {})
    per_class = pd.DataFrame(det.get("per_class", []))
    overall = pd.DataFrame([det.get("overall", {})])
    baselines = pd.DataFrame(METRICS.get("baselines_detection", []))
    image_level = pd.DataFrame(METRICS.get("baselines_image_level", []))
    cm = det.get("confusion_matrix")
    if cm:
        matrix = pd.DataFrame(cm["matrix"], columns=[f"true: {n}" for n in cm["labels"]])
        matrix.insert(0, "predicted", cm["labels"])
    else:
        matrix = pd.DataFrame()
    return per_class, overall, baselines, image_level, matrix


def dataset_tables():
    if not STATS:
        return pd.DataFrame(), pd.DataFrame()
    composition, sizes = [], []
    for split, s in STATS["splits"].items():
        composition.append({"split": split, "images": s["images"],
                            "images with a bad weld or defect": s["images_with_a_bad_weld_or_defect"],
                            "images with only good welds": s["images_with_only_good_welds_or_nothing"],
                            **s["instances_per_class"]})
        for name, median in s["median_box_area_share"].items():
            sizes.append({"split": split, "class": name,
                          "median box area (% of image)": None if median is None else round(100 * median, 2),
                          "10th percentile": None if s["p10_box_area_share"][name] is None
                          else round(100 * s["p10_box_area_share"][name], 2),
                          "90th percentile": None if s["p90_box_area_share"][name] is None
                          else round(100 * s["p90_box_area_share"][name], 2)})
    return pd.DataFrame(composition), pd.DataFrame(sizes)


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
low0, high0 = POINT["band"]
per_class, overall, baselines, image_level, matrix = performance_tables()
det = METRICS.get("detection", {})
composition, sizes = dataset_tables()

with gr.Blocks(title="Weld inspection: pass, review or fail") as demo:
    gr.Markdown(
        "# Weld inspection: pass, review or fail\n"
        f"Each photo gets a score: the model's highest confidence that it contains a "
        f"{' or '.join(NAMES[i] for i in POS_IDS)}. Two edges split scores into pass, "
        f"inspector review and fail. Costs use a production bad-weld rate of "
        f"{cfg.BASE_RATE:.1%}, not the test set's mix. Starting values come from the "
        f"{POINT['chosen_by']}."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=low0, step=0.01, label="Pass below this score")
        high = gr.Slider(0, 1, value=high0, step=0.01, label="Fail at or above this score")

    with gr.Tab("Costs and review load"):
        summary = gr.Markdown()
        with gr.Row():
            expected = gr.Dataframe(label="Expected per 1,000 production welds", interactive=False)
            counts = gr.Dataframe(label="Test-set counts", interactive=False)
        alternatives = gr.Dataframe(label="Against the alternatives (per 1,000 welds)", interactive=False)

    with gr.Tab("Cost against threshold"):
        plot = gr.Plot()
        gr.Markdown(
            "The line is the model deciding alone at each threshold. The diamond is the "
            "chosen threshold. The shaded strip is the review band set by the sliders."
        )

    with gr.Tab("Inspect a photo"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Weld photo")
            image_out = gr.Image(label="Boxes at or above the display threshold")
        box_threshold = gr.Slider(0.01, 1, value=max(POINT["threshold"], 0.05), step=0.01,
                                  label="Show boxes at or above this confidence")
        verdict = gr.Markdown()
        boxes = gr.Dataframe(label="Detections shown", interactive=False)

    with gr.Tab("Per-class results"):
        if not HAVE_RESULTS:
            gr.Markdown(MISSING)
        gr.Dataframe(per_class, label="Finetuned model, per class (test split)", interactive=False)
        gr.Dataframe(overall, label="Finetuned model, overall (test split)", interactive=False)
        gr.Dataframe(matrix, label=f"Detection confusion matrix, counts at confidence "
                     f"{det.get('confusion_matrix', {}).get('confidence', '?')}", interactive=False)
        if (RESULTS / "confusion_matrix_normalized.png").exists():
            gr.Image(str(RESULTS / "confusion_matrix_normalized.png"),
                     label="Normalised confusion matrix", interactive=False)
        gr.Dataframe(baselines, label="Baselines: detection metrics", interactive=False)
        gr.Dataframe(image_level, label="Baselines: pass/fail decision per image", interactive=False)

    with gr.Tab("Dataset"):
        gr.Dataframe(composition, label="Images and instances per split", interactive=False)
        gr.Dataframe(sizes, label="Box size relative to the image", interactive=False)

    with gr.Tab("Assumptions"):
        gr.Dataframe(pd.DataFrame(cfg.as_table(), columns=["Assumption", "Value"]),
                     interactive=False)
        gr.Markdown("Edit these in `config.py`, then restart the app.")

    band_outputs = [summary, expected, counts, alternatives, plot]
    gr.on([low.change, high.change, demo.load], band_view, [low, high], band_outputs)
    gr.on([image_in.change, box_threshold.change, low.change, high.change],
          inspect, [image_in, box_threshold, low, high], [image_out, verdict, boxes])


if __name__ == "__main__":
    demo.launch()
