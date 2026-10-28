"""
Assignment 2 dashboard. Run with `uv run python main.py app`.

Loads weights/best.pt exactly once at import. Everything else is read from
outputs/ (written by the training pipeline and the analysis scripts) and
src/config.py. Nothing is trained or downloaded here.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.costing import (  # noqa: E402
    UNITS,
    cost_per_1000,
    image_level_rates,
    load_scores,
    outcome_table,
    review_load,
    shares,
    unit_costs,
)
from src.data import DATASET_ID  # noqa: E402
from src.scoring import decide, image_score  # noqa: E402

WEIGHTS = ROOT / "weights" / "best.pt"
OUT = ROOT / "outputs"
PORT = 7860
SCORE_CONF = 0.001          # the score keeps every candidate, as in src.evaluate
IMGSZ = 640
COLORS = {"Box": (60, 160, 200), "Package": (200, 120, 200),
          "Box_broken": (230, 60, 40), "Open_package": (240, 160, 40)}

# ----------------------------------------------------------- load once ------

def _load_model():
    from ultralytics import YOLO

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"{WEIGHTS} is missing. The app serves committed weights only.")
    return YOLO(str(WEIGHTS))


MODEL = _load_model()
MODEL_LOADS = 1
SCORES = load_scores(OUT / "test_scores.csv")
METRICS = json.loads((OUT / "test_metrics_finetuned.json").read_text(encoding="utf-8"))
BASELINES = json.loads((OUT / "baselines.json").read_text(encoding="utf-8"))
SWEEP_SUMMARY = json.loads((OUT / "sweep_summary.json").read_text(encoding="utf-8"))
REVIEW_SUMMARY = json.loads((OUT / "review_summary.json").read_text(encoding="utf-8"))
SWEEP = pd.read_csv(OUT / "sweep.csv")
OPTIONS = pd.read_csv(OUT / "options.csv")
with (OUT / "failures" / "failures.csv").open(newline="", encoding="utf-8") as _f:
    FAILURES = list(csv.DictReader(_f))

BASE_RATES = sorted(set([config.BASE_RATE] + list(config.BASE_RATE_SWEEP)))
FALSE_CLAIMS = sorted(set([config.COST_FALSE_CLAIM_DOP] + list(config.COST_FALSE_CLAIM_SWEEP)))


# ----------------------------------------------------------- helpers --------

def fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def clamp_pair(t_low: float, t_high: float) -> tuple[float, float]:
    return (t_low, max(t_low, t_high))


def dop_eur(x: float) -> str:
    return f"{x:,.0f} DOP ({x / config.DOP_PER_EUR:,.2f} EUR)"


# ----------------------------------------------------------- tab 1 ----------

def inspect_image(image: np.ndarray | None, draw_conf: float, t_low: float, t_high: float, false_claim: float):
    if image is None:
        return None, "Upload an image."
    t_low, t_high = clamp_pair(t_low, t_high)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    res = MODEL.predict(bgr, conf=SCORE_CONF, imgsz=IMGSZ, verbose=False)[0]
    score = image_score(res)
    decision = decide(score, t_low, t_high)
    uc = unit_costs(false_claim)[decision]

    out = image.copy()
    shown = 0
    for cls, conf, xyxy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
        if conf < draw_conf:
            continue
        name = res.names[int(cls)]
        x0, y0, x1, y1 = (int(v) for v in xyxy)
        color = COLORS.get(name, (0, 255, 0))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        text = f"{name} {conf:.2f}"
        cv2.putText(out, text, (x0 + 3, max(y0 + 18, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x0 + 3, max(y0 + 18, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        shown += 1
    total = len(res.boxes)
    md = (
        f"**Image score: {score:.4f}** (max confidence over {', '.join(config.POSITIVE_CLASSES)} "
        f"detections at conf {SCORE_CONF}; 0 if none)\n\n"
        f"**Decision at t_low = {t_low:.2f}, t_high = {t_high:.2f}: `{decision.upper()}`**\n\n"
        f"Cost this decision carries: **if the unit is intact {uc['intact']:,.0f} DOP**, "
        f"**if damaged {uc['damaged']:,.0f} DOP** (false claim {false_claim:g} DOP).\n\n"
        f"Boxes drawn: {shown} of {total} candidates at conf >= {draw_conf:.2f}."
    )
    return out, md


# ----------------------------------------------------------- tab 2 ----------

def recompute(t_low: float, t_high: float, base_rate: float, false_claim: float):
    lo, hi = clamp_pair(float(t_low), float(t_high))
    base_rate, false_claim = float(base_rate), float(false_claim)
    sh = shares(SCORES, lo, hi)
    c = cost_per_1000(sh, base_rate, false_claim)
    table = pd.DataFrame(outcome_table(sh, base_rate))
    table = table.rename(columns={
        "truth": "truth", "pass_test_n": "pass (test n)", "pass_per_1000": "pass / 1,000",
        "review_test_n": "review (test n)", "review_per_1000": "review / 1,000",
        "fail_test_n": "fail (test n)", "fail_per_1000": "fail / 1,000"})
    load = pd.DataFrame(review_load(sh, base_rate))[
        ["container", "units", "units_to_review", "units_failed", "units_passed",
         "expected_missed_damaged", "person_hours", "days_to_clear", "fits_claim_window"]]
    load["fits_claim_window"] = load["fits_claim_window"].map({1: "yes", 0: "no"})
    md = (
        f"**Expected cost per {UNITS:,} units: {dop_eur(c['cost_dop_per_1000'])}** "
        f"at base rate {base_rate:.1%}, false claim {false_claim:g} DOP.\n\n"
        f"Sent to review: **{c['exp_review_per_1000'] / UNITS:.1%}** "
        f"({c['exp_review_per_1000']:.1f} per 1,000). Failed on the photo: "
        f"**{c['exp_fail_per_1000'] / UNITS:.1%}** ({c['exp_fail_per_1000']:.1f} per 1,000, "
        f"of which {c['exp_false_claims_per_1000']:.1f} false claims). "
        f"Missed damaged units: {c['exp_fn_per_1000']:.2f} per 1,000.\n\n"
        f"Load with {config.STAFF} staff at {config.HOURS_PER_DAY} h/day, "
        f"{config.MIN_PER_REVIEW} min per review, {config.MIN_PER_FAIL_CONFIRM} min per fail confirmation, "
        f"claim window {config.CLAIM_WINDOW_DAYS} days."
    )
    return gr.update(value=hi), md, table, load


# ----------------------------------------------------------- tab 3 ----------

def cost_curve(t_low: float, t_high: float, base_rate: float):
    lo, hi = clamp_pair(float(t_low), float(t_high))
    base_rate = float(base_rate)
    d = SWEEP[(SWEEP["subset"] == "all") & (np.isclose(SWEEP["base_rate"], base_rate))].sort_values("threshold")
    band = next((m for m in SWEEP_SUMMARY["min_by_base_rate"] if np.isclose(m["base_rate"], base_rate)), None)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d["threshold"], d["cost_dop_per_1000"], color="#4C78A8", lw=2, label="single threshold: flag if score >= t")
    if band:
        ax.axvspan(band["band_low"], band["band_high"], color="#F58518", alpha=0.18,
                   label=f"within one test positive of min ({band['band_low']}..{band['band_high']})")
        ax.plot([band["threshold_at_min"]], [band["min_cost_dop_per_1000"]], "o", color="#E45756",
                label=f"min {band['min_cost_dop_per_1000']:,.0f} DOP at t={band['threshold_at_min']}")
    ax.axvline(lo, color="#54A24B", ls="--", lw=1.5, label=f"t_low = {lo:.2f} (pass below)")
    ax.axvline(hi, color="#B279A2", ls="--", lw=1.5, label=f"t_high = {hi:.2f} (fail at or above)")
    ax.axhline(UNITS * config.COST_REVIEW_DOP, color="grey", ls=":", lw=1,
               label=f"person checks everything ({UNITS * config.COST_REVIEW_DOP:,})")
    ymax = max(UNITS * config.COST_REVIEW_DOP * 1.25, d["cost_dop_per_1000"].min() * 3)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("threshold")
    ax.set_ylabel(f"expected cost per {UNITS:,} units (DOP)")
    ax.set_title(f"Single-threshold cost curve at base rate {base_rate:.1%} (sweep.csv: FN x {config.COST_FN_DOP} + FP x {config.COST_FP_DOP})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    return fig_to_array(fig)


def options_table() -> pd.DataFrame:
    cols = [c for c in OPTIONS.columns if c != "note"]
    df = OPTIONS[cols].copy()
    df["note"] = OPTIONS["note"]
    return df


# ----------------------------------------------------------- tab 4 ----------

def per_class_table() -> pd.DataFrame:
    rows = []
    det = METRICS["all"]["detection"]
    for cls, m in det["per_class"].items():
        rows.append({"model": "fine-tuned yolo11n", "class": cls, "precision": m["precision"],
                     "recall": m["recall"], "AP50": m["ap50"], "AP50-95": m["ap50_95"]})
    rows.append({"model": "fine-tuned yolo11n", "class": "all (mAP)", "precision": det["mean_precision"],
                 "recall": det["mean_recall"], "AP50": det["map50"], "AP50-95": det["map50_95"]})
    for cls in det["per_class"]:
        rows.append({"model": "COCO yolo11n zero-shot", "class": cls, "precision": "n/a", "recall": "n/a",
                     "AP50": 0.0, "AP50-95": 0.0})
    rows.append({"model": "COCO yolo11n zero-shot", "class": "all (mAP)", "precision": "n/a", "recall": "n/a",
                 "AP50": 0.0, "AP50-95": 0.0})
    for name in ("trivial: predict nothing", "trivial: flag everything"):
        rows.append({"model": name, "class": "all", "precision": "n/a", "recall": "n/a", "AP50": 0.0, "AP50-95": 0.0})
    return pd.DataFrame(rows)


def model_comparison() -> pd.DataFrame:
    t = config.CHOSEN_T_HIGH
    ft = image_level_rates(SCORES, t)
    coco = BASELINES["coco_zero_shot"]
    rows = [
        {"model": "fine-tuned yolo11n", "mAP50": METRICS["all"]["detection"]["map50"],
         "mAP50-95": METRICS["all"]["detection"]["map50_95"],
         f"TPR @ t_high={t}": round(ft["tpr"], 4), f"FPR @ t_high={t}": round(ft["fpr"], 4),
         "note": f"{METRICS['all']['image_level']['images']} test images"},
        {"model": "COCO yolo11n zero-shot", "mAP50": 0.0, "mAP50-95": 0.0,
         f"TPR @ t_high={t}": 0.0, f"FPR @ t_high={t}": 0.0,
         "note": f"no package class, every image scores 0; it saw {coco['detections_total']} objects, "
                 f"top: {', '.join(f'{k} {v}' for k, v in list(coco['class_histogram'].items())[:4])}"},
        {"model": "trivial: predict nothing", "mAP50": 0.0, "mAP50-95": 0.0,
         f"TPR @ t_high={t}": 0.0, f"FPR @ t_high={t}": 0.0, "note": "every image passes"},
        {"model": "trivial: flag everything", "mAP50": 0.0, "mAP50-95": 0.0,
         f"TPR @ t_high={t}": 1.0, f"FPR @ t_high={t}": 1.0, "note": "every image fails"},
    ]
    return pd.DataFrame(rows)


def image_level_table(t_low: float, t_high: float) -> pd.DataFrame:
    lo, hi = clamp_pair(float(t_low), float(t_high))
    rows = []
    for label, t in ((f"fail only: score >= t_high ({hi:.2f})", hi),
                     (f"review or fail: score >= t_low ({lo:.2f})", lo)):
        r = image_level_rates(SCORES, t)
        rows.append({"rule": label, "TPR": round(r["tpr"], 4), "FPR": round(r["fpr"], 4),
                     **{f"FPR {c}": (round(v, 4) if v is not None else "n/a") for c, v in r["fpr_by_class"].items()},
                     "TP": r["tp"], "FN": r["fn"], "FP": r["fp"], "TN": r["tn"]})
    return pd.DataFrame(rows)


def failure_gallery() -> list[tuple[str, str]]:
    items = []
    for f in FAILURES:
        path = OUT / "failures" / f["filename"]
        if path.is_file():
            items.append((str(path), f"{f['kind']} | GT {f['ground_truth']} | score {f['score']} | "
                                     f"predicted {f['top3_predictions']}"))
    return items


# ----------------------------------------------------------- layout ---------

def header_md() -> str:
    conf = REVIEW_SUMMARY.get("chosen", {})
    return (
        f"# Package intake: pass / review / fail\n"
        f"Dataset `{DATASET_ID}` (Roboflow Universe, YOLOv11 export). "
        f"**Positive (compromised)** = an image with at least one `{'` or `'.join(config.POSITIVE_CLASSES)}` box; "
        f"**negative (intact)** = only `{'`/`'.join(config.NEGATIVE_CLASSES)}` boxes. "
        f"Image score = max confidence over positive-class detections.\n\n"
        f"**Owner-confirmed figures** (src/config.py): miss {config.COST_FN_DOP} DOP, false alarm unpack "
        f"{config.COST_FP_DOP}, review {config.COST_REVIEW_DOP}, fail confirmation {config.COST_FAIL_CONFIRM_DOP}, "
        f"wage {config.WAGE_DOP_PER_MIN} DOP/min, base rate {config.BASE_RATE:.1%}, containers "
        f"{config.CONTAINER_SIZES}, {config.STAFF} staff x {config.HOURS_PER_DAY} h, claim window "
        f"{config.CLAIM_WINDOW_DAYS} days, {config.DOP_PER_EUR} DOP/EUR (2026-10-23). "
        f"**To confirm**: false-claim cost {config.COST_FALSE_CLAIM_DOP} DOP (placeholder, see sweep "
        f"{config.COST_FALSE_CLAIM_SWEEP}), and the chosen thresholds t_low = {config.CHOSEN_T_LOW}, "
        f"t_high = {config.CHOSEN_T_HIGH}, each resting on one test image. "
        f"Model: committed `weights/best.pt`; all tables read from `outputs/`."
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="Assignment 2: package intake") as demo:
        gr.Markdown(header_md())
        with gr.Tab("Inspect"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="numpy", label="Upload a package photo")
                    draw_conf = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Draw boxes at confidence >=")
                    insp_btn = gr.Button("Run")
                with gr.Column():
                    img_out = gr.Image(type="numpy", label="Detections", interactive=False)
                    insp_md = gr.Markdown("Upload an image, then Run. Thresholds come from the Thresholds tab.")
        with gr.Tab("Thresholds"):
            with gr.Row():
                t_low = gr.Slider(0.0, 1.0, value=config.CHOSEN_T_LOW, step=0.01, label="t_low (pass below)")
                t_high = gr.Slider(0.0, 1.0, value=config.CHOSEN_T_HIGH, step=0.01, label="t_high (fail at or above)")
            with gr.Row():
                base_rate = gr.Dropdown(choices=BASE_RATES, value=config.BASE_RATE, label="Base rate (share of units damaged)")
                false_claim = gr.Dropdown(choices=FALSE_CLAIMS, value=config.COST_FALSE_CLAIM_DOP, label="False-claim cost (DOP)")
            thr_md = gr.Markdown()
            outcome_df = gr.Dataframe(label="Outcomes: truth x decision (test counts and per 1,000 units)", interactive=False)
            load_df = gr.Dataframe(label="Review load per container", interactive=False)
        with gr.Tab("Cost curve"):
            curve_img = gr.Image(type="numpy", label="Single-threshold cost vs threshold (sweep.csv)", interactive=False)
            gr.Image(value=str(OUT / "review_grid.png"), label=f"Two-threshold grid at false claim {config.COST_FALSE_CLAIM_DOP} DOP (review_grid.png)", interactive=False)
            gr.Dataframe(value=options_table(), label="Options (options.csv)", interactive=False, wrap=True)
        with gr.Tab("Per class"):
            gr.Dataframe(value=per_class_table(), label="Per-class precision / recall / AP50 / AP50-95 on the test split (mAP at conf 0.001)", interactive=False)
            gr.Dataframe(value=model_comparison(), label="Fine-tuned vs COCO zero-shot vs trivial", interactive=False, wrap=True)
            gr.Image(value=str(OUT / "confusion_matrix_all.png"), label="4-class confusion matrix, test split, conf 0.25", interactive=False)
            il_df = gr.Dataframe(label="Image-level TPR / FPR at the current thresholds, FPR split by ground-truth class", interactive=False)
            gr.Gallery(value=failure_gallery(), label="Worst test cases (outputs/failures/)", columns=5, height=360, object_fit="contain")

        controls = [t_low, t_high, base_rate, false_claim]
        insp_btn.click(inspect_image, inputs=[img_in, draw_conf, t_low, t_high, false_claim], outputs=[img_out, insp_md])
        draw_conf.release(inspect_image, inputs=[img_in, draw_conf, t_low, t_high, false_claim], outputs=[img_out, insp_md])
        for comp in controls:
            comp.change(recompute, inputs=controls, outputs=[t_high, thr_md, outcome_df, load_df])
            comp.change(cost_curve, inputs=[t_low, t_high, base_rate], outputs=[curve_img])
            comp.change(image_level_table, inputs=[t_low, t_high], outputs=[il_df])
        demo.load(recompute, inputs=controls, outputs=[t_high, thr_md, outcome_df, load_df])
        demo.load(cost_curve, inputs=[t_low, t_high, base_rate], outputs=[curve_img])
        demo.load(image_level_table, inputs=[t_low, t_high], outputs=[il_df])
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch(server_port=PORT, share=False, inbrowser=False)
