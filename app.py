"""
Assignment 2: Gradio dashboard for solder-joint inspection.

    uv run python main.py app

Never trains anything. It loads the committed weights in weights/ and the
test-set predictions and metrics that train.py saved in artifacts/, then
recomputes routing and cost live as you move the thresholds.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
import costing

# ----------------------------------------------------------------------------- load
for needed in (config.WEIGHTS, config.ARTIFACTS / "metrics.json", config.ARTIFACTS / "test_predictions.json"):
    if not needed.exists():
        raise SystemExit(f"Missing {needed.relative_to(config.ROOT)}. Run `uv run python main.py train` "
                         f"first and commit weights/ and artifacts/.")

ART =json.loads((config.ARTIFACTS / "metrics.json").read_text())
RECORDS = json.loads((config.ARTIFACTS / "test_predictions.json").read_text())
NAMES: list[str] = ART["dataset"]["names"]
DEFECTS = set(ART["dataset"]["defect_class_ids"])
SCORES = np.array([r["score"] for r in RECORDS])
LABELS = np.array([r["label"] for r in RECORDS], dtype=bool)
BAND = ART["operating_point"]["band"]
SINGLE = ART["operating_point"]["single"]

_model = None


def model():
    """Load the committed weights once, on first use."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(str(config.WEIGHTS))
    return _model


def costs_from(base_rate, fn, fp) -> costing.Costs:
    return costing.Costs(base_rate=float(base_rate) / 100, fn=float(fn), fp=float(fp), tp=float(fp))


def route(score: float, lo: float, hi: float) -> str:
    hi = max(hi, lo)
    if score >= hi:
        return "FAIL → rework"
    if score >= lo:
        return "REVIEW → inspector"
    return "PASS → next station"


# ----------------------------------------------------------------------------- views
def inspect(image, lo, hi):
    if image is None:
        return None, "Upload an image of solder joints.", pd.DataFrame()
    res = model().predict(image, conf=0.001, iou=0.6, imgsz=config.IMGSZ, verbose=False, max_det=100)[0]
    b = res.boxes
    rows = []
    for c, s in zip(b.cls.int().tolist(), b.conf.tolist()):
        rows.append({"class": NAMES[c], "defect": c in DEFECTS, "confidence": round(s, 3)})
    df = pd.DataFrame(rows)
    defect_scores = [r["confidence"] for r in rows if r["defect"]]
    score = max(defect_scores) if defect_scores else 0.0
    # draw only boxes that matter at the current band: everything >= lo
    keep = b.conf >= min(lo, hi)
    res.boxes = b[keep]
    annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
    msg = (f"### {route(score, lo, hi)}\n"
           f"Window score (most confident defect box): **{score:.3f}**  ·  "
           f"pass < {lo:.2f} ≤ review < {max(hi, lo):.2f} ≤ fail")
    shown = df[df.confidence >= min(lo, hi)] if len(df) else df
    return annotated, msg, shown.head(20)


def operating_point(lo, hi, base_rate, fn, fp):
    c = costs_from(base_rate, fn, fp)
    counts = costing.route_counts(SCORES, LABELS, lo, hi)
    res = costing.cost_per_case(counts, c)
    staff = costing.staffing(res["review_share"])
    alt = costing.alternatives(c)
    alone = min(costing.sweep_single(SCORES, LABELS, c), key=lambda r: r["cost"])

    cm = pd.DataFrame(
        [["defective window", counts["pos"]["pass"], counts["pos"]["review"], counts["pos"]["fail"], counts["pos"]["n"]],
         ["good window", counts["neg"]["pass"], counts["neg"]["review"], counts["neg"]["fail"], counts["neg"]["n"]]],
        columns=["actual \\ routed to", "pass", "review", "fail", "total"])

    warn = "  \n⚠️ The review edge is below the pass edge; treating them as one threshold." if hi < lo else ""
    summary = (
        f"### €{res['cost'] * 1000:,.2f} expected cost per 1,000 windows{warn}\n"
        f"- **Sent to a person:** {res['review_share']:.2%} of windows → "
        f"**{staff['reviews_per_day']:,.0f} reviews/day**, {staff['review_hours_per_day']:.1f} h at "
        f"{config.REVIEW_SECONDS} s each → **{staff['reviewers_needed']:.2f} reviewers** "
        f"({config.PRODUCTIVE_HOURS_PER_REVIEWER} productive h each)\n"
        f"- **Failed to rework automatically:** {res['fail_share']:.2%} of windows\n"
        f"- **Defects escaping per 1,000 windows:** {res['escape_rate'] * 1000:.3f} "
        f"(at a base rate of {c.base_rate:.2%})\n"
        f"- Cost re-weighted to the production base rate; counts above are the raw test split "
        f"({counts['pos']['n']} defective, {counts['neg']['n']} good windows)."
    )
    per_day = staff["windows_per_day"] / 1000
    comp = pd.DataFrame([
        ["do nothing (no inspection)", alt["do_nothing"]],
        ["person checks every window", alt["human_checks_all"]],
        ["fail everything", alt["fail_everything"]],
        [f"model decides alone (best t = {alone['threshold']:.2f})", alone["cost"]],
        [f"this band ({lo:.2f} – {max(hi, lo):.2f})", res["cost"]],
    ], columns=["policy", "cost"])
    comp["€ / 1,000 windows"] = (comp.cost * 1000).round(2)
    comp["€ / day"] = (comp.cost * 1000 * per_day).round(0)
    return cm, summary, comp.drop(columns="cost")


def cost_curve(lo, hi, base_rate, fn, fp):
    c = costs_from(base_rate, fn, fp)
    sweep = pd.DataFrame(costing.sweep_single(SCORES, LABELS, c))
    best = sweep.loc[sweep.cost.idxmin()]
    fig = go.Figure()
    fig.add_vrect(x0=lo, x1=max(hi, lo), fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.add_trace(go.Scatter(x=sweep.threshold, y=sweep.cost * 1000, mode="lines",
                             name="model decides alone", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[best.threshold], y=[best.cost * 1000], mode="markers+text",
                             marker=dict(size=12, color="crimson"), name="minimum (current assumptions)",
                             text=[f"t = {best.threshold:.2f}"], textposition="top center"))
    fig.add_vline(x=SINGLE["threshold"], line_dash="dash", line_color="crimson",
                  annotation_text=f"chosen in training: {SINGLE['threshold']:.2f}")
    alt = costing.alternatives(c)
    fig.add_hline(y=alt["do_nothing"] * 1000, line_dash="dot", line_color="gray",
                  annotation_text="do nothing", annotation_position="bottom right")
    band = costing.cost_per_case(costing.route_counts(SCORES, LABELS, lo, hi), c)
    fig.add_hline(y=band["cost"] * 1000, line_dash="dot", line_color="orange",
                  annotation_text="current band", annotation_position="top right")
    fig.update_layout(xaxis_title="confidence threshold (fail if window score ≥ t)",
                      yaxis_title="expected cost per 1,000 windows (€)", height=460,
                      margin=dict(l=40, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.2))
    return fig


def per_class(conf):
    pc = ART["metrics"]["per_class"]
    test_inst = ART["dataset"]["splits"]["test"]["instances"]
    cm = costing.detection_confusion(RECORDS, len(NAMES), float(conf))
    pr = costing.per_class_pr(cm)
    rows = []
    for i, n in enumerate(NAMES):
        u = pc.get(n, {})
        rows.append({
            "class": n, "role": "positive (defect)" if i in DEFECTS else "negative (good)",
            "test instances": test_inst.get(n, 0),
            f"precision @ {conf:.2f}": round(pr[i]["precision"], 3),
            f"recall @ {conf:.2f}": round(pr[i]["recall"], 3),
            "AP50": round(u.get("ap50", float("nan")), 3),
            "mAP50-95": round(u.get("ap50_95", float("nan")), 3),
        })
    m = ART["metrics"]
    rows.append({"class": "all", "role": "", "test instances": sum(test_inst.values()),
                 f"precision @ {conf:.2f}": None, f"recall @ {conf:.2f}": None,
                 "AP50": round(m["map50"], 3), "mAP50-95": round(m["map50_95"], 3)})
    labels = NAMES + ["background"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", text=cm,
                               texttemplate="%{text}", showscale=False))
    fig.update_layout(xaxis_title="predicted", yaxis_title="true", yaxis_autorange="reversed",
                      height=430, margin=dict(l=40, r=20, t=30, b=40),
                      title=f"Box-level confusion matrix at confidence ≥ {conf:.2f}, IoU ≥ 0.5")
    return pd.DataFrame(rows), fig


def recall_by_type(lo, hi):
    rows = []
    for c in sorted(DEFECTS):
        mask = np.array([c in r["gt_cls"] for r in RECORDS])
        if not mask.any():
            continue
        s = SCORES[mask]
        rows.append({"defect type": NAMES[c], "test windows": int(mask.sum()),
                     "failed automatically": f"{(s >= max(hi, lo)).mean():.1%}",
                     "sent to review": f"{((s >= lo) & (s < max(hi, lo))).mean():.1%}",
                     "passed (missed)": f"{(s < lo).mean():.1%}"})
    return pd.DataFrame(rows)


def refresh_band(lo, hi, base_rate, fn, fp):
    """Everything that depends on the two thresholds and the assumptions."""
    cm, summary, comp = operating_point(lo, hi, base_rate, fn, fp)
    return cm, summary, comp, recall_by_type(lo, hi), cost_curve(lo, hi, base_rate, fn, fp)


def baselines_table():
    b = ART["baselines"]
    rows = []
    for key, label in (("finetuned", "yolo11s finetuned"), ("coco_zero_shot", "yolo11s COCO zero-shot"),
                       ("predict_nothing", "predict nothing")):
        r = b[key]
        rows.append({"model": label, "AP50": round(r["map50"], 3), "mAP50-95": round(r["map50_95"], 3),
                     "image AUC": round(r["image"]["auc"], 3),
                     "recall @ 1% false-fail": round(r["image"]["recall_at_fpr_0.01"], 3),
                     "best €/1,000": round(r["best_cost"] * 1000, 2)})
    return pd.DataFrame(rows)


def dataset_table():
    s = ART["dataset"]["splits"]
    rows = []
    for n in NAMES:
        med = s["test"]["box_area_share_median"][n]
        rows.append({"class": n, "train": s["train"]["instances"][n], "val": s["val"]["instances"][n],
                     "test": s["test"]["instances"][n],
                     "median box area (test)": f"{med:.1%}" if med is not None else "–"})
    for sp in ("train", "val", "test"):
        rows.append({"class": f"[{sp}] images / with defect", "train": "", "val": "", "test": "",
                     "median box area (test)": f"{s[sp]['images']} / {s[sp]['images_with_defect']} "
                                               f"({s[sp]['defect_image_share']:.0%})"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- layout
with gr.Blocks(title="Solder joint inspection") as demo:
    gr.Markdown(
        f"# Solder joint inspection: pass, review or fail\n"
        f"YOLO11s finetuned on Roboflow `{ART['source']['workspace']}/{ART['source']['project']}` "
        f"v{ART['source']['version']}. Each case is one inspection window; its score is the most "
        f"confident **defect** box. Chosen in training: pass < **{BAND['lo']:.2f}** ≤ review < "
        f"**{BAND['hi']:.2f}** ≤ fail, €{BAND['cost'] * 1000:.2f} per 1,000 windows.")
    with gr.Row():
        lo = gr.Slider(0, 1, value=BAND["lo"], step=0.01, label="Pass / review edge (lo)")
        hi = gr.Slider(0, 1, value=BAND["hi"], step=0.01, label="Review / fail edge (hi)")
    with gr.Accordion("Business assumptions (defaults from REPORT.md)", open=False):
        with gr.Row():
            base_rate = gr.Slider(0.01, 5, value=config.BASE_RATE * 100, step=0.01,
                                  label="Base rate: % of windows with a defect")
            fn = gr.Number(value=config.COST_FN, label="Cost of a missed defect (€)")
            fp = gr.Number(value=config.COST_FP, label="Cost of a rework call (€, FP and TP)")

    with gr.Tab("Operating point"):
        op_summary = gr.Markdown()
        gr.Markdown("**Test-split confusion matrix at the current thresholds** (windows)")
        op_cm = gr.Dataframe(interactive=False)
        gr.Markdown("**Compared with the alternatives** (same assumptions)")
        op_comp = gr.Dataframe(interactive=False)
        gr.Markdown("**Where each defect type goes**")
        op_types = gr.Dataframe(interactive=False)

    with gr.Tab("Cost curve"):
        curve = gr.Plot()

    with gr.Tab("Inspect an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload a window")
            img_out = gr.Image(type="numpy", label="Predictions at or above the pass edge")
        verdict = gr.Markdown()
        dets = gr.Dataframe(interactive=False, label="Detections")
        go_btn = gr.Button("Inspect", variant="primary")

    with gr.Tab("Per-class results"):
        conf = gr.Slider(0.01, 0.99, value=ART["confusion"]["conf"], step=0.01,
                         label="Box confidence for precision / recall / confusion matrix")
        pc_table = gr.Dataframe(interactive=False)
        pc_cm = gr.Plot()

    with gr.Tab("Baselines & data"):
        gr.Markdown("**Same test split for all three.** COCO has no solder classes, so its AP is 0 "
                    "by construction; its image score is its most confident box of any class.")
        gr.Dataframe(value=baselines_table(), interactive=False)
        gr.Markdown("**Dataset composition**")
        gr.Dataframe(value=dataset_table(), interactive=False)

    # One handler updates every threshold-dependent view at once. While a
    # slider is dragged, "always_last" drops the stale positions in between
    # and runs only the newest one, and the loading timer is hidden, so the
    # tables update in place instead of flickering.
    band_inputs = [lo, hi, base_rate, fn, fp]
    band_outputs = [op_cm, op_summary, op_comp, op_types, curve]
    quiet = {"show_progress": "hidden", "trigger_mode": "always_last"}
    for ctrl in band_inputs:
        ctrl.change(refresh_band, band_inputs, band_outputs, **quiet)
    # Re-running the model on the uploaded image is slower, so do it once the
    # slider is let go rather than on every step of the drag.
    for ctrl in (lo, hi):
        ctrl.release(inspect, [img_in, lo, hi], [img_out, verdict, dets], show_progress="minimal")
    go_btn.click(inspect, [img_in, lo, hi], [img_out, verdict, dets])
    img_in.change(inspect, [img_in, lo, hi], [img_out, verdict, dets])
    conf.change(per_class, conf, [pc_table, pc_cm], **quiet)

    demo.load(refresh_band, band_inputs, band_outputs)
    demo.load(per_class, conf, [pc_table, pc_cm])


if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("APP_PORT", 7860)))
