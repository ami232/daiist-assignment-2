"""Dashboard: uv run python main.py app

Loads committed weights/best.pt and outputs/*. Never trains anything.
"""
from __future__ import annotations

import json

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import config as C
import costing

df_test = pd.read_csv(C.OUT / "test_scores.csv")
_vp = C.OUT / "valid_scores.csv"
df_valid = pd.read_csv(_vp) if _vp.exists() else None
SOURCES = ["Test only"] + (["Valid + Test (pooled)", "Valid only"] if df_valid is not None else [])


def get_df(source):
    if source == "Valid only" and df_valid is not None:
        return df_valid
    if source == "Valid + Test (pooled)" and df_valid is not None:
        return pd.concat([df_valid, df_test], ignore_index=True)
    return df_test


metrics = json.loads((C.OUT / "metrics.json").read_text())
model = YOLO(str(C.WEIGHTS))
POS_IDS = {i for i, n in model.names.items() if n in C.POSITIVE_CLASSES}


# ------------------------------------------------------------- predictions
def decide(score: float, lo: float, hi: float) -> str:
    lo, hi = min(lo, hi), max(lo, hi)
    return "PASS" if score < lo else ("FAIL (reject)" if score >= hi else "HUMAN REVIEW")


def predict(image, lo, hi):
    if image is None:
        return None, "Upload an image."
    # Gradio hands us RGB; Ultralytics treats numpy arrays as BGR (OpenCV convention).
    # Without this flip the model sees red/blue swapped (a yellow banana looks blue).
    bgr = image[..., ::-1]
    r = model.predict(bgr, conf=0.001, verbose=False)[0]
    score = 0.0
    if r.boxes is not None and len(r.boxes):
        for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if int(cls) in POS_IDS:
                score = max(score, float(conf))
        r.boxes = r.boxes[r.boxes.conf >= min(lo, hi)]  # show boxes at the current threshold
    shown = r.plot()[..., ::-1]
    return shown, f"**Decision: {decide(score, lo, hi)}**  \nDiseased score: {score:.3f}"


# -------------------------------------------------------------- cost views
def dashboard(lo, hi, base_rate, source):
    lo, hi = min(lo, hi), max(lo, hi)
    df = get_df(source)
    r = costing.evaluate(df, lo, hi, base_rate)
    st = costing.staffing(r["review_share"])

    banner = "⚠️ **Placeholder costs in config.py**\n\n" if C.PLACEHOLDER_COSTS else ""
    summary = (
        f"{banner}"
        f"**Expected cost per 1,000 cases: €{r['cost_per_1000']:,.0f}**  \n"
        f"Share sent for human review: **{r['review_share']:.1%}**  \n"
        f"Review load: {st['reviews_per_day']:,.0f} cases/day ≈ {st['person_hours']:.1f} h "
        f"≈ **{st['people_needed']:.2f} people** "
        f"({'within' if st['feasible'] else 'OVER'} the {C.REVIEWERS_AVAILABLE} available)  \n"
        f"After review, per 1,000: TP {r['tp']:.1f} · FN {r['fn']:.1f} · FP {r['fp']:.1f} · TN {r['tn']:.1f}  \n"
        f"Based on {len(df)} scored images ({int(df.label.sum())} diseased, {int((1 - df.label).sum())} healthy) — {source}"
    )

    # confusion matrix: actual x decision, per 1,000 cases at the assumed base rate
    fig1, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.imshow(r["grid"], cmap="Blues")
    ax.set_xticks(range(3), ["Pass", "Review", "Fail"])
    ax.set_yticks(range(2), ["Actually healthy", "Actually diseased"])
    ax.set_xlabel("Decision")
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{r['grid'][i, j]:.1f}", ha="center", va="center")
    ax.set_title(f"Per 1,000 cases (base rate {base_rate:.1%})")
    fig1.tight_layout()

    # cost curve (single threshold, no review band) with chosen + current thresholds
    ts, costs, t_best = costing.sweep(df, base_rate)
    fig2, ax2 = plt.subplots(figsize=(6, 3.4))
    ax2.plot(ts, costs, label="cost (no review band)")
    ax2.axvline(C.CHOSEN_SINGLE, color="green", ls="-", label=f"chosen = {C.CHOSEN_SINGLE:.2f}")
    ax2.axvline(t_best, color="grey", ls=":", label=f"sweep minimum (plateau middle) = {t_best:.2f}")
    ax2.axvspan(lo, hi, color="orange", alpha=0.2, label="current review band")
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("€ per 1,000 cases")
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    strat = costing.strategies(df, base_rate).round(2)
    return summary, fig1, fig2, strat


def uncertainty(base_rate, source):
    ths, mins = costing.bootstrap(get_df(source), base_rate)
    if len(ths) == 0:
        return "Not enough data."
    return (f"**Bootstrap ({len(ths)} resamples, {source})**  \n"
            f"Best single threshold: median {np.median(ths):.2f}, 90% interval "
            f"{np.percentile(ths, 5):.2f} to {np.percentile(ths, 95):.2f}  \n"
            f"Minimum cost per 1,000: median €{np.median(mins):,.0f}, 90% interval "
            f"€{np.percentile(mins, 5):,.0f} to €{np.percentile(mins, 95):,.0f}")


def per_class_table():
    t = pd.DataFrame(metrics["per_class"]).round(3)
    o = metrics["overall_detection"]
    note = (f"Overall: AP50 {o['map50']:.3f} · mAP50-95 {o['map50_95']:.3f}. "
            f"Positive class: {', '.join(metrics['positive_classes'])}. "
            "Precision/recall are at the max-F1 confidence.")
    b = pd.DataFrame(metrics["image_level_at_0.5"]).T.round(3).reset_index().rename(columns={"index": "model"})
    return note, t, b


# ---------------------------------------------------------------------- UI
def build():
    with gr.Blocks(title="Produce sorting: cost-based thresholds") as demo:
        gr.Markdown("# Diseased produce sorting\nscore < low → pass · between → human review · ≥ high → fail")
        with gr.Row():
            lo = gr.Slider(0, 1, value=C.CHOSEN_LO, step=0.01, label="Low threshold (below = pass)")
            hi = gr.Slider(0, 1, value=C.CHOSEN_HI, step=0.01, label="High threshold (above = fail)")
            base = gr.Slider(0.001, 0.5, value=C.BASE_RATE, step=0.001, label="Assumed base rate (diseased share)")
        source = gr.Radio(SOURCES, value=SOURCES[-2] if len(SOURCES) > 1 else SOURCES[0],
                          label="Scored images used for the cost curve")

        with gr.Tab("Try an image"):
            with gr.Row():
                inp = gr.Image(type="numpy", label="Upload")
                out_img = gr.Image(label="Predictions")
            out_txt = gr.Markdown()
            gr.Button("Predict").click(predict, [inp, lo, hi], [out_img, out_txt])

        with gr.Tab("Cost & review"):
            summary = gr.Markdown()
            with gr.Row():
                cm_plot = gr.Plot(label="Confusion matrix")
                curve_plot = gr.Plot(label="Cost vs threshold")
            strat = gr.Dataframe(label="Strategies compared (per 1,000 cases)")
            for s in (lo, hi, base, source):
                s.change(dashboard, [lo, hi, base, source], [summary, cm_plot, curve_plot, strat])
            demo.load(dashboard, [lo, hi, base, source], [summary, cm_plot, curve_plot, strat])
            boot_btn = gr.Button("How much does the best threshold depend on luck? (bootstrap)")
            boot_txt = gr.Markdown()
            boot_btn.click(uncertainty, [base, source], boot_txt)

        with gr.Tab("Per-class results"):
            note, t, b = per_class_table()
            gr.Markdown(note)
            gr.Dataframe(t, label="Per-class (test split)")
            gr.Dataframe(b, label="Image-level at threshold 0.5: finetuned vs baselines")
            cm = C.OUT / "confusion_matrix_normalized.png"
            if cm.exists():
                gr.Image(str(cm), label="YOLO confusion matrix (test split)")
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
