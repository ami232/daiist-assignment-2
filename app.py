"""
Assignment 2: Gradio dashboard.  Run with:  uv run python main.py app

Loads ONLY what train.py committed: weights/best.pt and artifacts/*. It never
trains, never touches runs/, never downloads the dataset.

Tabs:
  1. Predict      upload a plant photo, see detections at the current threshold
                  and the pass / review / fail verdict for the current band.
  2. Operating    move t_low and t_high; the 3-way confusion matrix, the cost
     point        per 1000 photos, the share sent to the agronomist and the
                  staffing all update together; cost-vs-threshold curve with
                  the chosen point marked.
  3. Per class    per-class precision / recall / AP50 / mAP50-95, the detection
                  confusion matrix, the baselines table, dataset composition.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from costing import COST_MODEL, evaluate_band, three_way_matrix_per_1000  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"

# ---------------------------------------------------------------------------
# Load the training stage's outputs
# ---------------------------------------------------------------------------
METRICS = json.loads((ART / "metrics.json").read_text())
SCORES_DF = pd.read_csv(ART / "test_scores.csv")
SWEEP = pd.read_csv(ART / "cost_sweep.csv")
PER_CLASS = pd.read_csv(ART / "per_class.csv")
BASELINES = pd.read_csv(ART / "baselines.csv")
ALTERNATIVES = pd.read_csv(ART / "alternatives.csv")
CONF = json.loads((ART / "confusion_matrix.json").read_text())
STATS = json.loads((ART / "dataset_stats.json").read_text())

SCORES = SCORES_DF["score"].to_numpy()
LABELS = SCORES_DF["label"].to_numpy()
OP = METRICS["operating_point"]
T_LOW0, T_HIGH0 = OP["band_chosen"]["t_low"], OP["band_chosen"]["t_high"]
T_STAR = OP["model_alone"]["threshold"]
CM = METRICS.get("cost_model", COST_MODEL)
POS_NAME = CM.get("POSITIVE_NAME", "positive")
NEG_NAME = CM.get("NEGATIVE_NAME", "negative")

NAMES = {int(k): v for k, v in METRICS["class_names"].items()}
POS_IDS = set(int(i) for i in METRICS["positive_class_ids"])
POS_CLASSES = METRICS["positive_classes"]

from ultralytics import YOLO  # noqa: E402

MODEL = YOLO(str(WEIGHTS))


# ---------------------------------------------------------------------------
# Tab 1: predictions on an uploaded image
# ---------------------------------------------------------------------------
def predict(image, t_low: float, t_high: float):
    if image is None:
        return None, "Upload a photo.", pd.DataFrame()
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    # Draw everything above the pass line so the reviewer sees what the band is about.
    # Gradio hands over RGB; ultralytics expects BGR for numpy input, and plots in BGR.
    bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    res = MODEL.predict(bgr, conf=min(t_low, 0.25), verbose=False)[0]
    annotated = np.ascontiguousarray(res.plot()[:, :, ::-1])  # BGR -> RGB

    rows, fungal_score, top_fungal = [], 0.0, None
    if res.boxes is not None and len(res.boxes):
        for c, s in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            c = int(c)
            rows.append({"class": NAMES[c], "group": "fungal" if c in POS_IDS else "non-fungal",
                         "confidence": round(s, 3)})
            if c in POS_IDS and s > fungal_score:
                fungal_score, top_fungal = s, NAMES[c]
    table = pd.DataFrame(rows).sort_values("confidence", ascending=False) if rows else pd.DataFrame(
        columns=["class", "group", "confidence"])

    what = f" (top fungal detection: {top_fungal})" if top_fungal else ""
    if fungal_score >= t_high:
        verdict = (f"🔴 FAIL — schedule the fungicide application for this bed. "
                   f"Max fungal confidence {fungal_score:.2f} ≥ t_high {t_high:.2f}{what}")
    elif fungal_score >= t_low:
        verdict = (f"🟠 REVIEW — send the photo to the agronomist. "
                   f"Max fungal confidence {fungal_score:.2f} in [{t_low:.2f}, {t_high:.2f}){what}")
    else:
        verdict = (f"🟢 PASS — no fungicide; follow the pest / bacterial protocol. "
                   f"Max fungal confidence {fungal_score:.2f} < t_low {t_low:.2f}")
    return annotated, verdict, table


# ---------------------------------------------------------------------------
# Tab 2: operating point
# ---------------------------------------------------------------------------
def cost_curve(t_low: float, t_high: float):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(SWEEP["threshold"], SWEEP["cost_per_1000"], color="#1f4e79", lw=2, label="model alone (one threshold)")
    ax.axvline(T_STAR, color="#c0392b", ls="--", label=f"cheapest single threshold t = {T_STAR:.2f}")
    ax.axvspan(t_low, t_high, color="#f39c12", alpha=0.25, label=f"current review band [{t_low:.2f}, {t_high:.2f})")
    e = evaluate_band(SCORES, LABELS, t_low, t_high, CM)
    ax.axhline(e["cost_per_1000"], color="#f39c12", lw=1, ls=":", label=f"cost with band = €{e['cost_per_1000']:.0f}")
    ax.axvspan(OP["band_chosen"]["t_low"], OP["band_chosen"]["t_high"], color="#27ae60", alpha=0.12,
               label=f"chosen band [{OP['band_chosen']['t_low']:.2f}, {OP['band_chosen']['t_high']:.2f})")
    ax.set_xlabel("confidence threshold on fungal detections")
    ax.set_ylabel("expected cost per 1000 photos (€)")
    ax.set_title("Cost against threshold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    return fig


def operating_point(t_low: float, t_high: float):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    e = evaluate_band(SCORES, LABELS, t_low, t_high, CM)
    fr = {k: e[k] for k in ("pos_pass", "pos_review", "pos_fail", "neg_pass", "neg_review", "neg_fail")}
    three = three_way_matrix_per_1000(fr, CM).reset_index().rename(columns={"index": "true class ↓ / outcome →"})

    final = pd.DataFrame(
        {"fungicide applied (FAIL)": [e["TP_per_1000"], e["FP_per_1000"]],
         "no fungicide (PASS)": [e["FN_per_1000"], e["TN_per_1000"]]},
        index=[POS_NAME, NEG_NAME],
    ).round(1).reset_index().rename(columns={"index": "after review"})

    fits = "✅ fits" if e["fits_available_people"] else "❌ exceeds"
    summary = (
        f"### € {e['cost_per_1000']:,.0f} per 1,000 photos\n"
        f"(decisions € {e['decision_cost_per_1000']:,.0f} + review € {e['review_cost_per_1000']:,.0f}; "
        f"≈ € {e['cost_per_1000'] * CM['CASES_PER_DAY'] / 1000:,.0f} per day at {CM['CASES_PER_DAY']} photos/day)\n\n"
        f"**Sent to the agronomist: {e['review_share']*100:.1f}%** → {e['review_cases_per_day']:.0f} photos/day, "
        f"{e['review_hours_per_day']:.2f} h/day, **{e['people_needed']:.2f} people** "
        f"at {CM['REVIEWER_HOURS_PER_DAY']} h each — {fits} the {CM['REVIEWERS_AVAILABLE']} available.\n\n"
        f"Pass {e['pass_share']*100:.1f}% · Review {e['review_share']*100:.1f}% · Fail {e['fail_share']*100:.1f}%  |  "
        f"final recall on fungal {e['TPR']*100:.1f}%, needless-spray rate {e['FPR']*100:.1f}%\n\n"
        f"Base rate assumed {CM['BASE_RATE']*100:.0f}% fungal (test split: {METRICS['case_level']['test_base_rate']*100:.0f}%); "
        f"C_FN €{CM['C_FN']}, C_FP €{CM['C_FP']}, C_TP €{CM['C_TP']}, review €{CM['C_REVIEW']}, "
        f"reviewer accuracy {CM['REVIEW_ACCURACY']*100:.0f}%."
    )
    return summary, three, final, cost_curve(t_low, t_high)


# ---------------------------------------------------------------------------
# Tab 3: per-class
# ---------------------------------------------------------------------------
def confusion_table() -> pd.DataFrame:
    labels = CONF["labels"]
    df = pd.DataFrame(CONF["matrix"], index=labels, columns=labels).fillna(0)
    return df.round(0).astype(int).reset_index().rename(columns={"index": "predicted ↓ / true →"})


def confusion_heatmap():
    labels = CONF["labels"]
    m = np.nan_to_num(np.array(CONF["matrix"], dtype=float))
    norm = m / np.maximum(m.sum(axis=0, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n + 2), max(5, 0.5 * n + 2)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    for i in range(n):
        for j in range(n):
            if m[i, j] > 0:
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Detection confusion matrix (column-normalised colour, raw counts)")
    fig.tight_layout()
    return fig


def dataset_table() -> pd.DataFrame:
    rows = []
    for split, d in STATS["splits"].items():
        rows.append({"split": split, "images": d["images"], "images with fungal": d["images_with_positive"],
                     "share fungal": round(d["share_images_with_positive"], 2),
                     "instances": d["instances_total"],
                     "median box area": f"{d['box_area_share_median']*100:.1f}%"})
    return pd.DataFrame(rows)


def instances_table() -> pd.DataFrame:
    inst = pd.DataFrame({s: d["instances_per_class"] for s, d in STATS["splits"].items()})
    inst.insert(0, "group", [STATS["class_group"][c] for c in inst.index])
    return inst.reset_index().rename(columns={"index": "class"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
o = METRICS["detection"]
HEADER = (
    f"# Greenhouse scouting: does this bed get the fungicide today?\n"
    f"Finetuned **{METRICS['settings']['model']}** on Roboflow Universe "
    f"`{METRICS['settings']['roboflow']}` · positive = any **fungal** class "
    f"({', '.join(POS_CLASSES)}) · negative = {', '.join(METRICS['negative_classes'])} · "
    f"test mAP50 {o['mAP50']:.3f}, mAP50-95 {o['mAP50-95']:.3f} · case-level AUC {METRICS['case_level']['auc']:.3f}\n\n"
    f"Chosen operating point: **pass below {T_LOW0:.2f}, agronomist review in [{T_LOW0:.2f}, {T_HIGH0:.2f}), "
    f"spray at ≥ {T_HIGH0:.2f}** (cheapest band the available agronomist time can actually cover)."
)

with gr.Blocks(title="Assignment 2 — plant disease scouting") as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        t_low = gr.Slider(0.0, 1.0, value=T_LOW0, step=0.01, label="t_low — below this: PASS (no fungicide)")
        t_high = gr.Slider(0.0, 1.0, value=T_HIGH0, step=0.01, label="t_high — at or above this: FAIL (spray); between: REVIEW")

    with gr.Tab("1 · Predict on a photo"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload a scouting photo")
            img_out = gr.Image(type="numpy", label="Detections (boxes drawn down to the pass line)")
        verdict = gr.Markdown()
        det_table = gr.Dataframe(label="Detections", interactive=False)
        run = gr.Button("Run model", variant="primary")
        run.click(predict, [img_in, t_low, t_high], [img_out, verdict, det_table])
        img_in.change(predict, [img_in, t_low, t_high], [img_out, verdict, det_table])

    with gr.Tab("2 · Operating point"):
        summary = gr.Markdown()
        with gr.Row():
            three_way = gr.Dataframe(label="Per 1,000 production photos: where do they go?", interactive=False)
            final_cm = gr.Dataframe(label="Final confusion matrix per 1,000 after the agronomist decides the band", interactive=False)
        curve = gr.Plot(label="Cost against threshold")
        gr.Markdown("### The design priced against the alternatives")
        gr.Dataframe(value=ALTERNATIVES.round(2), interactive=False)
        for comp in (t_low, t_high):
            comp.change(operating_point, [t_low, t_high], [summary, three_way, final_cm, curve])
        demo.load(operating_point, [t_low, t_high], [summary, three_way, final_cm, curve])

    with gr.Tab("3 · Per-class results"):
        gr.Markdown(f"Overall on the test split: precision {o['precision']:.3f}, recall {o['recall']:.3f}, "
                    f"mAP50 {o['mAP50']:.3f}, mAP50-95 {o['mAP50-95']:.3f}. "
                    f"The average hides the thin classes; read each row on its own.")
        gr.Dataframe(value=PER_CLASS.round(3), label="Per class (test split)", interactive=False)
        gr.Plot(value=confusion_heatmap())
        gr.Dataframe(value=confusion_table(), label=f"Detection confusion matrix ({CONF.get('note', '')})", interactive=False)
        gr.Markdown("### Baselines on the same test split (case level: does the photo show a fungal disease?)")
        gr.Dataframe(value=BASELINES.round(3), interactive=False)
        gr.Markdown("### Dataset composition")
        gr.Dataframe(value=dataset_table(), interactive=False)
        gr.Dataframe(value=instances_table(), label="Instances per class and split", interactive=False)

if __name__ == "__main__":
    demo.launch()
