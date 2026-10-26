"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Loads weights/best.pt and the artifacts/ that train.py wrote. Trains nothing.
Every number on screen is recomputed from the stored per-image probabilities
with the same functions train.py used (evaluation.py), so moving a slider
gives exactly what the pipeline would have reported at that setting.
"""

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from evaluation import (
    OUTCOMES, Costs, band_rates, cost_per_case, labels_and_scores, outcome_counts,
    per_class_outcomes, policy_costs, records_to_arrays, review_share, staffing, status_quo_from_config,
    sweep_single, thresholds,
)


def load_json(name: str):
    path = config.ARTIFACTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


METRICS = load_json("metrics.json")
STATS = load_json("dataset_stats.json")
OPERATING = load_json("operating_point.json")
TRAINING = load_json("training.json")
RECORDS = {split: load_json(f"records_{split}.json") for split in ("val", "test")}
READY = all(x is not None for x in (METRICS, STATS, OPERATING, *RECORDS.values()))

if READY:
    NAMES = {int(k): v for k, v in METRICS["names"].items()}
    ROTTEN_IDS = METRICS["rotten_ids"]
    SCORES = {split: labels_and_scores(RECORDS[split], ROTTEN_IDS) for split in RECORDS}
    TRUE = {split: records_to_arrays(RECORDS[split])[0] for split in RECORDS}
    DEFAULT_LO, DEFAULT_HI = OPERATING["lo"], OPERATING["hi"]
else:
    NAMES, ROTTEN_IDS, SCORES, TRUE = {}, [], {}, {}
    DEFAULT_LO, DEFAULT_HI = 0.3, 0.7

MODEL = None


def get_model():
    """Load the committed weights once, on first use, so startup stays fast."""
    global MODEL
    if MODEL is None and config.WEIGHTS_PATH.exists():
        from ultralytics import YOLO
        MODEL = YOLO(str(config.WEIGHTS_PATH))
    return MODEL


def layout(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(t=50, l=10, r=10, b=10))
    return fig


# --- Operating point -----------------------------------------------------------

def decision_view(lo, hi, base_rate, cost_fn, split):
    if not READY:
        return None, "Run `uv run python main.py train` first: artifacts/ is missing.", None, None, None
    if lo > hi:
        lo, hi = hi, lo
    y, s = SCORES[split]
    costs = Costs.from_config(config, fn=float(cost_fn))
    rates = band_rates(y, s, lo, hi)

    # Raw counts on the evaluation split, and what they mean per 1,000 pieces
    # once re-weighted to the production base rate.
    passed, failed = s < lo, s >= hi
    review = ~passed & ~failed
    counts = np.array([[(m & y).sum() for m in (passed, review, failed)],
                       [(m & ~y).sum() for m in (passed, review, failed)]])
    per1000 = outcome_counts(rates, base_rate)
    expected = np.array([[per1000["pos"][o] for o in OUTCOMES], [per1000["neg"][o] for o in OUTCOMES]])
    text = [[f"{counts[i, j]} images<br>{expected[i, j]:.1f} per 1,000" for j in range(3)] for i in range(2)]
    matrix = layout(go.Figure(go.Heatmap(
        z=expected, x=list(OUTCOMES), y=["rotten", "fresh"], text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False)),
        f"Decision matrix: {split} counts, and per 1,000 at {base_rate:.1%} rotten")
    matrix.update_yaxes(autorange="reversed")

    cost = 1000 * cost_per_case(rates, base_rate, costs)
    share = review_share(rates, base_rate)
    staff = staffing(share, config.DAILY_VOLUME, config.REVIEW_SECONDS, config.PRODUCTIVE_HOURS_PER_REVIEWER)
    capacity = config.REVIEWERS_AVAILABLE * config.PRODUCTIVE_HOURS_PER_REVIEWER
    fits = "fits within" if staff["review_hours_per_day"] <= capacity else "**exceeds**"
    shipped_rotten = rates["pos"]["pass"] + rates["pos"]["review"] * (1 - costs.reviewer_sensitivity)
    binned_fresh = rates["neg"]["fail"] + rates["neg"]["review"] * (1 - costs.reviewer_specificity)
    kpis = (
        f"### €{cost:,.1f} per 1,000 pieces · {share:.1%} sent to the grading table\n"
        f"- Band: pass < **{lo:.2f}** ≤ review < **{hi:.2f}** ≤ fail "
        f"(pipeline's choice on val: {DEFAULT_LO:.2f} / {DEFAULT_HI:.2f})\n"
        f"- {staff['cases_per_day']:,.0f} reviews/day × {config.REVIEW_SECONDS}s = "
        f"{staff['review_hours_per_day']:.1f} h/day = {staff['reviewers_needed']:.2f} graders; "
        f"{fits} the {capacity:.0f} h/day of {config.REVIEWERS_AVAILABLE} graders\n"
        f"- Rotten pieces that still ship: {shipped_rotten:.1%}; fresh pieces binned: {binned_fresh:.1%}\n"
        f"- At {config.DAILY_VOLUME:,} pieces/day: €{cost * config.DAILY_VOLUME / 1000:,.0f} per day"
    )

    single_t = OPERATING["single_threshold"]
    policies = {k: 1000 * v for k, v in policy_costs(y, s, base_rate, costs, single_t, lo, hi,
                                                     status_quo_from_config(config)).items()}
    labels = {"status_quo": "Today: 2 graders glance", "do_nothing": "Ship everything",
              "human_checks_all": f"Person checks all ({config.DAILY_VOLUME * config.REVIEW_SECONDS / 3600 / config.PRODUCTIVE_HOURS_PER_REVIEWER:.1f} graders)",
              "model_alone": f"Model alone (t={single_t:.2f})", "model_plus_review": "Model + review (sliders)"}
    bars = layout(go.Figure(go.Bar(x=[labels[k] for k in policies], y=list(policies.values()),
                                   text=[f"€{v:,.0f}" for v in policies.values()], textposition="outside")),
                  "Expected cost per 1,000 pieces by policy")
    bars.update_yaxes(title="EUR per 1,000", range=[0, max(policies.values()) * 1.2 or 1])

    sweep = sweep_single(y, s, base_rate, costs, thresholds(config.THRESHOLD_STEP))
    curve = go.Figure(go.Scatter(x=[r["threshold"] for r in sweep], y=[r["cost_per_1000"] for r in sweep],
                                 mode="lines", name="model alone"))
    at_chosen = min(sweep, key=lambda r: abs(r["threshold"] - single_t))
    curve.add_trace(go.Scatter(x=[single_t], y=[at_chosen["cost_per_1000"]], mode="markers",
                               marker=dict(size=14, symbol="star"), name=f"chosen on val: {single_t:.2f}"))
    best = min(sweep, key=lambda r: r["cost_per_1000"])
    if abs(best["threshold"] - single_t) > 1e-9:
        curve.add_trace(go.Scatter(x=[best["threshold"]], y=[best["cost_per_1000"]], mode="markers",
                                   marker=dict(size=9), name=f"minimum on {split} now: {best['threshold']:.2f}"))
    curve.add_vline(x=lo, line_dash="dot", annotation_text="lo")
    curve.add_vline(x=hi, line_dash="dash", annotation_text="hi")
    curve.add_hline(y=cost, line_dash="dot", annotation_text="current band", annotation_position="bottom right")
    layout(curve, f"Cost against threshold ({split}, {base_rate:.1%} rotten, missed rotten €{cost_fn:.2f})", 380)
    curve.update_xaxes(title="threshold on P(rotten)")
    curve.update_yaxes(title="EUR per 1,000 (log scale)", type="log")

    by_class = per_class_outcomes(TRUE[split], s, lo, hi, NAMES)
    table = pd.DataFrame([{"class": k, "rotten": NAMES_TO_ROTTEN[k], "images": v["n"],
                           **{o: f"{v[o]:.1%}" for o in OUTCOMES}} for k, v in by_class.items()])
    return matrix, kpis, bars, curve, table


NAMES_TO_ROTTEN = {n: c in ROTTEN_IDS for c, n in NAMES.items()}


# --- Per-class results -----------------------------------------------------------

def model_table():
    if not READY:
        return pd.DataFrame()
    return pd.DataFrame([{"model": m, "accuracy": round(v["accuracy"], 3),
                          **{f"macro {k}": round(x, 3) for k, x in v["macro"].items()},
                          "rotten ROC AUC": round(METRICS["binary_test"][m]["roc_auc"], 3)}
                         for m, v in METRICS["classification_test"].items()])


def per_class_table():
    if not READY:
        return pd.DataFrame()
    return pd.DataFrame([{"model": m, "class": name, "support": c["support"],
                          **{k: round(c[k], 3) for k in ("precision", "recall", "f1")}}
                         for m, v in METRICS["classification_test"].items()
                         for name, c in v["per_class"].items()])


def class_confusion_figure(model: str):
    if not READY:
        return None
    matrix = np.array(METRICS["classification_test"][model]["confusion"])
    labels = [NAMES[i] for i in range(len(NAMES))]
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, text=matrix, texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    layout(fig, f"Class confusion matrix, test split: {model}", 460)
    fig.update_xaxes(title="predicted")
    fig.update_yaxes(title="true", autorange="reversed")
    return fig


def error_frame():
    path = config.ARTIFACTS_DIR / "errors_test.csv"
    return pd.read_csv(path).drop(columns=["image"]) if path.exists() else pd.DataFrame()


# --- Try an image -------------------------------------------------------------------

def predict_image(image, lo, hi):
    if image is None:
        return "Upload a photo of one piece of fruit.", None
    model = get_model()
    if model is None:
        return "No weights/best.pt found.", None
    if lo > hi:
        lo, hi = hi, lo
    probs = model.predict(image, imgsz=config.IMGSZ, verbose=False)[0].probs.data.cpu().numpy()
    by_name = {model.names[i]: float(p) for i, p in enumerate(probs)}
    rotten = sum(by_name.get(NAMES[c], 0.0) for c in ROTTEN_IDS)
    outcome = ("FAIL: discard" if rotten >= hi else "REVIEW: send to the grading table" if rotten >= lo
               else "PASS: ship")
    top = max(by_name, key=by_name.get)
    text = (f"### {outcome}\nP(rotten) = **{rotten:.3f}** against pass < {lo:.2f} ≤ review < {hi:.2f} ≤ fail. "
            f"Top class: {top} ({by_name[top]:.2f}).")
    fig = layout(go.Figure(go.Bar(x=list(by_name.values()), y=list(by_name), orientation="h")),
                 "Class probabilities", 300)
    return text, fig


# --- Dataset ---------------------------------------------------------------------------

def dataset_table():
    if not READY:
        return pd.DataFrame()
    return pd.DataFrame([{"split": k, "images": v["images"], **v["per_class"],
                          "rotten share": f"{v['rotten_share']:.1%}"} for k, v in STATS["splits"].items()])


def leakage_note() -> str:
    if not READY or "published_leakage" not in STATS:
        return ""
    pub, used = STATS["published_leakage"], STATS["leakage"]
    return (f"**{pub['distinct_source_photos']:,} distinct source photos** behind "
            f"{sum(v['images'] for v in STATS['published_splits'].values()):,} images. In the published split, "
            f"{pub['val']['share']:.1%} of val and {pub['test']['share']:.1%} of test images are copies of a training "
            f"photo, so train.py re-splits by source photo: leakage {used['val']['share']:.0%} / "
            f"{used['test']['share']:.0%}. Test holds {used['test']['distinct_source_photos']} independent photos.")


def build() -> gr.Blocks:
    with gr.Blocks(title="Fruit grading: operating point") as demo:
        gr.Markdown("# Fruit grading line\nEvery piece is shipped (pass), sent to a person at the grading "
                    "table (review), or discarded (fail), from the model's probability that it is rotten.")
        if not READY:
            gr.Markdown("**artifacts/ not found. Run `uv run python main.py train` and commit its output.**")
        info = []
        if STATS:
            src = STATS["source"]
            info.append(f"Data: {src['workspace']}/{src['project']} v{src['version']}")
        if TRAINING:
            info.append(f"{TRAINING['base_model']} finetuned {TRAINING['epochs_run']} epochs at "
                        f"{TRAINING['imgsz']}px on {TRAINING['device']}")
        gr.Markdown(" · ".join(info))

        with gr.Row():
            lo = gr.Slider(0, 1, value=DEFAULT_LO, step=0.01, label="lo: below this P(rotten), ship")
            hi = gr.Slider(0, 1, value=DEFAULT_HI, step=0.01, label="hi: at or above this, discard")
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.5, value=config.BASE_RATE, step=0.005,
                                  label="Base rate: share of incoming pieces really rotten")
            cost_fn = gr.Slider(0.25, 20, value=config.COST_FN, step=0.25, label="Cost of shipping a rotten piece (€)")
            split = gr.Radio(["test", "val"], value="test",
                             label="Evaluation split (thresholds were chosen on val; test is the reported number)")

        with gr.Tab("Operating point"):
            kpis = gr.Markdown()
            with gr.Row():
                matrix = gr.Plot()
                bars = gr.Plot()
            curve = gr.Plot()
            gr.Markdown("**Where each class goes at the current band**")
            by_class = gr.Dataframe(interactive=False)
            gr.Markdown(f"Costs: shipped rotten €{config.COST_FN:.2f} (slider), binned fresh €{config.COST_FP:.2f}, "
                        f"binned rotten €{config.COST_TP:.2f}, shipped fresh €{config.COST_TN:.2f}, review "
                        f"€{config.COST_REVIEW:.3f} ({config.REVIEW_SECONDS}s at €{config.REVIEWER_HOURLY_COST:.0f}/h); "
                        f"graders catch {config.REVIEWER_SENSITIVITY:.0%} of rotten and clear "
                        f"{config.REVIEWER_SPECIFICITY:.0%} of fresh. Thresholds were chosen on val; "
                        "test is the honest number.")

        with gr.Tab("Per-class results"):
            gr.Markdown("### Test split: finetuned model against both baselines")
            gr.Dataframe(model_table(), interactive=False)
            gr.Dataframe(per_class_table(), interactive=False)
            which = gr.Radio(["finetuned", "coco_zero_shot", "majority_class"], value="finetuned", label="Model")
            confusion = gr.Plot(class_confusion_figure("finetuned"))
            which.change(class_confusion_figure, which, confusion)
            gr.Markdown("### Where it fails: every test image misclassified, reviewed or decided wrongly "
                        "at the pipeline's band (copies of one photo share the `photo` column)")
            gr.Dataframe(error_frame(), interactive=False)

        with gr.Tab("Try an image"):
            with gr.Row():
                upload = gr.Image(type="numpy", label="Photo of one piece of fruit")
                with gr.Column():
                    verdict = gr.Markdown()
                    probs = gr.Plot()

        with gr.Tab("Dataset"):
            gr.Markdown(leakage_note())
            gr.Dataframe(dataset_table(), interactive=False)

        controls = [lo, hi, base_rate, cost_fn, split]
        outputs = [matrix, kpis, bars, curve, by_class]
        for control in controls:
            control.change(decision_view, controls, outputs)
        for control in (upload, lo, hi):
            control.change(predict_image, [upload, lo, hi], [verdict, probs])
        demo.load(decision_view, controls, outputs)
    return demo


if __name__ == "__main__":
    build().launch()
