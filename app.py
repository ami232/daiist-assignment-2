"""
Assignment 2: Gradio dashboard for the skin lesion triage model.

Run with:  uv run python main.py app   (after main.py train)

Only LOADS the committed weights in weights/ and the files train.py saved in
artifacts/. Nothing is trained here.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO

from preprocess import remove_frame

ART = Path("artifacts")
cfg = json.load(open(ART / "config.json"))
model = YOLO(cfg["model_file"])  # the model train.py kept (lower validation cost)
test = pd.read_csv(ART / "predictions_test.csv")
sweep = pd.read_csv(ART / "cost_sweep_test.csv")
per_class = pd.read_csv(ART / "per_class_test.csv")
confusion = pd.read_csv(ART / "confusion_test.csv", index_col=0)
alternatives = pd.read_csv(ART / "alternatives_test.csv")
models = pd.read_csv(ART / "model_comparison.csv")
staffing = pd.read_csv(ART / "staffing_options_test.csv")

POSITIVE = cfg["positive_classes"]
NAMES = {"AK": "Actinic keratosis (pre-cancer)", "BCC": "Basal cell carcinoma", "MEL": "Melanoma",
         "SCC": "Squamous cell carcinoma", "NV": "Nevus (mole)", "BKL": "Benign keratosis",
         "DF": "Dermatofibroma", "VASC": "Vascular lesion"}


# ---------------------------------------------------------------------------
# Same cost logic as train.py
# ---------------------------------------------------------------------------
def case_weights(df):
    n_pos, n_neg = df["positive"].sum(), (1 - df["positive"]).sum()
    return np.where(df["positive"] == 1, cfg["base_rate"] / n_pos, (1 - cfg["base_rate"]) / n_neg)


def outcomes(scores, low, high):
    return np.where(scores < low, "pass", np.where(scores >= high, "fail", "review"))


def cost_per_1000(df, low, high):
    out = outcomes(df["score"].values, low, high)
    miss = df["true"].map(cfg["miss_cost"]).fillna(0).values
    pos = df["positive"].values
    # Review: the dermatologist catches (1 - review_miss) of cancers and misses the rest
    review = cfg["review"] + pos * ((1 - cfg["review_miss"]) * cfg["appointment"] + cfg["review_miss"] * miss)
    cost = np.select([out == "pass", out == "review"], [miss, review], cfg["appointment"])
    w = case_weights(df)
    return 1000 * np.sum(w * cost), np.sum(w * (out == "review")), out


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def classify(image, low, high):
    if image is None:
        return None, "Upload a photo of a skin lesion."
    if cfg["remove_frame"]:
        image = remove_frame(image)  # same preprocessing the model was trained with
    r = model.predict(image, imgsz=224, verbose=False)[0]
    probs = {model.names[k]: float(p) for k, p in enumerate(r.probs.data.cpu().numpy())}
    score = sum(probs[c] for c in POSITIVE)
    decision = outcomes(np.array([score]), low, high)[0]
    label = {"pass": "🟢 PASS: probably benign, no appointment",
             "review": "🟡 REVIEW: a dermatologist checks the photo",
             "fail": "🔴 FAIL: priority in-person appointment"}[decision]
    top = dict(sorted(((f"{k} – {NAMES[k]}", v) for k, v in probs.items()), key=lambda x: -x[1]))
    return top, (f"### {label}\nMalignant score (AK+BCC+MEL+SCC): **{score:.2f}** · "
                 f"thresholds: pass < {low:.2f} ≤ review < {high:.2f} ≤ fail\n\n"
                 f"*University exercise, not a medical device.*")


def band_view(low, high):
    high = max(high, low)
    cost, review_share, out = cost_per_1000(test, low, high)
    truth = np.where(test["true"] == "MEL", "Melanoma",
                     np.where(test["positive"] == 1, "Other malignant / pre-malignant", "Benign"))
    cm = pd.crosstab(pd.Categorical(truth, ["Benign", "Other malignant / pre-malignant", "Melanoma"]),
                     pd.Categorical(out, ["pass", "review", "fail"]), dropna=False)
    fig = px.imshow(cm.values, x=["Pass (home)", "Review (dermatologist)", "Fail (appointment)"],
                    y=list(cm.index), text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    title=f"Test images by true class and outcome (pass < {low:.2f} ≤ review < {high:.2f} ≤ fail)")
    reviews_day = review_share * cfg["daily_photos"]
    alone = sweep["cost_per_1000"].min()
    md = (f"### Expected cost: **€{cost:,.0f} per 1,000 photos**\n"
          f"- Sent to a dermatologist for review: **{review_share:.1%}** → "
          f"**{reviews_day:.0f} photos/day** of {cfg['daily_photos']} → "
          f"**{reviews_day / cfg['reviews_per_derm_per_day']:.2f} dermatologists** "
          f"(at {cfg['reviews_per_derm_per_day']} reviews each per day; the service has {cfg['dermatologists']})\n"
          f"- Melanomas sent home: **{int(cm.loc['Melanoma', 'pass'])} of {int(cm.loc['Melanoma'].sum())}** test melanomas\n"
          f"- Chosen on validation: pass < **{cfg['low']}**, fail ≥ **{cfg['high']}**\n\n"
          f"*Costs at a {cfg['base_rate']:.0%} base rate of malignant photos. Appointment €{cfg['appointment']}, "
          f"review €{cfg['review']} (dermatologist misses {cfg['review_miss']:.0%} of cancers), missed cancer: " +
          ", ".join(f"{k} €{v:,}" for k, v in cfg['miss_cost'].items()) + ".*")
    return fig, md


def sweep_plot():
    fig = go.Figure()
    fig.add_scatter(x=sweep["low"], y=sweep["cost_per_1000"], mode="lines", name="model alone (one threshold)")
    t = cfg["model_alone_threshold"]
    fig.add_vline(x=t, line_dash="dash", annotation_text=f"chosen threshold {t}")
    for name, v in [("do nothing", "do nothing"), ("dermatologist checks all", "dermatologist checks")]:
        y = alternatives.loc[alternatives["strategy"].str.startswith(v), "cost_per_1000"].iloc[0]
        fig.add_hline(y=y, line_dash="dot", annotation_text=f"{name}: €{y:,.0f}")
    ours = alternatives.iloc[-1]["cost_per_1000"]
    fig.add_hline(y=ours, line_color="green", annotation_text=f"model + review band: €{ours:,.0f}")
    fig.update_layout(title="Expected cost per 1,000 photos vs threshold (test set, re-weighted)",
                      xaxis_title="threshold on malignant score", yaxis_title="€ per 1,000 photos",
                      yaxis_range=[0, sweep["cost_per_1000"].quantile(0.9) * 1.1])
    return fig


def confusion_plot():
    return px.imshow(confusion.values, x=list(confusion.columns), y=list(confusion.index), text_auto=True,
                     color_continuous_scale="Blues", labels={"x": "predicted", "y": "true"},
                     title="8-class confusion matrix (test set, top-1 prediction)")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Skin lesion triage") as demo:
    gr.Markdown("# Skin lesion triage: pass, dermatologist review, or appointment\n"
                f"YOLO11n-cls finetuned on ISIC lesion images (model: `{cfg['model_file']}`). Loaded from `weights/` and `artifacts/`; "
                "nothing is trained here. **University exercise, not a medical device.**")
    with gr.Row():
        low = gr.Slider(0, 1, value=cfg["low"], step=0.01, label="Pass below (low threshold)")
        high = gr.Slider(0, 1, value=cfg["high"], step=0.01, label="Appointment from (high threshold)")

    with gr.Tab("Upload a photo"):
        with gr.Row():
            img = gr.Image(type="pil", label="Lesion photo")
            with gr.Column():
                decision = gr.Markdown()
                probs = gr.Label(num_top_classes=8, label="Class probabilities")
        for c in (img, low, high):
            c.change(classify, [img, low, high], [probs, decision])

    with gr.Tab("Thresholds & cost"):
        band_md = gr.Markdown()
        band_cm = gr.Plot()
        gr.Plot(sweep_plot())
        gr.Dataframe(alternatives, label="Strategies compared on the test set")
        gr.Dataframe(staffing, label="Best thresholds for each number of reviewing dermatologists (test set)")
        for c in (low, high):
            c.change(band_view, [low, high], [band_cm, band_md])
        demo.load(band_view, [low, high], [band_cm, band_md])

    with gr.Tab("Per-class results"):
        gr.Dataframe(per_class, label="Per-class precision and recall (test set)")
        gr.Plot(confusion_plot())
        gr.JSON(cfg["baselines"], label="Accuracy vs baselines (test set)")
        gr.Dataframe(models, label="Original images vs dermoscope frame removed (thresholds chosen on validation)")

if __name__ == "__main__":
    demo.launch()
