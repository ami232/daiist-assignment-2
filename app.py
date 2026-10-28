"""
Assignment 2: Gradio dashboard. Run with:  uv run python main.py app

Loads weights/best.pt and weights/results.json, both produced by train.py.
Never trains anything.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from train import alternatives, cost_per_car, rates, review_share, reviewers_needed

ROOT = Path(__file__).resolve().parent
model = YOLO(str(ROOT / "weights" / "best.pt"))
R = json.loads((ROOT / "weights" / "results.json").read_text())

C = R["costs"]
SCORES = np.array(R["test_cars"]["scores"])
LABELS = np.array(R["test_cars"]["labels"])
BAND = R["decision"]["best_band_with_staff"] or R["decision"]["best_band_unconstrained"]
DEFAULT_LOW, DEFAULT_HIGH = BAND["low"], BAND["high"]


# ------------------------------------------------------------ single image ---
def predict(image, low, high):
    if image is None:
        return None, "Upload a photo of a car."
    res = model.predict(image, conf=0.001, imgsz=R["training"]["imgsz"], verbose=False)[0]
    conf = res.boxes.conf.cpu().numpy()
    is_dmg = np.isin(res.boxes.cls.cpu().numpy().astype(int), R["damage_class_ids"])
    score = float(conf[is_dmg].max()) if is_dmg.any() else 0.0
    shown = res[conf >= low] if len(conf) else res
    plotted = shown.plot()[:, :, ::-1]  # BGR -> RGB
    if score >= high:
        verdict = "Fail: damage recorded, excluded from cover"
    elif score >= low:
        verdict = "Review: send to an underwriter"
    else:
        verdict = "Pass: issue the policy"
    return plotted, f"**{verdict}**  \nCar score (highest box confidence): {score:.2f}"


# ------------------------------------------------------- decision & cost ---
def outcome_table(low, high, base_rate):
    """Outcomes per 1,000 cars at the chosen base rate, plus raw test counts."""
    r = rates(SCORES, LABELS, low, high)
    rows = []
    for actual, key, weight in (("Damaged", "pos", base_rate), ("Clean", "neg", 1 - base_rate)):
        n_test = int((LABELS == (key == "pos")).sum())
        rows.append({"Actual": actual,
                     **{f"{col} (per 1,000)": round(1000 * weight * r[key][k], 1)
                        for col, k in (("Fail", "fail"), ("Review", "review"), ("Pass", "pass"))},
                     **{f"{col} (test count)": int(round(n_test * r[key][k]))
                        for col, k in (("Fail", "fail"), ("Review", "review"), ("Pass", "pass"))}})
    return pd.DataFrame(rows)


def cost_curve(low, high, base_rate):
    c = {**C, "base_rate": base_rate}
    ts = np.array(R["decision"]["thresholds"])
    curve = [1000 * cost_per_car(SCORES, LABELS, t, t, c) for t in ts]
    best = ts[int(np.argmin(curve))]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ts, curve, color="#1f4e79", lw=2, label="Model decides alone")
    ax.axvline(best, color="#1f4e79", ls="--", lw=1)
    ax.annotate(f"cheapest: {best:.2f}", (best, min(curve)), textcoords="offset points",
                xytext=(6, 10), color="#1f4e79")
    ax.axvspan(low, high, color="#e8a33d", alpha=0.25, label="Review band (current)")
    alts = alternatives(c)
    ax.axhline(1000 * alts["do_nothing"], color="#999", ls=":", label="Do nothing")
    ax.axhline(1000 * alts["human_checks_all"], color="#555", ls="-.", label="Human checks all")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("€ per 1,000 cars")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def update(low, high, base_rate):
    if low > high:
        low, high = high, low
    c = {**C, "base_rate": base_rate}
    band = 1000 * cost_per_car(SCORES, LABELS, low, high, c)
    share = review_share(SCORES, LABELS, low, high, base_rate)
    staff = reviewers_needed(share, c)
    alts = alternatives(c)
    ts = np.array(R["decision"]["thresholds"])
    alone = min(1000 * cost_per_car(SCORES, LABELS, t, t, c) for t in ts)
    summary = pd.DataFrame([
        {"Option": "Your band (pass / review / fail)", "€ per 1,000 cars": round(band, 2)},
        {"Option": "Model decides alone (best threshold)", "€ per 1,000 cars": round(alone, 2)},
        {"Option": "Do nothing", "€ per 1,000 cars": round(1000 * alts["do_nothing"], 2)},
        {"Option": "Human checks every car", "€ per 1,000 cars": round(1000 * alts["human_checks_all"], 2)},
    ])
    fits = "fits" if staff <= C["reviewers_available"] else "does NOT fit"
    text = (f"**{100 * share:.1f}%** of cars go to review: about "
            f"**{share * C['cars_per_day']:.0f} a day**, needing **{staff:.1f} reviewers** "
            f"({fits} the {C['reviewers_available']} available).")
    return outcome_table(low, high, base_rate), summary, text, cost_curve(low, high, base_rate)


# ----------------------------------------------------- model evaluation ---
def per_class_table():
    df = pd.DataFrame(R["detection"]["per_class"]).round(3)
    return df.rename(columns={"class": "Class", "precision": "Precision", "recall": "Recall",
                              "ap50": "AP50", "ap50_95": "AP50-95"})


def baseline_table():
    rows = []
    for name, label in (("finetuned", "Finetuned YOLO"), ("coco_zero_shot", "COCO zero-shot"),
                        ("predict_nothing", "Predict nothing")):
        b = R["baselines"][name]
        rows.append({"Model": label, "mAP50 (damage classes)": round(b["map50"], 3),
                     "Car-level AUC": None if b["auc"] is None else round(b["auc"], 3),
                     "Best € per 1,000 cars": round(1000 * b["cost_per_car"], 2)})
    return pd.DataFrame(rows)


def confusion_plot():
    m = R["detection"]["confusion_matrix"]
    if m is None:
        return None
    m, names = np.array(m), R["detection"]["confusion_labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black", fontsize=8)
    ax.set_title("Box-level confusion matrix, test split")
    fig.tight_layout()
    plt.close(fig)
    return fig


def dataset_table():
    rows = []
    for split, s in R["dataset"].items():
        rows.append({"Split": split, "Images": s["images"], "Damaged cars": s["damaged_cars"], "Clean cars": s["clean_cars"],
                     **s["instances"], "Median box % of image": s["median_box_area_pct"]})
    return pd.DataFrame(rows)


with gr.Blocks(title="Car damage pre-inspection") as demo:
    gr.Markdown("# Car damage pre-inspection\nDecides whether a car has pre-existing "
                "damage before a comprehensive policy is issued.")
    with gr.Row():
        low = gr.Slider(0, 1, value=DEFAULT_LOW, step=0.01, label="Review from (below this: pass)")
        high = gr.Slider(0, 1, value=DEFAULT_HIGH, step=0.01, label="Fail from (above this: damage recorded)")
        base = gr.Slider(0.01, 0.6, value=C["base_rate"], step=0.01, label="Base rate of damaged cars")

    with gr.Tab("Decision and cost"):
        staff_md = gr.Markdown()
        outcomes = gr.Dataframe(label="Outcomes")
        options = gr.Dataframe(label="Expected cost against the alternatives")
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Try a photo"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Car photo")
            img_out = gr.Image(label="Detected damage")
        verdict = gr.Markdown()
        gr.Button("Check this car").click(predict, [img_in, low, high], [img_out, verdict])

    with gr.Tab("Model evaluation"):
        d = R["detection"]
        ref = R["dataset_ref"]
        gr.Markdown(f"Dataset: [{ref['workspace']}/{ref['project']} v{ref['version']}]({ref['url']}). "
                    f"Test split: mAP50 **{d['map50']:.3f}**, mAP50-95 **{d['map50_95']:.3f}**. "
                    f"Positive class for the decision: *car has damage* (any damage box).")
        gr.Dataframe(per_class_table(), label="Per-class results (test split)")
        gr.Dataframe(baseline_table(), label="Against the baselines")
        gr.Plot(confusion_plot())
        gr.Dataframe(dataset_table(), label="Dataset composition")

    inputs, outputs = [low, high, base], [outcomes, options, staff_md, curve]
    for s in inputs:
        s.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)

if __name__ == "__main__":
    demo.launch()
