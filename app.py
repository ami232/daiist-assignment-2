"""
Assignment 2: Gradio dashboard for grading spruce boards (premium / review / standard).

Run with:  uv run python main.py app

It never trains anything. It loads the committed model in weights/ and the results
that train.py saved in outputs/, and applies the business rule from wood_grading.py.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after choosing the non-interactive backend)
from PIL import Image, ImageDraw  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from wood_grading import (  # noqa: E402
    ACCEPTABLE_CLASSES,
    board_score,
    decide,
    expected_cost,
    outcome_shares,
    positive_ids,
    to_landscape,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
WEIGHTS_PATH = ROOT / "weights" / "wood_yolo11s.pt"
EXAMPLES_DIR = ROOT / "examples"

model = YOLO(str(WEIGHTS_PATH))
POS_IDS = positive_ids(model.names)
IMGSZ = json.loads((OUTPUT_DIR / "metrics.json").read_text())["imgsz"]

# The two evaluation splits. Test comes from a different camera than training.
SPLITS = {
    "Validation (same camera as training)": ("_val", "val_scores_finetuned.csv"),
    "Test (different camera)": ("", "test_scores_finetuned.csv"),
}


def load_split(suffix: str, scores_file: str) -> dict:
    scores = pd.read_csv(OUTPUT_DIR / scores_file)
    return {
        "costs": json.loads((OUTPUT_DIR / f"costs{suffix}.json").read_text()),
        "scores": scores,
        "pos": np.sort(scores.loc[scores["y_true"] == 1, "score"].to_numpy()),
        "neg": np.sort(scores.loc[scores["y_true"] == 0, "score"].to_numpy()),
        "sweep": pd.read_csv(OUTPUT_DIR / f"cost_sweep{suffix}.csv"),
        "per_class": pd.read_csv(OUTPUT_DIR / f"per_class{suffix}.csv"),
        "alternatives": pd.read_csv(OUTPUT_DIR / f"alternatives{suffix}.csv"),
    }


DATA = {label: load_split(*files) for label, files in SPLITS.items()}
DEFAULT_SPLIT = "Validation (same camera as training)"
DEFAULT_BAND = DATA[DEFAULT_SPLIT]["costs"]["two_thresholds_with_2_graders"]


# ---------------------------------------------------------------------------
# Thresholds -> confusion matrix, cost, review load
# ---------------------------------------------------------------------------
def cost_view(split: str, lo: float, hi: float):
    d = DATA[split]
    c = d["costs"]
    hi = max(hi, lo)  # the review band cannot be negative
    base_rate = c["base_rate"]["production"]
    costs = dict(cost_fn=c["costs"]["FN"], cost_fp=c["costs"]["FP"], cost_review=c["costs"]["review"])
    s = {k: float(v) for k, v in outcome_shares(d["pos"], d["neg"], lo, hi).items()}
    cost, review = (float(x) for x in expected_cost(d["pos"], d["neg"], lo, hi, base_rate, **costs))

    # Confusion matrix with three outcomes, per 1,000 boards on the production line.
    n_def, n_good = 1000 * base_rate, 1000 * (1 - base_rate)
    matrix = pd.DataFrame({
        "true class": ["defective board", "good board"],
        "PREMIUM (pass)": [f"{n_def * s['pos_premium']:.0f}  ← FN", f"{n_good * s['neg_premium']:.0f}  ✓"],
        "REVIEW (grader)": [f"{n_def * s['pos_review']:.0f}", f"{n_good * s['neg_review']:.0f}"],
        "STANDARD (fail)": [f"{n_def * s['pos_standard']:.0f}  ✓", f"{n_good * s['neg_standard']:.0f}  ← FP"],
    })

    line = c["line"]
    reviews_per_minute = review * line["boards_per_minute"]
    graders = reviews_per_minute * line["review_seconds"] / 60
    all_standard = (1 - base_rate) * c["costs"]["FP"] * 1000
    alone = c["single_threshold"]
    status = "✅ fits" if graders <= line["graders"] + 1e-9 else "⚠️ more graders than the mill has"
    kpis = (
        f"### €{cost * 1000:,.0f} expected cost per 1,000 boards\n"
        f"- **Sent to review:** {review:.1%} of boards → {review * line['boards_per_day']:,.0f} per day, "
        f"{graders:.1f} graders busy at {line['boards_per_minute']} boards/min "
        f"({line['graders']} available: {status})\n"
        f"- **Compared with selling everything as standard** (€{all_standard:,.0f}): "
        f"{(all_standard - cost * 1000) / all_standard:+.0%} saving\n"
        f"- **Model deciding alone at its best threshold** ({alone['threshold']:.3f}): "
        f"€{alone['cost_per_1000']:,.0f} per 1,000 boards\n"
        f"- Production base rate: {base_rate:.1%} defective · FN €{c['costs']['FN']:.2f} · "
        f"FP €{c['costs']['FP']:.2f} · review €{c['costs']['review']:.3f}"
    )
    return matrix, kpis, cost_plot(d, lo, hi, cost * 1000)


def cost_plot(d: dict, lo: float, hi: float, current_cost: float):
    sweep = d["sweep"]
    best = sweep.loc[sweep["cost_per_1000"].idxmin()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep["threshold"], sweep["cost_per_1000"], label="model deciding alone (one threshold)")
    ax.scatter([best["threshold"]], [best["cost_per_1000"]], color="#d62728", zorder=3,
               label=f"chosen threshold {best['threshold']:.3f} (€{best['cost_per_1000']:,.0f})")
    ax.axvspan(lo, max(lo, hi), color="#ff7f0e", alpha=0.2, label=f"current review band [{lo:.3f}, {hi:.3f})")
    ax.axhline(current_cost, color="#2ca02c", linestyle="--",
               label=f"current design with review (€{current_cost:,.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("confidence threshold (board fails if score ≥ threshold)")
    ax.set_ylabel("expected cost per 1,000 boards (€, log scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Upload an image -> predictions at the current thresholds
# ---------------------------------------------------------------------------
def grade_image(image: Image.Image | None, lo: float, hi: float):
    if image is None:
        return None, "Upload a board image or pick an example."
    hi = max(hi, lo)
    image = to_landscape(image.convert("RGB"))
    result = model.predict(image, imgsz=IMGSZ, conf=0.001, verbose=False)[0]
    score = board_score(result, POS_IDS)
    outcome = decide(score, lo, hi)

    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    width = max(3, image.width // 400)
    rows = []
    for (x0, y0, x1, y1), cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                           result.boxes.conf.tolist()):
        name = model.names[int(cls)]
        if int(cls) in POS_IDS and conf >= max(lo, 0.05):
            color = "red" if conf >= hi else "orange"  # red decides STANDARD, orange alone means REVIEW
        elif name in ACCEPTABLE_CLASSES and conf >= 0.25:
            color = "lime"  # acceptable defect, never makes the board fail
        else:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        draw.text((x0 + 4, max(0, y0 - 14)), f"{name.split(' ', 1)[1]} {conf:.2f}", fill=color)
        rows.append(f"| {name} | {conf:.3f} | {'disqualifying' if int(cls) in POS_IDS else 'acceptable'} |")

    text = (f"## {outcome}\n"
            f"Board score **{score:.3f}** (highest confidence among disqualifying defects). "
            f"PREMIUM if < {lo:.3f}, REVIEW if < {hi:.3f}, otherwise STANDARD.\n\n"
            "Boxes: red = disqualifying defect above the upper threshold, orange = disqualifying "
            "defect in the review band, green = acceptable defect.\n\n")
    if rows:
        text += "| class | confidence | rule |\n|---|---|---|\n" + "\n".join(rows)
    else:
        text += "No defects shown at these thresholds."
    return drawn, text


def per_class_view(split: str):
    table = DATA[split]["per_class"][["Class", "label", "Images", "Instances", "Box-P", "Box-R", "mAP50", "mAP50-95"]]
    return table.round(3), DATA[split]["alternatives"].round(2)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Spruce board grading") as demo:
    gr.Markdown(
        "# Spruce board grading: premium, review or standard\n"
        "A YOLO11s model detects wood defects. A board's score is the highest confidence among "
        "the defects that disqualify it from the premium grade (cracks, dead or missing knots, "
        "knots with cracks, pith, resin, blue stain). Two thresholds turn that score into "
        "three outcomes. Live knots and quartzity never make a board fail."
    )
    with gr.Row():
        split = gr.Radio(list(SPLITS), value=DEFAULT_SPLIT, label="Evaluation data for the metrics")
        lo = gr.Slider(0, 1, value=DEFAULT_BAND["lo"], step=0.001, label="Lower threshold (below → PREMIUM)")
        hi = gr.Slider(0, 1, value=DEFAULT_BAND["hi"], step=0.001, label="Upper threshold (at or above → STANDARD)")
    with gr.Row():
        reset_band = gr.Button("Chosen operating point (2 graders)")
        reset_alone = gr.Button("Model deciding alone (one threshold)")

    with gr.Tabs():
        with gr.Tab("Thresholds and cost"):
            kpis = gr.Markdown()
            matrix = gr.Dataframe(label="Confusion matrix per 1,000 production boards (rows: truth, columns: decision)",
                                  interactive=False)
            curve = gr.Plot(label="Cost against threshold")
        with gr.Tab("Grade a board"):
            with gr.Row():
                upload = gr.Image(type="pil", label="Board image")
                shown = gr.Image(type="pil", label="Predictions at the current thresholds")
            verdict = gr.Markdown()
            examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg"))
            if examples:
                gr.Examples(examples=examples, inputs=upload)
        with gr.Tab("Per-class results"):
            gr.Markdown("Per-class precision, recall and AP on the selected split. Marrow has only "
                        "4 test instances and Blue_Stain 26 training instances: do not read much into them.")
            per_class = gr.Dataframe(interactive=False, label="Per class")
            alternatives = gr.Dataframe(interactive=False, label="Design against the alternatives (per 1,000 boards)")
            with gr.Row():
                gr.Image(str(OUTPUT_DIR / "confusion_matrix.png"), label="Confusion matrix between classes (test)")
                gr.Image(str(OUTPUT_DIR / "training" / "results.png"), label="Training curves")

    cost_outputs = [matrix, kpis, curve]
    for control in (split, lo, hi):
        control.change(cost_view, [split, lo, hi], cost_outputs)
        control.change(grade_image, [upload, lo, hi], [shown, verdict])
    split.change(per_class_view, split, [per_class, alternatives])
    upload.change(grade_image, [upload, lo, hi], [shown, verdict])

    def chosen_band(s):
        band = DATA[s]["costs"]["two_thresholds_with_2_graders"]
        return band["lo"], band["hi"]

    def chosen_alone(s):
        t = DATA[s]["costs"]["single_threshold"]["threshold"]
        return t, t

    reset_band.click(chosen_band, split, [lo, hi])
    reset_alone.click(chosen_alone, split, [lo, hi])
    demo.load(cost_view, [split, lo, hi], cost_outputs)
    demo.load(per_class_view, split, [per_class, alternatives])


if __name__ == "__main__":
    demo.launch()
