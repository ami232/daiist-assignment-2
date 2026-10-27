"""
Assignment 2: Gradio dashboard for the mask gate at the cosmetics plant.

This app NEVER trains. It only reads what the training stage produced and
committed:

    weights/best.pt                    the finetuned yolo11s detector
    outputs/config.json                cost assumptions, base rate, chosen thresholds
    outputs/metrics.json               box-level results, confusion matrix, baselines
    outputs/valid_scores.csv           one NO-Mask score per validation image
    outputs/test_scores.csv            one NO-Mask score per test image

The gate decision for an image uses its NO-Mask score (the highest NO-Mask
confidence found in the image):

    score <  lower edge                 -> PASS    (door opens)
    lower edge <= score < upper edge    -> REVIEW  (a supervisor looks at the photo)
    score >= upper edge                 -> FAIL    (door stays shut)

The confusion matrix, the cost per 1,000 cases and the review load are
recomputed live from the saved per-image scores, with the same cost maths as
train.py, using the constants saved in outputs/config.json.
"""
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
WEIGHTS_DIR = ROOT / "weights"
APP_PORT = int(os.environ.get("APP_PORT", "7860"))

POSITIVE = "NO-Mask"
COLORS = {"NO-Mask": (214, 39, 40), "Mask": (44, 160, 44)}


# ----------------------------------------------------------------- saved outputs
def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `uv run python main.py train` and commit "
            "weights/ and outputs/."
        )
    return path


CFG = json.loads(_require(OUT_DIR / "config.json").read_text())
METRICS = json.loads(_require(OUT_DIR / "metrics.json").read_text())
SCORES = {
    "test": pd.read_csv(_require(OUT_DIR / "test_scores.csv")),
    "validation": pd.read_csv(_require(OUT_DIR / "valid_scores.csv")),
}
ARRAYS = {k: (df["y_true"].to_numpy(), df["score_nomask"].to_numpy()) for k, df in SCORES.items()}

DEFAULT_LO = float(CFG["review_low"])
DEFAULT_HI = float(CFG["review_high"])
CHOSEN_SINGLE = float(CFG["best_single_threshold"])


def find_weights() -> Path:
    best = WEIGHTS_DIR / "best.pt"
    if best.exists():
        return best
    found = sorted(WEIGHTS_DIR.rglob("*.pt")) if WEIGHTS_DIR.is_dir() else []
    if not found:
        raise FileNotFoundError("No .pt file in weights/. Commit your trained weights there.")
    return found[0]


_MODEL = None


def get_model():
    """Load the committed weights once (inference only)."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(find_weights()))
    return _MODEL


# ----------------------------------------------------------------- cost maths
# Same definitions as train.py: image fails if score >= hi, is reviewed if
# lo <= score < hi, passes if score < lo. Costs are reweighted to the assumed base rate.
def outcome_rates(y, s, lo, hi) -> dict:
    pos = y == 1
    fail, review, passed = s >= hi, (s >= lo) & (s < hi), s < lo

    def share(mask, sel):
        return float(mask[sel].mean()) if sel.any() else 0.0

    return {"tpr": share(fail, pos), "rev_pos": share(review, pos), "fnr": share(passed, pos),
            "fpr": share(fail, ~pos), "rev_neg": share(review, ~pos), "tnr": share(passed, ~pos)}


def cost_per_case(r, base_rate, c_fn) -> float:
    reviewed_pos = CFG["C_REVIEW"] + CFG["HUMAN_MISS"] * c_fn + (1 - CFG["HUMAN_MISS"]) * CFG["C_TP"]
    pos = r["tpr"] * CFG["C_TP"] + r["rev_pos"] * reviewed_pos + r["fnr"] * c_fn
    neg = r["fpr"] * CFG["C_FP"] + r["rev_neg"] * CFG["C_REVIEW"] + r["tnr"] * CFG["C_TN"]
    return base_rate * pos + (1 - base_rate) * neg


def review_share(r, base_rate) -> float:
    return base_rate * r["rev_pos"] + (1 - base_rate) * r["rev_neg"]


def fix_band(lo, hi):
    """Keep the band valid: the upper edge can never sit below the lower edge."""
    lo, hi = float(lo), float(hi)
    return lo, max(lo, hi)


def design_row(name, y, s, lo, hi, base_rate, c_fn) -> dict:
    r = outcome_rates(y, s, lo, hi)
    share = review_share(r, base_rate)
    per_day = share * CFG["DAILY_CASES"]
    minutes = per_day * CFG["REVIEW_SECONDS"] / 60
    return {
        "Design": name,
        "Cost per 1,000 (EUR)": round(1000 * cost_per_case(r, base_rate, c_fn), 1),
        "Sent to a person": f"{100 * share:.1f}%",
        "Reviews/day": round(per_day, 1),
        "Minutes/day": round(minutes, 1),
        "People needed": round(minutes / CFG["PRODUCTIVE_MINUTES_PER_PERSON"], 2),
    }


# ----------------------------------------------------------------- live views
def refresh_ops(split, lo, hi, base_rate, c_fn):
    """Confusion matrix, cost per 1,000, review share, designs table and cost curve."""
    lo, hi = fix_band(lo, hi)
    y, s = ARRAYS[split]
    r = outcome_rates(y, s, lo, hi)
    cost = 1000 * cost_per_case(r, base_rate, c_fn)
    share = review_share(r, base_rate)
    per_day = share * CFG["DAILY_CASES"]
    minutes = per_day * CFG["REVIEW_SECONDS"] / 60
    people = minutes / CFG["PRODUCTIVE_MINUTES_PER_PERSON"]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())

    summary = (
        f"### At pass below **{lo:.2f}**, fail from **{hi:.2f}** ({split} split)\n\n"
        f"- **Expected cost per 1,000 cases: EUR {cost:,.0f}** "
        f"(base rate {base_rate:.1%}, a missed no-mask costs EUR {c_fn:,.0f})\n"
        f"- **Share sent for human review: {share:.1%}** of cases, "
        f"about {per_day:,.0f} a day, {minutes:,.0f} minutes of checking, "
        f"{people:.2f} people\n"
        f"- Of the {n_pos} images with a no-mask face, "
        f"{r['tpr']:.1%} fail, {r['rev_pos']:.1%} go to review and **{r['fnr']:.1%} pass (missed)**\n"
        f"- Of the {n_neg} images without one, {r['tnr']:.1%} pass, "
        f"{r['rev_neg']:.1%} go to review and **{r['fpr']:.1%} fail (false alarm)**\n\n"
        f"*The confusion matrix counts the {split} images. The cost and the review "
        f"share are reweighted to the assumed base rate, because the dataset has far "
        f"more no-mask images than the real gate would.*"
    )

    counts = np.array([
        [int(((s < lo) & (y == 1)).sum()), int(((s >= lo) & (s < hi) & (y == 1)).sum()), int(((s >= hi) & (y == 1)).sum())],
        [int(((s < lo) & (y == 0)).sum()), int(((s >= lo) & (s < hi) & (y == 0)).sum()), int(((s >= hi) & (y == 0)).sum())],
    ])
    fig = Figure(figsize=(5.6, 3.0))
    ax = fig.subplots()
    ratio = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if ratio[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), ["PASS", "REVIEW", "FAIL"])
    ax.set_yticks(range(2), [f"no-mask face\npresent (n={n_pos})", f"no no-mask\nface (n={n_neg})"])
    ax.set_xlabel("gate decision")
    ax.set_title("Confusion matrix (image counts)", fontsize=11)
    fig.tight_layout()

    designs = pd.DataFrame([
        design_row("Do nothing (everyone passes)", y, s, 1.01, 1.01, base_rate, c_fn),
        design_row("A person checks everyone", y, s, 0.0, 1.01, base_rate, c_fn),
        design_row(f"Model alone (threshold {CHOSEN_SINGLE:.2f}, chosen on validation)", y, s,
                   CHOSEN_SINGLE, CHOSEN_SINGLE, base_rate, c_fn),
        design_row(f"Model + human review (current band {lo:.2f} to {hi:.2f})", y, s, lo, hi, base_rate, c_fn),
    ])

    grid = np.round(np.linspace(0, 1, 101), 2)
    curve = Figure(figsize=(7, 4.8))
    cax = curve.subplots()
    for name, label, style in (("test", "test (reported)", "-"), ("validation", "validation (used to choose)", ":")):
        yy, ss = ARRAYS[name]
        cax.plot(grid, [1000 * cost_per_case(outcome_rates(yy, ss, t, t), base_rate, c_fn) for t in grid],
                 style, label=label)
    cax.axvspan(lo, max(hi, lo + 0.005), color="grey", alpha=0.25, label=f"review band {lo:.2f} to {hi:.2f}")
    cax.axvline(CHOSEN_SINGLE, color="red", linestyle="--",
                label=f"chosen threshold {CHOSEN_SINGLE:.2f} (minimum on validation)")
    cax.set_yscale("log")
    cax.set_xlabel("confidence threshold (flag NO-Mask at or above it)")
    cax.set_ylabel("expected cost per 1,000 cases (EUR, log scale)")
    cax.set_title("Cost against threshold")
    cax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    cax.grid(alpha=0.3)
    curve.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.34)
    return summary, fig, designs, curve


def run_detector(img: Image.Image):
    """One forward pass; the detections are cached so threshold changes don't re-run it."""
    r = get_model().predict(img, conf=0.001, iou=0.6, imgsz=640, device="cpu", verbose=False)[0]
    dets = []
    if r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, conf, cls):
            dets.append({"cls": r.names[int(k)], "conf": float(c), "box": [float(v) for v in box]})
    score = max([d["conf"] for d in dets if d["cls"] == POSITIVE], default=0.0)
    return dets, score


def decision(score, lo, hi) -> str:
    return "FAIL" if score >= hi else "REVIEW" if score >= lo else "PASS"


UPLOAD_PROMPT = ("**Upload a photo of a face below** (a stock photo or a photo of yourself, not a coursemate) "
                 "to see what the gate would decide.")

ACTIONS = {
    "PASS": "The door opens.",
    "REVIEW": "The model is unsure, so a supervisor looks at this photo and decides.",
    "FAIL": "The door stays shut until the operator puts a mask on.",
}


def render(state, lo, hi):
    """Annotated image, gate decision and detections table at the current thresholds."""
    if not state:
        return None, UPLOAD_PROMPT, pd.DataFrame(columns=["Class", "Confidence"])
    lo, hi = fix_band(lo, hi)
    img = state["img"].copy()
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 250)
    try:
        font = ImageFont.load_default(size=max(12, img.width // 40))
    except TypeError:
        font = ImageFont.load_default()
    shown = [d for d in state["dets"] if d["conf"] >= lo]
    for d in shown:
        color = COLORS.get(d["cls"], (255, 165, 0))
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{d['cls']} {d['conf']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 3 - top), label, fill=(255, 255, 255), font=font)
    verdict = decision(state["score"], lo, hi)
    text = (f"### Gate decision: **{verdict}**\n\n{ACTIONS[verdict]}\n\n"
            f"NO-Mask score for this image: **{state['score']:.3f}** "
            f"(pass below {lo:.2f}, fail from {hi:.2f}). Boxes shown: confidence at or above {lo:.2f}.")
    table = pd.DataFrame(
        [{"Class": d["cls"], "Confidence": round(d["conf"], 3)}
         for d in sorted(shown, key=lambda d: -d["conf"])[:30]],
        columns=["Class", "Confidence"])
    return img, text, table


def on_image(img, lo, hi):
    if img is None:
        return None, *render(None, lo, hi)
    img = img.convert("RGB")
    dets, score = run_detector(img)
    state = {"img": img, "dets": dets, "score": score}
    return (state, *render(state, lo, hi))


def on_thresholds(state, lo, hi):
    return render(state, lo, hi)


# ----------------------------------------------------------------- per-class views
def per_class_table() -> pd.DataFrame:
    box = METRICS["box_metrics"]
    rows = [{"Class": name, "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
             "AP50": round(v["ap50"], 3), "AP50-95": round(v["ap50_95"], 3)}
            for name, v in box["per_class"].items()]
    rows.append({"Class": "all classes", "Precision": "", "Recall": "",
                 "AP50": round(box["map50"], 3), "AP50-95": round(box["map50_95"], 3)})
    return pd.DataFrame(rows)


def box_confusion_table() -> pd.DataFrame:
    box = METRICS["box_metrics"]
    labels = box["confusion_matrix_labels"]
    df = pd.DataFrame(np.array(box["confusion_matrix_conf0.25"]).astype(int),
                      columns=[f"true {l}" for l in labels])
    df.insert(0, "", [f"predicted {l}" for l in labels])
    return df


def baselines_table() -> pd.DataFrame:
    rows = []
    for name, v in METRICS["image_level_baselines"].items():
        rows.append({
            "Model": name, "Caught": v["tp"], "False alarms": v["fp"], "Missed": v["fn"],
            "Correct passes": v["tn"],
            "Precision": "n/a" if v["precision"] is None else f"{v['precision']:.1%}",
            "Recall": f"{v['recall']:.1%}"})
    return pd.DataFrame(rows)


def band_text(lo, hi) -> str:
    lo_raw, hi_raw = float(lo), float(hi)
    lo, hi = fix_band(lo, hi)
    text = (f"**PASS** below {lo:.2f} (door opens)  ·  **REVIEW** from {lo:.2f} up to {hi:.2f} "
            f"(a supervisor looks)  ·  **FAIL** at {hi:.2f} and above (door stays shut)")
    if hi_raw < lo_raw:
        text += "  \n*The upper edge is below the lower edge, so the review band is empty.*"
    return text


def per_class_reading() -> str:
    box = METRICS["box_metrics"]
    pc = box["per_class"]
    labels = box["confusion_matrix_labels"]
    m = np.array(box["confusion_matrix_conf0.25"]).astype(int)
    mk, no, bg = labels.index("Mask"), labels.index("NO-Mask"), labels.index("background")
    return (
        f"**How to read this.** NO-Mask, the class the gate must catch, scores AP50 {pc['NO-Mask']['ap50']:.3f}; "
        f"Mask scores {pc['Mask']['ap50']:.3f}. The gate decision uses only the NO-Mask score. "
        f"The two classes are rarely confused with each other ({m[no][mk]} masked faces called NO-Mask, "
        f"{m[mk][no]} the other way). Most errors are faces the model misses entirely "
        f"({m[bg][mk]} Mask, {m[bg][no]} NO-Mask) or boxes where there is no face "
        f"({m[mk][bg]} Mask, {m[no][bg]} NO-Mask)."
    )


HEADER = f"""
# Mask gate for a luxury-cosmetics filling hall

A camera photographs each operator before they enter the hall. The model gives the photo a **NO-Mask score**
(how sure it is that someone is not wearing a mask), and two thresholds turn that score into a decision:
**PASS** (door opens), **REVIEW** (a supervisor looks at the photo) or **FAIL** (door stays shut).

**How to use it:** (1) upload a photo in tab 1, (2) move the two sliders below and watch every tab update,
(3) tab 2 shows what your choice costs and how many cases a person must check.

Model: yolo11s finetuned on Roboflow Universe `roboflow-universe-projects/mask-wearing-iskms`, version 6.
The starting sliders ({DEFAULT_LO:.2f} and {DEFAULT_HI:.2f}) were chosen on the validation split and are reported on the test split.
Photos you upload are processed in memory only and are not saved by this app.
"""


# ----------------------------------------------------------------- app
def build_app():
    with gr.Blocks(title="Mask gate dashboard") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            lo = gr.Slider(0.0, 1.0, value=DEFAULT_LO, step=0.01, label="Pass below this NO-Mask score")
            hi = gr.Slider(0.0, 1.0, value=DEFAULT_HI, step=0.01, label="Fail at or above this NO-Mask score")
        band = gr.Markdown(band_text(DEFAULT_LO, DEFAULT_HI))
        with gr.Accordion("Cost assumptions and data split (open this to test 'what if my assumption is wrong?')",
                          open=False):
            base = gr.Slider(0.001, 0.2, value=float(CFG["BASE_RATE"]), step=0.001,
                             label="Base rate: share of entries without a mask")
            cfn = gr.Slider(10, 3000, value=float(CFG["C_FN"]), step=10,
                            label="Cost of a missed no-mask (EUR, expected)")
            split = gr.Radio(["test", "validation"], value="test", label="Score the thresholds on")

        state = gr.State(None)
        ops_inputs = [split, lo, hi, base, cfn]

        with gr.Tabs():
            with gr.Tab("1. Try an image"):
                verdict = gr.Markdown(UPLOAD_PROMPT)
                with gr.Row():
                    upload = gr.Image(type="pil", label="Upload a photo", sources=["upload"], height=340)
                    annotated = gr.Image(type="pil", label="What the model sees", interactive=False, height=340)
                dets_table = gr.Dataframe(label="Faces found at or above the lower edge",
                                          headers=["Class", "Confidence"], interactive=False)
            with gr.Tab("2. Cost, errors and review load"):
                with gr.Row():
                    with gr.Column(scale=3):
                        summary = gr.Markdown()
                    with gr.Column(scale=2):
                        heat = gr.Plot(label="Confusion matrix")
                gr.Markdown("**How to read the matrix.** Each row is what is really in the photo; each column is what "
                            "the gate did. The costly mistake is the **top-left cell** (a no-mask face that passed). "
                            "The cheap mistake is the **bottom-right cell** (a masked operator who failed).")
                designs = gr.Dataframe(label="Priced against the alternatives (per 1,000 cases and per day)",
                                       interactive=False, wrap=True)
            with gr.Tab("3. Cost against threshold"):
                gr.Markdown("**How to read this.** Each line shows what the gate would cost if it used one single "
                            "threshold. The red dashed line is the threshold chosen on the validation split; "
                            "the grey band is your current review band. Lower is cheaper.")
                curve = gr.Plot(label="Cost curve")
            with gr.Tab("4. Per-class results"):
                gr.Markdown(per_class_reading())
                gr.Markdown("**Box-level results on the test split.** Precision and recall are at the confidence "
                            "that maximises F1; the confusion matrix is at confidence 0.25 "
                            "(rows predicted, columns true; *background* means no face).")
                gr.Dataframe(value=per_class_table(), label="Per class", interactive=False)
                gr.Dataframe(value=box_confusion_table(), label="Confusion matrix (boxes)", interactive=False)
                gr.Markdown("**Image-level baselines on the test split** (an image counts as positive if it has "
                            "at least one no-mask face; confidence 0.25).")
                gr.Dataframe(value=baselines_table(), label="Baselines", interactive=False)

        upload.change(on_image, [upload, lo, hi], [state, annotated, verdict, dets_table])
        for slider in (lo, hi):
            slider.change(on_thresholds, [state, lo, hi], [annotated, verdict, dets_table],
                          show_progress="hidden")
            slider.change(band_text, [lo, hi], band, show_progress="hidden")
        for comp in (split, lo, hi, base, cfn):
            comp.change(refresh_ops, ops_inputs, [summary, heat, designs, curve], show_progress="hidden")
        demo.load(refresh_ops, ops_inputs, [summary, heat, designs, curve])
    return demo


if __name__ == "__main__":
    get_model()  # fail fast if the committed weights are missing or do not load
    build_app().launch(server_port=APP_PORT)
