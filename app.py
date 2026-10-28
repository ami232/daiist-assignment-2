"""
Assignment 2: Gradio dashboard for garment quality control.

    uv run python main.py app

Loads the committed weights (weights/best.pt) and the results train.py saved in
results/. It never trains. Every cost, confusion matrix and per-class number is
recomputed live from the saved model scores and config.yaml, so moving a slider
or editing a cost in config.yaml needs no retraining.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import ImageDraw  # noqa: E402

import costing  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "best.pt"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


CFG = costing.load_config()
STATS = load("dataset_stats.json")
METRICS = load("metrics.json")
OP = load("operating_point.json")
SENS = load("sensitivity.json")
TRAINING = load("training.json")
PRED = {s: load(f"predictions_{s}.json") for s in ["valid", "test"]}
NAMES = STATS["classes"]
CHOSEN = OP["chosen"]
AREA = CFG["evaluation"]["negative_area_factor"]

POS = {s: costing.image_scores([im for im in PRED[s]["images"] if im["gt"]]) for s in PRED}
NEG = {s: np.array(PRED[s]["negative_crop_scores"], dtype=float) for s in PRED}

_model = None


def model():
    """Load the committed weights once, on first use (inference only)."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def cfg_with_base_rate(base_rate: float) -> dict:
    cfg = json.loads(json.dumps(CFG))
    cfg["business"]["base_rate"] = base_rate
    return cfg


def ordered(low: float, high: float) -> tuple[float, float]:
    return (low, high) if low <= high else (high, low)


def eur(x: float) -> str:
    return f"EUR {x:,.0f}"


# ---------------------------------------------------------------- inspect


def inspect(image, low, high):
    if image is None:
        return None, "Upload a photo of a garment or fabric, or click an example below.", None
    low, high = ordered(low, high)
    r = model().predict(image, imgsz=CFG["training"]["imgsz"], conf=CFG["evaluation"]["conf_floor"],
                        max_det=CFG["evaluation"]["max_det"], device="cpu", verbose=False)[0]
    boxes = [(NAMES[int(c)], float(s), [float(v) for v in xyxy])
             for c, s, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    score = max((b[1] for b in boxes), default=0.0)
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    rows = []
    for name, s, (x1, y1, x2, y2) in boxes:
        if s < low:
            continue
        colour = "red" if s >= high else "orange"
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        draw.text((x1 + 3, max(y1 - 12, 0)), f"{name} {s:.2f}", fill=colour)
        rows.append({"defect": name, "confidence": round(s, 3), "action": "reject" if s >= high else "review",
                     "box (x1,y1,x2,y2)": f"{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}"})
    if score >= high:
        verdict = f"## REJECT\nHighest defect confidence **{score:.2f}** is at or above the reject edge ({high:.2f})."
    elif score >= low:
        verdict = (f"## SEND TO HUMAN REVIEW\nHighest defect confidence **{score:.2f}** is in the review band "
                   f"({low:.2f}-{high:.2f}). A reviewer checks the orange boxes.")
    else:
        verdict = f"## PASS\nHighest defect confidence **{score:.2f}** is below the pass edge ({low:.2f})."
    table = pd.DataFrame(rows) if rows else pd.DataFrame([{"defect": "none at or above the pass edge"}])
    return out, verdict, table


# ---------------------------------------------------------------- cost + review


def confusion_figure(matrix: np.ndarray, raw: dict, low: float, high: float):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Oranges", vmin=0, vmax=1)
    cols, rows = ["pass", "review", "reject"], ["defective", "good"]
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.1f}\n({raw[rows[i]][cols[j]]} raw)", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(3), [f"pass\n< {low:.2f}", f"review\n{low:.2f}-{high:.2f}", f"reject\n>= {high:.2f}"])
    ax.set_yticks(range(2), ["actually defective", "actually good"])
    ax.set_title("Expected garments per 1,000 (raw test counts in brackets)", fontsize=10)
    fig.tight_layout()
    return fig


def cost_curve_figure(split: str, low: float, high: float, cfg: dict):
    sw = costing.single_sweep(POS[split], NEG[split], cfg)
    best = int(np.argmin(sw["cost"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sw["thresholds"], 1000 * sw["cost"], label=f"model alone ({split} split)")
    ax.axvline(CHOSEN["single"], color="red", ls="--", label=f"my chosen threshold {CHOSEN['single']:.2f}")
    ax.scatter([sw["thresholds"][best]], [1000 * sw["cost"][best]], color="black", zorder=5,
               label=f"minimum here: {sw['thresholds'][best]:.2f}")
    ax.axvspan(low, high, color="orange", alpha=0.2, label=f"current review band {low:.2f}-{high:.2f}")
    p = cfg["business"]["base_rate"]
    ax.axhline(1000 * p * cfg["costs_eur"]["false_negative"], color="grey", ls=":", label="do nothing")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("EUR per 1,000 garments")
    ax.set_title(f"Cost against threshold, base rate {p:.1%}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def cost_view(low, high, split, base_rate):
    low, high = ordered(low, high)
    cfg = cfg_with_base_rate(base_rate)
    pos, neg = POS[split], NEG[split]
    rates = costing.outcome_rates(pos, neg, low, high, AREA)
    share = costing.review_share(rates, base_rate)
    per_day = share * cfg["business"]["garments_per_day"]
    people = costing.reviewers_needed(share, cfg)
    strategies = costing.strategy_costs(pos, neg, cfg, CHOSEN["single"], low, high)
    current = strategies["Model + human review"]["cost_per_1000"]
    raw = {"defective": {"pass": int((pos < low).sum()), "review": int(((pos >= low) & (pos < high)).sum()),
                         "reject": int((pos >= high).sum())},
           "good": {"pass": int((neg < low).sum()), "review": int(((neg >= low) & (neg < high)).sum()),
                    "reject": int((neg >= high).sum())}}
    available = cfg["human"]["reviewers_available"]
    kpis = (
        f"### Expected cost at these thresholds: **{eur(current)} per 1,000 garments**\n"
        f"- Share sent to human review: **{share:.1%}** = **{per_day:,.0f} garments/day** "
        f"x {cfg['human']['review_minutes']} min = {per_day * cfg['human']['review_minutes']:,.0f} min/day\n"
        f"- Reviewers needed: **{people:.2f}** (available: {available}) "
        f"{'-- fits' if people <= available else '-- **exceeds capacity**'}\n"
        f"- Defects caught (reject + confirmed in review): "
        f"**{rates['pos']['reject'] + rates['pos']['review'] * cfg['human']['review_recall']:.1%}**; "
        f"good garments wrongly rejected: **{rates['neg']['reject'] + rates['neg']['review'] * (1 - cfg['human']['specificity']):.1%}**\n"
        f"- Saving vs doing nothing: **{eur(strategies['Do nothing (ship everything)']['cost_per_1000'] - current)}** "
        f"per 1,000; vs a person checking everything: "
        f"**{eur(strategies['Person checks every garment']['cost_per_1000'] - current)}**\n\n"
        f"*Good-garment rates come from {len(neg)} defect-free crops, scaled x{AREA} to a full photo "
        f"(the dataset has no defect-free images). Defective rates come from {len(pos)} {split} photos.*"
    )
    table = pd.DataFrame([{"strategy": k, "EUR per 1,000": round(v["cost_per_1000"], 1),
                           "share to a person": f"{v['review_share']:.1%}", "people needed": round(v["people"], 2)}
                          for k, v in strategies.items()])
    matrix = costing.expected_confusion(rates, base_rate)
    return kpis, confusion_figure(matrix, raw, low, high), cost_curve_figure(split, low, high, cfg), table


# ---------------------------------------------------------------- per class


def class_confusion_figure(cm: np.ndarray, conf: float, split: str):
    labels = NAMES + ["background"]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Box confusion matrix, {split}, conf >= {conf:.2f}, IoU >= 0.5", fontsize=10)
    fig.tight_layout()
    return fig


def per_class_view(conf, split):
    images = PRED[split]["images"]
    cm = costing.class_confusion(images, conf, CFG["evaluation"]["iou_match"], len(NAMES))
    at_t = costing.per_class_pr(cm)
    ul = METRICS[split]["per_class"]

    def fmt(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else round(v, 3)

    rows = [{"class": n, "instances": r["instances"], f"precision @ {conf:.2f}": fmt(r["precision"]),
             f"recall @ {conf:.2f}": fmt(r["recall"]), "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
             "AP50": fmt(ul[n]["ap50"]), "AP50-95": fmt(ul[n]["ap50_95"])} for n, r in zip(NAMES, at_t)]
    rows.append({"class": "ALL (mean)", "instances": sum(r["instances"] for r in at_t),
                 "AP50": round(METRICS[split]["map50"], 3), "AP50-95": round(METRICS[split]["map50_95"], 3)})
    return pd.DataFrame(rows), class_confusion_figure(cm, conf, split)


# ---------------------------------------------------------------- static tables


def baseline_table(split: str) -> pd.DataFrame:
    b = METRICS["baselines"][split]
    rows = []
    for name, v in b.items():
        if not isinstance(v, dict):
            continue
        rows.append({"model": name, "mAP50": round(v["map50"], 3), "mAP50-95": round(v["map50_95"], 3),
                     "class-agnostic AP50": round(v["agnostic_ap50"], 3),
                     "threshold": v["threshold"], "defect recall (image)": f"{v['image_recall']:.1%}",
                     "good garments rejected": f"{v['garment_false_reject_rate']:.1%}",
                     "EUR per 1,000 (model alone)": round(v["cost_per_1000"], 1)})
    return pd.DataFrame(rows)


def dataset_table() -> pd.DataFrame:
    rows = []
    for split, st in STATS["splits"].items():
        row = {"split": split, "images": st["images"], "source images": st["source_images"],
               "defect-free images": st["defect_free_images"], "boxes": st["boxes"]}
        for n in NAMES:
            pc = st["per_class"][n]
            row[f"{n} boxes"] = pc["boxes"]
            row[f"{n} median area %"] = None if pc["area_pct_median"] is None else round(pc["area_pct_median"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_table() -> pd.DataFrame:
    rows = []
    for d in SENS.values():
        base = min(d["rows"], key=lambda r: abs(r["value"] - d["baseline"]))
        rows.append({"assumption": d["label"], "my value": d["baseline"],
                     "best strategy at my value": d["baseline_best"],
                     "chosen threshold range over sweep": f"{min(r['single'] for r in d['rows']):.2f}-"
                                                          f"{max(r['single'] for r in d['rows']):.2f}",
                     "review band at my value": f"{base['low']:.2f}-{base['high']:.2f}",
                     "where the best strategy changes": "; ".join(d["switches"]) or "never in the range tested"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- layout


def header() -> str:
    d = CFG["dataset"]
    t = OP["test"]
    s = t["strategies"]
    return (
        f"# Garment quality control: pass, review or reject\n"
        f"Finetuned **yolo11n** on Roboflow Universe `{d['workspace']}/{d['project']}` v{d['version']} "
        f"({TRAINING['epochs_run']} epochs, {TRAINING['where'].split('|')[-2].strip()}). "
        f"Positive class = *garment has a defect*; a photo counts as defective if any box scores at or above "
        f"the threshold.\n\n"
        f"**My operating point** (chosen on the valid split by minimum expected cost): model alone rejects at "
        f"**{CHOSEN['single']:.2f}**; with people: pass < **{CHOSEN['low']:.2f}** <= review < "
        f"**{CHOSEN['high']:.2f}** <= reject. On test, at a {CFG['business']['base_rate']:.0%} base rate: "
        f"model + review **{eur(s['Model + human review']['cost_per_1000'])}**, model alone "
        f"{eur(s['Model alone']['cost_per_1000'])}, person checks all "
        f"{eur(s['Person checks every garment']['cost_per_1000'])}, do nothing "
        f"{eur(s['Do nothing (ship everything)']['cost_per_1000'])} per 1,000 garments."
    )


EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

with gr.Blocks(title="Garment QC dashboard") as demo:
    gr.Markdown(header())
    with gr.Row():
        low = gr.Slider(0, 1, value=CHOSEN["low"], step=0.01, label="Pass | review edge (low threshold)")
        high = gr.Slider(0, 1, value=CHOSEN["high"], step=0.01, label="Review | reject edge (high threshold)")
        base_rate = gr.Slider(0.005, 0.3, value=CFG["business"]["base_rate"], step=0.005,
                              label="Defect base rate (share of garments with a defect)")
        split = gr.Radio(["test", "valid"], value="test", label="Evaluate on split")

    with gr.Tab("Inspect a garment"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Upload a garment / fabric photo")
            image_out = gr.Image(label="Predictions (orange = review, red = reject)")
        verdict = gr.Markdown()
        boxes = gr.Dataframe(label="Boxes at or above the pass edge")
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=image_in, label="Test-split examples (CC BY 4.0, sqc on Roboflow Universe)")

    with gr.Tab("Cost, confusion matrix and review load"):
        kpis = gr.Markdown()
        with gr.Row():
            confusion_plot = gr.Plot(label="Image-level confusion matrix")
            curve_plot = gr.Plot(label="Cost against threshold")
        strategy_table = gr.Dataframe(label="Four ways to run inspection (cost per 1,000 garments)")

    with gr.Tab("Per-class results"):
        gr.Markdown("Box-level results per defect class. A box counts as found if it overlaps a labelled defect "
                    "of the **same class** with IoU >= 0.5. AP50 / AP50-95 come from Ultralytics and do not "
                    "depend on the slider. The test split has only 3 *hole* boxes: treat that row as anecdotal.")
        box_conf = gr.Slider(0.01, 1, value=max(CHOSEN["single"], 0.01), step=0.01,
                             label="Box confidence threshold for per-class precision / recall")
        class_table = gr.Dataframe()
        class_plot = gr.Plot()
        gr.Image(str(RESULTS / "plots" / "failures.png"), label="Where it fails: hardest test images")

    with gr.Tab("Baselines and dataset"):
        gr.Markdown("### Same test split, three models")
        gr.Dataframe(baseline_table("test"))
        gr.Markdown("The COCO model has no fabric-defect classes, so its class-aware mAP is 0 by construction. "
                    "Class-agnostic AP50 gives it credit for boxing *anything* where a defect is.")
        gr.Markdown("### Dataset composition")
        gr.Dataframe(dataset_table())
        gr.Image(str(RESULTS / "plots" / "dataset.png"), label="Boxes per class and object size")
        gr.Image(str(RESULTS / "plots" / "training_curves.png"), label="Training curves")

    with gr.Tab("Sensitivity"):
        gr.Markdown("Each assumption in config.yaml varied on its own; thresholds re-chosen on valid, "
                    "strategies re-priced on test.")
        gr.Dataframe(sensitivity_table())
        gr.Image(str(RESULTS / "plots" / "sensitivity.png"))

    cost_inputs = [low, high, split, base_rate]
    cost_outputs = [kpis, confusion_plot, curve_plot, strategy_table]
    gr.on([low.change, high.change, split.change, base_rate.change, demo.load], cost_view, cost_inputs, cost_outputs)
    gr.on([image_in.change, low.change, high.change], inspect, [image_in, low, high], [image_out, verdict, boxes])
    gr.on([box_conf.change, split.change, demo.load], per_class_view, [box_conf, split], [class_table, class_plot])


if __name__ == "__main__":
    demo.launch()
