"""Gradio dashboard. Loads the committed weights and the files in results/; never trains.

Run with: uv run python main.py app
"""
import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from costing import load_costs, load_scores, three_way  # noqa: E402

RESULTS = ROOT / "results"
model = YOLO(str(ROOT / "weights" / "best.pt"))
results = json.loads((RESULTS / "results.json").read_text())
sweep = json.loads((RESULTS / "cost_sweep.json").read_text())
costs = load_costs()
PI = costs["base_rate"]
SCORES = {s: load_scores(s) for s in ("valid", "test")}
LO0, HI0 = sweep["three_way"]["low"], sweep["three_way"]["high"]


def predict(img, lo, hi):
    """Run the model on one uploaded image and apply the two thresholds to its top confidence."""
    if img is None:
        return None, "Upload an image first."
    hi = max(hi, lo)
    r = model.predict(img, conf=max(lo, 0.001), imgsz=640, verbose=False)[0]
    shown = r.plot()[..., ::-1]
    if len(r.boxes) == 0:
        return shown, f"### PASS\nNo box with confidence of {lo:.2f} or more."
    top = float(r.boxes.conf.max())
    decision = "FAIL" if top >= hi else "HUMAN REVIEW"
    lines = [f"- {r.names[int(c)]}: {float(p):.2f}" for c, p in zip(r.boxes.cls, r.boxes.conf)]
    return shown, f"### {decision}\nHighest confidence: **{top:.2f}**\n\nBoxes shown (confidence at or above {lo:.2f}):\n" + "\n".join(lines)


def update_cost(split, lo, hi):
    """Outcome table, expected cost, review load and cost curve for the current thresholds."""
    hi = max(hi, lo)
    s, t = SCORES[split]
    masks = {"pass": s < lo, "review": (s >= lo) & (s < hi), "fail": s >= hi}
    table = pd.DataFrame({
        "truth": ["really defective", "really good"],
        **{k: [int((m & t).sum()), int((m & ~t).sum())] for k, m in masks.items()},
    })
    r = three_way(s, t, lo, hi, costs, PI)
    ok = "enough" if r["people_needed"] <= costs["max_reviewers"] else "NOT enough"
    summary = (
        f"**Expected cost per 1,000 boards: EUR {r['cost_per_1000']:.0f}** (at an assumed base rate of {PI*100:.1f}% defective)\n\n"
        f"- Share sent for human review: **{r['review_share']*100:.1f}%** "
        f"({r['reviews_per_day']:.0f} boards per day)\n"
        f"- Inspectors needed: **{r['people_needed']:.2f}** "
        f"({costs['max_reviewers']} available, which is {ok})\n"
        f"- Defective boards missed (passed): {r['defective']['pass']*100:.1f}% of defective\n"
        f"- Good boards failed automatically: {r['good']['fail']*100:.1f}% of good\n\n"
        f"Table counts come from the **{split}** split. Costs are re-weighted to the assumed base rate, not the dataset's."
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["single"]["grid"], sweep["single"]["cost_valid"], label="valid")
    ax.plot(sweep["single"]["grid"], sweep["single"]["cost_test"], label="test")
    ax.axvspan(lo, hi, alpha=0.15, color="orange", label="human review band")
    ax.axvline(sweep["single"]["chosen_threshold"], color="k", ls="--", label="chosen single threshold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("expected cost per 1,000 boards (EUR)")
    ax.set_title("Cost against threshold (flagged boards go to a person)")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return table, summary, fig


def per_class_tables():
    d = results["detection_test"]
    pc = pd.DataFrame(d["per_class"]).round(3)
    o = d["overall"]
    overall = (f"**All classes (test):** precision {o['precision']:.3f}, recall {o['recall']:.3f}, "
               f"AP50 {o['ap50']:.3f}, mAP50-95 {o['map50_95']:.3f}")
    base = pd.DataFrame([
        {"strategy": k.replace("_", " "), "cost per 1,000 (EUR)": round(v["cost_per_1000"]),
         "people needed": round(v["people_needed"], 1),
         "feasible": v["people_needed"] <= costs["max_reviewers"]}
        for k, v in sweep["baselines"].items()])
    il = results["image_level"]
    auroc = (f"Image-level AUROC on test (does the top score separate defective from good images?): "
             f"fine-tuned model **{il['model']['test']['auroc']:.3f}**, "
             f"COCO zero-shot **{il['coco_zero_shot']['test']['auroc']:.3f}**")
    return pc, overall, base, auroc


with gr.Blocks(title="PCB defect inspection") as demo:
    gr.Markdown("# PCB defect inspection\nBelow the lower threshold a board **passes**, from the upper threshold up it "
                "**fails**, and in between it goes to a **person**. Positive class for the cost analysis: image has at least one defect.")
    with gr.Row():
        lo = gr.Slider(0, 1, value=LO0, step=0.01, label="Lower threshold (below = PASS)")
        hi = gr.Slider(0, 1, value=HI0, step=0.01, label="Upper threshold (at or above = FAIL)")
        split = gr.Radio(["test", "valid"], value="test", label="Evaluate on")

    with gr.Tab("Try an image"):
        with gr.Row():
            img = gr.Image(type="pil", label="Upload a board image")
            out_img = gr.Image(label="Predictions")
        out_txt = gr.Markdown()
        gr.Button("Predict").click(predict, [img, lo, hi], [out_img, out_txt])
        img.change(predict, [img, lo, hi], [out_img, out_txt])
        lo.release(predict, [img, lo, hi], [out_img, out_txt])
        hi.release(predict, [img, lo, hi], [out_img, out_txt])

    with gr.Tab("Cost and human review"):
        table = gr.Dataframe(label="Outcomes: truth against decision (image counts)")
        summary = gr.Markdown()
        plot = gr.Plot()

    with gr.Tab("Per-class results"):
        pc, overall, base, auroc = per_class_tables()
        gr.Markdown(overall)
        gr.Dataframe(pc, label="Detection per class (test split)")
        cm = RESULTS / "confusion_matrix.png"
        if cm.exists():
            gr.Image(str(cm), label="Confusion matrix (test)")
        gr.Markdown(auroc)
        gr.Dataframe(base, label="Strategies compared (test split, assumed base rate)")

    for control in (lo, hi, split):
        control.change(update_cost, [split, lo, hi], [table, summary, plot])
    demo.load(update_cost, [split, lo, hi], [table, summary, plot])

demo.launch()