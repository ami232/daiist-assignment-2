"""Assignment 2: Gradio dashboard. Run: uv run python main.py app

Loads the committed weights and the artifacts written by train.py. Nothing is
trained here. Decision is per frame: the score is the highest 'pothole'
confidence. Below `lo` -> pass; from `lo` up to `hi` -> a person reviews the
photo; at or above `hi` -> a repair crew is sent.
"""
import json
import math
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
IMGSZ, IOU, POSITIVE = 512, 0.6, "pothole"

model = YOLO(str(ROOT / "weights" / "best.pt"))
metrics = json.loads((ART / "metrics.json").read_text())
baselines = json.loads((ART / "baselines.json").read_text())
sweep = json.loads((ART / "cost_sweep.json").read_text())
preds = json.loads((ART / "predictions.json").read_text())
stats_file = ART / "dataset_stats.json"
stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}

COST_FP, COST_FN = sweep["cost_fp"], sweep["cost_fn"]
REVIEW_COST = sweep["review_cost_per_frame"]
FRAMES_PER_DAY = sweep["frames_per_day"]
PER_PERSON = sweep["frames_per_person_per_day"]
N_REVIEWERS = sweep["n_reviewers"]
CH_LO, CH_HI = sweep["chosen"]["lo"], sweep["chosen"]["hi"]
CH_BR = sweep["central_base_rate"]

names = {int(k): v for k, v in model.names.items()}
pid = next(k for k, v in names.items() if v == POSITIVE)
scores, labels = [], []
for rec in preds.values():
    conf = [p[1] for p in rec["pred"] if p[0] == pid]
    scores.append(max(conf, default=0.0))
    labels.append(any(g[0] == pid for g in rec["gt"]))
scores, labels = np.array(scores), np.array(labels)
POS, NEG = scores[labels], scores[~labels]
THR = np.round(np.linspace(0, 1, 201), 3)
TPR = np.array([(POS >= t).mean() for t in THR])
FPR = np.array([(NEG >= t).mean() for t in THR])


def rates(s, lo, hi):
    """Share of frames that pass / go to review / get flagged."""
    return np.array([(s < lo).mean(), ((s >= lo) & (s < hi)).mean(), (s >= hi).mean()])


def evaluate(lo, hi, pi):
    rp, rn = rates(POS, lo, hi), rates(NEG, lo, hi)
    share = pi * rp[1] + (1 - pi) * rn[1]
    cost = 1000 * (pi * rp[0] * COST_FN + (1 - pi) * rn[2] * COST_FP + share * REVIEW_COST)
    return rp, rn, share, cost


def update(lo, hi, br_pct):
    lo = float(lo)
    hi = max(float(hi), lo)
    pi = float(br_pct) / 100
    rp, rn, share, cost = evaluate(lo, hi, pi)
    nothing = 1000 * pi * COST_FN
    review_all = 1000 * REVIEW_COST
    per_day = share * FRAMES_PER_DAY
    people = math.ceil(per_day / PER_PERSON)
    everyone = math.ceil(FRAMES_PER_DAY / PER_PERSON)

    summary = (
        f"### Expected cost per 1,000 frames: €{cost:,.0f}\n"
        f"Do nothing: €{nothing:,.0f} · a person checks every frame: "
        f"€{review_all:,.0f} (needs {everyone} people)\n\n"
        f"### Sent to a person: {share:.1%} of frames ≈ {per_day:,.0f} per day "
        f"→ {people} reviewer(s)\n"
        f"Staffed in this scenario: {N_REVIEWERS}. "
        + ("**Warning: this needs more people than are staffed.**"
           if people > N_REVIEWERS else "This fits the staffed team.")
        + f"\n\nBase rate {pi:.1%}, false positive €{COST_FP:,.0f}, "
        f"false negative €{COST_FN:,.0f}, review €{REVIEW_COST:.3f} per frame "
        f"(the reviewer looks at the photo and is assumed to be always right)."
    )

    cols = ["Pass", "Human review", "Flag (send crew)"]
    per1000 = pd.DataFrame({
        "Truth, per 1,000 frames": ["Pothole present", "No pothole"],
        **{c: [round(1000 * pi * rp[i], 1), round(1000 * (1 - pi) * rn[i], 1)]
           for i, c in enumerate(cols)}})
    n_pos, n_neg = len(POS), len(NEG)
    counts = pd.DataFrame({
        "Truth, test split": [f"Pothole present (n={n_pos})", f"No pothole (n={n_neg})"],
        **{c: [int(round(rp[i] * n_pos)), int(round(rn[i] * n_neg))]
           for i, c in enumerate(cols)}})

    curve = 1000 * (pi * (1 - TPR) * COST_FN + (1 - pi) * FPR * COST_FP)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=THR, y=curve, mode="lines",
                             name="model alone (one threshold)"))
    fig.add_hline(y=nothing, line_dash="dash", line_color="gray",
                  annotation_text="do nothing")
    fig.add_hline(y=cost, line_dash="dot", line_color="green",
                  annotation_text="current band", annotation_position="bottom right")
    fig.add_vline(x=CH_LO, line_color="green", annotation_text=f"chosen lo {CH_LO:.3f}")
    fig.add_vline(x=CH_HI, line_color="red", annotation_text=f"chosen hi {CH_HI:.3f}",
                  annotation_position="top right")
    fig.add_vline(x=lo, line_dash="dot", line_color="green")
    fig.add_vline(x=hi, line_dash="dot", line_color="red")
    fig.update_layout(xaxis_title="confidence threshold",
                      yaxis_title="EUR per 1,000 frames",
                      margin=dict(t=30, b=40), height=420)
    return summary, per1000, counts, fig


def predict(image, lo, hi):
    if image is None:
        return None, "Upload a road photo to see predictions."
    lo = float(lo)
    hi = max(float(hi), lo)
    bgr = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])  # Gradio is RGB, YOLO wants BGR
    r = model.predict(bgr, conf=max(lo, 0.001), iou=IOU, imgsz=IMGSZ, verbose=False)[0]
    shown = np.ascontiguousarray(r.plot()[:, :, ::-1])
    pot = [float(c) for k, c in zip(r.boxes.cls, r.boxes.conf) if int(k) == pid]
    score = max(pot, default=0.0)
    if score >= hi:
        verdict = f"**FLAG: send a repair crew** (pothole confidence {score:.2f} ≥ hi {hi:.3f})"
    elif pot:
        verdict = f"**HUMAN REVIEW** (pothole confidence {score:.2f}, between lo {lo:.3f} and hi {hi:.3f})"
    else:
        verdict = f"**PASS** (no pothole box at or above lo {lo:.3f})"
    return shown, verdict + "\n\nBoxes of all classes are drawn at confidence ≥ lo; the verdict uses pothole boxes only."


def per_class_table():
    test = stats.get("instances", {}).get("test", {})
    rows = [{"Class": c + (" (positive)" if c == POSITIVE else ""),
             "Test instances": test.get(c, ""),
             "Precision": round(m["precision"], 3), "Recall": round(m["recall"], 3),
             "AP50": round(m["ap50"], 3), "AP50-95": round(m["ap50_95"], 3)}
            for c, m in metrics["per_class"].items()]
    rows.append({"Class": "all (mAP)", "Test instances": sum(test.values()) if test else "",
                 "Precision": "", "Recall": "",
                 "AP50": round(metrics["map50"], 3), "AP50-95": round(metrics["map50_95"], 3)})
    return pd.DataFrame(rows)


def baseline_table():
    def fmt(v):
        return "n/a" if v is None else round(v, 3)
    p = metrics["per_class"][POSITIVE]["ap50"]
    cz = baselines["coco_zero_shot"].get("ap50")
    pn = baselines["predict_nothing"].get("ap50")
    return pd.DataFrame([
        {"Model": "Finetuned YOLO11s", "AP50 all classes": round(metrics["map50"], 3), "AP50 pothole": round(p, 3)},
        {"Model": "COCO zero-shot", "AP50 all classes": fmt(cz), "AP50 pothole": fmt(cz)},
        {"Model": "Predict nothing", "AP50 all classes": fmt(pn), "AP50 pothole": fmt(pn)}])


with gr.Blocks(title="Road damage triage") as demo:
    gr.Markdown("# Pothole triage: pass, human review or send a crew")
    with gr.Row():
        lo = gr.Slider(0, 1, value=CH_LO, step=0.005, label="Pass / review edge (lo)")
        hi = gr.Slider(0, 1, value=CH_HI, step=0.005, label="Review / flag edge (hi)")
        br = gr.Slider(0.5, 50, value=CH_BR * 100, step=0.5,
                       label="Assumed share of frames with a pothole (%)")
    with gr.Tab("Cost and review load"):
        summary = gr.Markdown()
        with gr.Row():
            m1 = gr.Dataframe(label="Confusion matrix per 1,000 frames at the assumed base rate", interactive=False)
            m2 = gr.Dataframe(label="Same matrix, raw counts on the test split", interactive=False)
        gr.Markdown("Pothole present + Pass = false negative · No pothole + Flag = false positive. "
                    "Frames in review are settled by the person.")
        plot = gr.Plot(label="Cost against threshold")
    with gr.Tab("Try an image"):
        img = gr.Image(type="numpy", label="Road photo")
        out_img = gr.Image(label="Predictions")
        verdict = gr.Markdown()
    with gr.Tab("Per-class results"):
        gr.Markdown("Test split. Precision and recall are Ultralytics' values at its best-F1 confidence.")
        gr.Dataframe(value=per_class_table(), interactive=False)
        gr.Markdown("Baselines on the same test split.")
        gr.Dataframe(value=baseline_table(), interactive=False)

    ins = [lo, hi, br]
    outs = [summary, m1, m2, plot]
    gr.on(triggers=[lo.change, hi.change, br.change], fn=update, inputs=ins, outputs=outs)
    demo.load(update, ins, outs)
    gr.on(triggers=[img.change, lo.change, hi.change], fn=predict,
          inputs=[img, lo, hi], outputs=[out_img, verdict])

if __name__ == "__main__":
    demo.launch(inbrowser=True)