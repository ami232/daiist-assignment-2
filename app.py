import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OP = json.loads((RES / "operating_point.json").read_text())
model = YOLO(str(ROOT / "weights" / "best.pt"))   # saved weights; nothing is trained here
NAMES = model.names
DISTRESS = {0, 1}                                  # Crawling_person, Lying_person
PER_CLASS = pd.read_csv(RES / "per_class.csv")


def load(split, n_all):
    gt = pd.read_csv(RES / f"{split}_gt.csv")
    pred = pd.read_csv(RES / f"{split}_pred.csv")
    return {
        "scores": gt.score.values,                         # model score of each distress person (0 = never found)
        "false_alarms": pred[pred.is_tp == 0].conf.values,  # confidence of each false alarm
        "n_pos": len(gt),                                   # distress people
        "n_neg": n_all - len(gt),                           # everyone else
    }


TEST = load("test", 129)        # 32 distress + 97 others
VALID = load("valid", 262)      # 56 distress + 206 others


def outcomes(S, lo, hi, base):
    """How many cases per 1,000 end up passed / reviewed / flagged."""
    rp = 1000 * base / S["n_pos"]
    rn = 1000 * (1 - base) / S["n_neg"]
    s, fa = S["scores"], S["false_alarms"]
    o = {
        "dist_pass": (s < lo).sum() * rp,
        "dist_rev": ((s >= lo) & (s < hi)).sum() * rp,
        "dist_flag": (s >= hi).sum() * rp,
        "other_rev": ((fa >= lo) & (fa < hi)).sum() * rn,
        "other_flag": (fa >= hi).sum() * rn,
    }
    o["other_pass"] = max(0.0, 1000 * (1 - base) - o["other_rev"] - o["other_flag"])
    return o


def cost_per_1000(o, c_miss, c_fa, rev_cost, miss_rate):
    return (o["dist_pass"] * c_miss
            + o["dist_rev"] * (rev_cost + miss_rate * c_miss)
            + o["other_rev"] * rev_cost
            + o["other_flag"] * c_fa)


def alone_curve(S, grid, c_miss, c_fa, base):
    """Cost per 1,000 cases if the model decides alone at each threshold."""
    return [cost_per_1000(outcomes(S, t, t, base), c_miss, c_fa, 0, 0) for t in grid]


def dark_axes(fig, ax):
    """Match the matplotlib charts to the dark navy theme."""
    fig.patch.set_facecolor("#0f1f40")
    ax.set_facecolor("#0f1f40")
    for part in (ax.xaxis.label, ax.yaxis.label, ax.title):
        part.set_color("#cfdcf7")
    ax.tick_params(colors="#9fb3d9")
    for spine in ax.spines.values():
        spine.set_color("#25407a")
    ax.grid(True, color="#1c3264", linewidth=0.6)
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor("#0a1630")
        legend.get_frame().set_edgecolor("#25407a")
        for text in legend.get_texts():
            text.set_color("#e8eefb")


def refresh(lo, hi, c_miss, c_fa, base, rev_cost, miss_rate, frames, secs):
    hi = max(hi, lo)
    o = outcomes(TEST, lo, hi, base)
    cost = cost_per_1000(o, c_miss, c_fa, rev_cost, miss_rate)

    share = (o["dist_rev"] + o["other_rev"]) / 1000
    reviews = share * frames
    hours = reviews * secs / 3600
    text = (
        f"### Expected cost per 1,000 cases: EUR {cost:,.0f}\n"
        f"Doing nothing would cost EUR {1000 * base * c_miss:,.0f}.\n\n"
        f"- **Share sent to a person: {share:.1%}**\n"
        f"- At {frames:,.0f} cases per day: {reviews:,.0f} reviews = {hours:.1f} hours "
        f"of screen time, about {hours / OP['hours']:.2f} reviewers (you have {OP['reviewers']}).\n"
        f"- Cost per day: EUR {cost * frames / 1000:,.0f}"
    )

    table = pd.DataFrame(
        [["Person in distress", o["dist_pass"], o["dist_rev"], o["dist_flag"]],
         ["Not in distress", o["other_pass"], o["other_rev"], o["other_flag"]]],
        columns=["Truth (per 1,000 cases)", "Pass", "Human review", "Flag: send rescuer"],
    ).round(1)

    grid = np.round(np.arange(0.01, 0.96, 0.01), 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, alone_curve(VALID, grid, c_miss, c_fa, base), color="#5b9bff", label="validation (used to choose)")
    ax.plot(grid, alone_curve(TEST, grid, c_miss, c_fa, base), "--", color="#f2b84b", label="test")
    ax.axvline(OP["t_alone"], color="#ff6b6b", ls="--", label=f"chosen threshold {OP['t_alone']}")
    ax.axvline(lo, color="#4ad295", ls=":", label=f"your lower threshold {lo:.2f}")
    ax.axhline(1000 * base * c_miss, color="#8fa3cc", ls=":", label="do nothing")
    ax.set_xlabel("confidence threshold (model decides alone)")
    ax.set_ylabel("expected cost per 1,000 cases (EUR)")
    ax.legend()
    dark_axes(fig, ax)
    fig.tight_layout()
    plt.close(fig)
    return text, table, fig


def predict(img, lo, hi):
    """Run the saved model on one uploaded image and apply the two thresholds."""
    if img is None:
        return None, "Upload an image first."
    hi = max(hi, lo)
    img = img.convert("RGB")
    res = model.predict(img, conf=0.001, verbose=False)[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    best, lines = 0.0, []
    for box, cls, p in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        cls = int(cls)
        if cls in DISTRESS:
            if p < lo:
                continue
            best = max(best, p)
            flagged = p >= hi
            status = "SEND RESCUER" if flagged else "HUMAN REVIEW"
            color = "red" if flagged else "orange"
        else:
            if p < 0.25:
                continue
            status, color = "not distress", "gray"
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 3, box[1] + 3), f"{NAMES[cls]} {p:.2f}", fill=color)
        lines.append(f"- {NAMES[cls]} {p:.2f}: {status}")
    if best >= hi:
        decision = "FLAG: send a rescuer"
    elif best >= lo:
        decision = "SEND FOR HUMAN REVIEW"
    else:
        decision = "PASS: no distress detected"
    detail = "\n".join(lines) if lines else "No detections above the thresholds."
    return out, f"### Decision: {decision}\n{detail}"


def per_class_fig():
    d = PER_CLASS[PER_CLASS["class"] != "all"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, d.precision, 0.4, color="#5b9bff", label="precision")
    ax.bar(x + 0.2, d.recall, 0.4, color="#4ad295", label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_person", "") for c in d["class"]], rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class results, test set (distress = Crawling, Lying)")
    ax.legend()
    dark_axes(fig, ax)
    fig.tight_layout()
    plt.close(fig)
    return fig


THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Helvetica Neue", "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Menlo", "monospace"],
).set(
    body_background_fill_dark="#0a1630",
    background_fill_primary_dark="#0f1f40",
    background_fill_secondary_dark="#132850",
    block_background_fill_dark="#0f1f40",
    block_border_color_dark="#25407a",
    border_color_primary_dark="#25407a",
    input_background_fill_dark="#0a1630",
    body_text_color_dark="#e8eefb",
    body_text_color_subdued_dark="#9fb3d9",
    block_title_text_color_dark="#cfdcf7",
    block_label_text_color_dark="#cfdcf7",
    button_primary_background_fill_dark="#2f6fed",
    button_primary_background_fill_hover_dark="#4a85f5",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill_dark="#1a3266",
    button_secondary_text_color_dark="#e8eefb",
    slider_color_dark="#4a85f5",
)

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#app-header { padding: 8px 0 4px 0; border-bottom: 1px solid #25407a; margin-bottom: 12px; }
#app-header h1 { font-weight: 700; letter-spacing: -0.01em; font-size: 1.9rem; margin: 0 0 6px 0; }
#app-header p { color: #9fb3d9; font-size: 0.98rem; line-height: 1.5; margin: 0 0 10px 0; }
"""

FORCE_DARK = "() => { document.body.classList.add('dark'); }"


with gr.Blocks(title="Search-and-rescue triage dashboard") as demo:
    gr.Markdown(
        "# Search-and-Rescue Triage Dashboard\n"
        "Decision support for finding people in distress. Each detection is passed, sent to a human "
        "reviewer, or escalated for dispatch. Distress is defined as a person lying or crawling. "
        "Model: YOLO11n fine-tuned on Roboflow search-rescue v14.",
        elem_id="app-header",
    )
    with gr.Row():
        lo = gr.Slider(0.01, 0.95, value=OP["t_low"], step=0.01,
                       label="Lower threshold (cases below it pass)")
        hi = gr.Slider(0.01, 1.0, value=OP["t_high"], step=0.01,
                       label="Upper threshold (cases above it are dispatched)")
    with gr.Accordion("Cost and workload assumptions (report defaults)", open=False):
        c_miss = gr.Slider(50, 10000, value=OP["c_miss"], step=50, label="Cost of a miss (EUR)")
        c_fa = gr.Slider(1, 50, value=OP["c_fa"], step=1, label="Cost of a false alarm (EUR)")
        base = gr.Slider(0.005, 0.2, value=OP["base"], step=0.005, label="Base rate of distress")
        rev_cost = gr.Slider(0.1, 5, value=OP["rev_cost"], step=0.1, label="Cost of one human review (EUR)")
        miss_rate = gr.Slider(0, 0.5, value=OP["hm_light"], step=0.01, label="Reviewer miss rate")
        frames = gr.Slider(100, 10000, value=OP["frames_per_day"], step=100, label="Cases per day")
        secs = gr.Slider(5, 180, value=OP["rev_sec"], step=5, label="Seconds per review")

    with gr.Tabs():
        with gr.Tab("Image inference"):
            img = gr.Image(type="pil", label="Image")
            btn = gr.Button("Run model", variant="primary")
            out_img = gr.Image(label="Model output at the current thresholds")
            out_txt = gr.Markdown()
            btn.click(predict, [img, lo, hi], [out_img, out_txt])
        with gr.Tab("Confusion matrix and cost"):
            summary = gr.Markdown()
            table = gr.Dataframe(interactive=False, label="Confusion matrix per 1,000 cases")
        with gr.Tab("Cost vs. threshold"):
            curve = gr.Plot()
        with gr.Tab("Per-class results"):
            gr.Markdown("Test split, from the saved evaluation. Distress = Crawling and Lying.")
            gr.Dataframe(PER_CLASS, interactive=False)
            gr.Plot(per_class_fig())

    inputs = [lo, hi, c_miss, c_fa, base, rev_cost, miss_rate, frames, secs]
    for component in inputs:
        component.change(refresh, inputs, [summary, table, curve])
    demo.load(refresh, inputs, [summary, table, curve])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_DARK)