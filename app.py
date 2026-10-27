"""Assignment 2 dashboard: PCB defect inspection with a human review band.

Uses only what train.py produced: weights/best.pt, artifacts/results.json and
artifacts/cost_sweep.json. Nothing is trained here. The cost maths and the
assumptions are imported from train.py, so the app and the report always agree.
"""
import json

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from train import (ARTIFACTS, BASE_RATE, BOARDS_PER_DAY, COST_MISS, COST_REVIEW,
                   DATA_DIR, IMGSZ, PER, REVIEWS_PER_INSPECTOR, THRESHOLDS, WEIGHTS,
                   outcome, share_at_or_above)

# ---------- Pipeline outputs, loaded once at startup ----------

RESULTS = json.loads((ARTIFACTS / "results.json").read_text())
SWEEP = json.loads((ARTIFACTS / "cost_sweep.json").read_text())
MODEL = YOLO(str(WEIGHTS))
CHOSEN = SWEEP["chosen"]

# Share of defective (D) and clean (F) boards scoring at or above each threshold
D = {t: share_at_or_above(RESULTS["scores"]["defect"], t) for t in THRESHOLDS}
F = {t: share_at_or_above(RESULTS["scores"]["false_alarm"], t) for t in THRESHOLDS}
PLOT_X = [t for t in THRESHOLDS if t <= 1.0]

NAVY, ACCENT, GREY = "#0F2A4A", "#2563EB", "#94A3B8"


def snap(x: float) -> float:
    """Round a slider value onto the 0.01 grid the sweep uses."""
    return round(min(max(float(x), 0.0), 1.0), 2)


def html_table(df: pd.DataFrame) -> str:
    table = df.to_html(index=False, classes="tbl", border=0, escape=False)
    return f'<div class="tbl-wrap">{table}</div>'


# ---------- Cost & decisions ----------

def cheapest(cost_miss: float, base_rate: float) -> dict:
    return min((outcome(D, F, lo, hi, cost_miss, base_rate)
                for lo in THRESHOLDS for hi in THRESHOLDS if hi >= lo),
               key=lambda o: o["cost"])


def kpi_cards(o: dict, best: dict, inspect_all: float) -> str:
    reviews_day = o["review_share"] * BOARDS_PER_DAY
    best_reviews_day = best["review_share"] * BOARDS_PER_DAY

    def delta(now, opt, fmt, judged=True):
        """Small line comparing the current value with the cost-optimal one."""
        diff = now - opt
        if abs(diff) < 1e-6:
            return '<div class="delta same">● at the cost-optimal point</div>'
        cls = ("worse" if diff > 0 else "better") if judged else "neutral"
        arrow = "▲" if diff > 0 else "▼"
        return f'<div class="delta {cls}">{arrow} {fmt(abs(diff))} vs optimal</div>'

    share_bar = (f'<div class="meter"><div style="width:{min(o["review_share"], 1) * 100:.1f}%">'
                 f'</div></div>')
    cards = [
        ("Expected cost per 1,000 boards", f"€{o['cost']:,.0f}",
         f"vs €{inspect_all:,.0f} if every board is inspected",
         delta(o["cost"], best["cost"], lambda d: f"€{d:,.0f}")),
        ("Sent to an inspector", f"{o['review_share']:.1%}",
         f"{reviews_day:,.0f} boards a day" + share_bar,
         delta(o["review_share"], best["review_share"], lambda d: f"{d:.1%}", judged=False)),
        ("Inspectors needed", f"{reviews_day / REVIEWS_PER_INSPECTOR:.1f}",
         f"vs {BOARDS_PER_DAY / REVIEWS_PER_INSPECTOR:.1f} to check every board",
         delta(reviews_day / REVIEWS_PER_INSPECTOR, best_reviews_day / REVIEWS_PER_INSPECTOR,
               lambda d: f"{d:.1f}", judged=False)),
        ("Defective boards shipped", f"{o['shipped_defects']:.2f}",
         f"per 1,000 · {o['shipped_defects'] * BOARDS_PER_DAY / PER:.0f} a day",
         delta(o["shipped_defects"], best["shipped_defects"], lambda d: f"{d:.2f}")),
    ]
    body = "".join(f'<div class="kpi"><div class="label">{l}</div>'
                   f'<div class="value">{v}</div>{dl}<div class="note">{n}</div></div>'
                   for l, v, n, dl in cards)
    return f'<div class="kpis">{body}</div>'


def cost_plot(low, high, cost_miss, base_rate, current, best):
    alone = [outcome(D, F, t, t, cost_miss, base_rate)["cost"] for t in PLOT_X]
    band_x = [t for t in PLOT_X if t <= high]
    band = [outcome(D, F, t, high, cost_miss, base_rate)["cost"] for t in band_x]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvspan(low, high, color=ACCENT, alpha=0.07, label="Inspector band")
    ax.plot(PLOT_X, alone, color=GREY, lw=2, label="Model alone (no inspectors)")
    ax.plot(band_x, band, color=ACCENT, lw=2.5,
            label=f"With inspectors (reject from {high:.2f})")
    ax.scatter([best["low"]], [best["cost"]], marker="*", s=260, color=NAVY, zorder=4,
               label=f"Cost-optimal: {best['low']:.2f} / {best['high']:.2f}")
    ax.scatter([low], [current["cost"]], s=90, facecolor="white",
               edgecolor=ACCENT, linewidth=2.5, zorder=5, label="Current setting")
    ax.set_xlabel("Lower threshold (model confidence)", color=NAVY)
    ax.set_ylabel("€ per 1,000 boards", color=NAVY)
    ax.set_ylim(0, max(alone) * 1.05)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color="#E5E9F0")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    return fig


def business_view(low, high, cost_miss, base_rate):
    low, high = snap(low), snap(max(low, high))
    o = outcome(D, F, low, high, cost_miss, base_rate)
    best = cheapest(cost_miss, base_rate)
    bad, good = PER * base_rate, PER * (1 - base_rate)
    inspect_all = PER * COST_REVIEW

    matrix = pd.DataFrame({
        "Actual board": ["<b>Defective</b>", "<b>Good</b>"],
        "Shipped": [f'<span class="err">{bad * (1 - D[low]):.1f}</span>',
                    f"{good * (1 - F[low]):.1f}"],
        "Sent to inspector": [f"{bad * (D[low] - D[high]):.1f}",
                              f"{good * (F[low] - F[high]):.1f}"],
        "Rejected automatically": [f"{bad * D[high]:.1f}",
                                   f'<span class="err">{good * F[high]:.1f}</span>'],
    })

    options = [("Do nothing: ship every board", bad * cost_miss),
               ("Inspector checks every board", inspect_all),
               ("Model alone: best single threshold",
                min(outcome(D, F, t, t, cost_miss, base_rate)["cost"] for t in THRESHOLDS)),
               ("Model + inspector band: current setting", o["cost"])]
    best_cost = min(c for _, c in options)
    alternatives = pd.DataFrame({
        "Option": [f'<b>{n}</b> <span class="tag">cheapest</span>' if c == best_cost else n
                   for n, c in options],
        "€ per 1,000 boards": [f"<b>€{c:,.0f}</b>" if c == best_cost else f"€{c:,.0f}"
                               for _, c in options],
    })

    return (kpi_cards(o, best, inspect_all), html_table(matrix), html_table(alternatives),
            cost_plot(low, high, cost_miss, base_rate, o, best))


# ---------- Spot-check a board ----------

def inspect_board(image, low, high):
    if image is None:
        return None, '<div class="hint">Upload a photo of a bare board to see the decision.</div>'
    low, high = snap(low), snap(max(low, high))
    r = MODEL.predict(image, imgsz=IMGSZ, conf=max(low, 0.01), verbose=False)[0]
    annotated = r.plot()[:, :, ::-1]  # YOLO draws in BGR, the browser wants RGB
    confs = r.boxes.conf.tolist()
    names = [MODEL.names[int(c)] for c in r.boxes.cls.tolist()]
    score = max(confs, default=0.0)  # a board is judged by its most confident defect

    if score >= high:
        badge = f'<span class="badge reject">Reject automatically</span> top confidence {score:.2f} ≥ {high:.2f}'
    elif score >= low:
        badge = f'<span class="badge review">Send to an inspector</span> top confidence {score:.2f} is between {low:.2f} and {high:.2f}'
    else:
        badge = f'<span class="badge ship">Ship</span> nothing reaches {low:.2f}'

    found = sorted(zip(names, confs), key=lambda x: -x[1])
    table = (html_table(pd.DataFrame({"Defect": [n for n, _ in found],
                                      "Confidence": [f"{c:.2f}" for _, c in found]}))
             if found else "")
    return annotated, f'<div class="verdict">{badge}</div>{table}'


# ---------- Model performance (test split) ----------

def recall_bar(v: float) -> str:
    """Bar scaled from 0.80 (empty) to 1.00 (full) so the gaps between classes show."""
    pct = max(0.0, min(1.0, (v - 0.8) / 0.2)) * 100
    return f'<div class="bar"><div class="fill" style="width:{pct:.0f}%"></div><span>{v:.3f}</span></div>'


def per_class_html() -> str:
    pc = RESULTS["finetuned"]["per_class"]
    weakest = min(pc, key=lambda c: pc[c]["recall"])
    df = pd.DataFrame([{
        "Defect": f'<b>{c}</b> <span class="tag warn">weakest</span>' if c == weakest else c,
        "Precision": f"{m['precision']:.3f}",
        "Recall": recall_bar(m["recall"]),
        "AP50": f"{m['ap50']:.3f}", "AP50-95": f"{m['ap50_95']:.3f}",
    } for c, m in pc.items()])
    return html_table(df)


def baselines_html() -> str:
    rows = []
    for name, key in [("<b>Finetuned YOLO11s</b>", "finetuned"),
                      ("Pretrained COCO, zero-shot", "coco_zero_shot"),
                      ("Trivial: predicts nothing", "trivial")]:
        r = RESULTS[key]
        rows.append({"Model": name, "mAP50": f"{r['map50']:.3f}",
                     "mAP50-95": f"{r['map50_95']:.3f}",
                     "Defects found (0.25)": f"{r['at_0.25']['defects_found']:.1%}",
                     "Boards with a false alarm (0.25)":
                         f"{r['at_0.25']['boards_with_false_alarm']:.1%}"})
    return html_table(pd.DataFrame(rows))


def confusion_html() -> str:
    labels = RESULTS["finetuned"]["confusion_labels"]
    m = RESULTS["finetuned"]["confusion_matrix"]  # rows = predicted, columns = true
    df = pd.DataFrame(m, index=[f"pred: {l}" for l in labels],
                      columns=[f"true: {l}" for l in labels]).round(0).astype(int)
    table = (df.style.background_gradient(cmap="Blues", axis=None)
             .set_table_attributes('class="tbl heat"').to_html())
    return f'<div class="tbl-wrap">{table}</div>'


# ---------- Look and feel ----------

THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate",
                       font=gr.themes.GoogleFont("Inter"))
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"
CSS = """
.gradio-container {max-width: 1400px !important; margin: auto !important; background: #FFFFFF;}

/* Header */
#header {background: #0F2A4A; border-radius: 14px; padding: 24px 30px; margin-bottom: 8px;}
#header h1 {color: #FFFFFF; font-size: 26px; font-weight: 700; margin: 0;}
#header p {color: #C9D6E8; margin: 6px 0 14px; font-size: 15px;}
#header .facts span {display: inline-block; background: rgba(255,255,255,.1); color: #FFFFFF;
    border-radius: 999px; padding: 5px 12px; margin-right: 8px; font-size: 13px;}

/* Sidebar */
#sidebar {background: #F7F9FC; border: 1px solid #E3E8EF; border-radius: 14px; padding: 20px !important;}
.section-title {font-size: 13px; font-weight: 700; color: #0F2A4A; text-transform: uppercase;
    letter-spacing: .05em; margin: 4px 0 8px;}
.rule {font-size: 13px; color: #334155; line-height: 1.9;}
.dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;}

/* KPI cards */
.kpis {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 6px;}
.kpi {background: #FFFFFF; border: 1px solid #E3E8EF; border-top: 4px solid #2563EB;
    border-radius: 14px; padding: 16px 18px; transition: box-shadow .2s;}
.kpi:hover {box-shadow: 0 6px 18px rgba(15,42,74,.08);}
.kpi .label {font-size: 12px; color: #5B6B7F; text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {font-size: 30px; font-weight: 700; color: #0F2A4A; margin-top: 4px;
    font-variant-numeric: tabular-nums;}
.kpi .note {font-size: 12px; color: #5B6B7F; margin-top: 4px;}
.delta {font-size: 12px; font-weight: 600; margin-top: 4px;}
.delta.better, .delta.same {color: #1B7A3A;}
.delta.worse {color: #B42318;}
.delta.neutral {color: #5B6B7F;}
.meter {height: 6px; background: #E5E9F0; border-radius: 999px; margin-top: 8px; overflow: hidden;}
.meter div {height: 100%; background: #2563EB; border-radius: 999px;}

/* Tabs */
#main-tabs button[role="tab"] {font-size: 16px !important; font-weight: 600 !important;
    padding: 12px 20px !important; color: #334155 !important;}
#main-tabs button[role="tab"][aria-selected="true"] {color: #2563EB !important;}

/* Tables */
.tbl-wrap {border: 1px solid #E3E8EF; border-radius: 12px; overflow: hidden; margin: 2px 0 20px;
    box-shadow: 0 1px 3px rgba(15,42,74,.05);}
.tbl {width: 100%; border-collapse: collapse; font-size: 14px; margin: 0 !important;}
.tbl th {background: #F1F4F8; color: #0F2A4A; text-align: left; padding: 10px 14px;
    font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
    border-bottom: 1px solid #E3E8EF;}
.tbl td {padding: 10px 14px; border-bottom: 1px solid #EEF1F5; color: #1E293B;
    font-variant-numeric: tabular-nums;}
.tbl td:not(:first-child), .tbl th:not(:first-child) {text-align: right;}
.tbl tbody tr:nth-child(even) td {background: #FAFBFD;}
.tbl tbody tr:hover td {background: #EEF4FF;}
.tbl tbody tr:last-child td {border-bottom: none;}
.heat td, .heat th {text-align: center !important; font-size: 12px; padding: 8px 6px;}
.err {color: #B42318; font-weight: 700;}
.tag {display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px;
    border-radius: 999px; background: #E7F6EC; color: #1B7A3A; margin-left: 6px;}
.tag.warn {background: #FDE8E8; color: #B42318;}
.bar {position: relative; height: 22px; background: #EEF1F5; border-radius: 6px;
    min-width: 150px; text-align: right;}
.bar .fill {position: absolute; left: 0; top: 0; bottom: 0; background: #BFD3FB; border-radius: 6px;}
.bar span {position: relative; padding-right: 8px; line-height: 22px;}
.caption {font-size: 13px; color: #5B6B7F; margin: -10px 0 18px;}

/* Spot-check */
.badge {display: inline-block; padding: 7px 16px; border-radius: 999px; font-weight: 700;
    margin-right: 10px;}
.ship {background: #E7F6EC; color: #1B7A3A;}
.review {background: #FFF4E0; color: #9A5B00;}
.reject {background: #FDE8E8; color: #B42318;}
.verdict {font-size: 15px; color: #334155; margin: 6px 0 14px;}
.hint {color: #5B6B7F; font-size: 14px;}
"""

GR6 = int(gr.__version__.split(".")[0]) >= 6
STYLE = dict(theme=THEME, css=CSS, js=FORCE_LIGHT)

ft = RESULTS["finetuned"]
HEADER = f"""
<div id="header">
  <h1>PCB Defect Inspection · Decision Dashboard</h1>
  <p>Which boards ship, which go to an inspector, and which are rejected, and what each choice costs.</p>
  <div class="facts">
    <span>Test mAP50 {ft['map50']:.3f}</span>
    <span>{ft['at_0.25']['defects_found']:.1%} of defects found</span>
    <span>{BOARDS_PER_DAY:,} boards / day</span>
  </div>
</div>
"""
RULE = """
<div class="rule">
  <span class="dot" style="background:#1B7A3A"></span>Below the lower threshold → <b>ship</b><br>
  <span class="dot" style="background:#9A5B00"></span>Between the two → <b>inspector</b><br>
  <span class="dot" style="background:#B42318"></span>From the upper threshold → <b>reject</b>
</div>
"""

# ---------- Layout ----------

with gr.Blocks(title="PCB Defect Inspection", **({} if GR6 else STYLE)) as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300, elem_id="sidebar"):
            gr.HTML('<div class="section-title">Decision thresholds</div>')
            gr.HTML(RULE)
            low = gr.Slider(0, 1, value=CHOSEN["low"], step=0.01,
                            label="Lower threshold: send to inspector from")
            high = gr.Slider(0, 1, value=CHOSEN["high"], step=0.01,
                             label="Upper threshold: reject automatically from")
            reset = gr.Button("Reset to cost-optimal", variant="primary")
            gr.HTML('<div class="section-title" style="margin-top:18px">Assumptions</div>')
            cost_miss = gr.Slider(10, 1000, value=COST_MISS, step=10,
                                  label="Cost of shipping a defective board (€)")
            base_rate = gr.Slider(0.01, 0.20, value=BASE_RATE, step=0.01,
                                  label="Share of boards that are defective")

        with gr.Column(scale=3):
            kpis = gr.HTML()
            with gr.Tabs(elem_id="main-tabs"):
                with gr.Tab("Cost & decisions"):
                    curve = gr.Plot(show_label=False)
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="section-title">What happens to 1,000 boards</div>')
                            matrix = gr.HTML()
                            gr.HTML('<div class="caption">Red: defects shipped and good boards thrown away.</div>')
                        with gr.Column():
                            gr.HTML('<div class="section-title">Against the alternatives</div>')
                            alternatives = gr.HTML()

                with gr.Tab("Spot-check a board"):
                    with gr.Row():
                        image_in = gr.Image(type="pil", label="Board photo", height=420)
                        image_out = gr.Image(label="Defects found", height=420)
                    verdict = gr.HTML()
                    samples = sorted((DATA_DIR / "test" / "images").glob("*.jpg"))[:4] \
                        if (DATA_DIR / "test" / "images").exists() else []
                    if samples:
                        gr.Examples([[str(p)] for p in samples], inputs=image_in,
                                    label="Test boards")

                with gr.Tab("Model performance"):
                    gr.HTML(f'<div class="caption" style="margin-top:6px">Test split: '
                            f'{RESULTS["test_images"]} boards, {RESULTS["test_defects"]} defects. '
                            f'Positive class: defect.</div>')
                    gr.HTML('<div class="section-title">Per defect type</div>')
                    gr.HTML(per_class_html())
                    gr.HTML('<div class="caption">Recall bars run from 0.80 (empty) to 1.00 (full).</div>')
                    gr.HTML('<div class="section-title">Against the baselines</div>')
                    gr.HTML(baselines_html())
                    gr.HTML('<div class="section-title">Confusion matrix (counts)</div>')
                    gr.HTML(confusion_html())

    business_inputs = [low, high, cost_miss, base_rate]
    business_outputs = [kpis, matrix, alternatives, curve]
    for control in business_inputs:
        control.change(business_view, business_inputs, business_outputs)
    for control in (image_in, low, high):
        control.change(inspect_board, [image_in, low, high], [image_out, verdict])
    reset.click(lambda: (CHOSEN["low"], CHOSEN["high"], COST_MISS, BASE_RATE),
                outputs=business_inputs)
    demo.load(business_view, business_inputs, business_outputs)
    demo.load(inspect_board, [image_in, low, high], [image_out, verdict])


if __name__ == "__main__":
    demo.launch(**(STYLE if GR6 else {}))