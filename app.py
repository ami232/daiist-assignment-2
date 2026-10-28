"""
Assignment 2 dashboard: Smoke & Fire Triage.

Run with `uv run python main.py app`. It reads only what train.py wrote to
weights/ (best.pt plus the JSON/CSV artifacts), never trains, never downloads,
and never needs the dataset. Every number on the Decide tab is recomputed with
numpy from weights/test_scores.csv using the same cost functions as train.py.
"""

import html
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw, ImageFont

from train import (derived, describe_policy, optimise, policy_cost, share_at_or_above,
                   sweep_thresholds, three_way_rates)

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
REQUIRED = ["best.pt", "run_meta.json", "dataset_stats.json", "test_scores.csv",
            "eval.json", "baselines.json", "decision.json"]
PRED_CONF = 0.001
MAX_BOXES_DRAWN = 50

# The three outcomes are the only colours on the page; everything else is ink on paper.
PASS_C, REVIEW_C, ALARM_C = "#1A8A6B", "#BF8700", "#D0352E"
INK, INK_2, MUTED, REFERENCE = "#1C1B19", "#55534E", "#8A8781", "#CFCCC4"
LINE, GRID, SURFACE, PAPER = "#E3E1DA", "#EFEDE7", "#FFFFFF", "#FAFAF8"
FONT = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"
OUTCOMES = [("Pass", PASS_C), ("Review", REVIEW_C), ("Alarm", ALARM_C)]
VIEWS = ["Per 1,000 frames at base rate", "Raw test-set counts"]
WHATIF_KEYS = ("cost_fn", "cost_fp", "base_rate", "review_seconds")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def load_artifacts():
    missing = [name for name in REQUIRED if not (WEIGHTS / name).is_file()]
    if missing:
        return None, missing
    art = {name.split(".")[0]: json.loads((WEIGHTS / name).read_text())
           for name in REQUIRED if name.endswith(".json")}
    art["scores"] = pd.read_csv(WEIGHTS / "test_scores.csv")
    return art, []


ART, MISSING = load_artifacts()
if ART:
    DEC = ART["decision"]
    REPORT_D = DEC["decisions"]
    SCORES = ART["scores"]
    _y, _s = SCORES.y_true.to_numpy(), SCORES.s.to_numpy()
    POS, NEG = np.sort(_s[_y == 1]), np.sort(_s[_y == 0])
    N_POS, N_NEG = POS.size, NEG.size
    CURVE_T = sweep_thresholds(_s)
    CLASS_NAMES = ART["eval"]["class_names"]


# ---------------------------------------------------------------------------
# Pure computation (cost maths lives in train.py and is shared)
# ---------------------------------------------------------------------------
def scenario_decisions(cost_fn, cost_fp, base_rate, review_seconds):
    D = dict(REPORT_D)
    D.update(cost_fn=float(cost_fn), cost_fp=float(cost_fp),
             base_rate=min(max(float(base_rate), 1e-6), 0.999), review_seconds=float(review_seconds))
    return D


@lru_cache(maxsize=64)
def scenario(cost_fn, cost_fp, base_rate, review_seconds):
    """Everything that depends on DECISIONS but not on the sliders."""
    D = scenario_decisions(cost_fn, cost_fp, base_rate, review_seconds)
    o = optimise(POS, NEG, D, CURVE_T)
    f = o["feasible_idx"]
    lo_f, hi_f = float(o["grid_lo"][f]), float(o["grid_hi"][f])
    t_star = o["t_star"]
    tpr, fpr = float(share_at_or_above(POS, t_star)), float(share_at_or_above(NEG, t_star))
    kw = {"n_pos": N_POS, "n_neg": N_NEG}
    policies = [
        describe_policy("(a) Do nothing", 0, 0, 0, 0, D, **kw),
        describe_policy("(b) Review everything", 0, 1, 0, 1, D, lo=0.0, hi=1.01, **kw),
        describe_policy(f"(c) Model alone, t* = {t_star:.3f}", tpr, 0, fpr, 0, D, lo=t_star, hi=t_star, **kw),
        describe_policy(f"(d) Three-way ({lo_f:.2f}, {hi_f:.2f})", *three_way_rates(POS, NEG, lo_f, hi_f), D,
                        lo=lo_f, hi=hi_f, **kw),
    ]
    return {"D": D, "t_star": t_star, "lo": lo_f, "hi": hi_f, "curve_cost": 1000 * o["cost"],
            "policies": policies, "recommended": o["recommended"]}


def current_policy(lo, hi, sc):
    return describe_policy(f"Current ({thr(lo)}, {thr(hi)})", *three_way_rates(POS, NEG, lo, hi), sc["D"],
                           lo=lo, hi=hi, n_pos=N_POS, n_neg=N_NEG)


def best_feasible(sc):
    return min((p for p in sc["policies"] if p["feasible"]), key=lambda p: p["cost_per_1000"], default=None)


def decide_frame(s, lo, hi):
    return "Alarm" if s >= hi else "Review" if s >= lo else "Pass"


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def eur(x):
    return f"€{x:,.0f}" if abs(x) >= 100 else f"€{x:,.2f}"


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def thr(x):
    """Slider thresholds have two decimals; t* can have three."""
    return f"{x:.2f}" if abs(x - round(x, 2)) < 1e-9 else f"{x:.3f}"


def sentence(text):
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def table(header, rows, left=1):
    """Leading `left` columns are labels; the rest are figures and align right."""
    cls = lambda i: ' class="l"' if i < left else ""
    head = "".join(f"<th{cls(i)}>{html.escape(str(h))}</th>" for i, h in enumerate(header))
    body = "".join("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(row)) + "</tr>" for row in rows)
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def panel(title, body, note=""):
    note = f'<p class="note">{note}</p>' if note else ""
    return f'<section class="panel"><div class="panel-head"><h3>{title}</h3>{note}</div>{body}</section>'


def stat_strip(cells):
    """cells: (label, value, [sub lines], extra css class)."""
    return '<div class="kpis">' + "".join(
        f'<div class="kpi{extra}"><div class="label">{label}</div><div class="value">{value}</div>'
        + "".join(f'<div class="sub">{s}</div>' for s in subs) + "</div>"
        for label, value, subs, extra in cells) + "</div>"


def style(fig, height, title, subtitle=""):
    fig.update_layout(
        template="none", height=height, margin=dict(l=64, r=24, t=76 if subtitle else 52, b=48),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, showlegend=False,
        font=dict(family=FONT, size=12, color=INK),
        title=dict(text=title, x=0, xref="container", xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(l=16, t=16), font=dict(size=14, color=INK, weight=600),
                   subtitle=dict(text=subtitle, font=dict(size=12, color=INK_2))),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(family=FONT, color=INK, size=12)),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=REFERENCE, activecolor=INK),
    )
    axis = dict(showgrid=False, zeroline=False, showline=True, linecolor=LINE, ticks="outside", ticklen=4,
                tickcolor=LINE, tickfont=dict(color=MUTED, size=11), title_font=dict(color=INK_2, size=12),
                title_standoff=12)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def kpi_html(cur, sc):
    D, best = sc["D"], best_feasible(sc)
    fte = D["available_reviewer_fte"]
    over = cur["fte_needed"] > fte + 1e-9
    gap = cur["cost_per_1000"] - best["cost_per_1000"] if best else 0.0
    vs = ("At the recommended operating point" if abs(gap) < 0.5 else
          f'{eur(abs(gap))} {"above" if gap > 0 else "below"} recommended')
    return stat_strip([
        ("Expected cost per 1,000 frames", eur(cur["cost_per_1000"]),
         [f'{eur(cur["cost_per_day"])} per day', vs], ""),
        ("Sent to human review", pct(cur["review_share"], 2),
         [f'{cur["review_frames_per_day"]:,.0f} frames per day'], ""),
        ("Reviewers needed", f'{cur["fte_needed"]:.1f}<small>FTE</small>',
         [f'<span class="flag">Over capacity, {fte:g} available</span>' if over else f"{fte:g} available"],
         " over" if over else ""),
        ("Missed fires per 1,000 fire frames", f'{cur["missed_per_1000_true_frames"]:.1f}',
         ["After human review"], ""),
    ])


def scale_html(lo, hi, mark=None):
    """The 0 to 1 frame-score axis split into pass, review and alarm, optionally pinned at one score."""
    segs = [(0.0, lo, PASS_C), (lo, hi, REVIEW_C), (hi, 1.0, ALARM_C)]
    track = "".join(f'<i style="width:{100 * (b - a):.2f}%;background:{c}"></i>' for a, b, c in segs if b > a)
    pin = "" if mark is None else f'<b class="pin" style="left:{100 * min(max(mark, 0.0), 1.0):.2f}%"></b>'
    return (f'<div class="scale"><div class="track">{track}</div>{pin}</div>'
            '<div class="scale-ends"><span>0</span><span>frame score</span><span>1</span></div>')


def zones_html(lo, hi, sc):
    rows = [("Pass", PASS_C, f"below {thr(lo)}"),
            ("Review", REVIEW_C, f"{thr(lo)} to {thr(hi)}" if hi > lo else "none, the model decides alone"),
            ("Alarm", ALARM_C, f"{thr(hi)} and above")]
    keys = "".join(f'<li><span class="sw" style="background:{c}"></span><span class="k">{n}</span>'
                   f'<span class="v">{r}</span></li>' for n, c, r in rows)
    return (scale_html(lo, hi) + f'<ul class="zones">{keys}</ul>'
            f'<p class="note">Recommended: review from {sc["lo"]:.2f}, alarm from {sc["hi"]:.2f}. '
            f'Model-alone optimum t* = {sc["t_star"]:.3f}.</p>')


def matrix_fig(cur, view):
    raw = view.startswith("Raw")
    m = cur["matrix_test_counts"] if raw else cur["matrix_per_1000"]
    rows = [("Fire / smoke", m["fire_smoke"]), ("Clear", m["clear"])]
    fig = go.Figure()
    for i, (row_name, vals) in enumerate(rows):
        total = sum(vals) or 1
        for j, (name, colour) in enumerate(OUTCOMES):
            share = vals[j] / total
            fig.add_shape(type="rect", x0=j - 0.49, x1=j + 0.49, y0=i - 0.47, y1=i + 0.47, line_width=0,
                          fillcolor=colour, opacity=0.05 + 0.45 * share, layer="below")
            text = f"{vals[j]:,}" if raw else (f"{vals[j]:,.1f}" if vals[j] >= 1 else f"{vals[j]:.3f}")
            fig.add_annotation(x=j, y=i, text=f"<b>{text}</b><br><span style='font-size:11px;color:{INK_2}'>"
                                              f"{pct(share)} of row</span>",
                               showarrow=False, font=dict(size=15, color=INK))
    hover_x, hover_y, hover_t = [], [], []
    for i, (row_name, vals) in enumerate(rows):
        for j, (name, _) in enumerate(OUTCOMES):
            hover_x.append(j), hover_y.append(i)
            hover_t.append(f"Actual {row_name.lower()}, outcome {name.lower()}<br>{vals[j]:,.3f}"
                           + (" test frames" if raw else " per 1,000 frames"))
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=60, opacity=0),
                             hovertext=hover_t, hoverinfo="text"))
    style(fig, 300, "Outcome by actual class",
          "Raw test-set frames" if raw else f"Expected frames per 1,000 at base rate {cur_base_rate(cur):g}")
    fig.update_xaxes(tickvals=[0, 1, 2], ticktext=[n for n, _ in OUTCOMES], range=[-0.52, 2.52], showline=False,
                     ticks="", side="top", tickfont=dict(size=12, color=INK))
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Fire / smoke", "Clear"], range=[1.52, -0.52],
                     showline=False, ticks="", tickfont=dict(size=12, color=INK))
    fig.update_layout(margin=dict(l=96, r=16, t=104, b=16))
    return fig


def cur_base_rate(cur):
    m = cur["matrix_per_1000"]
    return round(sum(m["fire_smoke"]) / 1000, 6)


def final_html(cur, D):
    f = cur["final_per_1000"]
    fmt = lambda x: f"{x:,.3f}" if x < 10 else f"{x:,.1f}"
    cell = lambda tag, x: f'<span class="tag">{tag}</span>{fmt(x)}'
    return panel("After human review, per 1,000 frames", table(["Actual", "Escalated", "Not escalated"],
                                             [["Fire / smoke", cell("TP", f["TP"]), cell("FN", f["FN"])],
                                              ["Clear", cell("FP", f["FP"]), cell("TN", f["TN"])]]),
                 f"Reviewers escalate {pct(D['human_recall'], 0)} of the real fires they see and "
                 f"{pct(D['human_fpr'], 0)} of the clear frames. Alarms escalate directly.")


def curve_fig(sc, lo, hi, cur):
    pol = {p["policy"][:3]: p for p in sc["policies"]}
    fig = go.Figure()
    if hi > lo:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=REVIEW_C, opacity=0.10, line_width=0, layer="below",
                      annotation_text="review band", annotation_position="top left",
                      annotation_font=dict(color=INK_2, size=11))
    fig.add_trace(go.Scatter(x=CURVE_T, y=sc["curve_cost"], mode="lines", line=dict(color=INK_2, width=1.75),
                             hovertemplate="t = %{x:.3f}<br>€%{y:,.2f} per 1,000<extra>model alone</extra>"))
    for key, dash, label in (("(a)", "dash", "do nothing"), ("(b)", "dot", "review everything")):
        y = pol[key]["cost_per_1000"]
        fig.add_hline(y=y, line=dict(color=MUTED, dash=dash, width=1))
        # log axis: annotations take log10(y); add_hline's own annotation mis-places them
        fig.add_annotation(x=0.99, y=np.log10(max(y, 1e-9)), xref="x", yref="y", text=f"{label} {eur(y)}",
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=INK_2, size=11))
    ts, cs = sc["t_star"], pol["(c)"]["cost_per_1000"]
    fig.add_vline(x=ts, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=[ts], y=[cs], mode="markers+text",
                             marker=dict(color=INK_2, size=9, line=dict(color=SURFACE, width=2)),
                             text=[f"t* = {ts:.3f}"], textposition="bottom right",
                             textfont=dict(color=INK, size=12),
                             hovertemplate=f"t* = {ts:.3f}<br>{eur(cs)} per 1,000<extra>model alone</extra>"))
    cc = cur["cost_per_1000"]
    fig.add_trace(go.Scatter(x=[lo, hi], y=[cc, cc], mode="lines+markers", line=dict(color=INK, width=2.5),
                             marker=dict(color=INK, size=8, line=dict(color=SURFACE, width=2)),
                             hovertemplate=f"current ({thr(lo)}, {thr(hi)})<br>"
                                           f"{eur(cc)} per 1,000<extra></extra>"))
    if hi > lo:  # with no review band the current policy sits on the curve, already labelled
        fig.add_annotation(x=(lo + hi) / 2, y=np.log10(max(cc, 1e-9)), text=f"current {eur(cc)}",
                           showarrow=False, yanchor="top", yshift=-8, font=dict(color=INK, size=12))
    style(fig, 400, "Expected cost against threshold",
          "Grey curve: the model alone, alarming at every frame scoring t or more. "
          "Black: the current three-way policy across its review band.")
    fig.update_xaxes(title_text="threshold t on frame score", range=[0, 1], dtick=0.1)
    fig.update_yaxes(title_text="€ per 1,000 frames (log)", type="log", dtick=1, tickprefix="€",
                     tickformat=",.0f", showgrid=True, gridcolor=GRID)
    fig.update_layout(margin=dict(l=88))
    return fig


def policy_fig(sc, cur):
    items = sc["policies"] + [cur]
    names = [p["policy"] for p in items]
    vals = [p["cost_per_1000"] for p in items]
    colours = [REFERENCE] * (len(items) - 1) + [INK]
    hover = [f"{p['policy']}<br>{eur(p['cost_per_1000'])} per 1,000, {eur(p['cost_per_day'])} per day<br>"
             f"review share {pct(p['review_share'], 2)}, {p['fte_needed']:.1f} FTE"
             f"{'' if p['feasible'] else ', over capacity'}" for p in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", width=0.56,
                           marker=dict(color=colours, cornerradius=3),
                           text=[eur(v) + ("" if p["feasible"] else ", over capacity") for v, p in zip(vals, items)],
                           textposition="outside", textfont=dict(color=INK, size=12), hovertext=hover,
                           hoverinfo="text", cliponaxis=False))
    style(fig, 320, "Policy comparison",
          "Expected cost per 1,000 frames, including reviewer staffing. Black is the current setting.")
    fig.update_xaxes(title_text="€ per 1,000 frames", tickprefix="€", tickformat=",.0f", rangemode="tozero",
                     showgrid=True, gridcolor=GRID, showline=False)
    fig.update_yaxes(autorange="reversed", showline=False, ticks="", tickfont=dict(color=INK, size=12))
    fig.update_layout(margin=dict(l=216, r=150, t=76, b=48))
    return fig


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def update_decide(lo, hi, view, *whatif):
    sc = scenario(*whatif)
    lo, hi = float(lo), float(max(lo, hi))
    cur = current_policy(lo, hi, sc)
    return (kpi_html(cur, sc), zones_html(lo, hi, sc), matrix_fig(cur, view), final_html(cur, sc["D"]),
            curve_fig(sc, lo, hi, cur), policy_fig(sc, cur))


def clamp_lo(lo, hi):
    return gr.update(value=max(lo, hi))


def clamp_hi(lo, hi):
    return gr.update(value=min(lo, hi))


def reset_recommended(*whatif):
    sc = scenario(*whatif)
    return sc["lo"], sc["hi"], False


def model_only(checked, lo, hi, *whatif):
    sc = scenario(*whatif)
    if checked:
        return sc["t_star"], sc["t_star"]
    return sc["lo"], sc["hi"]


def reset_whatif():
    return tuple(REPORT_D[k] for k in WHATIF_KEYS)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(WEIGHTS / "best.pt"))
    return _MODEL


def detect(image):
    if image is None:
        return []
    result = get_model().predict(image, conf=PRED_CONF, verbose=False)[0]
    names = result.names
    return [{"cls": names[int(c)], "conf": float(cf), "xyxy": [float(v) for v in b]}
            for c, cf, b in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                result.boxes.xyxy.cpu().numpy())]


def class_colour(name):
    n = name.lower()
    return ALARM_C if ("fire" in n or "flame" in n) else INK_2 if "smoke" in n else MUTED


EMPTY_VERDICT = ('<div class="verdict empty">Upload a camera frame. It is scored against the operating point '
                 'on the left and rescored whenever the thresholds move.</div>')


def render_frame(image, dets, lo, hi):
    if image is None:
        return None, EMPTY_VERDICT
    lo, hi = float(lo), float(max(lo, hi))
    s_fire = max([d["conf"] for d in dets if "fire" in d["cls"].lower() or "flame" in d["cls"].lower()],
                 default=0.0)
    s_smoke = max([d["conf"] for d in dets if "smoke" in d["cls"].lower()], default=0.0)
    s = max(s_fire, s_smoke)
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(3, round(min(img.size) / 160))
    try:
        font = ImageFont.load_default(size=max(12, round(min(img.size) / 40)))
    except TypeError:
        font = ImageFont.load_default()
    shown = sorted((d for d in dets if d["conf"] >= lo), key=lambda d: -d["conf"])[:MAX_BOXES_DRAWN]
    for d in reversed(shown):
        colour = class_colour(d["cls"])
        x0, y0, x1, y1 = d["xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        label = f" {d['cls']} {d['conf']:.2f} "
        tb = draw.textbbox((x0, y0), label, font=font)
        ty = y0 - (tb[3] - tb[1]) - 4 if y0 - (tb[3] - tb[1]) - 4 > 0 else y0
        tb = draw.textbbox((x0, ty), label, font=font)
        draw.rectangle([tb[0], tb[1] - 2, tb[2], tb[3] + 2], fill=colour)
        draw.text((x0, ty), label, fill="#FFFFFF", font=font)
    outcome = decide_frame(s, lo, hi)
    facts = [("Frame score s", f"{s:.3f}"), ("Fire score", f"{s_fire:.3f}"), ("Smoke score", f"{s_smoke:.3f}"),
             ("Boxes drawn", f"{len(shown)}")]
    meta = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
    verdict = (f'<div class="verdict" style="--c:{dict(OUTCOMES)[outcome]}">'
               f'<div class="verdict-main"><div class="label">Decision</div>'
               f'<div class="outcome"><span class="sw"></span>{outcome}</div></div>'
               f'<div class="verdict-body"><dl>{meta}</dl>{scale_html(lo, hi, s)}</div></div>')
    return img, verdict


# ---------------------------------------------------------------------------
# Static content (header, Model and Assumptions tabs)
# ---------------------------------------------------------------------------
def header_html():
    meta = ART["run_meta"]
    facts = [f'{html.escape(meta["model"].removesuffix(".pt"))}, {meta["epochs_completed"]} epochs',
             f'{ART["dataset_stats"]["splits"]["test"]["images"]:,} test frames',
             f'{REPORT_D["n_cameras"]:,} cameras, {derived(REPORT_D)["frames_per_day"]:,} frames a day']
    return ('<header class="top"><div><h1>Smoke &amp; Fire Triage</h1>'
            '<p>Per-frame triage for fixed cameras: pass, send to a person, or raise an alarm. Figures are '
            'expected values at the production base rate, computed from the test-split scores.</p></div>'
            '<ul class="facts">' + "".join(f"<li>{f}</li>" for f in facts) + "</ul></header>")


def model_tab_html():
    ev, base, stats = ART["eval"], ART["baselines"], ART["dataset_stats"]
    inst = stats["splits"]["test"]["instances"]
    ov = ev["overall"]
    overall = stat_strip([(label, num(ov.get(key)), [sub], "") for label, key, sub in (
        ("mAP50", "mAP50", "Test split, all classes"), ("mAP50-95", "mAP50-95", "Test split, all classes"),
        ("Precision", "precision", "At the best-F1 confidence"), ("Recall", "recall", "At the best-F1 confidence"))])
    per_class = table(["Class", "Precision", "Recall", "AP50", "AP50-95", "Test instances"],
                      [[html.escape(c["class"]), num(c["precision"]), num(c["recall"]),
                        num(c["AP50"]), num(c["AP50-95"]), f"{inst.get(c['class'], c['test_instances']):,}"]
                       for c in ev["per_class"]])
    op = DEC["recommended_operating_point"]
    frame_rows = []
    for c in DEC["frame_level_per_class"]:
        for key, label in (("at_t_star", f"Model alone, s ≥ t* ({DEC['t_star']:.3f})"),
                           ("flagged_at_lo", f"Flagged, s ≥ lo ({op['lo']:.2f})"),
                           ("alarm_at_hi", f"Alarm, s ≥ hi ({op['hi']:.2f})")):
            r = c[key]
            frame_rows.append([html.escape(c["class"]), label, num(r["precision"]),
                               num(r["recall"]), num(r["fpr"]), f"{r['TP']} / {r['FN']} / {r['FP']} / {r['TN']}"])
    frame_tbl = table(["Class", "Rule", "Precision", "Recall", "FPR", "TP / FN / FP / TN"], frame_rows, left=2)
    bl = [("Finetuned, model alone at t*", base["finetuned"]), ("COCO zero-shot", base["coco_zero_shot"]),
          ("Predict nothing", base["predict_nothing"])]
    if "always_alarm" in base:
        bl.append(("Always alarm", base["always_alarm"]))
    base_tbl = table(["Model or policy", "mAP50", "Frame recall", "Frame FPR", "Cost per 1,000"],
                     [[n, num(b["mAP50"]), pct(b["frame_recall"]), pct(b["frame_fpr"]),
                       eur(b["cost_per_1000"])] for n, b in bl])
    coco = base["coco_zero_shot"]
    top = lambda key: ", ".join(f"{html.escape(n)} ({c})" for n, c in coco[key]) or "none"
    coco_note = (f"COCO zero-shot top classes on positive frames: {top('top_classes_positive_frames')}. "
                 f"On negative frames: {top('top_classes_negative_frames')}.")
    return (overall +
            panel("Detection by class", per_class, f"Test split. {sentence(ev['per_class_note'])}") +
            panel("Frame-level decisions by class", frame_tbl, sentence(DEC["frame_level_note"])) +
            panel("Baselines on the same test split", base_tbl, coco_note))


def detection_cm_fig():
    cm = ART["eval"]["confusion_matrix"]
    m = np.array(cm["matrix"])
    labels = cm["labels"]
    fig = go.Figure(go.Heatmap(z=np.log1p(m), x=labels, y=labels, colorscale=[[0, SURFACE], [1, "#3A3935"]],
                               showscale=False, customdata=m, xgap=2, ygap=2,
                               hovertemplate="true %{y}, predicted %{x}: %{customdata:,}<extra></extra>"))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            fig.add_annotation(x=labels[j], y=labels[i], text=f"{m[i, j]:,}", showarrow=False,
                               font=dict(color=SURFACE if np.log1p(m[i, j]) > 0.6 * np.log1p(m.max()) else INK,
                                         size=13))
    style(fig, 380, "Detection confusion matrix",
          f"Test split, boxes with confidence of at least {cm['conf']}. Shading is log-scaled.")
    fig.update_xaxes(title_text="predicted", showline=False, ticks="", tickfont=dict(color=INK, size=12))
    fig.update_yaxes(title_text="true", autorange="reversed", showline=False, ticks="",
                     tickfont=dict(color=INK, size=12))
    fig.update_layout(margin=dict(l=96, r=24, t=76, b=56))
    return fig


def dataset_html():
    stats = ART["dataset_stats"]
    sp, names = stats["splits"], stats["class_names"]
    split_tbl = table(["Split", "Images", "Background", "Fire-or-smoke prevalence"] + names,
                      [[s, f"{v['images']:,}", f"{v['background_images']:,} ({v['background_pct']:.1f}%)",
                        pct(v["positive_prevalence"])] + [f"{v['instances'][n]:,}" for n in names]
                       for s, v in sp.items()])
    small = pct(stats["small_box_threshold"], 0)
    box_tbl = table(["Class", "Median box area", f"Share under {small} of image"],
                    [[n, pct(a["all"].get("median"), 2), pct(a["all"].get("small_share"))]
                     for n, a in stats["box_area_relative"].items()])
    low = stats["low_count_classes_test"]
    warn = (f'<span class="flag">Fewer than {stats["low_count_threshold"]} test instances: {", ".join(low)}. '
            f'Treat those per-class numbers as unreliable.</span>' if low else
            f'Every class has at least {stats["low_count_threshold"]} test instances.')
    src = stats["source"]
    return (panel("Splits", split_tbl, f'Roboflow Universe {html.escape(src["workspace"])}/'
                  f'{html.escape(src["project"])}, version {src["version"]}, licence {html.escape(str(src["licence"]))}.')
            + panel("Box size relative to the image", box_tbl, warn))


def assumptions_html():
    D = REPORT_D
    rows = [[f"<code>{k}</code>", f"{v:,}" if isinstance(v, (int, float)) else html.escape(str(v))]
            for k, v in D.items()]
    rows.append(["<code>cost_review</code>, derived", f"€{derived(D)['cost_review']:.4f} per reviewed frame"])
    rows.append(["<code>frames_per_day</code>, derived", f"{derived(D)['frames_per_day']:,}"])
    b = DEC["bootstrap"]
    ci = lambda x, f=eur: f"{f(x['mean'])} <span class=\"tag\">95% CI {f(x['lo95'])} to {f(x['hi95'])}</span>"
    boot = table(["Quantity", "Bootstrap estimate"], [
        [f"Cost per 1,000, model alone at t* = {DEC['t_star']:.3f}", ci(b["model_alone_cost_per_1000"])],
        [f"Cost per 1,000, three-way ({DEC['recommended_operating_point']['lo']:.2f}, "
         f"{DEC['recommended_operating_point']['hi']:.2f})", ci(b["three_way_cost_per_1000"])],
        ["Frame recall at t*", ci(b["model_alone_recall"], pct)],
    ])
    return (panel("Decision inputs used in training", table(["Input", "Value"], rows),
                  "Read-only. These are the values the report and the recommended thresholds rest on.") +
            panel("Uncertainty", boot, f"{b['n_resamples']} stratified bootstrap resamples of the test split."))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def light_only(theme):
    """Gradio follows the OS dark mode; giving dark mode the light values keeps the page light everywhere."""
    for key in [k for k in vars(theme) if k.endswith("_dark")]:
        if hasattr(theme, key[:-5]):
            setattr(theme, key, getattr(theme, key[:-5]))
    return theme


THEME = light_only(gr.themes.Base(
    primary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill=PAPER, body_text_color=INK, body_text_color_subdued=INK_2,
    background_fill_primary=SURFACE, background_fill_secondary=PAPER,
    border_color_primary=LINE, border_color_accent=INK, color_accent=INK, color_accent_soft=GRID,
    block_background_fill=SURFACE, block_border_color=LINE, block_border_width="1px", block_shadow="none",
    block_radius="6px", block_title_text_color=INK, block_title_text_weight="500", block_label_text_color=INK_2,
    block_info_text_color=MUTED, block_label_background_fill=SURFACE, block_label_border_color=LINE,
    block_label_shadow="none", panel_background_fill=SURFACE, panel_border_color=LINE,
    input_background_fill=SURFACE, input_border_color=LINE, input_border_color_focus=INK,
    input_border_width="1px", button_border_width="1px",
    input_shadow="none", input_shadow_focus="none",
    button_primary_background_fill=INK, button_primary_background_fill_hover="#000000",
    button_primary_text_color=SURFACE, button_primary_border_color=INK, button_primary_shadow="none",
    button_secondary_background_fill=SURFACE, button_secondary_background_fill_hover=PAPER,
    button_secondary_border_color=LINE, button_secondary_text_color=INK, button_secondary_shadow="none",
    slider_color=INK, loader_color=MUTED,
    checkbox_background_color_selected=INK, checkbox_border_color_selected=INK, checkbox_border_color=REFERENCE,
    checkbox_label_background_fill=SURFACE, checkbox_label_background_fill_hover=PAPER,
    checkbox_label_background_fill_selected=SURFACE, checkbox_label_border_color=LINE,
    checkbox_label_text_color_selected=INK, table_border_color=LINE, shadow_drop="none", shadow_spread="0px",
))

CSS = f"""
:root, .dark {{ color-scheme: light; }}
.gradio-container {{ width: 100%; max-width: 1280px !important; min-width: 0; margin: 0 auto; overflow: visible !important;
                     overflow-x: clip !important; }}
.gradio-container .prose, .gradio-container p {{ font-size: inherit; }}
.gradio-container .html-container {{ padding: 0 !important; }}

header.top {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px 32px; flex-wrap: wrap;
              padding: 8px 0 20px; border-bottom: 1px solid {LINE}; }}
header.top h1 {{ font-size: 1.375rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; color: {INK}; }}
header.top p {{ margin: 4px 0 0; max-width: 64ch; font-size: 0.875rem; line-height: 1.5; color: {INK_2}; }}
ul.facts {{ display: flex; flex-wrap: wrap; gap: 4px 24px; margin: 0; padding: 0; list-style: none;
            font-size: 0.8125rem; color: {MUTED}; }}

.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 16px;
         background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; }}
.kpi {{ min-width: 0; padding: 14px 18px 16px; border-left: 1px solid {LINE}; }}
.kpi:first-child {{ border-left: 0; }}
.kpi .label {{ font-size: 0.75rem; font-weight: 500; color: {INK_2}; }}
.kpi .value {{ margin: 6px 0 4px; font-size: 1.75rem; font-weight: 500; letter-spacing: -0.02em; line-height: 1.15;
               color: {INK}; }}
.kpi .value small {{ margin-left: 4px; font-size: 0.8125rem; font-weight: 400; letter-spacing: 0; color: {INK_2}; }}
.kpi .sub {{ font-size: 0.75rem; line-height: 1.5; color: {MUTED}; }}
.kpi.over {{ box-shadow: inset 0 2px 0 {ALARM_C}; }}
.flag {{ font-weight: 500; color: {ALARM_C}; }}
@media (max-width: 860px) {{
  .modebar-container {{ display: none !important; }}
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .kpi:nth-child(3) {{ border-left: 0; }}
  .kpi:nth-child(n+3) {{ border-top: 1px solid {LINE}; }}
}}

.rail {{ align-self: flex-start; gap: 14px !important; padding: 16px !important;
         background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; }}
@media (min-width: 960px) and (min-height: 720px) {{ .rail {{ position: sticky; top: 16px; }} }}
.rail .block, .rail .form {{ border: 0 !important; padding: 0 !important; background: transparent !important;
                             overflow: visible !important; }}
.rail .form {{ gap: 16px !important; }}
.rail-head h2 {{ margin: 0; font-size: 0.875rem; font-weight: 600; color: {INK}; }}
.rail-head p {{ margin: 2px 0 0; font-size: 0.8125rem; color: {INK_2}; }}

.scale {{ position: relative; margin-top: 4px; }}
.scale .track {{ display: flex; height: 8px; border-radius: 2px; overflow: hidden; }}
.scale .track i {{ display: block; height: 100%; }}
.scale .track i + i {{ box-shadow: inset 2px 0 0 {SURFACE}; }}
.scale .pin {{ position: absolute; top: -4px; width: 3px; height: 16px; margin-left: -1.5px; border-radius: 1px;
               background: {INK}; box-shadow: 0 0 0 2px {SURFACE}; }}
.scale-ends {{ display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.6875rem; color: {MUTED}; }}
ul.zones {{ display: grid; gap: 6px; margin: 12px 0 0; padding: 0; list-style: none; }}
ul.zones li {{ display: grid; grid-template-columns: 8px 52px 1fr; align-items: center; gap: 8px;
               font-size: 0.8125rem; }}
ul.zones .k {{ font-weight: 500; color: {INK}; }}
ul.zones .v {{ color: {INK_2}; font-variant-numeric: tabular-nums; }}
.sw {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; }}

.note {{ margin: 8px 0 0; font-size: 0.8125rem; line-height: 1.5; color: {INK_2}; }}
.tag {{ margin-right: 6px; font-size: 0.75rem; color: {MUTED}; }}
code {{ font-family: var(--font-mono); font-size: 0.8125rem; color: {INK}; background: none; padding: 0; }}

.panel {{ margin-bottom: 16px; overflow-x: auto; background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; }}
.panel-head {{ padding: 14px 16px 10px; }}
.panel-head h3 {{ margin: 0; font-size: 0.875rem; font-weight: 600; color: {INK}; }}
.panel-head .note {{ margin-top: 2px; }}
table.tbl {{ width: 100%; border-collapse: collapse; font-size: 0.8125rem; font-variant-numeric: tabular-nums; }}
.gradio-container table.tbl {{ margin: 0; border: 0 !important; }}
.gradio-container table.tbl th, .gradio-container table.tbl td {{ padding: 8px 16px; text-align: right;
    white-space: nowrap; background: none; border: 0 !important; }}
.gradio-container table.tbl th {{ font-weight: 500; color: {MUTED}; border-top: 1px solid {LINE} !important;
    border-bottom: 1px solid {LINE} !important; }}
.gradio-container table.tbl td {{ color: {INK}; border-bottom: 1px solid {GRID} !important; }}
.gradio-container table.tbl tbody tr:last-child td {{ border-bottom: 0 !important; }}
table.tbl .l {{ text-align: left; }}
table.tbl td.l:first-child {{ font-weight: 500; }}

.verdict {{ display: flex; flex-wrap: wrap; gap: 16px 40px; padding: 18px 20px; background: {SURFACE};
            border: 1px solid {LINE}; border-left: 3px solid var(--c); border-radius: 6px; }}
.verdict.empty {{ --c: {LINE}; font-size: 0.875rem; color: {INK_2}; }}
.verdict .label {{ font-size: 0.75rem; font-weight: 500; color: {INK_2}; }}
.verdict .outcome {{ display: flex; align-items: center; gap: 10px; margin-top: 2px; font-size: 1.75rem;
                     font-weight: 500; letter-spacing: -0.02em; color: {INK}; }}
.verdict .outcome .sw {{ width: 12px; height: 12px; background: var(--c); }}
.verdict-body {{ flex: 1 1 320px; min-width: 0; }}
.verdict dl {{ display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 0 0 14px; }}
.verdict dt {{ font-size: 0.75rem; color: {MUTED}; }}
.verdict dd {{ margin: 0; font-size: 0.9375rem; font-weight: 500; color: {INK}; font-variant-numeric: tabular-nums; }}

.tabs > .tab-wrapper, .tab-wrapper {{ border-bottom: 1px solid {LINE}; }}
button[role="tab"] {{ font-size: 0.875rem !important; color: {INK_2} !important; }}
button[role="tab"][aria-selected="true"] {{ color: {INK} !important; font-weight: 500 !important; }}
button[role="tab"][aria-selected="true"]::after {{ background: {INK} !important; height: 2px !important; }}
"""


def build_missing():
    with gr.Blocks(title="Smoke & Fire Triage") as demo:
        gr.HTML('<header class="top"><div><h1>Smoke &amp; Fire Triage</h1></div></header>')
        gr.Markdown(
            "### The trained model and its artifacts are not here yet\n\n"
            f"Missing from `weights/`: {', '.join(f'`{m}`' for m in MISSING)}.\n\n"
            "Run the training pipeline first:\n\n```bash\nuv run python main.py train\n```")
    return demo


def build():
    report_whatif = tuple(REPORT_D[k] for k in WHATIF_KEYS)
    sc0 = scenario(*report_whatif)
    init = update_decide(sc0["lo"], sc0["hi"], VIEWS[0], *report_whatif)
    with gr.Blocks(title="Smoke & Fire Triage") as demo:
        gr.HTML(header_html())
        kpis = gr.HTML(init[0])

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=272, elem_classes="rail"):
                gr.HTML('<div class="rail-head"><h2>Operating point</h2>'
                        '<p>Drives every figure on the page.</p></div>')
                lo = gr.Slider(0, 1, value=sc0["lo"], step=0.01, label="Review from", buttons=[],
                               info="Frames scoring below this pass.")
                hi = gr.Slider(0, 1, value=sc0["hi"], step=0.01, label="Alarm from", buttons=[],
                               info="Frames at or above this alarm. In between, a person reviews.")
                zones = gr.HTML(init[1])
                only = gr.Checkbox(label="Model alone (lo = hi = t*)", value=False)
                reset = gr.Button("Reset to recommended", size="sm")

            with gr.Column(scale=3, min_width=320):
                with gr.Tabs():
                    with gr.Tab("Decide"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=3, min_width=300):
                                view = gr.Radio(VIEWS, value=VIEWS[0], show_label=False, container=False)
                                matrix = gr.Plot(init[2], show_label=False)
                            with gr.Column(scale=2, min_width=260):
                                final = gr.HTML(init[3])
                        curve = gr.Plot(init[4], show_label=False)
                        policies = gr.Plot(init[5], show_label=False)

                    with gr.Tab("Try a frame"):
                        with gr.Row(equal_height=False):
                            upload = gr.Image(type="pil", label="Camera frame", sources=["upload", "clipboard"])
                            annotated = gr.Image(type="pil", label="Detections at or above the review threshold",
                                                 interactive=False)
                        badge = gr.HTML(EMPTY_VERDICT)
                        dets = gr.State([])

                    with gr.Tab("Model"):
                        gr.HTML(model_tab_html())
                        gr.Plot(detection_cm_fig(), show_label=False)
                        with gr.Accordion("Dataset summary", open=False):
                            gr.HTML(dataset_html())

                    with gr.Tab("Assumptions"):
                        gr.HTML(assumptions_html())
                        gr.HTML('<div class="panel"><div class="panel-head"><h3>What-if</h3><p class="note">'
                                'Changes recompute t*, the optimum, the figures and the charts live. '
                                'The report is not affected.</p></div></div>')
                        with gr.Row():
                            w_fn = gr.Number(REPORT_D["cost_fn"], label="cost_fn, € per missed frame", minimum=0)
                            w_fp = gr.Number(REPORT_D["cost_fp"], label="cost_fp, € per false alarm", minimum=0)
                            w_pi = gr.Number(REPORT_D["base_rate"], label="base_rate, share of positive frames",
                                             minimum=1e-6, maximum=0.999)
                            w_rs = gr.Number(REPORT_D["review_seconds"], label="review_seconds, per frame",
                                             minimum=1)
                        reset_w = gr.Button("Reset to report values", size="sm")

        whatif = [w_fn, w_fp, w_pi, w_rs]
        decide_in = [lo, hi, view, *whatif]
        decide_out = [kpis, zones, matrix, final, curve, policies]
        frame_in, frame_out = [upload, dets, lo, hi], [annotated, badge]
        quiet = dict(show_progress="hidden")

        lo.release(clamp_lo, [lo, hi], hi, **quiet).then(update_decide, decide_in, decide_out, **quiet) \
            .then(render_frame, frame_in, frame_out, **quiet)
        hi.release(clamp_hi, [lo, hi], lo, **quiet).then(update_decide, decide_in, decide_out, **quiet) \
            .then(render_frame, frame_in, frame_out, **quiet)
        view.change(update_decide, decide_in, decide_out, **quiet)
        reset.click(reset_recommended, whatif, [lo, hi, only], **quiet) \
            .then(update_decide, decide_in, decide_out, **quiet).then(render_frame, frame_in, frame_out, **quiet)
        only.input(model_only, [only, lo, hi, *whatif], [lo, hi], **quiet) \
            .then(update_decide, decide_in, decide_out, **quiet).then(render_frame, frame_in, frame_out, **quiet)
        for w in whatif:
            w.change(update_decide, decide_in, decide_out, **quiet)
        reset_w.click(reset_whatif, None, whatif, **quiet)
        upload.change(detect, upload, dets, show_progress="minimal").then(render_frame, frame_in, frame_out, **quiet)
    return demo


demo = build() if ART else build_missing()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, footer_links=[])
