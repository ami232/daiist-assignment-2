"""Assignment 2: dashboard. Loads the committed weights and the saved outputs; trains nothing."""
import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
POSITIVE = "head"
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

COSTS = json.loads((ROOT / "costs.json").read_text())
HM = COSTS.get("human_miss_rate", 0.05)
PROD_HOURS = COSTS.get("productive_hours_per_person", 6)
CASES_DAY = COSTS["cameras"] * COSTS["hours_per_day"] * COSTS["frames_per_hour"]

MODEL = YOLO(str(ROOT / "weights" / "best.pt"))  # loaded, never trained here


# ---------- data saved by the training pipeline ----------
def load_scores(split):
    d = pd.read_csv(OUT / f"image_scores_{split}.csv")
    return (d[f"gt_{POSITIVE}"] > 0).astype(int).values, d[f"score_{POSITIVE}"].values


YV, SV = load_scores("valid")
YT, ST = load_scores("test")
POS_V, NEG_V = np.sort(SV[YV == 1]), np.sort(SV[YV == 0])
POS_T, NEG_T = np.sort(ST[YT == 1]), np.sort(ST[YT == 0])


# ---------- cost model (same logic as the training notebook) ----------
def fractions(sorted_scores, tl, th):
    """Share of cases with score < tl (pass), tl <= score < th (review), >= th (alert)."""
    n = len(sorted_scores)
    a = np.searchsorted(sorted_scores, tl, side="left") / n
    b = np.searchsorted(sorted_scores, th, side="left") / n
    return np.array([a, b - a, 1 - b])


def evaluate(pos, neg, tl, th):
    c, pi = COSTS, COSTS["base_rate"]
    p, n = fractions(pos, tl, th), fractions(neg, tl, th)
    cost_pos = (p[0] * c["c_fn"]
                + p[1] * (c["c_review"] + (1 - HM) * c["c_tp"] + HM * c["c_fn"])
                + p[2] * c["c_tp"])
    cost_neg = n[0] * c["c_tn"] + n[1] * (c["c_review"] + c["c_tn"]) + n[2] * c["c_fp"]
    return {"cost_per_1000": 1000 * (pi * cost_pos + (1 - pi) * cost_neg),
            "share": pi * p + (1 - pi) * n,  # share of ALL production cases
            "missed": pi * (p[0] + p[1] * HM)}


def best_bands():
    """Choose both thresholds on the validation split."""
    best, single = None, None
    for i, tl in enumerate(GRID):
        for th in GRID[i:]:
            v = evaluate(POS_V, NEG_V, tl, th)["cost_per_1000"]
            if best is None or v < best[0]:
                best = (v, float(tl), float(th))
            if tl == th and (single is None or v < single[0]):
                single = (v, float(tl), float(th))
    return best, single


BEST, SINGLE = best_bands()
DEF_TL, DEF_TH = BEST[1], BEST[2]
CURVE = [evaluate(POS_T, NEG_T, t, t)["cost_per_1000"] for t in GRID]


def day(per_1000):
    return per_1000 * CASES_DAY / 1000


# ---------- tab 2: thresholds, confusion matrix, cost ----------
def update(tl, th):
    th = max(th, tl)
    r = evaluate(POS_T, NEG_T, tl, th)
    pi = COSTS["base_rate"]
    pred = np.where(ST < tl, 0, np.where(ST < th, 1, 2))
    cm = pd.DataFrame(
        [["Real case: head without helmet"] + [int(((YT == 1) & (pred == j)).sum()) for j in range(3)],
         ["Real case: no head"] + [int(((YT == 0) & (pred == j)).sum()) for j in range(3)]],
        columns=["(test split, counts)", "Pass", "Human review", "Alert"])

    reviews = CASES_DAY * r["share"][1]
    people = reviews * COSTS["review_seconds"] / 3600 / PROD_HOURS
    nothing = 1000 * pi * COSTS["c_fn"]
    always = 1000 * (pi * COSTS["c_tp"] + (1 - pi) * COSTS["c_fp"])
    person_all = 1000 * (COSTS["c_review"] + pi * ((1 - HM) * COSTS["c_tp"] + HM * COSTS["c_fn"]))
    people_all = CASES_DAY * COSTS["review_seconds"] / 3600 / PROD_HOURS
    model_alone = evaluate(POS_T, NEG_T, SINGLE[1], SINGLE[1])["cost_per_1000"]

    md = f"""
### This design (thresholds above), measured on the test split
- **Expected cost: {r['cost_per_1000']:.0f} EUR per 1,000 cases** ({day(r['cost_per_1000']):.0f} EUR/day at {CASES_DAY:,} cases/day)
- **Sent to a person: {100 * r['share'][1]:.1f}% of cases** = {reviews:.0f} photos/day = {people:.2f} people
- Alerts to the supervisor: {CASES_DAY * r['share'][2]:.0f}/day; passed without a look: {100 * r['share'][0]:.1f}%
- Real cases that slip through: {CASES_DAY * r['missed']:.1f}/day

| Strategy | EUR/day | People |
|---|---|---|
| Do nothing | {day(nothing):.0f} | 0 |
| Alert on every photo | {day(always):.0f} | 0 |
| A person checks every photo | {day(person_all):.0f} | {people_all:.2f} |
| Model alone (threshold {SINGLE[1]:.2f}) | {day(model_alone):.0f} | 0 |
| **This design** | **{day(r['cost_per_1000']):.0f}** | **{people:.2f}** |

Costs come from `costs.json`: false alarm {COSTS['c_fp']} EUR, missed case {COSTS['c_fn']} EUR, looking at a photo {COSTS['c_review']} EUR,
base rate {100 * pi:.1f}%, human miss rate {100 * HM:.0f}%. They are assumptions; see REPORT.md.
"""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(GRID, CURVE, label="model alone, one threshold (test)")
    ax.scatter([SINGLE[1]], [model_alone], marker="*", s=180, color="gold", edgecolor="black",
               zorder=5, label=f"best single threshold {SINGLE[1]:.2f} (chosen on valid)")
    ax.axvline(tl, color="tab:green", ls="--", label=f"t_low = {tl:.2f}")
    ax.axvline(th, color="tab:red", ls="--", label=f"t_high = {th:.2f}")
    ax.axhline(r["cost_per_1000"], color="gray", ls=":", label="this design (review band)")
    ax.set_yscale("log")
    ax.set_xlabel("confidence threshold on the 'head' score")
    ax.set_ylabel("EUR per 1,000 cases (log scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return cm, md, fig


# ---------- tab 1: one image ----------
COLORS = {"helmet": "#2ca02c", "person": "#1f77b4"}


def detect(img):
    if img is None:
        return []
    res = MODEL.predict(img, conf=0.001, imgsz=640, verbose=False)[0]
    return [(res.names[int(c)], float(p), [float(v) for v in b])
            for b, p, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]


def render(img, dets, tl, th):
    if img is None:
        return None, "Upload an image."
    th = max(th, tl)
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    head_score = max([p for n, p, _ in dets if n == POSITIVE], default=0.0)
    for name, conf, (x1, y1, x2, y2) in dets:
        if conf < tl:
            continue
        color = ("#d62728" if conf >= th else "#ff7f0e") if name == POSITIVE else COLORS.get(name, "#7f7f7f")
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        d.text((x1 + 3, y1 + 2), f"{name} {conf:.2f}", fill=color)
    if head_score >= th:
        verdict = f"**ALERT**: head score {head_score:.2f} >= t_high {th:.2f}. Send the supervisor."
    elif head_score >= tl:
        verdict = f"**HUMAN REVIEW**: head score {head_score:.2f} is between t_low {tl:.2f} and t_high {th:.2f}. A person looks at this photo."
    else:
        verdict = f"**PASS**: head score {head_score:.2f} < t_low {tl:.2f}."
    return out, verdict + "\n\nBoxes below t_low are hidden. Red = head above t_high, orange = head in review band."


# ---------- tab 3: per-class results ----------
def per_class_tables():
    m = json.loads((OUT / "metrics.json").read_text())
    stats = json.loads((OUT / "dataset_stats.json").read_text())
    n_test = stats["splits"]["test"]["instances"]
    det = pd.DataFrame([{"class": k, "test instances": n_test.get(k),
                         "precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                         "AP50": round(v["ap50"], 3), "mAP50-95": round(v["ap50_95"], 3)}
                        for k, v in m["detection_test"]["per_class"].items()])
    b = m["baselines_test"]
    img_rows = pd.DataFrame([{"class": k,
                              "finetuned": round(b["finetuned_image_ap"][k], 3),
                              "COCO zero-shot": round(b["coco_zero_shot_image_ap"][k], 3),
                              "predict nothing": round(b["trivial_predict_nothing_image_ap"][k], 3)}
                             for k in b["finetuned_image_ap"]])
    head = (f"Overall mAP50 = {m['detection_test']['map50']:.3f}, but it averages three very different classes. "
            f"**`head` carries the headline number**; `person` has too few examples to trust (see test instances).")
    return det, img_rows, head


DET_DF, IMG_DF, HEADLINE = per_class_tables()

with gr.Blocks(title="Hard-hat alerts: threshold and cost dashboard") as demo:
    gr.Markdown("# Hard-hat alerts: thresholds, cost and human review\n"
                "A camera photo is scored for 'head without helmet'. "
                "Below **t_low** it passes, above **t_high** it alerts, in between a person looks at it.")
    with gr.Row():
        t_low = gr.Slider(0.01, 0.99, value=DEF_TL, step=0.01, label="t_low (below: pass)")
        t_high = gr.Slider(0.01, 0.99, value=DEF_TH, step=0.01, label="t_high (above: alert)")
    dets = gr.State([])
    with gr.Tabs():
        with gr.Tab("Try an image"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Upload a photo")
                img_out = gr.Image(label="Prediction")
            verdict = gr.Markdown()
        with gr.Tab("Cost and human review"):
            cm_out = gr.Dataframe(label="Confusion matrix, three outcomes")
            summary = gr.Markdown()
            plot = gr.Plot(label="Cost against threshold")
        with gr.Tab("Per-class results"):
            gr.Markdown(HEADLINE)
            gr.Dataframe(DET_DF, label="Detection on the test split (precision/recall at the best-F1 confidence)")
            gr.Dataframe(IMG_DF, label="Image-level AP: is the class present? Baselines on the same test split")
            gr.Image(str(OUT / "test_eval_conf025" / "confusion_matrix_normalized.png"),
                     label="Confusion matrix at confidence 0.25 (normalized)")

    img_in.change(detect, img_in, dets).then(render, [img_in, dets, t_low, t_high], [img_out, verdict])
    for s in (t_low, t_high):
        s.change(render, [img_in, dets, t_low, t_high], [img_out, verdict])
        s.change(update, [t_low, t_high], [cm_out, summary, plot])
    demo.load(update, [t_low, t_high], [cm_out, summary, plot])

if __name__ == "__main__":
    demo.launch()