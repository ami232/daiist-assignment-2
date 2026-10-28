"""
Assignment 2: Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file. main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via uv run python main.py app, this file must build and launch a Gradio app
(a demo that calls .launch()) using only what your training stage already
produced. It must never train anything itself. Load the committed weights in
weights/, never whatever the training run left in runs/.

At minimum, the dashboard must let you:

- Upload an image and see the model's predictions at the current threshold.
- Move the two edges of your escalation band and watch the confusion matrix,
  the expected cost per 1,000 units, and the share of cases escalated to a
  human all change together.
- See the cost-against-threshold curve with your chosen operating point marked
  on it.
- See per-class performance, not just the overall number.

How you structure the code beyond that is your call to make and be able to
defend: file layout, function boundaries, naming.
"""

"""
Assignment 2: Gradio dashboard.   Run with:  uv run python main.py app

This app never trains and never downloads a dataset. It uses only what the
training stage already produced and committed:

  weights/best.pt          the finetuned model (used for the image-upload tab)
  outputs/results.json     per-image scores, assumptions, sweeps and metrics

Case definition (same as train.py): one case = one image. An image is truly
"bad" if it contains a rotten label; the model's score for an image is the
highest rotten confidence it produced. Below the low threshold the image
passes, between the two it goes to a person, at or above the high one it fails.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import gradio as gr  # noqa: E402

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
RESULTS = ROOT / "outputs" / "results.json"

for path in (WEIGHTS, RESULTS):
    if not path.exists():
        raise SystemExit(
            f"Missing {path.relative_to(ROOT)}. Commit the files your training run produced "
            f"(weights/best.pt and outputs/results.json)."
        )

R = json.loads(RESULTS.read_text())
A0 = R["assumptions"]
PRODUCTIVE_H = R["productive_hours_per_person"]
NAMES = {int(k): v for k, v in R["names"].items()}
POS_NAME = R["positive_class"]
POS_ID = [k for k, v in NAMES.items() if v == POS_NAME][0]
CH = R["chosen"]
TS = np.array(R["sweep"]["thresholds"])
yv, sv = np.array(R["scores"]["valid"]["y"], bool), np.array(R["scores"]["valid"]["s"])
yt, st = np.array(R["scores"]["test"]["y"], bool), np.array(R["scores"]["test"]["s"])

from ultralytics import YOLO  # noqa: E402  (inference only)

MODEL = YOLO(str(WEIGHTS))


# ----------------------------------------------------------------------------
# Costing. The same formulas as train.py, so the dashboard matches the report.
# ----------------------------------------------------------------------------
def evaluate(y, s, lo, hi, A):
    """Pass if s < lo, human review if lo <= s < hi, fail if s >= hi."""
    r, g = s[y], s[~y]
    chk = A["check_s"] * A["wage_h"] / 3600
    n_rot, n_good = 1000 * A["base_rate"], 1000 * (1 - A["base_rate"])
    r_fail, r_pass = (r >= hi).mean(), (r < lo).mean()
    r_rev = 1 - r_fail - r_pass
    g_fail, g_pass = (g >= hi).mean(), (g < lo).mean()
    g_rev = 1 - g_fail - g_pass
    cost = (n_rot * (r_pass * A["c_miss"] + r_rev * (chk + (1 - A["h_catch"]) * A["c_miss"]))
            + n_good * (g_fail * A["c_fa"] + g_rev * (chk + A["h_false"] * A["c_fa"])))
    return {
        "cost": float(cost),
        "share": float((n_rot * r_rev + n_good * g_rev) / 1000),
        "misses": float(n_rot * (r_pass + r_rev * (1 - A["h_catch"]))),
        "false_alarms": float(n_good * (g_fail + g_rev * A["h_false"])),
        "rot": (int((r < lo).sum()), int(((r >= lo) & (r < hi)).sum()), int((r >= hi).sum())),
        "good": (int((g < lo).sum()), int(((g >= lo) & (g < hi)).sum()), int((g >= hi).sum())),
    }


def do_nothing(A):
    return 1000 * A["base_rate"] * A["c_miss"]


def check_everything(A):
    chk = A["check_s"] * A["wage_h"] / 3600
    n_rot, n_good = 1000 * A["base_rate"], 1000 * (1 - A["base_rate"])
    return 1000 * chk + n_rot * (1 - A["all_catch"]) * A["c_miss"] + n_good * A["all_false"] * A["c_fa"]


def people(share, A):
    hands = share * A["hands_day"]
    return hands, hands * A["check_s"] / (PRODUCTIVE_H * 3600)


# ----------------------------------------------------------------------------
# Tab 1: upload an image
# ----------------------------------------------------------------------------
def predict_image(img, lo, hi):
    if img is None:
        return None, "Upload a photo of a hand of bananas to see the model's predictions."
    lo, hi = sorted((float(lo), float(hi)))
    bgr = np.ascontiguousarray(img[..., ::-1])
    r = MODEL.predict(bgr, conf=0.05, iou=0.7, imgsz=416, verbose=False)[0]
    conf, cls = r.boxes.conf.tolist(), [int(c) for c in r.boxes.cls.tolist()]
    keep = [i for i, c in enumerate(conf) if c >= lo]
    shown = r[keep].plot() if keep else r.orig_img
    score = max([c for c, k in zip(conf, cls) if k == POS_ID], default=0.0)
    if score >= hi:
        verdict = "FAIL: pull this hand out"
    elif score >= lo:
        verdict = "SEND TO A PERSON: check it by eye"
    else:
        verdict = "PASS: goes in the box"
    lines = [f"### {verdict}",
             f"Highest **{POS_NAME}** confidence: **{score:.2f}** "
             f"(pass below {lo:.2f}, review from {lo:.2f}, fail from {hi:.2f})."]
    if keep:
        lines.append("\nBoxes shown (confidence at or above the lower threshold):")
        lines += [f"- {NAMES[cls[i]]}: {conf[i]:.2f}" for i in sorted(keep, key=lambda j: -conf[j])]
    else:
        lines.append("\nNo detections at or above the lower threshold.")
    return np.ascontiguousarray(shown[..., ::-1]), "\n".join(lines)


# ----------------------------------------------------------------------------
# Tab 2: cost and human review (all outputs move together with the sliders)
# ----------------------------------------------------------------------------
def cost_plot(lo, hi, A):
    cv = [evaluate(yv, sv, t, t, A)["cost"] for t in TS]
    ct = [evaluate(yt, st, t, t, A)["cost"] for t in TS]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvspan(lo, hi, color="orange", alpha=0.15, label="human-review band (current)")
    ax.plot(TS, cv, label="model alone, validation")
    ax.plot(TS, ct, alpha=0.6, label="model alone, test")
    ax.axhline(do_nothing(A), ls="--", c="gray", label="do nothing")
    ax.axvline(CH["single_threshold"], c="red", ls=":",
               label=f"chosen single threshold ({CH['single_threshold']:.2f})")
    best = int(np.argmin(cv))
    ax.scatter([TS[best]], [cv[best]], c="green", zorder=5, label=f"cheapest under these assumptions ({TS[best]:.2f})")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("expected cost per 1,000 hands (EUR)")
    ax.set_title("Cost against threshold")
    ax.legend(loc="upper left", fontsize=8)
    plt.close(fig)
    return fig


def update(lo, hi, base_rate, c_miss, c_fa, h_catch, check_s):
    lo, hi = float(lo), float(hi)
    note = ""
    if lo > hi:
        lo, hi = hi, lo
        note = "\n\n*The low edge was above the high edge, so I swapped them.*"
    A = {**A0, "base_rate": float(base_rate), "c_miss": float(c_miss), "c_fa": float(c_fa),
         "h_catch": float(h_catch), "check_s": float(check_s)}
    ev = evaluate(yt, st, lo, hi, A)
    n_rot, n_good = int(yt.sum()), int((~yt).sum())

    confusion = pd.DataFrame({
        "Truth (test images)": [f"Truly rotten ({n_rot})", f"Truly good ({n_good})"],
        "Pass": [ev["rot"][0], ev["good"][0]],
        "Send to a person": [ev["rot"][1], ev["good"][1]],
        "Fail": [ev["rot"][2], ev["good"][2]],
    })
    hands, ppl = people(ev["share"], A)
    summary = (
        f"### Expected cost: **EUR {ev['cost']:.2f} per 1,000 hands** (EUR {ev['cost'] * A['hands_day'] / 1000:,.0f} per day)\n"
        f"- **Sent to a person:** {ev['share']:.1%} of hands = {hands:,.0f} per day, about **{ppl:.2f} people** "
        f"({A['check_s']:.0f} s each)\n"
        f"- **Per 1,000 hands:** {ev['misses']:.1f} rotten hands still reach the customer, "
        f"{ev['false_alarms']:.1f} good hands wrongly failed\n"
        f"- Base rate assumed: {A['base_rate']:.1%} rotten. The table counts are real test images; "
        f"the euro figures re-weight them to that base rate.{note}"
    )
    alone = evaluate(yt, st, CH["single_threshold"], CH["single_threshold"], A)
    chk_all = check_everything(A)
    comparison = pd.DataFrame({
        "Strategy": ["Do nothing (ship everything)",
                     "A person checks every hand",
                     f"Model alone (threshold {CH['single_threshold']:.2f})",
                     f"Model + review [{lo:.2f}, {hi:.2f})  (current)"],
        "EUR per 1,000 hands": [do_nothing(A), chk_all, alone["cost"], ev["cost"]],
        "EUR per day": [v * A["hands_day"] / 1000 for v in (do_nothing(A), chk_all, alone["cost"], ev["cost"])],
        "People checking": [0.0, people(1.0, A)[1], 0.0, ppl],
    }).round(2)
    return confusion, summary, comparison, cost_plot(lo, hi, A)


# ----------------------------------------------------------------------------
# Tab 3: per-class results
# ----------------------------------------------------------------------------
def per_class_table():
    det = R["detection_test"]
    inst = R["dataset"]["instances"]["test"]
    imgs = R["dataset"]["images_with_class"]["test"]
    rows = [{"Class": d["class"] + ("  (positive)" if d["class"] == POS_NAME else ""),
             "Precision": d["precision"], "Recall": d["recall"],
             "AP50": d["ap50"], "AP50-95": d["ap50_95"],
             "Test instances": inst[d["class"]], "Test images": imgs[d["class"]]}
            for d in det["per_class"]]
    o = det["overall"]
    rows.append({"Class": "ALL", "Precision": o["precision"], "Recall": o["recall"], "AP50": o["map50"],
                 "AP50-95": o["map50_95"], "Test instances": sum(inst.values()),
                 "Test images": R["dataset"]["images"]["test"]})
    return pd.DataFrame(rows).round(3)


def confusion_heatmap():
    cm = R["detection_test"].get("confusion_matrix_conf025")
    if not cm:
        return None
    m, labels = np.array(cm["matrix"], dtype=float), cm["labels"]
    norm = m / np.maximum(m.sum(axis=0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j] > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("true class")
    ax.set_ylabel("predicted class")
    ax.set_title("Class confusion matrix on test (confidence 0.25)")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ----------------------------------------------------------------------------
# Tab 4: assumptions and baselines
# ----------------------------------------------------------------------------
def assumptions_markdown():
    a, s = A0, R["strategies_test"]
    d, t = R["dataset"], R["training"]
    out = [
        "### Assumptions behind every euro figure",
        f"- Positive (bad) class: **{POS_NAME}**. One case = one image.",
        f"- Rotten share of hands: **{a['base_rate']:.1%}** (an assumption; the test set is "
        f"{yt.mean():.0%} rotten images)",
        f"- A miss costs **EUR {a['c_miss']:.2f}**. A false alarm costs **EUR {a['c_fa']:.2f}**.",
        f"- {a['hands_day']:,} hands per day. A human check takes {a['check_s']} s at EUR {a['wage_h']}/hour. "
        f"A reviewer catches {a['h_catch']:.0%} of rotten hands and wrongly fails {a['h_false']:.0%} of good ones.",
        f"- Thresholds chosen on the **validation** split, reported on the **test** split.",
        "",
        "### Baselines on the test split (EUR per 1,000 hands)",
        f"- Do nothing / predict nothing / majority class: **{s['do_nothing']['cost_per_1000']:.2f}**",
        f"- Fail every hand: **{s['fail_everything']['cost_per_1000']:.2f}**",
        f"- A person checks every hand: **{s['person_checks_every_hand']['cost_per_1000']:.2f}**",
        f"- Model alone: **{s['model_alone']['cost_per_1000']:.2f}**",
        f"- Model + review: **{s['model_plus_review']['cost_per_1000']:.2f}**",
        "",
        "Pretrained COCO model, zero-shot, flagging an image when it sees a banana "
        "(COCO has no ripeness classes, so this is all it can say):",
    ]
    for row in R["coco_flag_if_banana_test"]:
        out.append(f"- banana confidence >= {row['banana_conf']:.2f}: recall {row['recall']:.2f}, "
                   f"false-alarm rate {row['false_alarm_rate']:.2f}, EUR {row['cost_per_1000']:.2f} per 1,000")
    out += ["", "### Data and training",
            f"- Roboflow `{d['workspace']}/{d['project']}` version {d['version']}",
            f"- Images: train {d['images']['train']:,} (includes 3x augmented copies), "
            f"valid {d['images']['valid']:,}, test {d['images']['test']:,}",
            f"- Model: {t['base_model']}, {t['epochs']} epochs, {t['imgsz']} px, batch {t['batch']}, "
            f"seed {t['seed']}; trained on {t['device']}"]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
with gr.Blocks(title="Banana packing station: costed decision") as demo:
    gr.Markdown(
        "# Banana packing station: pass, check, or fail\n"
        f"A camera photographs each hand of bananas. The model scores it for **{POS_NAME}**. "
        "Below the low threshold the hand passes, between the two a person checks it, "
        "and at or above the high threshold it is failed. Move the two edges and watch everything update."
    )
    with gr.Row():
        lo_sl = gr.Slider(0.05, 0.95, value=CH["review_low"], step=0.01, label="Low threshold (pass below this)")
        hi_sl = gr.Slider(0.05, 0.95, value=CH["review_high"], step=0.01, label="High threshold (fail from this)")

    with gr.Accordion("What if my assumptions are wrong?", open=False):
        with gr.Row():
            br_sl = gr.Slider(0.005, 0.20, value=A0["base_rate"], step=0.005, label="Share of hands that are rotten")
            cm_sl = gr.Slider(1, 100, value=A0["c_miss"], step=1, label="Cost of a miss (EUR)")
            cfa_sl = gr.Slider(0.1, 5, value=A0["c_fa"], step=0.1, label="Cost of a false alarm (EUR)")
        with gr.Row():
            hc_sl = gr.Slider(0.5, 1.0, value=A0["h_catch"], step=0.01, label="Reviewer catch rate")
            ck_sl = gr.Slider(3, 60, value=A0["check_s"], step=1, label="Seconds per human check")
        reset = gr.Button("Reset to my chosen settings")

    inputs = [lo_sl, hi_sl, br_sl, cm_sl, cfa_sl, hc_sl, ck_sl]

    with gr.Tab("Cost and human review"):
        summary_md = gr.Markdown()
        confusion_df = gr.Dataframe(label="Confusion matrix on the test images (pass / review / fail)", interactive=False)
        comparison_df = gr.Dataframe(label="Compared with the alternatives", interactive=False)
        curve = gr.Plot(label="Cost against threshold")
    outputs = [confusion_df, summary_md, comparison_df, curve]

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(label="Upload a photo", type="numpy")
            img_out = gr.Image(label="Predictions at the current thresholds")
        verdict_md = gr.Markdown()

    with gr.Tab("Per-class results"):
        gr.Markdown("Test split. Precision and recall are at the confidence that maximises F1. "
                    "The positive class is the one the costing is built on.")
        gr.Dataframe(value=per_class_table(), interactive=False)
        gr.Plot(value=confusion_heatmap(), label="Which classes the model confuses")

    with gr.Tab("Assumptions and baselines"):
        gr.Markdown(assumptions_markdown())

    for comp in inputs:
        comp.change(update, inputs, outputs)
    for comp in (img_in, lo_sl, hi_sl):
        comp.change(predict_image, [img_in, lo_sl, hi_sl], [img_out, verdict_md])
    reset.click(lambda: (CH["review_low"], CH["review_high"], A0["base_rate"], A0["c_miss"],
                         A0["c_fa"], A0["h_catch"], A0["check_s"]), None, inputs)
    demo.load(update, inputs, outputs)


if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("APP_PORT", "7860")))
