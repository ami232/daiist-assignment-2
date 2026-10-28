"""
Assignment 2: Gradio dashboard.   Run with:  uv run python main.py app

Loads weights/best.pt and the files train.py saved in results/. It never
trains anything and never downloads a dataset: everything it shows comes from
the real outputs of the training pipeline.

Tabs
  1. Try an image      upload a metre of denim, see boxes and the decision
  2. Cost and review   move the two thresholds, watch the confusion matrix,
                       the cost per 1,000 metres and the share sent to people
  3. Per-class         precision / recall / AP per class + class confusion matrix
  4. Baselines         finetuned model vs COCO zero-shot vs trivial, and the
                       four strategies priced against each other
  5. Sensitivity       what happens when each assumption changes
  6. Dataset           what the data looks like and what that implies

The two thresholds: a metre is PASSED if its score is below the first one,
FAILED (scrapped) if its score is at or above the second one, and sent to a
person in between. Its score is the highest box confidence of any class.
"""

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
RESULTS = ROOT / "results"
WEIGHT_FILE = ROOT / "weights" / "best.pt"

for needed in (WEIGHT_FILE, RESULTS / "metrics.json", RESULTS / "operating_point.json",
               RESULTS / "test_scores.csv"):
    if not needed.exists():
        raise SystemExit(f"Missing {needed.relative_to(ROOT)}. Run `uv run python main.py train` first.")

# ----------------------------- load saved outputs -------------------------
metrics = json.loads((RESULTS / "metrics.json").read_text())
op = json.loads((RESULTS / "operating_point.json").read_text())
class_cm = json.loads((RESULTS / "class_confusion.json").read_text())
stats = json.loads((RESULTS / "dataset_stats.json").read_text())
scores = pd.read_csv(RESULTS / "test_scores.csv")
per_class = pd.read_csv(RESULTS / "per_class.csv")
baselines = pd.read_csv(RESULTS / "baselines.csv")
strategies = pd.read_csv(RESULTS / "strategies.csv")
sensitivity_table = pd.read_csv(RESULTS / "sensitivity.csv")

P = metrics["params"]
THRESHOLDS = np.array(metrics["thresholds"])
IMGSZ = metrics["training"]["imgsz"]
METRES_PER_DAY = metrics["metres_per_day"]
SHIFT_HOURS = metrics["shift_hours"]
REVIEW_SECONDS = metrics["review_seconds"]
DISPLAY = metrics.get("display_names") or {}
FEW_INSTANCES = 50   # per-class numbers on fewer test instances are flagged

POS = scores.loc[scores["y_true"] == 1, "score"].to_numpy()   # defective metres
NEG = scores.loc[scores["y_true"] == 0, "score"].to_numpy()   # clean metres

model = YOLO(str(WEIGHT_FILE))   # loading weights is NOT training


def nice(name):
    return DISPLAY.get(name, name)


# ----------------------------- cost maths ---------------------------------
# Same formula as train.py. Checked against operating_point.json at startup.
def expected_cost(pass_p, fail_p, rev_p, pass_n, fail_n, rev_n):
    cost_pos = (pass_p * P["c_fn"] + fail_p * P["c_tp"]
                + rev_p * (P["c_review"] + P["h_tpr"] * P["c_tp"] + (1 - P["h_tpr"]) * P["c_fn"]))
    cost_neg = (pass_n * P["c_tn"] + fail_n * P["c_fp"]
                + rev_n * (P["c_review"] + P["h_fpr"] * P["c_fp"] + (1 - P["h_fpr"]) * P["c_tn"]))
    cost = 1000 * (P["base_rate"] * cost_pos + (1 - P["base_rate"]) * cost_neg)
    review = P["base_rate"] * rev_p + (1 - P["base_rate"]) * rev_n
    return cost, review


def shares(s, low, high):
    passed = float((s < low).mean()); failed = float((s >= high).mean())
    return passed, failed, max(0.0, 1 - passed - failed)


def cost_at(low, high):
    return expected_cost(*shares(POS, low, high), *shares(NEG, low, high))


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (centre + margin) / (1 + z * z / n)))


def pessimistic_at(low, high):
    ub = wilson_upper(int((NEG >= high).sum()), len(NEG))
    pass_n, fail_seen, rev_n = shares(NEG, low, high)
    extra = max(0.0, ub - fail_seen)          # extra clean metres that would be scrapped
    take = min(extra, rev_n); rev_n -= take   # they come from the review band first,
    pass_n = max(0.0, pass_n - (extra - take))  # then from the passed ones
    fail_n = fail_seen + extra
    cost, _ = expected_cost(*shares(POS, low, high), pass_n, fail_n, rev_n)
    return float(cost), ub


def full_grid():
    K = len(THRESHOLDS)
    pass_p = (POS[None, :] < THRESHOLDS[:, None]).mean(1)[:, None]
    fail_p = (POS[None, :] >= THRESHOLDS[:, None]).mean(1)[None, :]
    pass_n = (NEG[None, :] < THRESHOLDS[:, None]).mean(1)[:, None]
    fail_n = (NEG[None, :] >= THRESHOLDS[:, None]).mean(1)[None, :]
    cost, _ = expected_cost(pass_p, fail_p, 1 - pass_p - fail_p, pass_n, fail_n, 1 - pass_n - fail_n)
    return np.where(np.triu(np.ones((K, K), dtype=bool)), cost, np.nan)


GRID = full_grid()
CURVE_NO_REVIEW = np.diag(GRID)                          # low == high
CURVE_WITH_REVIEW = np.nanmin(GRID, axis=0)              # best `low` for each `high`

_check, _ = cost_at(op["review_low"], op["review_high"])
if abs(_check - op["cost_per_1000"]) > 0.5:
    print(f"WARNING: app recomputes {_check:.2f} EUR but operating_point.json says "
          f"{op['cost_per_1000']:.2f}. train.py and app.py disagree.")


# ----------------------------- tab 1: one image ---------------------------
def predict(image, low, high):
    if image is None:
        return None, "Upload an image of a metre of denim."
    low, high = min(low, high), high
    result = model.predict(image, imgsz=IMGSZ, conf=0.001, iou=0.6, verbose=False)[0]
    boxes = result.boxes
    confs = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else np.array([])
    score = float(confs.max()) if len(confs) else 0.0
    decision = "FAIL (scrap)" if score >= high else "REVIEW (send to a person)" if score >= low else "PASS"

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    show_from = max(low, 0.05)            # do not draw hundreds of near-zero boxes
    lines = []
    for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy() if len(confs) else [], confs,
                               boxes.cls.cpu().numpy() if len(confs) else []):
        if conf < show_from:
            continue
        colour = "red" if conf >= high else "orange"
        name = nice(model.names[int(cls)])
        draw.rectangle([float(v) for v in xyxy], outline=colour, width=4)
        draw.text((float(xyxy[0]) + 4, float(xyxy[1]) + 4), f"{name} {conf:.2f}", fill=colour)
        lines.append(f"| {name} | {conf:.2f} | {'fail' if conf >= high else 'review'} |")
    table = ("\n\n| Box | Confidence | Counts as |\n|---|---|---|\n" + "\n".join(lines)) if lines else ""
    summary = (f"### Decision: {decision}\n"
               f"Score of this metre (highest box confidence): **{score:.3f}**  \n"
               f"Pass below **{low:.2f}** · fail from **{high:.2f}** · in between goes to a person."
               f"{table}")
    return canvas, summary


# ----------------------------- tab 2: thresholds, cost, review ------------
def update(low, high):
    low = min(low, high)
    pass_p, fail_p, rev_p = shares(POS, low, high)
    pass_n, fail_n, rev_n = shares(NEG, low, high)
    n_pos, n_neg = len(POS), len(NEG)
    confusion = pd.DataFrame(
        {"Pass": [round(pass_p * n_pos), round(pass_n * n_neg)],
         "Review": [round(rev_p * n_pos), round(rev_n * n_neg)],
         "Fail (scrap)": [round(fail_p * n_pos), round(fail_n * n_neg)]},
        index=[f"Really defective ({n_pos})", f"Really clean ({n_neg})"]).reset_index(
        names="Test metres")

    cost, review = cost_at(low, high)
    pess, ub = pessimistic_at(low, high)
    cases = METRES_PER_DAY * review
    people = cases * REVIEW_SECONDS / (SHIFT_HOURS * 3600)
    by_name = {r["strategy"].split(" (")[0]: r["cost_per_1000_metres"] for _, r in strategies.iterrows()}
    nothing = by_name.get("do_nothing", np.nan); everything = by_name.get("person checks everything", np.nan)

    text = (
        f"### Expected cost: **{cost:,.0f} € per 1,000 metres** ({cost * METRES_PER_DAY / 1000:,.0f} €/day)\n"
        f"- Share sent to a person: **{review:.1%}** → {cases:,.0f} metres/day, "
        f"about **{people:.1f} reviewers** ({REVIEW_SECONDS} s each, {SHIFT_HOURS} h shifts)\n"
        f"- Reference: do nothing {nothing:,.0f} € · a person checks everything {everything:,.0f} €"
        f" (per 1,000 m, same assumptions)\n"
        f"- Assumes **{P['base_rate']:.0%}** of metres are defective, not the dataset's ~70%.\n\n"
        f"**How much to trust it.** Only {len(NEG)} test metres are clean. None of the clean ones may have "
        f"been scrapped here ({(NEG >= high).mean():.1%}), but with so few the true rate could be up to "
        f"**{ub:.1%}**, which would make the cost **{pess:,.0f} € per 1,000 m**."
    )
    return text, confusion, cost_figure(low, high)


def cost_figure(low, high):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(THRESHOLDS[:-1], CURVE_NO_REVIEW[:-1], color="tab:blue", label="Model alone (pass/fail)")
    ax.plot(THRESHOLDS[:-1], CURVE_WITH_REVIEW[:-1], color="tab:green",
            label="Model + review (best 'pass below' per 'fail from', picked on test: optimistic)")
    for label, name, style in (("Do nothing", "do_nothing", ":"),
                               ("Person checks everything", "person checks everything", "--")):
        row = strategies[strategies["strategy"].str.startswith(name.split(" (")[0])]
        if len(row):
            ax.axhline(row["cost_per_1000_metres"].iloc[0], color="grey", linestyle=style,
                       linewidth=1, label=label)
    ax.scatter([op["best_single_threshold"]], [op["single_cost_per_1000"]], marker="o", s=70,
               color="tab:blue", zorder=5, label=f"Chosen single threshold ({op['best_single_threshold']:.2f})")
    ax.scatter([op["review_high"]], [op["cost_per_1000"]], marker="*", s=220, color="tab:green",
               zorder=6, label=f"Chosen operating point (pass < {op['review_low']:.2f}, fail ≥ {op['review_high']:.2f})")
    ax.set_ylim(0, 1.8 * max(1500.0, float(np.nanmin(CURVE_NO_REVIEW[:-1])) * 3))   # the far-left spike is not the story
    ax.axvline(high, color="red", linewidth=1, alpha=0.6)
    ax.annotate("your 'fail from'", (high, ax.get_ylim()[1]), rotation=90, va="top", ha="right",
                color="red", fontsize=8)
    ax.axvline(low, color="orange", linewidth=1, alpha=0.8)
    ax.annotate("your 'pass below'", (low, ax.get_ylim()[1]), rotation=90, va="top", ha="right",
                color="orange", fontsize=8)
    ax.set_xlabel("'Fail from' confidence threshold")
    ax.set_ylabel("Expected cost, € per 1,000 metres (test split)")
    ax.set_title("Cost against threshold (y-axis cut off: scrapping everything costs far more)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ----------------------------- tab 3: per class ---------------------------
def per_class_table():
    t = per_class.copy()
    t["class"] = t["class"].map(nice)
    t["trust"] = np.where(t["instances_in_test"] < FEW_INSTANCES,
                          "⚠ too few instances for a stable number", "ok")
    t[["precision", "recall", "ap50", "map50_95"]] = t[["precision", "recall", "ap50", "map50_95"]].round(3)
    return t.rename(columns={"class": "Class", "instances_in_test": "Test instances",
                             "precision": "Precision", "recall": "Recall", "ap50": "AP50",
                             "map50_95": "mAP50-95", "trust": "Reliability"})


def class_confusion_figure():
    labels = [nice(x) for x in class_cm["labels"]]
    m = np.array(class_cm["matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted class")
    ax.set_title(f"Class confusion matrix (test, confidence ≥ {class_cm['conf']})")
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            ax.text(c, r, int(m[r, c]), ha="center", va="center",
                    color="white" if m[r, c] > m.max() / 2 else "black")
    fig.tight_layout()
    plt.close(fig)
    return fig


def overall_text():
    o = metrics["overall_box_metrics_test"]
    return (f"### Overall (test split, box level)\n"
            f"**AP50 {o['ap50']:.3f}** · mAP50-95 {o['map50_95']:.3f} · precision {o['precision']:.3f} · "
            f"recall {o['recall']:.3f}\n\n"
            f"Positive class for the business decision: **any defect** (any of the classes). "
            f"The overall number hides the weakest class, so read the table below.")


# ----------------------------- tab 6: dataset -----------------------------
def dataset_tables():
    rows = []
    for split, v in stats["splits"].items():
        rows.append({"Split": split, "Images": v["images"],
                     "Without defect": f"{v['images_without_defect']} ({v['share_without_defect']:.0%})",
                     **{f"{nice(c)} instances": n for c, n in v["instances_per_class"].items()},
                     "Median box area (% of image)": round(100 * v["box_area_share_p10_p50_p90"][1], 1)})
    leak = stats["leakage_check"]
    note = (f"**Leakage check.** {leak['test_images_whose_base_name_is_also_in_train_or_valid']} of the "
            f"{stats['splits']['test']['images']} test images ({leak['share_of_test']:.0%}) share a source "
            f"file name with a train or valid image. Part of the test split is therefore not independent "
            f"of training, and the test numbers are probably optimistic.")
    return pd.DataFrame(rows), note


dataset_df, leak_note = dataset_tables()

# ----------------------------- build the app ------------------------------
with gr.Blocks(title="Denim defect inspection") as demo:
    gr.Markdown(
        "# Denim defect inspection: pass, review or scrap\n"
        "One unit = one metre of denim (one image). A metre with **any** defect is scrapped. "
        "Move the two thresholds and watch the decision, the confusion matrix and the cost change.")
    with gr.Row():
        low = gr.Slider(0, 1, value=op["review_low"], step=0.01,
                        label="Pass below (scores under this are passed without a person)")
        high = gr.Slider(0, 1, value=op["review_high"], step=0.01,
                         label="Fail from (scores at or above this are scrapped)")

    with gr.Tabs():
        with gr.Tab("Try an image"):
            with gr.Row():
                image_in = gr.Image(type="pil", label="Upload a metre of denim")
                image_out = gr.Image(label="Predictions at the current thresholds")
            decision_md = gr.Markdown()
        with gr.Tab("Cost and review"):
            cost_md = gr.Markdown()
            confusion_df = gr.Dataframe(label="Confusion matrix on the test split (3 outcomes)",
                                        interactive=False)
            cost_plot = gr.Plot(label="Cost against threshold")
        with gr.Tab("Per-class results"):
            gr.Markdown(overall_text())
            gr.Dataframe(value=per_class_table(), label="Per class (test split)", interactive=False)
            gr.Plot(value=class_confusion_figure(), label="Class confusion matrix")
        with gr.Tab("Baselines and strategies"):
            gr.Markdown("### Same test split, image level: does the metre have any defect?")
            gr.Dataframe(value=baselines, interactive=False)
            gr.Markdown(f"### Four strategies, € per 1,000 metres at a {P['base_rate']:.0%} defect rate")
            gr.Dataframe(value=strategies, interactive=False)
        with gr.Tab("Sensitivity"):
            gr.Markdown("Thresholds are chosen on validation and the cost is measured on test, "
                        "re-chosen for every row. `review_share` 0 means a person is not worth it.")
            gr.Dataframe(value=sensitivity_table, interactive=False)
        with gr.Tab("Dataset"):
            gr.Dataframe(value=dataset_df, interactive=False)
            gr.Markdown(leak_note)

    outputs_cost = [cost_md, confusion_df, cost_plot]
    image_in.change(predict, [image_in, low, high], [image_out, decision_md])
    low.change(predict, [image_in, low, high], [image_out, decision_md])
    high.change(predict, [image_in, low, high], [image_out, decision_md])
    low.change(update, [low, high], outputs_cost)
    high.change(update, [low, high], outputs_cost)
    demo.load(update, [low, high], outputs_cost)

if __name__ == "__main__":
    demo.launch()
