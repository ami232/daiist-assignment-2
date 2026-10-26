"""
Dashboard for Assignment 2.

    uv run python main.py app

Loads weights/best.pt and the files train.py saved in outputs/. It never trains
and never needs the dataset: every number on screen is recomputed from the
saved per-image scores, so moving a slider takes milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as cfg
import costing
import evaluation as ev

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WEIGHTS = ROOT / "weights" / "best.pt"
UPLOAD_CONF = 0.01
POS = cfg.POSITIVE_NAME.capitalize()
BLUE, GREEN, ORANGE = "#1f77b4", "#2ca02c", "#ff7f0e"


# The Colab run's results can also be committed as the results.zip it downloads.
# If outputs/ is missing, unpack that zip's outputs/ folder: reading saved
# results, never training.
if not (OUTPUTS / "image_scores.json").is_file() and (ROOT / "results.zip").is_file():
    import zipfile

    with zipfile.ZipFile(ROOT / "results.zip") as archive:
        archive.extractall(ROOT, members=[m for m in archive.namelist() if m.startswith("outputs/")])


def read(name: str):
    path = OUTPUTS / name
    return json.loads(path.read_text()) if path.is_file() else None


EVAL = read("evaluation.json")
STATS = read("dataset_stats.json")
OFFICIAL = (EVAL or {}).get("ultralytics_validator")
SCORES = costing.load_scores(OUTPUTS) if (OUTPUTS / "image_scores.json").is_file() else None
READY = EVAL is not None and SCORES is not None

MODEL = None
if WEIGHTS.is_file():
    from ultralytics import YOLO

    MODEL = YOLO(str(WEIGHTS))

NAMES = EVAL["names"] if EVAL else (list(MODEL.names.values()) if MODEL else [])
POSITIVE_IDS = [NAMES.index(c) for c in cfg.POSITIVE_CLASSES if c in NAMES]
A0 = costing.Assumptions.from_config()
NOT_READY = "outputs/ has no results yet. Run `uv run python main.py train` first."


def chosen(a: costing.Assumptions) -> tuple[float, float, float]:
    """Model-alone threshold and review band chosen on the validation split."""
    val = SCORES["finetuned"]["val"]
    t, _ = costing.best_threshold(*val, a)
    low, high, _ = costing.best_band(*val, a)
    return t, low, high


T0, LOW0, HIGH0 = chosen(A0) if READY else (0.5, 0.3, 0.7)


def make_assumptions(base_rate_pct, c_fp, c_fn, c_tp, c_tn, seconds, per_hour, cases, reviewers):
    def num(x, default=0.0):
        return default if x is None else float(x)
    return costing.Assumptions(
        base_rate=min(max(num(base_rate_pct) / 100, 0.0), 1.0),
        cost_fp=num(c_fp), cost_fn=num(c_fn), cost_tp=num(c_tp), cost_tn=num(c_tn),
        seconds_per_review=max(num(seconds, 1.0), 1e-6), reviewer_cost_per_hour=num(per_hour),
        cases_per_day=num(cases), productive_hours=cfg.PRODUCTIVE_HOURS_PER_REVIEWER,
        reviewers_available=num(reviewers))


def eur(x: float) -> str:
    return f"€{x:,.2f}" if abs(x) < 100 else f"€{x:,.0f}"


def fmt(x, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


# ------------------------------------------------------------ check an image --

def run_model(image):
    if image is None or MODEL is None:
        return None
    r = MODEL.predict(image, imgsz=cfg.IMAGE_SIZE, conf=UPLOAD_CONF, verbose=False)[0]
    return {"cls": r.boxes.cls.cpu().numpy().astype(int),
            "conf": r.boxes.conf.cpu().numpy().astype(float),
            "boxes": r.boxes.xyxyn.cpu().numpy().astype(float).reshape(-1, 4)}


def render(image, dets, t_low, t_high):
    low, high = sorted((float(t_low), float(t_high)))
    if image is None:
        return None, "Upload an image or pick one of the test images below."
    if dets is None:
        return image, "weights/best.pt is missing, so there is no model to run."
    drawn = ev.draw(np.array(image.convert("RGB")), dets["boxes"], dets["cls"], dets["conf"],
                    NAMES, POSITIVE_IDS, min_conf=low)
    score = float(dets["conf"][np.isin(dets["cls"], POSITIVE_IDS)].max(initial=0.0))
    what = " or ".join(cfg.POSITIVE_CLASSES)
    if score >= high:
        verdict = (f"### Fail\nThe most confident {what} box scores {score:.2f}, at or above "
                   f"{high:.2f}, so the model flags this case without a person.")
    elif score >= low:
        verdict = (f"### Send to a person\nThe most confident {what} box scores {score:.2f}, "
                   f"between {low:.2f} and {high:.2f}: too unsure to decide automatically.")
    else:
        verdict = (f"### Pass\nNo {what} box scores {low:.2f} or more (highest: {score:.2f}), "
                   f"so the case passes without a person.")
    shown = int((dets["conf"] >= low).sum())
    verdict += (f"\n\nShowing {shown} boxes with confidence of at least {low:.2f}. "
                f"Red boxes are {what}; blue boxes are other classes.")
    return drawn, verdict


def on_upload(image, t_low, t_high):
    dets = run_model(image)
    drawn, verdict = render(image, dets, t_low, t_high)
    return dets, drawn, verdict


# ------------------------------------------------------ cost and review load --

def cost_figure(a: costing.Assumptions, low: float, high: float, t_best: float) -> go.Figure:
    grid = costing.GRID
    fig = go.Figure()
    fig.add_vrect(x0=low, x1=high, fillcolor=ORANGE, opacity=0.15, line_width=0)
    curves = {}
    for model, split, name, dash, color, width in (
        ("finetuned", "test", "Finetuned, test", "solid", BLUE, 3),
        ("finetuned", "val", "Finetuned, validation (used to choose)", "dash", BLUE, 1.5),
        ("coco_zero_shot", "test", "COCO zero-shot, test", "solid", GREEN, 1.5),
    ):
        curves[(model, split)] = y = costing.sweep(*SCORES[model][split], a)["cost_per_1000"]
        fig.add_trace(go.Scatter(x=grid, y=y, name=name, mode="lines",
                                 line=dict(dash=dash, color=color, width=width)))
    nothing = 1000 * (a.base_rate * a.cost_fn + (1 - a.base_rate) * a.cost_tn)
    everyone = 1000 * (a.cost_review + a.base_rate * a.cost_tp + (1 - a.base_rate) * a.cost_tn)
    fig.add_hline(y=nothing, line_dash="dot", line_color="grey",
                  annotation_text="do nothing", annotation_position="top left")
    fig.add_hline(y=everyone, line_dash="dashdot", line_color="grey",
                  annotation_text="person checks everything", annotation_position="bottom left")
    i = int(np.abs(grid - t_best).argmin())
    fig.add_vline(x=t_best, line_color="black", line_width=1)
    fig.add_trace(go.Scatter(x=[t_best], y=[curves[("finetuned", "test")][i]], mode="markers+text",
                             marker=dict(size=11, color="black"), text=[f"chosen t = {t_best:.2f}"],
                             textposition="top right", name="Chosen threshold (validation)"))
    top = 1.2 * max(nothing, min(everyone, 3 * nothing), float(np.min(curves[("finetuned", "test")])))
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=60, r=20, t=30, b=40),
                      xaxis_title="Confidence threshold: fail at or above",
                      yaxis_title="Expected cost per 1,000 cases (€)",
                      yaxis_range=[0, top if top > 0 else None],
                      legend=dict(orientation="h", yanchor="top", y=-0.18))
    return fig


def operating_view(t_low, t_high, *values):
    if not READY:
        return NOT_READY, None, None, None
    a = make_assumptions(*values)
    low, high = sorted((float(t_low), float(t_high)))
    test = SCORES["finetuned"]["test"]
    result = costing.evaluate_band(*test, a, low, high)
    t_best, low_best, high_best = chosen(a)
    staff = result["staffing"]
    fits = "within the team" if staff["fits_team"] else "more than the team has"

    kpis = (
        f"| Expected cost per 1,000 cases | Sent to a person | Reviews per day | People needed |\n"
        f"|---|---|---|---|\n"
        f"| **{eur(result['cost_per_1000'])}** | **{result['review_share_pct']:.1f}%** "
        f"| {staff['reviews_per_day']:,.0f} | {staff['people_needed']:.2f} of "
        f"{staff['people_available']:g} available ({fits}) |\n\n"
        f"With these thresholds the model passes {result['positives_missed_pct']:.1f}% of the "
        f"{cfg.POSITIVE_NAME} cases and fails {result['negatives_flagged_pct']:.1f}% of the fine ones. "
        f"A review costs {eur(a.cost_review)} of a person's time.\n\n"
        f"Cheapest band for these assumptions, chosen on the validation split: pass below "
        f"**{low_best:.2f}**, fail at or above **{high_best:.2f}**. Cheapest single threshold "
        f"with no review: **{t_best:.2f}**."
    )

    counts = result["counts"]

    def cell(truth: str, decision: str) -> str:
        total = sum(counts[truth].values())
        n = counts[truth][decision]
        return f"{n}  ({100 * n / total:.1f}%)" if total else "0"

    matrix = pd.DataFrame({
        "Truth (test split)": [f"{POS} (n = {sum(counts['positive'].values())})",
                               f"Fine (n = {sum(counts['negative'].values())})"],
        "Pass": [cell("positive", "pass"), cell("negative", "pass")],
        "Person reviews": [cell("positive", "review"), cell("negative", "review")],
        "Fail": [cell("positive", "fail"), cell("negative", "fail")],
    })

    rows = costing.strategy_table(*test, a, t_best, (low_best, high_best))
    rows.append({"strategy": f"Your sliders (pass < {low:.2f}, fail ≥ {high:.2f})",
                 "cost_per_1000": result["cost_per_1000"], "cost_per_day": result["cost_per_day"],
                 "review_share_pct": result["review_share_pct"], "people_needed": staff["people_needed"],
                 "fits_team": staff["fits_team"], "positives_missed_pct": result["positives_missed_pct"],
                 "negatives_flagged_pct": result["negatives_flagged_pct"]})
    strategies = pd.DataFrame([{
        "Strategy": r["strategy"],
        "Cost per 1,000": eur(r["cost_per_1000"]),
        "Cost per day": eur(r["cost_per_day"]),
        "Sent to a person": f"{r['review_share_pct']:.1f}%",
        "People needed": f"{r['people_needed']:.2f}",
        "Fits the team": "yes" if r["fits_team"] else "no",
        f"{POS} missed": f"{r['positives_missed_pct']:.1f}%",
        "Fine cases failed": f"{r['negatives_flagged_pct']:.1f}%",
    } for r in rows])
    return kpis, matrix, strategies, cost_figure(a, low, high, t_best)


# ------------------------------------------------------ per class, baselines --

def per_class_view(t_low, t_high, *values):
    if not READY:
        return NOT_READY, None, None, None
    a = make_assumptions(*values)
    low, high = sorted((float(t_low), float(t_high)))
    fin = EVAL["models"]["finetuned"]
    grid = np.array(fin["pr_grid"]["thresholds"])
    i_low, i_high = int(np.abs(grid - low).argmin()), int(np.abs(grid - high).argmin())

    rows = []
    for c, pc in enumerate(fin["per_class"]):
        precision, recall = fin["pr_grid"]["precision"][c], fin["pr_grid"]["recall"][c]
        rows.append({
            "Class": pc["class"] + ("  (positive)" if c in POSITIVE_IDS else ""),
            "Test instances": pc["instances"],
            "AP50": fmt(pc["ap50"]), "AP50-95": fmt(pc["ap50_95"]),
            "Precision, lower t": fmt(precision[i_low]), "Recall, lower t": fmt(recall[i_low]),
            "Precision, upper t": fmt(precision[i_high]), "Recall, upper t": fmt(recall[i_high]),
        })
    rows.append({"Class": "All classes (mean)", "Test instances": sum(p["instances"] for p in fin["per_class"]),
                 "AP50": fmt(fin["map50"]), "AP50-95": fmt(fin["map50_95"]),
                 "Precision, lower t": "", "Recall, lower t": "", "Precision, upper t": "", "Recall, upper t": ""})
    note = (f"Box-level results on the test split, IoU ≥ 0.5. Lower t = {low:.2f} (pass below), "
            f"upper t = {high:.2f} (fail at or above). A dash means no box reaches that confidence.")

    cm = fin["confusion"]
    cm_grid = np.array(cm["thresholds"])
    k = int(np.abs(cm_grid - high).argmin())
    matrix = np.array(cm["matrices"][k])
    labels = cm["labels"]
    confusion = go.Figure(go.Heatmap(z=matrix, x=[f"predicted {n}" for n in labels],
                                     y=[f"true {n}" for n in labels], text=matrix,
                                     texttemplate="%{text}", colorscale="Blues", showscale=False))
    confusion.update_layout(template="plotly_white", height=360, margin=dict(l=110, r=20, t=40, b=40),
                            yaxis_autorange="reversed",
                            title=f"Objects at confidence ≥ {cm_grid[k]:.2f}, matched at IoU > {cm['iou']}")

    models = []
    for key, label in (("finetuned", "Finetuned"), ("coco_zero_shot", "COCO zero-shot"),
                       ("predict_nothing", "Predict nothing")):
        m = EVAL["models"][key]
        if key == "predict_nothing":   # no box is ever found, so every case passes
            t, threshold_text = np.inf, "none: all pass"
        else:
            t, _ = costing.best_threshold(*SCORES[key]["val"], a)
            threshold_text = f"{t:.2f}"
        result = costing.evaluate_band(*SCORES[key]["test"], a, t, t)
        models.append({
            "Model": label, "mAP50": fmt(m["map50"]), "mAP50-95": fmt(m["map50_95"]),
            "Image AUC": fmt(EVAL["image_level"][key]["test"]["roc_auc"]),
            "Best threshold (val)": threshold_text,
            "Cost per 1,000 (test)": eur(result["cost_per_1000"]),
            f"{POS} caught": f"{100 - result['positives_missed_pct']:.1f}%",
            "Fine cases failed": f"{result['negatives_flagged_pct']:.1f}%",
        })
    return note, pd.DataFrame(rows), confusion, pd.DataFrame(models)


# ------------------------------------------------------------------ dataset --

def dataset_tables():
    if not STATS:
        return None, None
    splits = pd.DataFrame([{
        "Split": split, "Images": s["images"], "Images with a positive": s["positive_images"],
        "Share positive": f"{s['positive_image_pct']}%", "Objects per image": s["objects_per_image"],
        "Images with no objects": s["images_without_objects"]} for split, s in STATS["splits"].items()])
    classes = []
    for i, name in enumerate(STATS["names"]):
        train = STATS["splits"]["train"]["per_class"][i]
        classes.append({"Class": name, **{f"{split} instances": s["per_class"][i]["instances"]
                                          for split, s in STATS["splits"].items()},
                        "Median box area (train)": f"{train['median_area_pct']}%",
                        "Boxes under 1% of image (train)": f"{train['small_pct']}%"})
    return splits, pd.DataFrame(classes)


def official_table():
    if not OFFICIAL:
        return None
    rows = [{"Class": p["class"], "Precision": fmt(p["precision"]), "Recall": fmt(p["recall"]),
             "AP50": fmt(p["ap50"]), "AP50-95": fmt(p["ap50_95"])} for p in OFFICIAL["per_class"]]
    rows.append({"Class": "All", "Precision": fmt(OFFICIAL["precision"]), "Recall": fmt(OFFICIAL["recall"]),
                 "AP50": fmt(OFFICIAL["map50"]), "AP50-95": fmt(OFFICIAL["map50_95"])})
    return pd.DataFrame(rows)


def existing(path: Path) -> str | None:
    return str(path) if path.is_file() else None


# ---------------------------------------------------------------------- UI --

with gr.Blocks(title="Decision dashboard") as demo:
    gr.Markdown(f"# Decision dashboard\n{cfg.DECISION}\n\nA case is one image. It counts as "
                f"**{cfg.POSITIVE_NAME}** when it contains at least one "
                f"**{' or '.join(cfg.POSITIVE_CLASSES)}**. Thresholds are chosen on the validation "
                f"split; every result shown is measured on the test split.")
    if cfg.PLACEHOLDERS:
        gr.Markdown("> **The cost figures in config.py are still placeholders.** "
                    "Replace them before reading anything into the euros.")
    if not READY:
        gr.Markdown(f"> {NOT_READY}")

    with gr.Row(equal_height=True):
        t_low = gr.Slider(0, 1, value=LOW0, step=0.01, label="Pass below (lower threshold)", scale=3)
        t_high = gr.Slider(0, 1, value=HIGH0, step=0.01, label="Fail at or above (upper threshold)", scale=3)
        reset = gr.Button("Use the cost-minimising band", scale=1)

    with gr.Accordion("Assumptions (defaults come from config.py)", open=False):
        with gr.Row():
            base_rate = gr.Number(round(A0.base_rate * 100, 4), label="Base rate: % of cases that are positive", minimum=0, maximum=100)
            c_fp = gr.Number(A0.cost_fp, label="€ per false positive")
            c_fn = gr.Number(A0.cost_fn, label="€ per false negative")
            c_tp = gr.Number(A0.cost_tp, label="€ per true positive")
            c_tn = gr.Number(A0.cost_tn, label="€ per true negative")
        with gr.Row():
            seconds = gr.Number(A0.seconds_per_review, label="Seconds per review", minimum=0)
            per_hour = gr.Number(A0.reviewer_cost_per_hour, label="Reviewer cost, € per hour", minimum=0)
            cases = gr.Number(A0.cases_per_day, label="Cases per day", minimum=0)
            reviewers = gr.Number(A0.reviewers_available, label="Reviewers available", minimum=0)
    assumption_inputs = [base_rate, c_fp, c_fn, c_tp, c_tn, seconds, per_hour, cases, reviewers]

    with gr.Tabs():
        with gr.Tab("Check an image"):
            dets_state = gr.State()
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="pil", label="Image to check", sources=["upload", "clipboard"])
                    example_files = sorted((OUTPUTS / "examples").glob("*.jpg"))
                    if example_files:
                        gr.Examples([[str(p)] for p in example_files], inputs=image_in,
                                    label="Test images the model caught, missed or falsely flagged")
                with gr.Column():
                    image_out = gr.Image(label="Predictions at the current thresholds", interactive=False)
                    verdict = gr.Markdown("Upload an image or pick one of the test images.")

        with gr.Tab("Cost and review load"):
            kpi_md = gr.Markdown()
            gr.Markdown("**Decision matrix on the test split.** Rows are the truth, columns what the "
                        "thresholds do. Percentages are of each row. The euros reweight those row "
                        "rates to the assumed base rate, not to the test split's own mix.")
            matrix_df = gr.Dataframe(interactive=False)
            cost_plot = gr.Plot(label="Cost against threshold")
            gr.Markdown("**The same images under four ways of running the decision**, plus your sliders. "
                        "A person is assumed to decide every reviewed case correctly.")
            strategy_df = gr.Dataframe(interactive=False, wrap=True)

        with gr.Tab("Per class and baselines"):
            class_note = gr.Markdown()
            class_df = gr.Dataframe(interactive=False)
            confusion_plot = gr.Plot(label="Object-level confusion matrix at the upper threshold")
            gr.Markdown("The last column counts true objects the model missed; the last row counts "
                        "boxes that matched no true object. **Model comparison on the same test "
                        "split.** Each model's threshold is chosen on validation with the current "
                        f"assumptions. COCO stand-ins: {(EVAL or {}).get('coco_stand_ins') or 'none'}.")
            models_df = gr.Dataframe(interactive=False, wrap=True)
            with gr.Accordion("Cross-check: the Ultralytics validator on the test split", open=False):
                gr.Markdown((OFFICIAL or {}).get("note", ""))
                gr.Dataframe(official_table(), interactive=False)
                normalised = existing(OUTPUTS / "ultralytics_test" / "confusion_matrix_normalized.png")
                if normalised:
                    gr.Image(normalised, label="Ultralytics confusion matrix (normalised)", interactive=False)

        with gr.Tab("Where it fails"):
            gr.Markdown("The most confident misses and false alarms on the test split at the "
                        "model-alone threshold. White boxes are the truth, red boxes predicted "
                        f"{' or '.join(cfg.POSITIVE_CLASSES)}, blue boxes other predicted classes. "
                        "File names carry the image's score.")
            failure_files = sorted((OUTPUTS / "failures").glob("*.jpg"))
            gr.Gallery([str(p) for p in failure_files], columns=3, height="auto",
                       label="Failures", show_label=False)

        with gr.Tab("Dataset"):
            splits_table, classes_table = dataset_tables()
            gr.Dataframe(splits_table, interactive=False)
            gr.Dataframe(classes_table, interactive=False)
            overview = existing(OUTPUTS / "dataset_overview.png")
            if overview:
                gr.Image(overview, label="Class balance and object size", interactive=False)

    view_inputs = [t_low, t_high, *assumption_inputs]
    triggers = [t_low.change, t_high.change, *[x.change for x in assumption_inputs]]
    gr.on(triggers, operating_view, view_inputs, [kpi_md, matrix_df, strategy_df, cost_plot])
    gr.on(triggers, per_class_view, view_inputs, [class_note, class_df, confusion_plot, models_df])
    gr.on([t_low.change, t_high.change], render, [image_in, dets_state, t_low, t_high], [image_out, verdict])
    image_in.change(on_upload, [image_in, t_low, t_high], [dets_state, image_out, verdict])
    reset.click(lambda *v: chosen(make_assumptions(*v))[1:] if READY else (LOW0, HIGH0),
                assumption_inputs, [t_low, t_high])
    demo.load(operating_view, view_inputs, [kpi_md, matrix_df, strategy_df, cost_plot])
    demo.load(per_class_view, view_inputs, [class_note, class_df, confusion_plot, models_df])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"),
                allowed_paths=[str(OUTPUTS)])
