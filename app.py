import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = ROOT / "weights"
PORT = int(os.environ.get("APP_PORT", "7860"))
IMGSZ = 512
DISPLAY_FLOOR = 0.01
PALETTE = [(220, 50, 50), (40, 120, 220), (40, 170, 90), (230, 160, 30), (150, 80, 200)]


def read_results(name: str, required: bool = True):
    path = RESULTS_DIR / name
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"{path} is missing. Run `uv run python main.py train` and commit the results folder."
            )
        return None
    return json.loads(path.read_text())


def find_weights() -> Path:
    best = WEIGHTS_DIR / "best.pt"
    if best.is_file():
        return best
    found = sorted(WEIGHTS_DIR.glob("*.pt"))
    if not found:
        raise FileNotFoundError(f"No .pt file in {WEIGHTS_DIR}. Commit the weights from your training run.")
    return found[0]


COSTING = read_results("costing.json")
METRICS = read_results("metrics.json")
MODEL = YOLO(str(find_weights()))
NAMES = dict(MODEL.names)
POSITIVE = COSTING["positive_class"]
POSITIVE_ID = next(index for index, name in NAMES.items() if name == POSITIVE)
POS = np.array(COSTING["scores"]["positive"])
NEG = np.array(COSTING["scores"]["negative"])
CONFIG = COSTING["config"]
DERIVED = COSTING["derived"]
GRID = np.array(COSTING["grid"])
GRID_TPR = np.array(COSTING["tpr"])
GRID_FPR = np.array(COSTING["fpr"])
SCENARIOS = list(CONFIG["sensitivity"]["miss_scenarios"])
EVENT_CHOICES = [int(e) for e in CONFIG["sensitivity"]["events_per_year"]]
CHOSEN = COSTING["band_optimum_within_review_capacity"]


def build_params(scenario: str, events: int) -> dict:
    spec = CONFIG["sensitivity"]["miss_scenarios"][scenario]
    total = CONFIG["costs_eur"]["compensation_per_death"] * spec["death_given_event"]
    per_image = total / CONFIG["events"]["images_per_event"] if spec["spread_over_event"] else total
    return {
        "pi": events * CONFIG["events"]["images_per_event"] / DERIVED["cases_per_year"],
        "c_fn": per_image,
        "c_tp": DERIVED["c_alert"],
        "c_fp": DERIVED["c_alert"],
        "c_tn": DERIVED["c_true_negative"],
        "c_rev": DERIVED["c_review"],
        "sens": CONFIG["review"]["sensitivity"],
        "spec": CONFIG["review"]["specificity"],
    }


def design_outcome(lo: float, hi: float, p: dict) -> dict:
    ti, tj = float((POS >= lo).mean()), float((POS >= hi).mean())
    fi, fj = float((NEG >= lo).mean()), float((NEG >= hi).mean())
    pi = p["pi"]
    positive = (1 - ti) * p["c_fn"] + tj * p["c_tp"] + (ti - tj) * (
        p["c_rev"] + p["sens"] * p["c_tp"] + (1 - p["sens"]) * p["c_fn"]
    )
    negative = (1 - fi) * p["c_tn"] + fj * p["c_fp"] + (fi - fj) * (
        p["c_rev"] + p["spec"] * p["c_tn"] + (1 - p["spec"]) * p["c_fp"]
    )
    cost = pi * positive + (1 - pi) * negative
    review_share = pi * (ti - tj) + (1 - pi) * (fi - fj)
    alert_share = pi * tj + (1 - pi) * fj
    per_day = DERIVED["cases_per_day"]
    review_seconds = CONFIG["costs_eur"]["review_seconds"]
    return {
        "cost_per_case": cost,
        "cost_per_1000": cost * 1000,
        "cost_per_year": cost * DERIVED["cases_per_year"],
        "review_share": review_share,
        "alert_share": alert_share,
        "reviews_per_day": review_share * per_day,
        "review_hours_per_day": review_share * per_day * review_seconds / 3600,
        "alerts_per_day": alert_share * per_day,
        "false_alerts_per_day": (1 - pi) * fj * per_day,
        "missed_images_per_year": pi * DERIVED["cases_per_year"] * ((1 - ti) + (ti - tj) * (1 - p["sens"])),
        "tpr_alert": tj,
        "tpr_review_or_alert": ti,
        "fpr_alert": fj,
        "fpr_review_or_alert": fi,
    }


def do_nothing_per_case(p: dict) -> float:
    return p["pi"] * p["c_fn"] + (1 - p["pi"]) * p["c_tn"]


def human_all_per_case(p: dict) -> float:
    return (
        p["c_rev"]
        + p["pi"] * (p["sens"] * p["c_tp"] + (1 - p["sens"]) * p["c_fn"])
        + (1 - p["pi"]) * (p["spec"] * p["c_tn"] + (1 - p["spec"]) * p["c_fp"])
    )


def binary_curve(p: dict) -> np.ndarray:
    pi = p["pi"]
    return 1000 * (
        pi * (GRID_TPR * p["c_tp"] + (1 - GRID_TPR) * p["c_fn"])
        + (1 - pi) * (GRID_FPR * p["c_fp"] + (1 - GRID_FPR) * p["c_tn"])
    )


def euro(value: float) -> str:
    return f"€{value:,.0f}" if abs(value) >= 100 else f"€{value:,.2f}"


def operating_point(low: float, high: float, scenario: str, events: int):
    lo, hi = sorted((float(low), float(high)))
    p = build_params(scenario, int(events))
    current = design_outcome(lo, hi, p)
    chosen = design_outcome(CHOSEN["threshold_low"], CHOSEN["threshold_high"], p)
    nothing = do_nothing_per_case(p)
    everything = human_all_per_case(p)
    per_year = DERIVED["cases_per_year"]
    camera_year = DERIVED["camera_cost_per_year"]

    lines = [
        f"### Current design: pass below **{lo:.3f}**, review from {lo:.3f} to {hi:.3f}, alert at **{hi:.3f}** or above",
        f"Scenario: **{scenario}** cost of a missed image ({euro(p['c_fn'])} per image), **{int(events)}** real incidents a year (base rate {p['pi']:.2e} per image).",
        "",
        f"- Expected cost: **{euro(current['cost_per_1000'])} per 1,000 cases**, about **{euro(current['cost_per_year'])} a year** in losses and labour, plus {euro(camera_year)} a year for cameras",
        f"- Sent to a person: **{current['review_share'] * 100:.3f}%** of cases, about **{current['reviews_per_day']:,.0f} a day** ({current['review_hours_per_day']:.2f} hours of checking a day)",
        f"- Alerts: about {current['alerts_per_day']:,.2f} a day, of which {current['false_alerts_per_day']:,.2f} are false alarms",
        f"- Missed drowning images: about {current['missed_images_per_year']:,.1f} a year",
        f"- Measured on the test set: {current['tpr_alert'] * 100:.1f}% of drowning images alerted, {current['tpr_review_or_alert'] * 100:.1f}% alerted or reviewed; {current['fpr_alert'] * 100:.1f}% of other images alerted",
        "",
        "| Strategy | Expected cost per year |",
        "|---|---|",
        f"| Do nothing | {euro(nothing * per_year)} |",
        f"| This design | {euro(current['cost_per_year'] + camera_year)} |",
        f"| Chosen operating point ({CHOSEN['threshold_low']:.3f} / {CHOSEN['threshold_high']:.3f}) | {euro(chosen['cost_per_year'] + camera_year)} |",
        f"| A person checks every image | {euro(everything * per_year + camera_year)} |",
    ]
    if hi < lo or float(low) > float(high):
        lines.append("")
        lines.append("The two edges were swapped so the review band stays valid.")
    if current["fpr_alert"] == 0:
        bound = (1 - p["pi"]) * 3 / len(NEG) * DERIVED["cases_per_day"]
        lines.append("")
        lines.append(
            f"**Caution:** none of the {len(NEG)} test images without drowning triggered an alert, but that only bounds the "
            f"false-alarm rate at about {3 / len(NEG) * 100:.1f}%, which could mean up to about {bound:,.0f} false alerts a day."
        )

    matrix = pd.DataFrame(
        {
            "pass": [int((POS < lo).sum()), int((NEG < lo).sum())],
            "review": [int(((POS >= lo) & (POS < hi)).sum()), int(((NEG >= lo) & (NEG < hi)).sum())],
            "alert": [int((POS >= hi).sum()), int((NEG >= hi).sum())],
        },
        index=[f"drowning in image ({len(POS)})", f"no drowning in image ({len(NEG)})"],
    ).reset_index(names="test set images")

    curve = binary_curve(p)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=GRID, y=np.maximum(curve, 1e-9), mode="lines", name="alert if score >= threshold"))
    figure.add_hline(y=nothing * 1000, line_dash="dash", line_color="gray", annotation_text="do nothing")
    figure.add_hline(y=everything * 1000, line_dash="dot", line_color="crimson", annotation_text="person checks everything")
    figure.add_trace(
        go.Scatter(
            x=[CHOSEN["threshold_high"]],
            y=[max(chosen["cost_per_1000"], 1e-9)],
            mode="markers",
            marker=dict(size=16, symbol="star", color="green"),
            name="chosen operating point",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[hi],
            y=[max(current["cost_per_1000"], 1e-9)],
            mode="markers",
            marker=dict(size=11, symbol="diamond", color="purple"),
            name="current design",
        )
    )
    figure.update_yaxes(type="log", title="expected cost per 1,000 cases (EUR)")
    figure.update_xaxes(title="confidence threshold")
    figure.update_layout(title="Expected cost against threshold", height=430, legend=dict(orientation="h", y=-0.25))
    return "\n".join(lines), matrix, figure


def run_model(image, low: float, high: float):
    if image is None:
        return None, "Upload an image to see the model's predictions."
    lo, hi = sorted((float(low), float(high)))
    height, width = image.shape[:2]
    resized = cv2.resize(image, (IMGSZ, IMGSZ))
    result = MODEL.predict(resized, imgsz=IMGSZ, conf=0.001, iou=0.6, verbose=False)[0]
    found = result.boxes
    boxes = found.xyxyn.cpu().numpy()
    confidences = found.conf.cpu().numpy()
    classes = found.cls.cpu().numpy().astype(int)

    positive_scores = confidences[classes == POSITIVE_ID]
    score = float(positive_scores.max()) if len(positive_scores) else 0.0
    if score >= hi:
        verdict = "ALERT"
    elif score >= lo:
        verdict = "REVIEW"
    else:
        verdict = "PASS"

    annotated = image.copy()
    shown = []
    for box, confidence, class_id in zip(boxes, confidences, classes):
        if confidence < max(lo, DISPLAY_FLOOR):
            continue
        x1, y1, x2, y2 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
        colour = PALETTE[class_id % len(PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, max(2, width // 300))
        cv2.putText(
            annotated,
            f"{NAMES[class_id]} {confidence:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, width / 900),
            colour,
            2,
        )
        shown.append(f"{NAMES[class_id]} {confidence:.2f}")

    text = [
        f"## {verdict}",
        f"Highest {POSITIVE} score: **{score:.3f}** (pass below {lo:.3f}, review {lo:.3f} to {hi:.3f}, alert at {hi:.3f} or above)",
        f"Detections shown at confidence {max(lo, DISPLAY_FLOOR):.3f} or higher: {', '.join(shown) if shown else 'none'}",
        "",
        "The image is stretched to 512 by 512 before prediction, as the training images were. "
        "The training data also had contrast equalisation applied, which is not reproduced here, so photos "
        "unlike the training images can score differently.",
    ]
    return annotated, "\n".join(text)


def fmt(value):
    return "n/a" if value is None else value


def per_class_views(view: str):
    key = "models" if view.startswith("Full") else "models_deduplicated"
    images = METRICS["test_images_full"] if key == "models" else METRICS["test_images_deduplicated"]
    models = METRICS[key]
    tuned = models["finetuned"]

    classes = pd.DataFrame(
        [
            {
                "class": name,
                "test instances": row["instances"],
                "AP50": fmt(row["ap50"]),
                "AP50-95": fmt(row["ap50_95"]),
                "precision": fmt(row["precision"]),
                "recall": fmt(row["recall"]),
                "TP": row["tp"],
                "FP": row["fp"],
                "FN": row["fn"],
            }
            for name, row in tuned["per_class"].items()
        ]
    )

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "mAP50": fmt(result["map50"]),
                "mAP50-95": fmt(result["map50_95"]),
                **{f"AP50 {cls}": fmt(row["ap50"]) for cls, row in result["per_class"].items()},
            }
            for name, result in models.items()
        ]
    )

    labels = tuned["confusion_labels"]
    matrix = np.array(tuned["confusion_matrix"])
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=matrix.astype(str),
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    figure.update_yaxes(autorange="reversed", title="true")
    figure.update_xaxes(title=f"predicted (confidence {tuned['conf_threshold']})")
    figure.update_layout(title=f"Finetuned model confusion matrix, {images} test images", height=420)
    return f"{images} test images", classes, comparison, figure


def duplicate_sweep_table() -> pd.DataFrame:
    rows = []
    for threshold, result in METRICS["finetuned_by_duplicate_threshold"].items():
        row = {"distance above": threshold, "test images": result["test_images"], "mAP50": fmt(result["map50"])}
        for name, values in result["per_class"].items():
            row[f"{name} AP50"] = fmt(values["ap50"])
            row[f"{name} recall"] = fmt(values["recall"])
        rows.append(row)
    return pd.DataFrame(rows)


def static_cost_tables():
    per_case = pd.DataFrame(
        [
            {"outcome": "missed drowning image (false negative)", "EUR": round(COSTING["costs_per_case"]["false_negative_missed_image"], 4)},
            {"outcome": "false alarm", "EUR": round(COSTING["costs_per_case"]["false_positive_alert"], 4)},
            {"outcome": "correct alert", "EUR": round(COSTING["costs_per_case"]["true_positive_alert"], 4)},
            {"outcome": "correct all-clear", "EUR": round(COSTING["costs_per_case"]["true_negative"], 4)},
            {"outcome": "one human check", "EUR": round(COSTING["costs_per_case"]["human_review"], 4)},
        ]
    )
    strategies = pd.DataFrame(
        [
            {"strategy": name.replace("_", " "), "expected cost per year (EUR)": row["cost_per_year"]}
            for name, row in COSTING["alternatives_per_year"].items()
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "missed-image cost": row["miss_scenario"],
                "EUR per missed image": row["miss_cost_per_image"],
                "incidents per year": row["events_per_year"],
                "review band": f"{row['threshold_low']:.3f} to {row['threshold_high']:.3f}",
                "reviews per day": row["reviews_per_day"],
                "system EUR per year": row["system_cost_per_year"],
                "do nothing EUR per year": row["do_nothing_cost_per_year"],
                "person checks everything EUR per year": row["human_checks_everything_per_year"],
                "system beats doing nothing": row["review_system_beats_doing_nothing"],
            }
            for row in COSTING["sensitivity"]
        ]
    )
    staffing = COSTING["staffing"]
    even = COSTING["break_even"]
    notes = [
        f"**Cases:** {DERIVED['cases_per_day']:,.0f} a day ({CONFIG['operating']['cameras']} cameras, one image every {CONFIG['operating']['seconds_between_images']} seconds, "
        f"{CONFIG['operating']['hours_per_day']} open hours), base rate {DERIVED['base_rate']:.2e} per image.",
        f"**Staffing:** covering one dedicated monitoring seat for {staffing['open_hours_per_year']:,.0f} open hours a year takes "
        f"{staffing['people_needed_to_cover_one_seat']} people and costs about {euro(staffing['cost_of_one_dedicated_seat_per_year'])} a year. "
        f"At the chosen operating point the checking itself takes {staffing['review_hours_per_day_at_optimum']:.2f} hours a day.",
    ]
    variable = even["miss_cost_per_image_where_review_system_beats_doing_nothing"]
    covers = even["miss_cost_per_image_where_it_also_covers_camera_cost"]
    if variable is not None:
        notes.append(f"**Break-even:** the system beats doing nothing once a missed image costs more than about {euro(variable)}, and also covers its camera cost above about {euro(covers) if covers else 'the searched range'}.")
    duplicate = COSTING.get("deduplicated_test_set")
    if duplicate:
        band = duplicate["band"]
        notes.append(
            f"**Deduplicated test set ({duplicate['test_images']} images):** best thresholds {band['threshold_low']:.3f} / {band['threshold_high']:.3f}, "
            f"alert-only recall {band['image_recall_alert_only'] * 100:.0f}%, about {euro(band['cost_per_year'])} a year in losses and labour."
        )
    return per_case, strategies, sensitivity, "\n\n".join(notes)


INTRO = (
    "# Pool drowning alerts: operating point dashboard\n"
    f"One case is one camera image. A case is an alert when the model's highest **{POSITIVE}** confidence reaches the upper edge, "
    "goes to a person when it falls in the review band, and passes otherwise. "
    "Move the two edges and choose a cost scenario: every table and chart below updates together. "
    f"The green star marks the chosen operating point ({CHOSEN['threshold_low']:.3f} / {CHOSEN['threshold_high']:.3f}), "
    "the minimum of the cost sweep under the middle scenario, within the review capacity."
)


def build_app() -> gr.Blocks:
    per_case, strategies, sensitivity, notes = static_cost_tables()
    with gr.Blocks(title="Pool drowning alerts") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            low = gr.Slider(0, 1, value=CHOSEN["threshold_low"], step=0.005, label="Review band: lower edge (below this a case passes)")
            high = gr.Slider(0, 1, value=CHOSEN["threshold_high"], step=0.005, label="Alert threshold (at or above this a case is an alert)")
        with gr.Row():
            scenario = gr.Dropdown(SCENARIOS, value="middle", label="Cost of a missed image")
            events = gr.Dropdown(EVENT_CHOICES, value=int(CONFIG["events"]["events_per_year"]), label="Real incidents per year")

        with gr.Tabs():
            with gr.Tab("Operating point"):
                summary = gr.Markdown()
                matrix = gr.Dataframe(label="Confusion matrix on the test set", interactive=False)
                curve = gr.Plot()
            with gr.Tab("Try an image"):
                with gr.Row():
                    upload = gr.Image(type="numpy", label="Upload an image")
                    annotated = gr.Image(label="Predictions", interactive=False)
                verdict = gr.Markdown()
            with gr.Tab("Per-class results"):
                view = gr.Radio(
                    ["Full test split", "Deduplicated test split"],
                    value="Full test split",
                    label="Which test images",
                )
                view_caption = gr.Markdown()
                class_table = gr.Dataframe(label="Finetuned model, per class", interactive=False)
                comparison_table = gr.Dataframe(label="Finetuned model against the two baselines", interactive=False)
                confusion = gr.Plot()
                gr.Markdown("### How the score changes as near-duplicates of training images are removed")
                gr.Dataframe(value=duplicate_sweep_table(), interactive=False)
            with gr.Tab("Costs and sensitivity"):
                gr.Markdown(notes)
                gr.Dataframe(value=per_case, label="Cost per case (EUR)", interactive=False)
                gr.Dataframe(value=strategies, label="Strategies under the middle scenario", interactive=False)
                gr.Dataframe(value=sensitivity, label="Sensitivity: cost of a miss against incidents per year", interactive=False)

        controls = [low, high, scenario, events]
        for control in controls:
            control.change(operating_point, controls, [summary, matrix, curve])
        for slider in (low, high):
            slider.release(run_model, [upload, low, high], [annotated, verdict])
        upload.change(run_model, [upload, low, high], [annotated, verdict])
        view.change(per_class_views, view, [view_caption, class_table, comparison_table, confusion])
        demo.load(operating_point, controls, [summary, matrix, curve])
        demo.load(per_class_views, view, [view_caption, class_table, comparison_table, confusion])
    return demo


if __name__ == "__main__":
    build_app().launch(server_port=PORT)
