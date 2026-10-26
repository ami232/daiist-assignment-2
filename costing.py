import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from dataset import DATASET_DIR, RESULTS_DIR, class_names
from evaluate import load_ground_truth, predict

ROOT = Path(__file__).resolve().parent
COSTS_PATH = ROOT / "costs.json"
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
DEFAULT_IMGSZ = 512
GRID = np.round(np.linspace(0.0, 1.0, 201), 3)
BREAK_EVEN_GRID = np.logspace(-1, 6, 281)


def load_config() -> dict:
    return json.loads(COSTS_PATH.read_text())


def derive(config: dict) -> dict:
    op, ev, co = config["operating"], config["events"], config["costs_eur"]
    cases_per_day = 3600 / op["seconds_between_images"] * op["cameras"] * op["hours_per_day"]
    cases_per_year = cases_per_day * op["days_per_year"]
    hourly = co["monitor_hourly"]
    open_hours_per_year = op["hours_per_day"] * op["days_per_year"]
    return {
        "cases_per_day": cases_per_day,
        "cases_per_year": cases_per_year,
        "base_rate": ev["events_per_year"] * ev["images_per_event"] / cases_per_year,
        "c_review": hourly * co["review_seconds"] / 3600,
        "c_alert": hourly * co["alert_response_seconds"] / 3600,
        "c_true_negative": co["true_negative"],
        "miss_cost_per_image": miss_cost(config, co["death_given_event"], True),
        "open_hours_per_year": open_hours_per_year,
        "seat_fte": open_hours_per_year / config["staffing"]["annual_hours_per_person"],
        "seat_cost_per_year": open_hours_per_year * hourly,
        "camera_cost_per_year": config["hardware"]["cameras_total_eur"] / config["hardware"]["amortisation_years"],
    }


def miss_cost(config: dict, death_given_event: float, spread: bool) -> float:
    total = config["costs_eur"]["compensation_per_death"] * death_given_event
    return total / config["events"]["images_per_event"] if spread else total


def params_from(config: dict, derived: dict, c_miss: float = None, base_rate: float = None) -> dict:
    return {
        "pi": derived["base_rate"] if base_rate is None else base_rate,
        "c_fn": derived["miss_cost_per_image"] if c_miss is None else c_miss,
        "c_tp": derived["c_alert"],
        "c_fp": derived["c_alert"],
        "c_tn": derived["c_true_negative"],
        "c_rev": derived["c_review"],
        "sens": config["review"]["sensitivity"],
        "spec": config["review"]["specificity"],
    }


def image_scores(dataset_dir: Path, weights: Path, names: list[str], positive: str, imgsz: int):
    truth = load_ground_truth(dataset_dir, "test")
    images = list(truth)
    predictions = predict(YOLO(str(weights)), images, imgsz)
    positive_id = names.index(positive)
    scores, labels = [], []
    for path in images:
        found = predictions[path]
        mask = found["classes"] == positive_id
        scores.append(float(found["conf"][mask].max()) if mask.any() else 0.0)
        labels.append(bool((truth[path]["classes"] == positive_id).any()))
    return images, np.array(scores), np.array(labels)


def rates(scores: np.ndarray, labels: np.ndarray):
    positives, negatives = scores[labels], scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("The test split needs both positive and negative images")
    tpr = np.array([(positives >= t).mean() for t in GRID])
    fpr = np.array([(negatives >= t).mean() for t in GRID])
    return tpr, fpr


def binary_cost(tpr, fpr, p) -> np.ndarray:
    return p["pi"] * (tpr * p["c_tp"] + (1 - tpr) * p["c_fn"]) + (1 - p["pi"]) * (
        fpr * p["c_fp"] + (1 - fpr) * p["c_tn"]
    )


def band_costs(tpr, fpr, p):
    ti, tj = tpr[:, None], tpr[None, :]
    fi, fj = fpr[:, None], fpr[None, :]
    positive = (1 - ti) * p["c_fn"] + tj * p["c_tp"] + (ti - tj) * (
        p["c_rev"] + p["sens"] * p["c_tp"] + (1 - p["sens"]) * p["c_fn"]
    )
    negative = (1 - fi) * p["c_tn"] + fj * p["c_fp"] + (fi - fj) * (
        p["c_rev"] + p["spec"] * p["c_tn"] + (1 - p["spec"]) * p["c_fp"]
    )
    cost = p["pi"] * positive + (1 - p["pi"]) * negative
    review_share = p["pi"] * (ti - tj) + (1 - p["pi"]) * (fi - fj)
    order = np.arange(len(GRID))
    cost = np.where(order[:, None] <= order[None, :], cost, np.inf)
    return cost, review_share


def do_nothing_cost(p) -> float:
    return p["pi"] * p["c_fn"] + (1 - p["pi"]) * p["c_tn"]


def alert_everything_cost(p) -> float:
    return p["pi"] * p["c_tp"] + (1 - p["pi"]) * p["c_fp"]


def human_checks_everything_cost(p) -> float:
    return (
        p["c_rev"]
        + p["pi"] * (p["sens"] * p["c_tp"] + (1 - p["sens"]) * p["c_fn"])
        + (1 - p["pi"]) * (p["spec"] * p["c_tn"] + (1 - p["spec"]) * p["c_fp"])
    )


def best_band(tpr, fpr, p, derived, config, constrained: bool):
    cost, review_share = band_costs(tpr, fpr, p)
    if constrained:
        review_hours = review_share * derived["cases_per_day"] * config["costs_eur"]["review_seconds"] / 3600
        limit = config["review"]["max_utilisation"] * config["operating"]["hours_per_day"]
        cost = np.where(review_hours <= limit, cost, np.inf)
    index = np.unravel_index(np.argmin(cost), cost.shape)
    return int(index[0]), int(index[1]), float(cost[index])


def describe_band(i, j, tpr, fpr, p, derived, config) -> dict:
    ti, tj, fi, fj = tpr[i], tpr[j], fpr[i], fpr[j]
    pi = p["pi"]
    review_share = pi * (ti - tj) + (1 - pi) * (fi - fj)
    alert_share = pi * tj + (1 - pi) * fj
    reviews_per_day = review_share * derived["cases_per_day"]
    review_hours = reviews_per_day * config["costs_eur"]["review_seconds"] / 3600
    cost = band_costs(tpr, fpr, p)[0][i, j]
    return {
        "threshold_low": float(GRID[i]),
        "threshold_high": float(GRID[j]),
        "cost_per_1000_cases": round(float(cost) * 1000, 4),
        "cost_per_year": round(float(cost) * derived["cases_per_year"], 2),
        "pass_share": round(float(1 - review_share - alert_share), 6),
        "review_share": round(float(review_share), 6),
        "alert_share": round(float(alert_share), 6),
        "reviews_per_day": round(float(reviews_per_day), 2),
        "review_hours_per_day": round(float(review_hours), 3),
        "monitor_utilisation": round(float(review_hours / config["operating"]["hours_per_day"]), 4),
        "alerts_per_day": round(float(alert_share * derived["cases_per_day"]), 2),
        "false_alerts_per_day": round(float((1 - pi) * fj * derived["cases_per_day"]), 2),
        "image_recall_alert_only": round(float(tj), 4),
        "image_recall_alert_or_review": round(float(ti), 4),
        "false_positive_rate_alert_only": round(float(fj), 4),
        "false_positive_rate_alert_or_review": round(float(fi), 4),
        "expected_missed_images_per_year": round(
            float(pi * derived["cases_per_year"] * ((1 - ti) + (ti - tj) * (1 - p["sens"]))), 3
        ),
    }


def alternatives(p, derived, binary_best: float, band_best: float) -> dict:
    cameras = derived["camera_cost_per_year"]
    per_year = derived["cases_per_year"]
    return {
        "do_nothing": {"cost_per_year": round(do_nothing_cost(p) * per_year, 2), "fixed_per_year": 0.0},
        "model_alone": {
            "cost_per_year": round(binary_best * per_year + cameras, 2),
            "variable_per_year": round(binary_best * per_year, 2),
            "fixed_per_year": round(cameras, 2),
        },
        "model_with_review": {
            "cost_per_year": round(band_best * per_year + cameras, 2),
            "variable_per_year": round(band_best * per_year, 2),
            "fixed_per_year": round(cameras, 2),
        },
        "human_checks_everything": {
            "cost_per_year": round(human_checks_everything_cost(p) * per_year + cameras, 2),
            "variable_per_year": round(human_checks_everything_cost(p) * per_year, 2),
            "fixed_per_year": round(cameras, 2),
        },
        "alert_on_everything": {
            "cost_per_year": round(alert_everything_cost(p) * per_year + cameras, 2),
            "variable_per_year": round(alert_everything_cost(p) * per_year, 2),
            "fixed_per_year": round(cameras, 2),
        },
    }


def evaluate_setting(tpr, fpr, p, derived, config) -> dict:
    binary = binary_cost(tpr, fpr, p)
    best_binary = int(np.argmin(binary))
    i, j, band_best = best_band(tpr, fpr, p, derived, config, constrained=True)
    return {
        "binary_threshold": float(GRID[best_binary]),
        "binary_cost_per_case": float(binary[best_binary]),
        "band": describe_band(i, j, tpr, fpr, p, derived, config),
        "band_cost_per_case": band_best,
        "do_nothing_per_case": do_nothing_cost(p),
        "human_all_per_case": human_checks_everything_cost(p),
    }


def break_even(tpr, fpr, derived, config) -> dict:
    variable = None
    with_cameras = None
    for c_miss in BREAK_EVEN_GRID:
        p = params_from(config, derived, c_miss=c_miss)
        _, _, system = best_band(tpr, fpr, p, derived, config, constrained=True)
        nothing = do_nothing_cost(p)
        if variable is None and system < nothing:
            variable = float(c_miss)
        if with_cameras is None and system * derived["cases_per_year"] + derived["camera_cost_per_year"] < nothing * derived["cases_per_year"]:
            with_cameras = float(c_miss)
        if variable is not None and with_cameras is not None:
            break
    return {
        "miss_cost_per_image_where_review_system_beats_doing_nothing": variable,
        "miss_cost_per_image_where_it_also_covers_camera_cost": with_cameras,
        "searched_range_eur": [float(BREAK_EVEN_GRID[0]), float(BREAK_EVEN_GRID[-1])],
    }


def sensitivity_grid(tpr, fpr, derived, config) -> list[dict]:
    rows = []
    for name, scenario in config["sensitivity"]["miss_scenarios"].items():
        c_miss = miss_cost(config, scenario["death_given_event"], scenario["spread_over_event"])
        for events in config["sensitivity"]["events_per_year"]:
            base_rate = events * config["events"]["images_per_event"] / derived["cases_per_year"]
            p = params_from(config, derived, c_miss=c_miss, base_rate=base_rate)
            result = evaluate_setting(tpr, fpr, p, derived, config)
            system_year = result["band_cost_per_case"] * derived["cases_per_year"] + derived["camera_cost_per_year"]
            nothing_year = result["do_nothing_per_case"] * derived["cases_per_year"]
            rows.append(
                {
                    "miss_scenario": name,
                    "miss_cost_per_image": round(c_miss, 4),
                    "events_per_year": events,
                    "base_rate": base_rate,
                    "threshold_low": result["band"]["threshold_low"],
                    "threshold_high": result["band"]["threshold_high"],
                    "review_share": result["band"]["review_share"],
                    "reviews_per_day": result["band"]["reviews_per_day"],
                    "system_cost_per_year": round(system_year, 2),
                    "do_nothing_cost_per_year": round(nothing_year, 2),
                    "human_checks_everything_per_year": round(
                        result["human_all_per_case"] * derived["cases_per_year"] + derived["camera_cost_per_year"], 2
                    ),
                    "review_system_beats_doing_nothing": bool(system_year < nothing_year),
                }
            )
    return rows


def plot_curve(tpr, fpr, p, derived, binary_index: int, band: dict, path: Path) -> None:
    binary = binary_cost(tpr, fpr, p) * 1000
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(GRID, np.maximum(binary, 1e-9), label="alert if score >= threshold")
    axis.axhline(do_nothing_cost(p) * 1000, linestyle="--", color="tab:gray", label="do nothing")
    axis.axhline(human_checks_everything_cost(p) * 1000, linestyle=":", color="tab:red", label="a person checks everything")
    axis.axvline(GRID[binary_index], color="tab:green", alpha=0.6, label=f"minimum at {GRID[binary_index]:.3f}")
    axis.axhline(band["cost_per_1000_cases"], linestyle="-.", color="tab:purple", label="best pass/review/fail design")
    axis.set_yscale("log")
    axis.set_xlabel("confidence threshold")
    axis.set_ylabel("expected cost per 1,000 cases (EUR)")
    axis.set_title("Expected cost against threshold")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_costing(dataset_dir: Path = DATASET_DIR, names: list[str] = None, weights: Path = WEIGHTS_PATH, imgsz: int = DEFAULT_IMGSZ) -> dict:
    config = load_config()
    names = names or class_names(dataset_dir)
    derived = derive(config)
    p = params_from(config, derived)

    images, scores, labels = image_scores(dataset_dir, weights, names, config["positive_class"], imgsz)
    tpr, fpr = rates(scores, labels)

    binary = binary_cost(tpr, fpr, p)
    binary_index = int(np.argmin(binary))
    i, j, band_best = best_band(tpr, fpr, p, derived, config, constrained=False)
    ci, cj, constrained_best = best_band(tpr, fpr, p, derived, config, constrained=True)
    band = describe_band(i, j, tpr, fpr, p, derived, config)
    constrained_band = describe_band(ci, cj, tpr, fpr, p, derived, config)

    deduplicated = None
    duplicates_file = RESULTS_DIR / "duplicates.json"
    if duplicates_file.is_file():
        flagged = set(json.loads(duplicates_file.read_text())["flagged_test_images_at_default"])
        keep = np.array([Path(path).name not in flagged for path in images])
        if keep.sum() > 0 and labels[keep].any() and (~labels[keep]).any():
            d_tpr, d_fpr = rates(scores[keep], labels[keep])
            di, dj, _ = best_band(d_tpr, d_fpr, p, derived, config, constrained=False)
            deduplicated = {
                "test_images": int(keep.sum()),
                "positive_images": int(labels[keep].sum()),
                "band": describe_band(di, dj, d_tpr, d_fpr, p, derived, config),
            }

    result = {
        "positive_class": config["positive_class"],
        "case_definition": "one camera image; positive if it contains the positive class, flagged if any detection of that class has confidence >= threshold",
        "config": config,
        "derived": {key: float(value) for key, value in derived.items()},
        "costs_per_case": {
            "false_negative_missed_image": p["c_fn"],
            "false_positive_alert": p["c_fp"],
            "true_positive_alert": p["c_tp"],
            "true_negative": p["c_tn"],
            "human_review": p["c_rev"],
        },
        "test_images": {"total": int(len(labels)), "positive": int(labels.sum()), "negative": int((~labels).sum())},
        "scores": {
            "positive": [round(float(s), 4) for s in scores[labels]],
            "negative": [round(float(s), 4) for s in scores[~labels]],
        },
        "grid": GRID.tolist(),
        "tpr": tpr.tolist(),
        "fpr": fpr.tolist(),
        "binary_cost_per_1000": [round(float(c) * 1000, 6) for c in binary],
        "binary_optimum": {
            "threshold": float(GRID[binary_index]),
            "cost_per_1000_cases": round(float(binary[binary_index]) * 1000, 6),
            "image_recall": round(float(tpr[binary_index]), 4),
            "false_positive_rate": round(float(fpr[binary_index]), 4),
        },
        "band_optimum_unconstrained": band,
        "band_optimum_within_review_capacity": constrained_band,
        "alternatives_per_year": alternatives(p, derived, float(binary.min()), band_best),
        "staffing": {
            "open_hours_per_year": derived["open_hours_per_year"],
            "people_needed_to_cover_one_seat": round(derived["seat_fte"], 2),
            "cost_of_one_dedicated_seat_per_year": round(derived["seat_cost_per_year"], 2),
            "review_hours_per_day_at_optimum": band["review_hours_per_day"],
            "seat_utilisation_at_optimum": band["monitor_utilisation"],
        },
        "break_even": break_even(tpr, fpr, derived, config),
        "sensitivity": sensitivity_grid(tpr, fpr, derived, config),
        "deduplicated_test_set": deduplicated,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "costing.json").write_text(json.dumps(result, indent=2))
    plot_curve(tpr, fpr, p, derived, binary_index, band, RESULTS_DIR / "cost_curve.png")
    show(result)
    return result


def show(result: dict) -> None:
    derived = result["derived"]
    print("\n=== Costing ===")
    print(f"cases per day {derived['cases_per_day']:.0f}, base rate {derived['base_rate']:.3e}")
    print(f"per-case costs (EUR): {json.dumps({k: round(v, 4) for k, v in result['costs_per_case'].items()})}")
    print(f"test images {result['test_images']}")
    binary = result["binary_optimum"]
    print(f"\ntwo-outcome sweep minimum: threshold {binary['threshold']}  cost per 1000 cases EUR {binary['cost_per_1000_cases']}  image recall {binary['image_recall']}  false positive rate {binary['false_positive_rate']}")
    for label, key in [("pass/review/fail, unconstrained", "band_optimum_unconstrained"), ("pass/review/fail, within review capacity", "band_optimum_within_review_capacity")]:
        print(f"\n{label}: {json.dumps(result[key])}")
    print("\nannual cost by strategy (EUR):")
    for name, row in result["alternatives_per_year"].items():
        print(f"  {name:26s} {row['cost_per_year']:>14,.2f}")
    print(f"\nstaffing: {json.dumps(result['staffing'])}")
    print(f"\nbreak-even: {json.dumps(result['break_even'])}")
    print("\nsensitivity (miss scenario x events per year):")
    for row in result["sensitivity"]:
        print(
            f"  {row['miss_scenario']:12s} events/yr {row['events_per_year']:>3}  miss EUR/img {row['miss_cost_per_image']:>10}  "
            f"thresholds {row['threshold_low']:.3f}/{row['threshold_high']:.3f}  review {row['reviews_per_day']:>8}/day  "
            f"system EUR {row['system_cost_per_year']:>12,.0f}  nothing EUR {row['do_nothing_cost_per_year']:>12,.0f}  beats nothing: {row['review_system_beats_doing_nothing']}"
        )
    if result["deduplicated_test_set"]:
        print(f"\ndeduplicated test set: {json.dumps(result['deduplicated_test_set'])}")


if __name__ == "__main__":
    run_costing()
