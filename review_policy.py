import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

REVIEW = {
    "human_sensitivity": 0.99,
    "human_specificity": 0.995,
    "review_seconds": 15,
    "hourly_cost": 24,
    "daily_cases": 10000,
    "productive_hours": 6,
    "available_reviewers": 1,
    "pocket_guard": 0.25,
}


def evaluate_policy(
    frame, lower, upper, config, use_guard=True, review_all=False
):
    if lower > upper:
        raise ValueError("Lower threshold cannot exceed upper threshold.")

    positive = frame["positive"].to_numpy() == 1
    scores = frame["score"].to_numpy()

    if not positive.any() or positive.all():
        raise ValueError("Both actual classes must be present.")

    # Each actual class is reweighted to the operational base rate.
    prevalence = config["base_rate"]
    weights = np.where(
        positive,
        prevalence / positive.sum(),
        (1 - prevalence) / (~positive).sum(),
    )

    review = (scores >= lower) & (scores < upper)
    if use_guard:
        review |= (
            frame["max_pocket_confidence"].to_numpy()
            < config["pocket_guard"]
        )
    if review_all:
        review[:] = True

    hold = (~review) & (scores >= upper)
    release = ~(review | hold)

    # Review outcomes are expected values under assumed human error rates.
    hold_probability = np.where(
        review,
        np.where(
            positive,
            config["human_sensitivity"],
            1 - config["human_specificity"],
        ),
        hold.astype(float),
    )

    review_cost = (
        config["review_seconds"] * config["hourly_cost"] / 3600
    )
    case_cost = np.where(
        positive,
        hold_probability * config["cost_tp"]
        + (1 - hold_probability) * config["cost_fn"],
        hold_probability * config["cost_fp"]
        + (1 - hold_probability) * config["cost_tn"],
    ) + review * review_cost

    review_share = float(weights[review].sum())
    reviews_daily = review_share * config["daily_cases"]
    hours = reviews_daily * config["review_seconds"] / 3600
    fte = hours / config["productive_hours"]

    result = {
        "lower": float(lower),
        "upper": float(upper),
        "cost_per_1000": float(1000 * (weights * case_cost).sum()),
        "dataset_review_count": int(review.sum()),
        "operational_review_share": review_share,
        "reviews_per_day": reviews_daily,
        "review_hours_per_day": hours,
        "review_FTE": fte,
        "dedicated_people": math.ceil(max(0, fte - 1e-12)),
        "within_capacity": bool(
            fte <= config["available_reviewers"] + 1e-12
        ),
    }

    for label, mask in (("positive", positive), ("negative", ~positive)):
        for action, selected in (
            ("pass", release), ("review", review), ("fail", hold)
        ):
            result[f"{label}_{action}"] = int((mask & selected).sum())

    for label, probabilities in {
        "TP": positive * hold_probability,
        "FN": positive * (1 - hold_probability),
        "FP": (~positive) * hold_probability,
        "TN": (~positive) * (1 - hold_probability),
    }.items():
        result[f"expected_{label}_per_1000"] = float(
            1000 * (weights * probabilities).sum()
        )

    return result


def main():
    saved = json.loads((OUT / "chosen_threshold.json").read_text())
    config = {**saved["assumptions"], **REVIEW}
    binary_threshold = saved["chosen_threshold"]

    scores = pd.read_csv(OUT / "image_scores.csv")
    validation = scores[scores["split"] == "valid"]
    test = scores[scores["split"] == "test"]

    # Search in 0.01 increments, also including the binary optimum.
    grid = sorted(set(np.linspace(0, 1, 101)) | {binary_threshold})
    rows = []

    for index, lower in enumerate(grid):
        for upper in grid[index:]:
            rows.append(evaluate_policy(
                validation, lower, upper, config
            ))

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "review_threshold_sweep.csv", index=False)

    feasible = sweep[sweep["within_capacity"]].copy()
    if feasible.empty:
        raise RuntimeError(
            "No candidate policy meets the assumed review capacity."
        )

    # Equal costs: prefer less review, then lower thresholds.
    feasible["cost_key"] = feasible["cost_per_1000"].round(9)
    selected = feasible.sort_values([
        "cost_key", "operational_review_share", "lower", "upper"
    ]).iloc[0]

    lower = float(selected["lower"])
    upper = float(selected["upper"])

    validation_result = evaluate_policy(
        validation, lower, upper, config
    )
    test_result = evaluate_policy(test, lower, upper, config)

    comparisons = []
    for split_name, frame in (("valid", validation), ("test", test)):
        policies = {
            "Release everything": evaluate_policy(
                frame, 1.000001, 1.000001, config, use_guard=False
            ),
            "Human checks everything": evaluate_policy(
                frame, 0, 1, config, review_all=True
            ),
            "Model alone": evaluate_policy(
                frame, binary_threshold, binary_threshold,
                config, use_guard=False
            ),
            "Model with review": evaluate_policy(
                frame, lower, upper, config
            ),
        }
        for policy, result in policies.items():
            comparisons.append({
                "split": split_name, "policy": policy, **result
            })

    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "policy_comparison.csv", index=False)

    (OUT / "review_policy.json").write_text(
        json.dumps({
            "config": config,
            "selection_split": "valid",
            "threshold_grid_step": 0.01,
            "lower": lower,
            "upper": upper,
            "validation": validation_result,
            "test": test_result,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSELECTED POLICY — VALIDATION")
    print(json.dumps(validation_result, indent=2))
    print("\nSAME FIXED POLICY — TEST")
    print(json.dumps(test_result, indent=2))
    print("\nPOLICY COMPARISON — TEST")
    columns = [
        "policy", "cost_per_1000", "reviews_per_day",
        "review_hours_per_day", "review_FTE", "within_capacity",
    ]
    print(comparison[comparison["split"] == "test"][columns].to_string(
        index=False
    ))
    print("\nHuman outcomes are scenario expectations, not measured results.")
    print("Test results do not change the selected thresholds.")


if __name__ == "__main__":
    main()



def run_sensitivity():
    saved = json.loads((OUT / "review_policy.json").read_text())
    binary = json.loads((OUT / "chosen_threshold.json").read_text())

    central = saved["config"]
    lower, upper = saved["lower"], saved["upper"]
    threshold = binary["chosen_threshold"]

    scores = pd.read_csv(OUT / "image_scores.csv")
    test = scores[scores["split"] == "test"]

    scenarios = [("Central", {})]

    for rate in (0.001, 0.005, 0.01, 0.05):
        scenarios.append((
            f"Positive rate {rate:.1%}", {"base_rate": rate}
        ))

    for cost in (5, 100, 1000):
        scenarios.append((
            f"FN cost EUR {cost}", {"cost_fn": cost}
        ))

    for cost in (0.25, 5):
        scenarios.append((
            f"FP cost EUR {cost}", {"cost_fp": cost}
        ))

    for seconds in (5, 30, 60):
        scenarios.append((
            f"Review time {seconds}s", {"review_seconds": seconds}
        ))

    scenarios.extend([
        ("Human sensitivity 90%", {"human_sensitivity": 0.90}),
        ("Human specificity 98%", {"human_specificity": 0.98}),
    ])

    rows = []

    for scenario, changes in scenarios:
        config = {**central, **changes}

        policies = {
            "Release everything": evaluate_policy(
                test, 1.000001, 1.000001, config, use_guard=False
            ),
            "Human checks everything": evaluate_policy(
                test, 0, 1, config, review_all=True
            ),
            "Model alone": evaluate_policy(
                test, threshold, threshold, config, use_guard=False
            ),
            "Model with review": evaluate_policy(
                test, lower, upper, config
            ),
        }

        for policy, result in policies.items():
            rows.append({
                "scenario": scenario,
                "policy": policy,
                **config,
                **result,
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "sensitivity.csv", index=False)

    comparison = table.pivot(
        index="scenario",
        columns="policy",
        values="cost_per_1000",
    )
    print("\nCOST PER 1,000 CASES — FIXED POLICIES")
    print(comparison.round(2).to_string())

    # Cost is linear in FN cost when routing and base rate stay fixed.
    def fn_cost_line(human):
        values = []
        for fn_cost in (0, 1):
            config = {**central, "cost_fn": fn_cost}
            if human:
                result = evaluate_policy(
                    test, 0, 1, config, review_all=True
                )
            else:
                result = evaluate_policy(
                    test, threshold, threshold,
                    config, use_guard=False
                )
            values.append(result["cost_per_1000"])
        return values[0], values[1] - values[0]

    model_intercept, model_slope = fn_cost_line(False)
    human_intercept, human_slope = fn_cost_line(True)

    denominator = model_slope - human_slope
    crossing = None
    if abs(denominator) > 1e-12:
        crossing = (
            human_intercept - model_intercept
        ) / denominator

    negatives = int((test["positive"] == 0).sum())
    upper_fpr = 1 - 0.05 ** (1 / negatives)

    notes = {
        "thresholds_remain_fixed": True,
        "one_factor_at_a_time": True,
        "human_vs_model_FN_cost_crossing_eur": crossing,
        "test_complete_images": negatives,
        "zero_FP_one_sided_95pct_upper_bound": upper_fpr,
    }
    (OUT / "sensitivity_notes.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )

    print("\nHUMAN-ALL VS MODEL-ALONE COST CROSSING")
    if crossing is not None and crossing > 0:
        print(f"FN cost: EUR {crossing:,.2f}")
        print("This comparison ignores the human-all staffing constraint.")
    else:
        print("No positive crossing under the central assumptions.")

    print("\nSAMPLING UNCERTAINTY")
    print(
        f"Zero false positives among {negatives} complete test images "
        f"gives a one-sided 95% upper FPR bound of {upper_fpr:.2%}."
    )
    print(
        "This assumes independent representative cases, which the "
        "dataset audit has not established."
    )
    print("\nNo thresholds were selected or changed using test results.")