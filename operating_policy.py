import json
from pathlib import Path

RESULTS = Path("results")

# Illustrative business assumptions.
COST_REVIEW = 15           # Desk review of one image
COST_INSPECTION = 150      # Physical inspection
COST_MISSED_DEFECT = 3000  # Expected cost of an overlooked defect

CASES_PER_DAY = 1000
MINUTES_PER_REVIEW = 3
PRODUCTIVE_HOURS_PER_PERSON = 6

# Detection confidence thresholds.
THRESHOLDS = [
    0.001, 0.01, 0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.40, 0.50, 0.60,
    0.70, 0.80, 0.90, 0.95
]


def load_predictions(filename):
    with open(RESULTS / filename) as f:
        return json.load(f)


def evaluate(records, low, high):
    if low > high:
        raise ValueError("Low threshold cannot exceed high threshold")

    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    costs = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    errors = {
        "missed_defects": 0,
        "unnecessary_inspections": 0,
    }

    for record in records:
        score = record["score"]
        defect = record["positive"]

        if score < low:
            decision = "PASS"
            cost = COST_MISSED_DEFECT if defect else 0
            if defect:
                errors["missed_defects"] += 1

        elif score < high:
            decision = "REVIEW"
            # Assumption: desk review resolves the case correctly.
            cost = COST_REVIEW + (
                COST_INSPECTION if defect else 0
            )

        else:
            decision = "FAIL"
            cost = COST_INSPECTION
            if not defect:
                errors["unnecessary_inspections"] += 1

        counts[decision] += 1
        costs[decision] += cost

    total = len(records)
    total_cost = sum(costs.values())
    review_rate = counts["REVIEW"] / total

    daily_reviews = CASES_PER_DAY * review_rate
    review_hours = daily_reviews * MINUTES_PER_REVIEW / 60
    inspectors_fte = review_hours / PRODUCTIVE_HOURS_PER_PERSON

    return {
        "low_threshold": low,
        "high_threshold": high,
        "decisions": counts,
        "decision_costs_eur": costs,
        "errors": errors,
        "review_share": round(review_rate, 4),
        "daily_reviews": round(daily_reviews, 1),
        "daily_review_hours": round(review_hours, 2),
        "review_staff_fte": round(inspectors_fte, 2),
        "cost_per_1000_eur": round(
            total_cost * 1000 / total, 2
        ),
    }


def alternatives(records, selected_single_threshold):
    n = len(records)
    positives = sum(r["positive"] for r in records)

    # No image-based inspection at all.
    no_automation = positives * COST_MISSED_DEFECT

    # Every image reviewed by a person.
    # Assumes reviewers identify all critical defects.
    full_manual = n * COST_REVIEW + (
        positives * COST_INSPECTION
    )

    # Automatic PASS / FAIL without human desk review.
    automatic = evaluate(
        records,
        selected_single_threshold,
        selected_single_threshold,
    )

    return {
        "no_automation_eur_per_1000": round(
            no_automation * 1000 / n, 2
        ),
        "full_manual_eur_per_1000": round(
            full_manual * 1000 / n, 2
        ),
        "full_automation_eur_per_1000": automatic[
            "cost_per_1000_eur"
        ],
    }


def main():
    valid = load_predictions("validation_predictions.json")
    test = load_predictions("test_predictions.json")

    operating_point = json.loads(
        (RESULTS / "operating_point.json").read_text()
    )
    single_threshold = operating_point["threshold"]

    # Select both thresholds using validation, never test.
    sweep = [
        evaluate(valid, low, high)
        for low in THRESHOLDS
        for high in THRESHOLDS
        if low < high
    ]

    best = min(
        sweep,
        key=lambda item: (
            item["cost_per_1000_eur"],
            item["review_share"],
            item["low_threshold"],
            item["high_threshold"],
        ),
    )

    test_result = evaluate(
        test,
        best["low_threshold"],
        best["high_threshold"],
    )

    output = {
        "assumptions": {
            "review_cost_eur": COST_REVIEW,
            "inspection_cost_eur": COST_INSPECTION,
            "missed_defect_cost_eur": COST_MISSED_DEFECT,
            "cases_per_day": CASES_PER_DAY,
            "minutes_per_review": MINUTES_PER_REVIEW,
            "productive_hours_per_person":
                PRODUCTIVE_HOURS_PER_PERSON,
            "human_review_assumed_perfect": True,
        },
        "selected_using": "validation",
        "validation": best,
        "test": test_result,
        "alternatives_test": alternatives(
            test, single_threshold
        ),
    }

    (RESULTS / "policy_sweep.json").write_text(
        json.dumps(sweep, indent=2)
    )
    (RESULTS / "policy_results.json").write_text(
        json.dumps(output, indent=2)
    )

    print("\nSELECTED OPERATING POLICY")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
