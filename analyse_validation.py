"""Reproduce validation-only photo analysis from saved predictions, no inference."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/.matplotlib"))
OUT = ROOT / "results/validation_analysis"

from costing import alternatives, binary_counts, evaluate, sweep, photo_workflow, capacity_limit


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def model_inputs(bundle, model):
    names = sorted(k for k, v in bundle["truths"].items() if v["condition_ground_truth_known"])
    return ([bundle["models"][model]["scores"][name] for name in names],
            [bundle["truths"][name]["has_damage_label"] for name in names])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = read(OUT / "photo_predictions.json")
    assert bundle["evaluation_split"] == "valid" and bundle["test_evaluated"] is False
    s = read(ROOT / "scenario.json")
    result = {"evaluation_split": "valid", "test_evaluated": False,
              "policy_status": "Exploration under current scenario assumptions; original frozen-policy test evaluation is retained separately.",
              "assumptions": s, "alternatives": alternatives(s), "models": {},
              "unknown_images_excluded_from_photo_rates": [k for k,v in bundle["truths"].items()
                                                          if not v["condition_ground_truth_known"]],
              "confusion_definition": ("Expected provisional damage incidents after simulated photo review; rows=true damage/undamaged, columns=incident/pass. FP is an unnecessary incident, not a customer charge." if photo_workflow(s) else "Expected physical escalation after simulated photo review, rows=true damage/undamaged; columns=escalate/pass. FP is an unnecessary physical check."),
              "limitations": ["Scores are uncalibrated, truncated below inference confidence .001.",
                              "Validation used for selection; these minima are optimistic, not test performance.",
                              "Photo reviewer accuracy is assumed. Automatic high-score incidents can be false positives even with a perfect reviewer in the middle band.",
                              "Incident handling time and false-incident losses are scenario estimates, not measured operations.",
                              "Current scenario was revised after viewing the original test; it has no new independent holdout evaluation.",
                              "Workload is cumulative expected staff time; no arrivals or queue simulation.",
                              "Single photo does not establish whole-vehicle condition or customer liability."]}
    rows = []
    if s.get("damage_scope_basis"):
        result["limitations"].append("The current loss assumption targets minor bodywork at ordinary returns, but the unfiltered public dataset also contains severe damage; class labels do not provide severity-specific rates.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for model in bundle["models"]:
        scores, truth = model_inputs(bundle, model)
        analysis = sweep(scores, truth, s)
        analysis["common_display_threshold"] = binary_counts(scores, truth, .4)
        candidate = analysis["candidate_feasible"]
        scenarios = []
        variations = {"damage_prevalence": [.01,.03,.05,.10],
                      "missed_damage_euros": sorted({s["missed_damage_euros"]*.5,
                                                     s["missed_damage_euros"],
                                                     s["missed_damage_euros"]*2,
                                                     s.get("previous_missed_damage_euros", s["missed_damage_euros"])})}
        if capacity_limit(s) is not None:
            variations["available_staff_minutes_per_day"] = [60.,75.,92.,120.]
        if photo_workflow(s):
            variations.update(incident_handling_minutes=[2.,5.,10.], false_incident_extra_euros=[0.,5.,20.])
        for field, values in variations.items():
            for value in values:
                changed = {**s, field: value}
                scenarios.append({"changed": {field: value},
                                  "nominal_policy_held_fixed": evaluate(scores, truth, candidate["lower"], candidate["upper"], changed),
                                  "reoptimised_candidate": sweep(scores, truth, changed)["candidate_feasible"]})
        for accuracy in (.8,.9,.95,.99,1.):
            changed = {**s, "photo_reviewer_sensitivity": accuracy, "photo_reviewer_specificity": accuracy}
            scenarios.append({"changed": {"reviewer_sensitivity_and_specificity": accuracy},
                              "nominal_policy_held_fixed": evaluate(scores, truth, candidate["lower"], candidate["upper"], changed),
                              "reoptimised_candidate": sweep(scores, truth, changed)["candidate_feasible"]})
        analysis["sensitivity_analysis"] = scenarios
        result["models"][model] = analysis
        for policy in ("candidate_feasible", "candidate_unconstrained", "single_threshold_feasible", "single_threshold_unconstrained"):
            v = analysis[policy]
            if v:
                rows.append({"model": model, "policy": policy, **{k:v[k] for k in (
                    "lower", "upper", "cost_per_1000", "cost_per_day", "minutes_per_day",
                    "photo_reviews_per_day", "physical_checks_per_day", "incidents_per_day", "false_incidents_per_day", "missed_damage_cases_per_day", "capacity_feasible")}})
        curve = analysis["single_threshold_curve"]
        axes[0].plot([v["threshold"] for v in curve], [v["cost_per_1000"] for v in curve], label=model)
        axes[1].plot([v["threshold"] for v in curve], [v["minutes_per_day"] for v in curve], label=model)
        single = analysis["single_threshold_feasible"]
        if single:
            axes[0].scatter(single["upper"], single["cost_per_1000"], s=45)
        print(f"{model}: .40 confusion={analysis['common_display_threshold']}", flush=True)
        print(f"{model}: numerical candidate lower={candidate['lower']:.6f}, upper={candidate['upper']:.6f}; "
              f"{candidate['cost_per_day']:.2f} euros/day, {candidate['minutes_per_day']:.2f} minutes/day; "
              f"{candidate['photo_reviews_per_day']:.2f} photo reviews/day; "
              f"misses={candidate['missed_damage_cases_per_day']:.3f}/day", flush=True)
    axes[0].set(xlabel="Single damage threshold (no photo review band)", ylabel="Expected euros / 1,000 photos")
    axes[1].set(xlabel="Single damage threshold (no photo review band)", ylabel="Expected staff minutes / day")
    if capacity_limit(s) is not None:
        axes[1].axhline(capacity_limit(s), color="red", linestyle="--", label=f"{capacity_limit(s):g} minute capacity")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Validation exploration — single-threshold policies; dots mark minimum expected cost")
    fig.savefig(OUT / "single_threshold_curves.png", dpi=170)
    plt.close(fig)
    write(OUT / "analysis.json", result)
    with (OUT / "policy_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print("VALIDATION ANALYSIS COMPLETE. This stage uses no test data.", flush=True)


if __name__ == "__main__":
    main()
