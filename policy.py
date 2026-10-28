"""Freeze a validation choice before this evaluation; disclose prior test use."""
from datetime import datetime, timezone
from pathlib import Path

from photo_predictions import ROOT, digest, read, write
from costing import capacity_limit


def freeze_policy(selection_note, model_key=None):
    bundle_path = ROOT / "results/validation_analysis/photo_predictions.json"
    analysis_path = ROOT / "results/validation_analysis/analysis.json"
    analysis, bundle = read(analysis_path), read(bundle_path)
    assert analysis["evaluation_split"] == "valid" and not analysis["test_evaluated"]
    assert bundle["evaluation_split"] == "valid" and not bundle["test_evaluated"]
    scenario = read(ROOT / "scenario.json")
    assert analysis["assumptions"] == scenario, "Regenerate validation analysis after a scenario change."
    if model_key is None:
        model_key = min(analysis["models"], key=lambda k: analysis["models"][k]["candidate_feasible"]["cost_per_case"])
    candidate = analysis["models"][model_key]["candidate_feasible"]
    metadata = bundle["models"][model_key]
    architecture = metadata.get("model_architecture", "yolo11n")
    single = analysis["models"][model_key].get("single_threshold_feasible")
    limit = capacity_limit(scenario)
    policy = {"status": "frozen_before_test", "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection_split": "valid", "model_key": model_key,
              "weights": metadata["weights"], "weights_sha256": metadata["weights_sha256"],
              "model_architecture": architecture,
              "pretrained_weights": f"data/pretrained/{architecture}.pt",
              "lower": candidate["lower"], "upper": candidate["upper"],
              "scenario_sha256": digest(ROOT / "scenario.json"),
              "validation_predictions_sha256": digest(bundle_path),
              "validation_analysis_sha256": digest(analysis_path),
              "selection_rule": ("Lowest validation expected cost among all observed threshold partitions. Staff time is costed and reported; no fixed daily workload limit."
                                 if limit is None else f"Lowest validation expected cost among all observed threshold partitions, subject to expected total staff workload <= {limit:g} min/day."),
              "available_staff_minutes_per_day": limit,
              "workload_constraint_applied": limit is not None,
              "selection_note": selection_note,
              "validation_expected_cost_per_day": candidate["cost_per_day"],
              "validation_expected_minutes_per_day": candidate["minutes_per_day"],
              "evaluation_context": "secondary_reuse_of_previously_evaluated_test" if (ROOT / "results/final_evaluation.json").exists() else "first_holdout_evaluation",
              "test_must_not_select_or_tune_policy": True}
    if single:
        policy["model_only_threshold"] = single["upper"]
        policy["model_only_selection_split"] = "valid"
    if metadata.get("pretrained_weights_sha256"):
        policy["pretrained_weights_sha256"] = metadata["pretrained_weights_sha256"]
    catalog_file = ROOT / "results/experiment_catalog.json"
    if catalog_file.exists():
        directory = read(catalog_file).get(model_key)
        if directory:
            environment = ROOT / directory / "environment.json"
            if environment.exists():
                policy["pretrained_weights_sha256"] = read(environment)["pretrained_sha256"]
    assert digest(ROOT / policy["weights"]) == policy["weights_sha256"]
    write(ROOT / "operating_policy.json", policy)
    print(f"POLICY FROZEN: {model_key}; lower={policy['lower']:.9f}; upper={policy['upper']:.9f}",flush=True)
    return policy
