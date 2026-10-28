"""Post-training exports for main.py train; app.py never imports this stage."""
from pathlib import Path
from photo_predictions import ROOT, read, write


def finish_training(summary):
    from photo_predictions import main as predict_validation
    from analyse_validation import main as analyse_validation
    from policy import freeze_policy
    from final_evaluation import main as evaluate_test

    run = ROOT / "results" / Path(summary["run_directory"]).name
    catalog_file = ROOT / "results/experiment_catalog.json"
    catalog = read(catalog_file)
    architecture = summary.get("model_architecture", "yolo11n")
    model_key = f"{architecture}_{summary['completed_epochs']}_epochs"
    catalog[model_key] = run.relative_to(ROOT).as_posix()
    write(catalog_file,catalog)
    predict_validation()
    analyse_validation()
    freeze_policy("Reproduction of the student-selected architecture and duration. Use the newly trained model and choose its thresholds by minimum validation cost under the current scenario. Staff time is costed and reported; a workload cap is applied only if configured. No test output enters selection.", model_key=model_key)
    final = evaluate_test()
    summary["stage"] = "completed_training_and_evaluation_pipeline"
    summary["pipeline_exports"] = {"validation":"results/validation_analysis/analysis.json",
                                    "policy":"operating_policy.json","test":"results/final_evaluation.json"}
    summary["pending"] = ["student-written analysis and conclusions", "submission"]
    write(run / "summary.json",summary)
    print("PIPELINE COMPLETE: training, validation selection, frozen-policy test evaluation, dashboard artifacts.",flush=True)
