"""Run an authorised YOLO11s validation trial; never import final test stages."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import ultralytics
    from ultralytics.utils.downloads import attempt_download_asset
    from train import main as train_model
    from photo_predictions import digest, read, records_from_model, write
    from costing import binary_counts, sweep

    config = read(ROOT / "training_config.json")
    scenario = read(ROOT / "scenario.json")
    reference = read(ROOT / "reference_validation_analysis.json")
    reference_models = read(ROOT / "reference_models.json")
    epochs = config["training"]["epochs"]
    assert config["model"] == "yolo11s.pt" and epochs in (20, 40)
    assert reference["evaluation_split"] == "valid" and not reference["test_evaluated"]
    assert reference["assumptions"] == scenario
    assert torch.cuda.is_available(), "This trial is authorised for the Colab GPU."
    dataset = ROOT / "data/car-damage-v1"
    assert not (dataset / "test").exists(), "This isolated trial must contain train/valid only."
    pretrained = ROOT / "data/pretrained/yolo11s.pt"
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    pretrained = Path(attempt_download_asset(str(pretrained)))
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "pretrained_sha256": digest(pretrained),
                   "test_evaluated": False}
    print("EXPERIMENT ENVIRONMENT " + json.dumps(environment), flush=True)
    summary = train_model()  # Function call avoids train.py's final-submission entry point.
    out = ROOT / "results" / Path(summary["run_directory"]).name
    summary.update(model_architecture="yolo11s", stage=("validation_only_duration_experiment"
                   if epochs == 40 else "validation_only_architecture_experiment"))
    write(out / "summary.json", summary)
    write(out / "environment.json", environment)
    (out / "packages.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    files = sorted(p for p in (dataset / "valid/images").iterdir()
                   if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"})
    assert len(files) == 366
    predictions = records_from_model(ROOT / summary["weights"], files,
                                     out / "review/validation_predictions.json",
                                     ["Damage-detected","Undamaged"], "0")
    known = [record for record in predictions["images"] if record["condition_ground_truth_known"]]
    scores = [record["max_damage_confidence"] for record in known]
    truth = [record["has_damage_label"] for record in known]
    assert len(known) == 365 and sum(truth) == 135
    analysis = sweep(scores, truth, scenario)
    analysis["common_display_threshold"] = binary_counts(scores, truth, .4)
    candidate = analysis["candidate_feasible"]
    comparison = {"evaluation_split":"valid", "test_evaluated":False,
                  "scenario":scenario, "scenario_sha256":digest(ROOT / "scenario.json"),
                  "model_key":f"yolo11s_{epochs}_epochs", "summary":summary,
                  "analysis":analysis, "reference_models":reference_models,
                  "reference_analysis":reference["models"],
                  "interpretation_status":"Pending discussion with the student; no policy replaced.",
                  "limitations":["One training seed per configuration; no proof that a difference is reproducible.",
                                 "The learning-rate schedule scales with the epoch budget; this is a fresh run, not a continuation of the first twenty epochs.",
                                 "Validation checkpoint and policy selection can be optimistic.",
                                 "The current business assumptions were revised after an earlier model's test evaluation."]}
    write(out / "architecture_comparison.json", comparison)
    print("EXPERIMENT VALIDATION " + json.dumps({"run_id":out.name,"metrics":summary["metrics"],
          "cost_per_day":candidate["cost_per_day"],"minutes_per_day":candidate["minutes_per_day"],
          "photo_reviews_per_day":candidate["photo_reviews_per_day"],
          "miss_rate":candidate["conditional_damage_miss_rate"],"test_evaluated":False}),flush=True)


if __name__ == "__main__":
    main()
