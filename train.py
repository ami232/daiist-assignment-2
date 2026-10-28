"""Assignment pipeline: training, validation choice, then frozen-policy test.

Run with ``uv run python main.py train``. Test is excluded during training and
threshold/model selection. After training, pipeline.py exports validation
analysis, freezes a policy, evaluates test and saves dashboard artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keep caches and local settings inside the ignored data directory.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / ".ultralytics"))


def write_json(path: Path, value: dict) -> None:
    """Convert NumPy scalar metrics without losing their numeric type."""
    path.write_text(
        json.dumps(value, indent=2, default=lambda item: item.item()) + "\n",
        encoding="utf-8",
    )


def prepare_training_yaml(dataset: Path, classes: list[str], destination: Path) -> dict:
    """Resolve the supplied folders without altering the public export."""
    import yaml

    source = yaml.safe_load((dataset / "data.yaml").read_text())
    names = source["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]
    if names != classes:
        raise ValueError("Dataset class names/order differ from dataset_config.json.")

    counts = {}
    for split in ("train", "valid"):
        images = dataset / split / "images"
        counts[split] = sum(
            file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            for file in images.iterdir()
        )
        if not counts[split]:
            raise ValueError(f"Empty {split} image directory.")
    runtime = {
        "path": str(dataset.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(names),
        "names": names,
    }
    # There is intentionally no test path in this experiment's configuration.
    destination.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    return counts


def choose_device(preference: list[str]) -> str:
    import torch

    for device in preference:
        if device == "cuda" and torch.cuda.is_available():
            return "0"
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        if device == "cpu":
            return "cpu"
    raise RuntimeError("None of the configured devices is available.")


def epoch_progress(trainer) -> None:
    """Print one concise summary per epoch, including validation performance."""
    if trainer.epoch >= trainer.epochs:  # skip final best-checkpoint validation callback
        return
    metrics = trainer.metrics
    print(
        f"PILOT epoch={trainer.epoch + 1}/{trainer.epochs} "
        f"val_mAP50={metrics.get('metrics/mAP50(B)', 0):.4f} "
        f"val_mAP50_95={metrics.get('metrics/mAP50-95(B)', 0):.4f}",
        flush=True,
    )


def save_comparison(summary: dict, config: dict, output: Path) -> None:
    """Export measured validation results without selecting a business policy."""
    previous = config.get("compare_with")
    if not previous:
        return
    reference = json.loads((ROOT / previous / "summary.json").read_text())
    rows = []
    for label, result in ((f"reference_{reference['completed_epochs']}_epochs", reference),
                          (f"new_{summary['completed_epochs']}_epochs", summary)):
        if result["evaluation_split"] != "valid" or result["test_evaluated"]:
            raise ValueError("This comparison must use validation only.")
        for cls in result["per_class"]:
            rows.append({
                "experiment": label,
                "completed_epochs": result["completed_epochs"],
                "best_epoch": result["best_epoch"],
                "class": cls["Class"],
                "precision": cls["Box-P"],
                "recall": cls["Box-R"],
                "AP50": cls["mAP50"],
                "AP50_95": cls["mAP50-95"],
            })
    with (output / "comparison_to_10epochs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_speed_trial_callbacks(model, output: Path, batch: int) -> None:
    """Measure one complete resumed epoch, including validation and saving."""
    path = output / "speed_trial.json"
    if not path.exists():
        return
    trial = json.loads(path.read_text())
    if trial["status"] != "awaiting_batch4_epoch" or batch != trial["new_batch"]:
        return
    expected_epoch = trial["resume_after_epoch"] + 1
    epoch_started = None

    def start(trainer):
        nonlocal epoch_started
        if trainer.epoch + 1 == expected_epoch:
            epoch_started = time.monotonic()

    def finish(trainer):
        if trainer.epoch + 1 != expected_epoch or epoch_started is None:
            return
        elapsed = time.monotonic() - epoch_started
        trial.update({
            "status": "measured",
            "measured_epoch": expected_epoch,
            "measured_batch": trainer.batch_size,
            "new_epoch_seconds": elapsed,
            "speedup_factor": trial["baseline_epoch_seconds"] / elapsed,
            "time_reduction_percent": 100 * (1 - elapsed / trial["baseline_epoch_seconds"]),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(path, trial)
        print(
            f"SPEED_TRIAL epoch={expected_epoch} batch={trainer.batch_size} "
            f"seconds={elapsed:.1f} baseline_seconds={trial['baseline_epoch_seconds']:.1f} "
            f"speedup={trial['speedup_factor']:.2f}x",
            flush=True,
        )

    model.add_callback("on_train_epoch_start", start)
    model.add_callback("on_fit_epoch_end", finish)


def main() -> dict:
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.utils.downloads import attempt_download_asset

    from audit_dataset import prepare_dataset

    config = json.loads((ROOT / "training_config.json").read_text())
    dataset_config = json.loads((ROOT / config["dataset_config"]).read_text())
    if config["evaluate_test_during_initial_run"]:
        raise ValueError("This pilot must not evaluate test performance.")
    if config["checkpoint_selection_split"] != "valid":
        raise ValueError("Select checkpoints using validation only.")

    settings.update({
        "sync": False,
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": False,
    })
    started = datetime.now(timezone.utc)
    resume = ROOT / config["resume_from"] if config.get("resume_from") else None
    prefix = config.get("experiment_name", "pilot_yolo11n")
    if not prefix or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in prefix):
        raise ValueError("Use a simple lowercase experiment name.")
    run_id = resume.parent.parent.name if resume else prefix + "_" + started.strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "results" / run_id
    output.mkdir(parents=True, exist_ok=bool(resume))
    previous_seconds = 0.0
    if resume:
        if not resume.is_file():
            raise FileNotFoundError("The requested resume checkpoint is missing.")
        with (resume.parent.parent / "results.csv").open(newline="") as handle:
            previous_epochs = list(csv.DictReader(handle))
        old_state = output / "run_state.json"
        preceding_segments = json.loads(old_state.read_text()).get("elapsed_seconds_before_segment", 0) if old_state.exists() else 0
        previous_seconds = preceding_segments + float(previous_epochs[-1]["time"])
        started = datetime.strptime(run_id.rsplit("_", 1)[-1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        print(f"Resuming after {len(previous_epochs)} saved epochs; target total={config['training']['epochs']}.", flush=True)
    dataset = prepare_dataset()
    runtime_yaml = ROOT / "data" / "runtime" / (run_id + ".yaml")
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    counts = prepare_training_yaml(dataset, dataset_config["classes"], runtime_yaml)
    if counts != {key: dataset_config["measured_export_images"][key] for key in counts}:
        raise ValueError("Image counts differ from the audited export.")

    device = choose_device(config["device_preference"])
    print(f"PILOT device={device}; images={counts}; test_evaluated=False", flush=True)
    pretrained = ROOT / "data" / "pretrained" / config["model"]
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        pretrained = Path(attempt_download_asset(pretrained))
    model = YOLO(str(resume or pretrained), task=config["task"])
    model.add_callback("on_fit_epoch_end", epoch_progress)
    add_speed_trial_callbacks(model, output, config["training"]["batch"])
    write_json(output / ("resume_configuration.json" if resume else "configuration.json"), config)
    write_json(output / "dataset.json", dataset_config)
    state = {
        "status": "running",
        "directory": str(output.relative_to(ROOT)),
        "target_epochs": config["training"]["epochs"],
        "started_utc": started.isoformat(),
        "elapsed_seconds_before_segment": previous_seconds,
        "test_evaluated": False,
    }
    write_json(output / "run_state.json", state)
    write_json(ROOT / "results" / "latest_training.json", state)
    config["status"] = "in_progress"
    config["last_run_id"] = run_id
    write_json(ROOT / "training_config.json", config)

    timer = time.monotonic()
    try:
        metrics = model.train(
            data=str(runtime_yaml),
            device=device,
            project=str(ROOT / "runs" / "detect"),
            name=run_id,
            # A resumed checkpoint can contain paths from another computer.
            save_dir=str(ROOT / "runs" / "detect" / run_id),
            exist_ok=False,
            split="val",
            val=True,
            plots=True,
            save=True,
            resume=str(resume) if resume else False,
            **config["training"],
        )
    except (Exception, KeyboardInterrupt) as error:
        state.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      "error_type": type(error).__name__})
        write_json(output / "run_state.json", state)
        write_json(ROOT / "results" / "latest_training.json", state)
        current = json.loads((ROOT / "training_config.json").read_text())
        if current.get("last_run_id") == run_id:
            current["status"] = state["status"]
            write_json(ROOT / "training_config.json", current)
        raise
    trainer = model.trainer
    best = Path(trainer.best)
    if not best.is_file():
        raise RuntimeError("Training did not produce a best checkpoint.")

    # Each experiment gets a distinct file; previous trained models stay intact.
    saved_weights = ROOT / "weights" / (run_id + ".pt")
    saved_weights.parent.mkdir(parents=True, exist_ok=True)
    if saved_weights.exists():
        raise FileExistsError("Refusing to overwrite an exported model.")
    shutil.copy2(best, saved_weights)
    shutil.copy2(trainer.csv, output / "epochs.csv")
    shutil.copy2(Path(trainer.save_dir) / "args.yaml", output / "training_args.yaml")
    for pattern in ("*.png", "val_batch*.jpg"):
        for plot in Path(trainer.save_dir).glob(pattern):
            shutil.copy2(plot, output / plot.name)

    validator = trainer.validator
    matrix = validator.confusion_matrix
    with Path(trainer.csv).open(newline="") as handle:
        epochs = list(csv.DictReader(handle))
    best_epoch = max(
        epochs, key=lambda row: (float(row["metrics/mAP50-95(B)"]), int(row["epoch"]))
    )["epoch"]
    summary = {
        "stage": "training_experiment_not_final_submission",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "training_and_validation_seconds": previous_seconds + time.monotonic() - timer,
        "timing_note": "Excludes discarded work in an interrupted partial epoch when resuming.",
        "execution_interface": config.get("execution_interface", "terminal"),
        "resumed_from": str(resume.relative_to(ROOT)) if resume else None,
        "experiment_name": prefix,
        "model_architecture": Path(config["model"]).stem,
        "pretrained_weights": pretrained.relative_to(ROOT).as_posix(),
        "pretrained_weights_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest() if pretrained.exists() else None,
        "compare_with": config.get("compare_with"),
        "final_batch": trainer.batch_size,
        "configuration_files": ["configuration.json"] + (["resume_configuration.json"] if resume else []),
        "speed_trial": "speed_trial.json" if (output / "speed_trial.json").exists() else None,
        "device": device,
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "train_images": counts["train"],
        "validation_images": counts["valid"],
        "completed_epochs": trainer.epoch + 1,
        "best_epoch": int(best_epoch),
        "checkpoint_selection": "Highest validation mAP50-95 (Ultralytics 8.4.173 default fitness). Ties keep the latest checkpoint.",
        "evaluation_split": "valid",
        "test_evaluated": False,
        "weights": saved_weights.relative_to(ROOT).as_posix(),
        "weights_sha256": hashlib.sha256(saved_weights.read_bytes()).hexdigest(),
        "run_directory": Path(trainer.save_dir).relative_to(ROOT).as_posix(),
        "metrics": metrics.results_dict,
        "per_class": metrics.summary(decimals=6),
        "precision_recall_note": "Ultralytics detection P/R at its validation F1 operating point, not the future business threshold.",
        "confusion_matrix": {
            "values": matrix.matrix.tolist(),
            "rows": "predicted class; last row = background/missed object",
            "columns": "true class; last column = background/unmatched prediction",
            "class_order": dataset_config["classes"] + ["background"],
            "confidence": validator.args.conf,
            "iou_threshold": 0.45,
            "settings_note": "Ultralytics ConfusionMatrix.process_batch defaults; differs from AP matching thresholds.",
        },
        "limitations": dataset_config["limitations"],
        "pending": ["baselines", "test evaluation", "cost sweep", "human review assumptions", "dashboard", "student analysis"],
    }
    write_json(output / "summary.json", summary)
    save_comparison(summary, config, output)
    state.update({"status": "completed", "completed_utc": summary["completed_utc"],
                  "completed_epochs": summary["completed_epochs"],
                  "best_epoch": summary["best_epoch"], "weights": summary["weights"]})
    write_json(output / "run_state.json", state)
    write_json(ROOT / "results" / "latest_training.json", state)
    current = json.loads((ROOT / "training_config.json").read_text())
    if current.get("last_run_id") == run_id:
        current["status"] = "completed"
        write_json(ROOT / "training_config.json", current)
    print("PILOT COMPLETE " + str(output.relative_to(ROOT)), flush=True)
    print(json.dumps(summary["metrics"]), flush=True)
    return summary


if __name__ == "__main__":
    from pipeline import finish_training
    finish_training(main())
