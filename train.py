"""
Assignment 2: training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file. main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via uv run python main.py train, this file must, start to finish, with no
manual steps in between:

- Get your Roboflow API key via from api_key import load_api_key, so that it
  works whether the key is in your .env file or in the environment, and
  download your dataset version from Roboflow Universe.
- Finetune yolo11n/yolo11s (or the -cls variant) on that dataset, from the
  pretrained checkpoint.
- Evaluate on the test split against the two baselines named in the brief: the
  pretrained COCO model zero-shot, and a trivial predict-nothing / majority
  class baseline.
- Sweep the confidence threshold, compute the expected cost at each point from
  the cost matrix you defined in REPORT.md, and save the sweep.
- Save everything app.py needs to build its dashboard without retraining:
  the metrics, the per-class numbers, the cost sweep, and the chosen operating
  point.

This is your real training pipeline. It takes as long as it takes, and it runs
wherever training is practical for you: your own machine, a GPU you have access
to, or a hosted notebook. Nothing re-runs it to check your submission.

The weights it produces are what you commit to weights/ and what your app
serves. Say in REPORT.md where you trained and what settings you used.

Save the trained model into weights/. That is the file you commit and the file
app.py loads.

How you structure the code beyond that is your call to make and be able to
defend: file layout, function boundaries, naming.

"""


import json
import shutil
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from download_data import main as download_dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "missing-pills-v1"


def main():
    if not (DATA / "data.yaml").exists():
        download_dataset()

    # Use absolute paths so the pipeline works locally and in Colab.
    config = yaml.safe_load((DATA / "data.yaml").read_text())
    config["path"] = str(DATA.resolve())
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    config_path = DATA / "training.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    device = 0 if torch.cuda.is_available() else "cpu"

    from inspect_data import main as inspect_dataset

    inspect_dataset()

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(config_path),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="missing_pills",
        plots=True,
        save=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"

    if not best.exists():
        raise FileNotFoundError(f"Training produced no best weights: {best}")

    weights_dir = ROOT / "weights"
    weights_dir.mkdir(exist_ok=True)
    shutil.copy2(best, weights_dir / "best.pt")

    metadata = {
        "initial_model": "yolo11n.pt",
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "run_directory": str(run_dir),
        "settings_file": str(run_dir / "args.yaml"),
    }
    (weights_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Best model saved to: {weights_dir / 'best.pt'}")

    from evaluate_model import main as evaluate_model
    from evaluate_baselines import main as evaluate_baselines
    from collect_scores import main as collect_scores
    from choose_threshold import main as choose_threshold
    from review_policy import main as review_policy, run_sensitivity

    evaluate_model()
    evaluate_baselines()
    collect_scores()
    choose_threshold()
    review_policy()
    run_sensitivity()

    print("Training, evaluation and cost analysis completed.")


if __name__ == "__main__":
    main()
