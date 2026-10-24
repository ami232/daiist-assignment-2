"""Assignment 2 training: finetune yolo11n on Roboflow search-rescue v14."""
import shutil
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"

EPOCHS = 50
IMGSZ = 640
BATCH = 16
SEED = 0


def main():
    if not (DATA_DIR / "data.yaml").exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace("trash-32e51").project("search-rescue").version(14)
        version.download("yolov11", location=str(DATA_DIR))

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_DIR / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        project=str(ROOT / "runs"),
        name="finetune",
        exist_ok=True,
    )

    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS_DIR / "best.pt")
    print("Saved weights/best.pt")


if __name__ == "__main__":
    main()