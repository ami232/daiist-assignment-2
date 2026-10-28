from pathlib import Path

from roboflow import Roboflow

from api_key import load_api_key


ROOT = Path(__file__).resolve().parent


def main():
    rf = Roboflow(api_key=load_api_key())

    project = rf.workspace("roboflow100vl-full").project(
        "pill-j8vgy-o5udx-jbna"
    )

    dataset = project.version(1).download(
        "yolov11",
        location=str(ROOT / "datasets" / "missing-pills-v1"),
    )

    print(f"Dataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()