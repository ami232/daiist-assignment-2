"""
Assignment 2: training pipeline.   Run:  uv run python main.py train

1. Download the Roboflow dataset (project-irggm/-w4yds, version 1).
2. Finetune YOLO11 from the pretrained COCO checkpoint -> weights/best.pt
3. Evaluate on the TEST split: finetuned model, COCO zero-shot, predict-nothing.
4. Score every test image ("how rotten does it look?") for the business decision.
5. Sweep the two thresholds (pass / review / fail) and save the cost sweep.
Everything app.py needs is saved in results/.

Settings can be changed with environment variables, e.g.
    EPOCHS=1 uv run python main.py train      (quick test run)
    SKIP_TRAIN=1 uv run python main.py train  (reuse weights/best.pt, redo
                                               evaluation + cost sweep only)
"""

import json
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import costs  # noqa: E402
from api_key import load_api_key  # noqa: E402

WORKSPACE, PROJECT, VERSION = "project-irggm", "-w4yds", 1
DATA_DIR = Path("data") / f"fruit-rot-v{VERSION}"
RESULTS, WEIGHTS = Path("results"), Path("weights")
MODEL = os.environ.get("YOLO_MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("EPOCHS", 20))
IMG_SIZE, SEED = 640, 0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_IDS = {"apple": 47, "banana": 46, "orange": 49}  # COCO class ids of our fruits


# ---------------------------------------------------------------- data
def download() -> None:
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset already in {DATA_DIR}, skipping download.")
        return
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DATA_DIR)
    )


def write_data_yaml() -> tuple[Path, list[str]]:
    """Rewrite Roboflow's data.yaml with absolute paths so it works anywhere."""
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cfg = {
        "path": str(DATA_DIR.resolve()),
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": names,
    }
    out = DATA_DIR / "data_local.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out, names


# ---------------------------------------------------------------- training
def finetune(data_yaml: Path) -> YOLO:
    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMG_SIZE, seed=SEED,
        patience=10, project="runs", name="finetune", exist_ok=True,
    )
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS / "best.pt")
    return YOLO(WEIGHTS / "best.pt")


# ---------------------------------------------------------------- evaluation
def evaluate_finetuned(model: YOLO, data_yaml: Path, names: list[str]) -> dict:
    m = model.val(
        data=str(data_yaml), split="test", imgsz=IMG_SIZE,
        project="runs", name="test_eval", exist_ok=True,
    )
    per_class = pd.DataFrame({
        "class": [names[i] for i in m.box.ap_class_index],
        "precision": m.box.p, "recall": m.box.r,
        "ap50": m.box.ap50, "ap50_95": m.box.ap,
    }).round(4)
    per_class.to_csv(RESULTS / "per_class.csv", index=False)
    np.savetxt(RESULTS / "confusion_matrix.csv", m.confusion_matrix.matrix, delimiter=",", fmt="%.0f")
    (RESULTS / "confusion_matrix_labels.json").write_text(json.dumps(names + ["background"]))
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map)}


def evaluate_coco_zero_shot(names: list[str]) -> dict:
    """Pretrained COCO model, no finetuning. COCO knows apple/banana/orange but
    not fresh vs rotten, so we score it on finding the right FRUIT TYPE."""
    root = DATA_DIR / "coco_test"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    remap = {i: next(v for k, v in COCO_IDS.items() if k in n) for i, n in enumerate(names)}
    for img in (DATA_DIR / "test" / "images").iterdir():
        link = root / "images" / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
    for lbl in (DATA_DIR / "test" / "labels").glob("*.txt"):
        lines = []
        for line in lbl.read_text().splitlines():
            v = line.split()
            if v:
                lines.append(" ".join([str(remap[int(float(v[0]))])] + v[1:]))
        (root / "labels" / lbl.name).write_text("\n".join(lines))

    coco = YOLO(MODEL)
    cfg = {"path": str(root.resolve()), "train": "images", "val": "images",
           "test": "images", "names": coco.names}
    (root / "data.yaml").write_text(yaml.safe_dump(cfg))
    m = coco.val(data=str(root / "data.yaml"), split="test", imgsz=IMG_SIZE,
                 plots=False, project="runs", name="coco_eval", exist_ok=True)
    return {
        "fruit_type_map50": float(m.box.map50),
        "fruit_type_ap50": {coco.names[c]: round(float(a), 4)
                            for c, a in zip(m.box.ap_class_index, m.box.ap50)},
        "can_detect_rot": False,
    }


def score_test_images(model: YOLO, names: list[str]) -> pd.DataFrame:
    """One row per test image: is it really rotten, and the model's highest
    confidence for any rotten box (0 if it found none)."""
    rotten_ids = {i for i, n in enumerate(names) if costs.POSITIVE in n}
    img_dir, lbl_dir = DATA_DIR / "test" / "images", DATA_DIR / "test" / "labels"
    rows = []
    # Pass the folder, not a list: a list is treated as ONE batch and runs out of memory.
    for res in model.predict(source=str(img_dir), stream=True, conf=0.001, imgsz=IMG_SIZE, verbose=False):
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        mask = np.isin(cls, list(rotten_ids))
        lbl = lbl_dir / f"{Path(res.path).stem}.txt"
        truth = lbl.exists() and any(
            int(float(line.split()[0])) in rotten_ids
            for line in lbl.read_text().splitlines() if line.strip()
        )
        rows.append({
            "image": Path(res.path).name,
            "is_rotten": bool(truth),
            "rotten_score": float(conf[mask].max()) if mask.any() else 0.0,
        })
    scores = pd.DataFrame(rows)
    scores.to_csv(RESULTS / "test_scores.csv", index=False)
    return scores


# ---------------------------------------------------------------- costing
def sweep_thresholds(scores: pd.DataFrame) -> dict:
    s, y = scores["rotten_score"].values, scores["is_rotten"].values
    grid = np.round(np.arange(0, 1.0001, 0.02), 2)
    rows = []
    for lo in grid:
        for hi in grid[grid >= lo]:
            r = costs.outcome_rates(s, y, lo, hi)
            rows.append({"t_low": lo, "t_high": hi, **r,
                         "review_share": costs.review_share(r),
                         "cost_per_1000": costs.cost_per_1000(r)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULTS / "cost_sweep.csv", index=False)

    single = sweep[sweep.t_low == sweep.t_high]
    best_single = single.loc[single.cost_per_1000.idxmin()]
    allowed = sweep if costs.MAX_REVIEW_SHARE is None else sweep[sweep.review_share <= costs.MAX_REVIEW_SHARE]
    best_review = allowed.loc[allowed.cost_per_1000.idxmin()]

    plt.figure(figsize=(7, 4))
    plt.plot(single.t_high, single.cost_per_1000)
    plt.axvline(best_single.t_high, linestyle="--", label=f"cheapest = {best_single.t_high:.2f}")
    plt.xlabel("Confidence threshold (rotten)")
    plt.ylabel("Expected cost per 1,000 fruits (EUR)")
    plt.title("Cost against threshold (model decides alone)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "cost_vs_threshold.png", dpi=150)
    plt.close()

    point = {
        "model_alone": {"threshold": float(best_single.t_high),
                        "cost_per_1000": float(best_single.cost_per_1000)},
        "model_plus_review": {"t_low": float(best_review.t_low), "t_high": float(best_review.t_high),
                              "review_share": float(best_review.review_share),
                              "cost_per_1000": float(best_review.cost_per_1000)},
        "alternatives_per_1000": costs.alternatives_per_1000(),
    }
    if costs.CASES_PER_DAY:
        daily = best_review.review_share * costs.CASES_PER_DAY
        point["model_plus_review"]["reviews_per_day"] = float(daily)
        point["model_plus_review"]["reviewer_hours_per_day"] = float(daily * costs.REVIEW_SECONDS / 3600)
    (RESULTS / "operating_point.json").write_text(json.dumps(point, indent=2))
    return point


# ---------------------------------------------------------------- main
def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    download()
    data_yaml, names = write_data_yaml()

    if os.environ.get("SKIP_TRAIN") and (WEIGHTS / "best.pt").exists():
        print("SKIP_TRAIN set: using existing weights/best.pt")
        model = YOLO(WEIGHTS / "best.pt")
    else:
        model = finetune(data_yaml)
    metrics = {
        "settings": {"model": MODEL, "epochs": EPOCHS, "imgsz": IMG_SIZE, "seed": SEED,
                     "trained_on_gpu": torch.cuda.is_available(),
                     "dataset": f"{WORKSPACE}/{PROJECT} v{VERSION}"},
        "finetuned": evaluate_finetuned(model, data_yaml, names),
        "coco_zero_shot": evaluate_coco_zero_shot(names),
        "predict_nothing": {"map50": 0.0, "map50_95": 0.0, "recall": 0.0},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    scores = score_test_images(model, names)
    if costs.costs_ready():
        print(json.dumps(sweep_thresholds(scores), indent=2))
    else:
        print("\nCosts in costs.py are not filled in yet: skipped the cost sweep.\n"
              "Fill them in and run again (training is the slow part; the sweep is quick).")
    print(f"\nDone. Weights in {WEIGHTS}/best.pt, results in {RESULTS}/")


if __name__ == "__main__":
    main()