"""
Training pipeline for Assignment 2.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download the dataset version named in config.py from Roboflow Universe.
2. Describe it: images per split, instances per class, object size.
3. Finetune the pretrained YOLO checkpoint and copy the best weights to weights/.
4. Score three models on the same validation and test images: the finetuned
   model, the pretrained COCO model zero-shot, and a model that predicts nothing.
5. Cost the errors (costing.py): sweep the threshold, choose the operating point
   on validation, report it on test.
6. Save everything the dashboard needs into outputs/, plus examples of failures.

SKIP_TRAINING=1 reuses weights/best.pt and reruns steps 4 to 6 only, which is
handy after changing the evaluation. A normal run always trains.
"""

from __future__ import annotations

import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import config as cfg
import costing
import evaluation as ev

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"
OUTPUTS = ROOT / "outputs"
WEIGHTS = ROOT / "weights" / "best.pt"
PREDICT_CONF = 0.001   # keep nearly every box; the thresholds are applied later, in costing
MODELS = ("finetuned", "coco_zero_shot", "predict_nothing")


def main() -> None:
    check_config()
    OUTPUTS.mkdir(exist_ok=True)

    data_yaml, names = get_dataset()
    dataset_dir = data_yaml.parent
    positive_ids = [names.index(c) for c in cfg.POSITIVE_CLASSES]

    print("\n[2/6] Describing the dataset")
    truth = {split: ev.load_ground_truth(dataset_dir, split) for split in ev.SPLIT_DIRS}
    stats = {split: ev.describe_split(truth[split], names, positive_ids) for split in truth}
    costing.save_json(OUTPUTS / "dataset_stats.json", {
        "dataset": dataset_info(), "names": names, "positive_classes": cfg.POSITIVE_CLASSES,
        "small_box_means": f"under {ev.SMALL_AREA:.0%} of the image area",
        "large_box_means": f"over {ev.LARGE_AREA:.0%} of the image area",
        "splits": stats})
    plot_dataset(truth, names, OUTPUTS / "dataset_overview.png")
    for split, s in stats.items():
        print(f"  {split:<5} {s['images']:>6} images, {s['positive_image_pct']}% contain a positive")

    device = pick_device()
    if os.environ.get("SKIP_TRAINING") == "1" and WEIGHTS.is_file():
        print(f"\n[3/6] SKIP_TRAINING=1: reusing {WEIGHTS.relative_to(ROOT)}")
    else:
        train(data_yaml, device)

    print("\n[4/6] Scoring the finetuned model and both baselines on the same images")
    from ultralytics import YOLO
    finetuned = YOLO(str(WEIGHTS))
    coco = YOLO(cfg.BASE_MODEL)
    coco_map = coco_class_map(coco.names, names)
    print(f"  COCO zero-shot stands in for: {describe_map(coco_map, coco.names, names) or 'none of your classes'}")

    dets: dict[str, dict[str, list[dict]]] = {m: {} for m in MODELS}
    for split in ("val", "test"):
        images = [t["path"] for t in truth[split]]
        dets["finetuned"][split] = predict(finetuned, images, device)
        dets["coco_zero_shot"][split] = predict(coco, images, device, coco_map)
        dets["predict_nothing"][split] = [empty_detection() for _ in images]

    image_scores = {"positive_classes": cfg.POSITIVE_CLASSES,
                    "images": {s: [t["path"].name for t in truth[s]] for s in ("val", "test")},
                    "models": {}}
    evaluation = {"names": names, "positive_classes": cfg.POSITIVE_CLASSES,
                  "coco_stand_ins": describe_map(coco_map, coco.names, names),
                  "models": {}, "image_level": {}}
    for model in MODELS:
        image_scores["models"][model] = {}
        evaluation["image_level"][model] = {}
        for split in ("val", "test"):
            labels, scores = ev.image_level(dets[model][split], truth[split], positive_ids)
            image_scores["models"][model][split] = {"labels": labels, "scores": scores}
            evaluation["image_level"][model][split] = {"roc_auc": ev.roc_auc(labels, scores),
                                                       "positives": int(labels.sum()),
                                                       "negatives": int((1 - labels).sum())}
        evaluation["models"][model] = ev.evaluate_detections(
            dets[model]["test"], truth["test"], names, with_curves=(model == "finetuned"))
        m = evaluation["models"][model]
        print(f"  {model:<16} test mAP50 {m['map50']:.3f}  mAP50-95 {m['map50_95']:.3f}  "
              f"image-level AUC {evaluation['image_level'][model]['test']['roc_auc']:.3f}")

    warn_if_one_sided(image_scores)
    evaluation["ultralytics_validator"] = ultralytics_test_metrics(data_yaml, device)
    costing.save_json(OUTPUTS / "evaluation.json", evaluation)
    costing.save_json(OUTPUTS / "image_scores.json", image_scores)

    print("\n[5/6] Costing the errors")
    operating_point = costing.run(OUTPUTS)

    print("\n[6/6] Saving examples of where the model fails")
    test_scores = image_scores["models"]["finetuned"]["test"]
    labels, scores = np.asarray(test_scores["labels"]), np.asarray(test_scores["scores"])
    save_examples(truth["test"], dets["finetuned"]["test"], labels, scores,
                  operating_point["model_alone"]["threshold"],
                  operating_point["with_review"]["pass_below"], names, positive_ids)
    print(f"\nDone. Weights in {WEIGHTS.relative_to(ROOT)}, results in {OUTPUTS.relative_to(ROOT)}/")


# ------------------------------------------------------------------ setup --

def check_config() -> None:
    missing = []
    if "TODO" in f"{cfg.ROBOFLOW_WORKSPACE}{cfg.ROBOFLOW_PROJECT}" or not cfg.ROBOFLOW_VERSION:
        missing.append("ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT and ROBOFLOW_VERSION")
    if not cfg.POSITIVE_CLASSES or any("TODO" in c for c in cfg.POSITIVE_CLASSES):
        missing.append("POSITIVE_CLASSES")
    if missing:
        raise SystemExit("Fill these in config.py first: " + "; ".join(missing))
    if cfg.PLACEHOLDERS:
        print("WARNING: config.PLACEHOLDERS is True. The cost figures are still placeholders.")


def dataset_info() -> dict:
    return {"workspace": cfg.ROBOFLOW_WORKSPACE, "project": cfg.ROBOFLOW_PROJECT,
            "version": cfg.ROBOFLOW_VERSION,
            "url": f"https://universe.roboflow.com/{cfg.ROBOFLOW_WORKSPACE}/"
                   f"{cfg.ROBOFLOW_PROJECT}/dataset/{cfg.ROBOFLOW_VERSION}"}


def get_dataset() -> tuple[Path, list[str]]:
    """Download once into datasets/, then write a data file with absolute paths."""
    target = DATASETS / f"{cfg.ROBOFLOW_PROJECT}-v{cfg.ROBOFLOW_VERSION}"
    if (target / "data.yaml").is_file():
        print(f"[1/6] Dataset already in {target.relative_to(ROOT)}, not downloading again")
    else:
        print(f"[1/6] Downloading {cfg.ROBOFLOW_WORKSPACE}/{cfg.ROBOFLOW_PROJECT} "
              f"version {cfg.ROBOFLOW_VERSION} from Roboflow Universe")
        from roboflow import Roboflow

        from api_key import load_api_key

        rf = Roboflow(api_key=load_api_key())
        project = rf.workspace(cfg.ROBOFLOW_WORKSPACE).project(cfg.ROBOFLOW_PROJECT)
        try:
            version = project.version(cfg.ROBOFLOW_VERSION)
        except Exception as error:  # noqa: BLE001 - say which versions do exist
            available = ", ".join(str(getattr(v, "version", v)) for v in project.versions()) or "none"
            raise SystemExit(f"Version {cfg.ROBOFLOW_VERSION} of {cfg.ROBOFLOW_PROJECT} not found ({error}). "
                             f"Versions on Universe: {available}") from error
        version.download("yolov11", location=str(target), overwrite=True)

    raw = yaml.safe_load((target / "data.yaml").read_text(encoding="utf-8"))
    names = raw["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for split, folder in ev.SPLIT_DIRS.items():
        if not ev.split_images(target, split):
            raise SystemExit(f"No {split} images in {target / folder / 'images'}. "
                             f"Pick a dataset version on Universe that has a {split} split.")
    unknown = [c for c in cfg.POSITIVE_CLASSES if c not in names]
    if unknown:
        raise SystemExit(f"POSITIVE_CLASSES {unknown} are not in this dataset. Its classes are {names}.")

    local = target / "data_local.yaml"
    local.write_text(yaml.safe_dump({
        "path": str(target.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "names": dict(enumerate(names))}, sort_keys=False), encoding="utf-8")
    return local, names


def pick_device():
    if cfg.DEVICE != "auto":
        return cfg.DEVICE
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------- training --

def train(data_yaml: Path, device) -> None:
    import torch
    import ultralytics
    from ultralytics import YOLO

    print(f"\n[3/6] Finetuning {cfg.BASE_MODEL} for up to {cfg.EPOCHS} epochs on {device}")
    started = time.time()
    model = YOLO(cfg.BASE_MODEL)
    model.train(data=str(data_yaml), epochs=cfg.EPOCHS, imgsz=cfg.IMAGE_SIZE, batch=cfg.BATCH,
                patience=cfg.PATIENCE, seed=cfg.SEED, deterministic=True, device=device,
                project=str(RUNS), name="finetune", exist_ok=True, plots=True)
    save_dir = Path(model.trainer.save_dir)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy2(save_dir / "weights" / "best.pt", WEIGHTS)

    results_csv = save_dir / "results.csv"
    for source, target in ((results_csv, "training_results.csv"), (save_dir / "results.png", "training_curves.png")):
        if source.is_file():
            shutil.copy2(source, OUTPUTS / target)
    epochs_done = max(len(results_csv.read_text().strip().splitlines()) - 1, 0) if results_csv.is_file() else None

    costing.save_json(OUTPUTS / "training_run.json", {
        "dataset": dataset_info(), "base_model": cfg.BASE_MODEL,
        "epochs_requested": cfg.EPOCHS, "epochs_completed": epochs_done, "patience": cfg.PATIENCE,
        "image_size": cfg.IMAGE_SIZE, "batch": cfg.BATCH, "seed": cfg.SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "where_trained": cfg.WHERE_TRAINED,
        "minutes": round((time.time() - started) / 60, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                     "python": platform.python_version()},
    })


# ------------------------------------------------------------- evaluation --

def empty_detection() -> dict:
    return {"cls": np.zeros(0, dtype=int), "conf": np.zeros(0), "boxes": np.zeros((0, 4))}


def predict(model, images: list[Path], device, class_map: dict[int, int] | None = None,
            batch: int = 16) -> list[dict]:
    """Boxes for every image, with class ids translated to the dataset's if a map is given."""
    out = []
    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start:start + batch]]
        for r in model.predict(chunk, imgsz=cfg.IMAGE_SIZE, conf=PREDICT_CONF, device=device, verbose=False):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            boxes = r.boxes.xyxyn.cpu().numpy().astype(float).reshape(-1, 4)
            if class_map is not None:
                keep = np.array([c in class_map for c in cls], dtype=bool)
                cls = np.array([class_map[c] for c in cls[keep]], dtype=int)
                conf, boxes = conf[keep], boxes[keep]
            out.append({"cls": cls, "conf": conf, "boxes": boxes})
    return out


def coco_class_map(coco_names: dict[int, str], names: list[str]) -> dict[int, int]:
    """COCO class id -> dataset class id: identical names, plus the stand-ins in config."""
    ours = {n.lower(): i for i, n in enumerate(names)}
    coco_ids = {n.lower(): i for i, n in coco_names.items()}
    mapping = {cid: ours[n.lower()] for cid, n in coco_names.items() if n.lower() in ours}
    for our_class, stand_ins in cfg.COCO_STAND_INS.items():
        if our_class not in names:
            raise SystemExit(f"COCO_STAND_INS uses {our_class!r}, which is not one of {names}.")
        for coco_name in stand_ins:
            if coco_name.lower() not in coco_ids:
                raise SystemExit(f"COCO has no class called {coco_name!r}.")
            mapping[coco_ids[coco_name.lower()]] = names.index(our_class)
    return mapping


def describe_map(mapping: dict[int, int], coco_names: dict[int, str], names: list[str]) -> str:
    return ", ".join(f"COCO '{coco_names[c]}' -> '{names[d]}'" for c, d in mapping.items())


def ultralytics_test_metrics(data_yaml: Path, device) -> dict:
    """The official Ultralytics validator on the test split, kept as a cross-check."""
    from ultralytics import YOLO

    metrics = YOLO(str(WEIGHTS)).val(data=str(data_yaml), split="test", imgsz=cfg.IMAGE_SIZE,
                                     batch=cfg.BATCH, device=device, project=str(RUNS),
                                     name="test_eval", exist_ok=True, plots=True, verbose=False)
    per_class = []
    for i, c in enumerate(metrics.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class.append({"class": metrics.names[int(c)], "precision": p, "recall": r,
                          "ap50": ap50, "ap50_95": ap})
    plots = OUTPUTS / "ultralytics_test"
    shutil.rmtree(plots, ignore_errors=True)
    plots.mkdir()
    for png in Path(metrics.save_dir).glob("*.png"):
        shutil.copy2(png, plots / png.name)
    return {"precision": metrics.box.mp, "recall": metrics.box.mr,
            "map50": metrics.box.map50, "map50_95": metrics.box.map, "per_class": per_class,
            "note": "Ultralytics validator at conf 0.001. Its precision and recall are taken "
                    "at the single confidence that maximises F1, unlike the dashboard's."}


def warn_if_one_sided(image_scores: dict) -> None:
    for split, v in image_scores["models"]["finetuned"].items():
        labels = np.asarray(v["labels"])
        if labels.sum() == 0 or labels.sum() == len(labels):
            print(f"  WARNING: every {split} image is {'positive' if labels.sum() else 'negative'}. "
                  "False alarm or miss rates cannot be measured on it.")


# ------------------------------------------------------------------ visuals --

def plot_dataset(truth: dict, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    x, width = np.arange(len(names)), 0.27
    for k, split in enumerate(("train", "val", "test")):
        counts = [len(a) for a in ev.box_areas(truth[split], len(names))]
        bars = left.bar(x + (k - 1) * width, counts, width, label=split)
        left.bar_label(bars, fontsize=7)
    left.set_xticks(x, names)
    left.set_ylabel("Instances")
    left.set_title("Instances per class and split")
    left.legend()

    bins = np.logspace(-4, 0, 30)
    for name, areas in zip(names, ev.box_areas(truth["train"], len(names))):
        if len(areas):
            right.hist(areas[areas > 0], bins=bins, histtype="step", linewidth=1.6, label=name)
    right.axvline(ev.SMALL_AREA, color="grey", ls=":", label="1% of the image")
    right.set_xscale("log")
    right.set_xlabel("Box area as a share of the image (train split)")
    right.set_ylabel("Boxes")
    right.set_title("Object size relative to the image")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_examples(truth: list[dict], dets: list[dict], labels: np.ndarray, scores: np.ndarray,
                  threshold: float, pass_below: float, names: list[str], positive_ids: list[int]) -> None:
    """Worst misses and false alarms at the model-alone threshold, with the truth drawn in white.

    Also copies a few raw test images to outputs/examples/ so the dashboard has
    something to try without the dataset.
    """
    failures, examples = OUTPUTS / "failures", OUTPUTS / "examples"
    for folder in (failures, examples):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()

    order = np.argsort(scores, kind="stable")
    missed = [i for i in order if labels[i] == 1 and scores[i] < threshold][:6]
    false_alarms = [i for i in order[::-1] if labels[i] == 0 and scores[i] >= threshold][:6]
    caught = [i for i in order[::-1] if labels[i] == 1 and scores[i] >= threshold][:2]
    show_from = max(0.10, pass_below)

    for kind, picks in (("missed", missed), ("false_alarm", false_alarms)):
        for rank, i in enumerate(picks, start=1):
            rgb = np.array(Image.open(truth[i]["path"]).convert("RGB"))
            drawn = ev.draw(rgb, dets[i]["boxes"], dets[i]["cls"], dets[i]["conf"],
                            names, positive_ids, min_conf=show_from, truth=truth[i])
            Image.fromarray(drawn).save(failures / f"{kind}_{rank:02d}_score_{scores[i]:.2f}.jpg", quality=88)

    for kind, picks in (("caught", caught), ("missed", missed[:2]), ("false_alarm", false_alarms[:2])):
        for rank, i in enumerate(picks, start=1):
            image = Image.open(truth[i]["path"]).convert("RGB")
            image.thumbnail((800, 800))
            image.save(examples / f"{kind}_{rank}.jpg", quality=85)
    print(f"  {len(missed)} misses and {len(false_alarms)} false alarms saved to {failures.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
