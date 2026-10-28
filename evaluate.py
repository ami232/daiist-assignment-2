"""
Evaluation on the test split: finetuned models against two baselines.

- finetuned yolo11s / yolo11n (weights/),
- the pretrained COCO model zero-shot: apple / banana / orange are mapped to the
  three *fresh* classes, since COCO knows the fruit but not its state,
- a trivial baseline that predicts nothing (every fruit passes).

Predictions are made once at conf=0.001 and cached in artifacts/predictions/, so
every metric and the cost sweep are computed offline from the same detections.
Also writes one row per labelled fruit with its rotten score (highest confidence
of any rotten-class box overlapping it at IoU >= 0.5), used by costs.py and the app.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PRED_DIR = ARTIFACTS / "predictions"
EVAL_DIR = ARTIFACTS / "eval"

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
FRUIT_IOU = 0.5  # a box "is on" a fruit if it overlaps it at least this much
CM_CONF, CM_IOU = 0.25, 0.45  # Ultralytics' defaults for its confusion matrix
COCO_TO_FRESH = {47: 0, 46: 1, 49: 2}  # COCO apple, banana, orange -> fresh_apples, fresh_banana, fresh_oranges

INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes a (N, 4) and b (M, 4)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_ground_truth(dataset: Path, split: str, images: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for img in sorted((dataset / split / "images").iterdir()):
        if images is not None and img.name not in images:
            continue
        label = dataset / split / "labels" / (img.stem + ".txt")
        for line in (label.read_text().splitlines() if label.exists() else []):
            parts = line.split()
            if not parts:
                continue
            cls, c = int(parts[0]), list(map(float, parts[1:]))
            if len(c) == 4:
                x1, y1, x2, y2 = c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2
            else:
                x1, y1, x2, y2 = min(c[0::2]), min(c[1::2]), max(c[0::2]), max(c[1::2])
            rows.append({"image": img.name, "cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return pd.DataFrame(rows)


def image_list(dataset: Path, split: str, list_file: Path | None = None) -> list[Path]:
    if list_file is not None:
        return [Path(p) for p in list_file.read_text().splitlines() if p]
    return sorted((dataset / split / "images").iterdir())


def predict(weights: str, images: list[Path], cache: Path, class_map: dict[int, int] | None = None,
            imgsz: int = 640) -> pd.DataFrame:
    """All detections at conf >= 0.001 (normalized xyxy), cached as CSV."""
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    model = YOLO(weights)
    batches = [images[i:i + 16] for i in range(0, len(images), 16)]  # a list source is one batch otherwise
    results = (r for batch in batches
               for r in model.predict(source=[str(p) for p in batch], conf=0.001, iou=0.7,
                                      max_det=300, imgsz=imgsz, verbose=False))
    for result in results:
        boxes = result.boxes
        name = Path(result.path).name
        for (x1, y1, x2, y2), cls, conf in zip(boxes.xyxyn.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist()):
            if class_map is not None:
                if cls not in class_map:
                    continue
                cls = class_map[cls]
            rows.append({"image": name, "cls": cls, "conf": round(conf, 5),
                         "x1": round(x1, 5), "y1": round(y1, 5), "x2": round(x2, 5), "y2": round(y2, 5)})
    preds = pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cache, index=False)
    return preds


def _match(gt_cls: np.ndarray, pred_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """Ultralytics' greedy matching: (n_pred, 10) bool, true where a prediction is a TP at each IoU threshold."""
    correct = np.zeros((len(pred_cls), len(IOU_THRESHOLDS)), dtype=bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for i, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.argwhere(iou >= threshold)
        if len(matches):
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1], i] = True
    return correct


def detection_metrics(preds: pd.DataFrame, gts: pd.DataFrame, names: list[str],
                      class_map: dict[int, int] | None = None) -> pd.DataFrame:
    """Per-class precision, recall (at the max-F1 confidence), AP50 and AP50-95, plus an 'all' row.

    class_map merges classes before scoring (e.g. the six classes into fresh / rotten).
    """
    if class_map is not None:
        preds = preds.assign(cls=preds.cls.map(class_map))
        gts = gts.assign(cls=gts.cls.map(class_map))
    tp, conf, pred_cls = [], [], []
    pred_groups = dict(tuple(preds.groupby("image")))
    for image, gt in gts.groupby("image"):
        pred = pred_groups.get(image)
        if pred is None:
            continue
        iou = iou_matrix(gt[["x1", "y1", "x2", "y2"]].to_numpy(), pred[["x1", "y1", "x2", "y2"]].to_numpy())
        tp.append(_match(gt.cls.to_numpy(), pred.cls.to_numpy(), iou))
        conf.append(pred.conf.to_numpy())
        pred_cls.append(pred.cls.to_numpy())
    # predictions on images without any label are all false positives
    unlabeled = preds[~preds.image.isin(set(gts.image))]
    tp.append(np.zeros((len(unlabeled), len(IOU_THRESHOLDS)), dtype=bool))
    conf.append(unlabeled.conf.to_numpy())
    pred_cls.append(unlabeled.cls.to_numpy())

    counts = gts.cls.value_counts()
    rows = {names[c]: {"instances": int(counts.get(c, 0)), "precision": 0.0, "recall": 0.0, "ap50": 0.0, "ap50_95": 0.0}
            for c in sorted(gts.cls.unique())}
    if sum(len(c) for c in conf):
        _, _, p, r, _, ap, classes, *_ = ap_per_class(np.concatenate(tp), np.concatenate(conf),
                                                      np.concatenate(pred_cls), gts.cls.to_numpy())
        for i, c in enumerate(classes):
            rows[names[c]].update(precision=float(p[i]), recall=float(r[i]),
                                  ap50=float(ap[i, 0]), ap50_95=float(ap[i].mean()))
    table = pd.DataFrame(rows).T
    table.loc["all"] = [table.instances.sum(), *table[["precision", "recall", "ap50", "ap50_95"]].mean()]
    return table


def confusion_matrix(preds: pd.DataFrame, gts: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class; index n_classes is background (missed / spurious box)."""
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    bg = n_classes
    preds = preds[preds.conf >= CM_CONF]
    pred_groups = dict(tuple(preds.groupby("image")))
    for image in set(gts.image) | set(preds.image):
        gt = gts[gts.image == image] if image in set(gts.image) else gts.iloc[:0]
        pred = pred_groups.get(image, preds.iloc[:0])
        matched_gt, matched_pred = set(), set()
        if len(gt) and len(pred):
            iou = iou_matrix(gt[["x1", "y1", "x2", "y2"]].to_numpy(), pred[["x1", "y1", "x2", "y2"]].to_numpy())
            pairs = np.argwhere(iou >= CM_IOU)
            for g, p in pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]:
                if g in matched_gt or p in matched_pred:
                    continue
                matched_gt.add(g)
                matched_pred.add(p)
                cm[gt.cls.iloc[g], pred.cls.iloc[p]] += 1
        for g in set(range(len(gt))) - matched_gt:
            cm[gt.cls.iloc[g], bg] += 1
        for p in set(range(len(pred))) - matched_pred:
            cm[bg, pred.cls.iloc[p]] += 1
    return cm


def fruit_scores(preds: pd.DataFrame, gts: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One row per labelled fruit: its true state and the highest rotten / fresh confidence on it."""
    rotten_ids = [i for i, n in enumerate(names) if n.startswith("rotten")]
    pred_groups = dict(tuple(preds.groupby("image")))
    rows = []
    for image, gt in gts.groupby("image"):
        pred = pred_groups.get(image)
        if pred is not None:
            iou = iou_matrix(gt[["x1", "y1", "x2", "y2"]].to_numpy(), pred[["x1", "y1", "x2", "y2"]].to_numpy())
            on_fruit = iou >= FRUIT_IOU
            is_rotten_box = pred.cls.isin(rotten_ids).to_numpy()
            conf = pred.conf.to_numpy()
        for k, (_, fruit) in enumerate(gt.iterrows()):
            rotten_score = fresh_score = 0.0
            if pred is not None:
                rotten_score = float(conf[on_fruit[k] & is_rotten_box].max(initial=0.0))
                fresh_score = float(conf[on_fruit[k] & ~is_rotten_box].max(initial=0.0))
            name = names[int(fruit.cls)]
            rows.append({"image": image, "true_class": name, "fruit": name.split("_")[1],
                         "rotten": name.startswith("rotten"),
                         "rotten_score": rotten_score, "fresh_score": fresh_score})
    return pd.DataFrame(rows)


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=SURFACE)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=INK_2)
    ax.set_xlabel("predicted", color=INK_2)
    ax.set_ylabel("true", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(models: dict[str, Path], coco_weights: str, dataset: Path, data_config: Path, imgsz: int = 640) -> dict:
    """Test-split metrics for each model and both baselines, plus fruit-level scores on val and test."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    names = yaml.safe_load(data_config.read_text())["names"]
    state = {i: int(n.startswith("rotten")) for i, n in enumerate(names)}
    test_images = image_list(dataset, "test")
    gts = load_ground_truth(dataset, "test")

    candidates = {name: (str(w), None) for name, w in models.items()}
    candidates[f"coco_{Path(coco_weights).stem}_zero_shot"] = (coco_weights, COCO_TO_FRESH)

    per_class, overall, by_state = [], [], []
    predictions = {}
    for name, (weights, class_map) in candidates.items():
        print(f"Evaluating {name} on test ...")
        preds = predict(weights, test_images, PRED_DIR / f"test_{name}.csv", class_map, imgsz)
        predictions[name] = preds
        table = detection_metrics(preds, gts, names)
        per_class.append(table.drop(index="all").assign(model=name))
        overall.append({"model": name, **table.loc["all"].drop("instances").to_dict()})
        by_state.append(detection_metrics(preds, gts, ["fresh", "rotten"], state).drop(index="all").assign(model=name))

        cm = confusion_matrix(preds, gts, len(names))
        labels = [*names, "background"]
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(EVAL_DIR / f"confusion_{name}.csv")
        plot_confusion(cm, labels, f"{name}: confusion matrix on test (conf ≥ {CM_CONF}, IoU ≥ {CM_IOU})",
                       EVAL_DIR / f"confusion_{name}.png")

    overall.append({"model": "trivial_predict_nothing", "precision": 0.0, "recall": 0.0, "ap50": 0.0, "ap50_95": 0.0})
    pd.DataFrame(overall).to_csv(EVAL_DIR / "overall_test.csv", index=False)
    pd.concat(per_class).rename_axis("class").reset_index().to_csv(EVAL_DIR / "per_class_test.csv", index=False)
    pd.concat(by_state).rename_axis("state").reset_index().to_csv(EVAL_DIR / "by_state_test.csv", index=False)

    # Fruit-level scores for the cost sweep: chosen threshold on val, reported on test.
    chosen = next(iter(models))
    fruit_scores(predictions[chosen], gts, names).to_csv(EVAL_DIR / "fruit_scores_test.csv", index=False)
    val_list = data_config.parent / "clean_valid.txt"
    val_images = image_list(dataset, "valid", val_list)
    val_preds = predict(str(models[chosen]), val_images, PRED_DIR / f"val_{chosen}.csv", imgsz=imgsz)
    val_gts = load_ground_truth(dataset, "valid", {p.name for p in val_images})
    fruit_scores(val_preds, val_gts, names).to_csv(EVAL_DIR / "fruit_scores_val.csv", index=False)

    print("\n== Overall (test) ==\n" + pd.DataFrame(overall).round(3).to_string(index=False))
    print("\n== Per class (test) ==\n" + pd.concat(per_class).round(3).to_string())
    print("\n== Fresh vs rotten (test) ==\n" + pd.concat(by_state).round(3).to_string())
    return {"overall": overall}
