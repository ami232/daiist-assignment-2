from pathlib import Path

import numpy as np

from dataset import IMAGE_SUFFIXES, read_boxes

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
BATCH_SIZE = 32


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def load_ground_truth(dataset_dir: Path, split: str) -> dict:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    truth = {}
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_dir / f"{image.stem}.txt"
        rows = read_boxes(label) if label.is_file() else []
        if rows:
            array = np.array(rows, dtype=float)
            classes = array[:, 0].astype(int)
            boxes = xywh_to_xyxy(array[:, 1:5])
        else:
            classes = np.zeros(0, dtype=int)
            boxes = np.zeros((0, 4))
        truth[str(image)] = {"classes": classes, "boxes": boxes}
    return truth


def predict(model, image_paths, imgsz, conf=0.001, iou=0.6, classes=None, class_map=None) -> dict:
    predictions = {}
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[start : start + BATCH_SIZE]
        results = model.predict(batch, imgsz=imgsz, conf=conf, iou=iou, classes=classes, verbose=False)
        for path, result in zip(batch, results):
            found = result.boxes
            predicted = found.cls.cpu().numpy().astype(int)
            if class_map is not None:
                predicted = np.array([class_map[int(c)] for c in predicted], dtype=int)
            predictions[path] = {
                "boxes": found.xyxyn.cpu().numpy(),
                "conf": found.conf.cpu().numpy(),
                "classes": predicted,
            }
    return predictions


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(grid, mrec, mpre), grid))


def match(ious: np.ndarray, threshold: float) -> np.ndarray:
    true_positive = np.zeros(ious.shape[0], dtype=bool)
    pairs = np.argwhere(ious >= threshold)
    if len(pairs) == 0:
        return true_positive
    scores = ious[pairs[:, 0], pairs[:, 1]]
    pairs = pairs[np.argsort(-scores)]
    pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
    true_positive[pairs[:, 0]] = True
    return true_positive


def class_ap(predictions: dict, truth: dict, class_id: int):
    total = sum(int((t["classes"] == class_id).sum()) for t in truth.values())
    if total == 0:
        return None, None
    confidences = []
    hits = [[] for _ in IOU_THRESHOLDS]
    for path, item in truth.items():
        found = predictions.get(path)
        if found is None:
            continue
        mask = found["classes"] == class_id
        boxes, conf = found["boxes"][mask], found["conf"][mask]
        order = np.argsort(-conf)
        boxes, conf = boxes[order], conf[order]
        ious = box_iou(boxes, item["boxes"][item["classes"] == class_id])
        confidences.append(conf)
        for k, threshold in enumerate(IOU_THRESHOLDS):
            hits[k].append(match(ious, threshold))
    if not confidences or sum(len(c) for c in confidences) == 0:
        return 0.0, 0.0
    all_conf = np.concatenate(confidences)
    order = np.argsort(-all_conf)
    scores = []
    for k in range(len(IOU_THRESHOLDS)):
        tp = np.concatenate(hits[k])[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(~tp)
        recall = tp_cum / total
        precision = tp_cum / (tp_cum + fp_cum)
        scores.append(average_precision(recall, precision))
    return scores[0], float(np.mean(scores))


def confusion_matrix(predictions: dict, truth: dict, n_classes: int, conf_threshold: float, iou_threshold: float = 0.5) -> np.ndarray:
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    background = n_classes
    for path, item in truth.items():
        gt_classes, gt_boxes = item["classes"], item["boxes"]
        found = predictions.get(path)
        if found is not None:
            keep = found["conf"] >= conf_threshold
            boxes, classes = found["boxes"][keep], found["classes"][keep]
            conf = found["conf"][keep]
        else:
            boxes, classes, conf = np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
        ious = box_iou(boxes, gt_boxes)
        pairs = np.argwhere(ious >= iou_threshold)
        if len(pairs):
            pairs = pairs[np.argsort(-ious[pairs[:, 0], pairs[:, 1]])]
        used_pred, used_gt = set(), set()
        for i, j in pairs:
            if i in used_pred or j in used_gt:
                continue
            used_pred.add(i)
            used_gt.add(j)
            matrix[gt_classes[j], classes[i]] += 1
        for j in range(len(gt_classes)):
            if j not in used_gt:
                matrix[gt_classes[j], background] += 1
        for i in range(len(classes)):
            if i not in used_pred:
                matrix[background, classes[i]] += 1
    return matrix


def precision_recall(matrix: np.ndarray, n_classes: int) -> list[dict]:
    rows = []
    for c in range(n_classes):
        tp = int(matrix[c, c])
        fp = int(matrix[:, c].sum() - tp)
        fn = int(matrix[c, :].sum() - tp)
        rows.append(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
            }
        )
    return rows


def rounded(value):
    return None if value is None else round(float(value), 4)


def evaluate_predictions(predictions: dict, truth: dict, class_names: list[str], conf_threshold: float) -> dict:
    n_classes = len(class_names)
    matrix = confusion_matrix(predictions, truth, n_classes, conf_threshold)
    counts = precision_recall(matrix, n_classes)
    per_class = {}
    for c, name in enumerate(class_names):
        ap50, ap5095 = class_ap(predictions, truth, c)
        per_class[name] = {
            "instances": int(sum((t["classes"] == c).sum() for t in truth.values())),
            "ap50": rounded(ap50),
            "ap50_95": rounded(ap5095),
            "precision": rounded(counts[c]["precision"]),
            "recall": rounded(counts[c]["recall"]),
            "tp": counts[c]["tp"],
            "fp": counts[c]["fp"],
            "fn": counts[c]["fn"],
        }
    scored = [v for v in per_class.values() if v["ap50"] is not None]
    return {
        "map50": rounded(np.mean([v["ap50"] for v in scored])) if scored else None,
        "map50_95": rounded(np.mean([v["ap50_95"] for v in scored])) if scored else None,
        "conf_threshold": conf_threshold,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": class_names + ["background"],
    }
