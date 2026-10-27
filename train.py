"""
Assignment 2: training pipeline.

    uv run python main.py train

Two models, one per piece of protective equipment, each trained and costed on
its own dataset from Roboflow Universe:

- helmet: Hard Hat Workers v10 (`head` = a worker without a hard hat)
- vest:   Safety Vests v1 (`NO-Safety Vest` = a worker without a vest)

For each one:
1. Download the dataset.
2. Finetune yolo11s on it.
3. Evaluate on the test split against two baselines: COCO yolo11s zero-shot and
   a model that predicts nothing.
4. Turn detections into one score per snapshot, sweep the thresholds on the
   validation split with the cost model in costing.py, and report the chosen
   operating point on the test split.
5. Save weights/<task>.pt and everything app.py needs into artifacts/<task>/.

Set SKIP_TRAIN to a comma-separated list of tasks (e.g. SKIP_TRAIN=helmet) to
reuse their committed weights and only redo steps 3 to 5.
Set RESUME=1 to continue interrupted runs from runs/<task>/weights/last.pt.
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "yolo11s.pt"
TRAIN_ARGS = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "patience": 15,
    "seed": 0,
    "workers": 2,  # 4 ran the laptop out of RAM
}
COCO_PERSON = 0
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
PRED_CONF = 0.001             # keep low-confidence boxes so AP and the sweep see the whole curve


@dataclass
class Task:
    key: str
    workspace: str
    project: str
    version: int
    names: dict[int, str]
    positive: int
    # Box area as a share of the image, for the recall-by-size breakdown.
    size_bins: list[tuple[float, float, str]]

    @property
    def data_dir(self) -> Path:
        return ROOT / "datasets" / f"{self.project}-{self.version}"

    @property
    def weights(self) -> Path:
        return ROOT / "weights" / f"{self.key}.pt"

    @property
    def artifacts(self) -> Path:
        return ROOT / "artifacts" / self.key

    @property
    def costs(self):
        return costing.TASK_COSTS[self.key]


TASKS = [
    Task("helmet", "joseph-nelson", "hard-hat-workers", 10,
         names={0: "head", 1: "helmet", 2: "person"}, positive=0,
         size_bins=[(0, 0.004, "small (<0.4% of image)"), (0.004, 0.02, "medium (0.4-2%)"), (0.02, 1.01, "large (>2%)")]),
    # Vest boxes cover the torso, so they are about ten times larger than heads.
    Task("vest", "roboflow-universe-projects", "safety-vests", 1,
         names={0: "NO-Safety Vest", 1: "Safety Vest"}, positive=0,
         size_bins=[(0, 0.02, "small (<2% of image)"), (0.02, 0.10, "medium (2-10%)"), (0.10, 1.01, "large (>10%)")]),
]


def download_dataset(task: Task) -> Path:
    if (task.data_dir / "data.yaml").exists():
        print(f"Dataset already at {task.data_dir}")
    else:
        from roboflow import Roboflow

        rf = Roboflow(api_key=load_api_key())
        rf.workspace(task.workspace).project(task.project).version(task.version).download(
            "yolov11", location=str(task.data_dir)
        )
    # Roboflow writes relative split paths; make them absolute so Ultralytics
    # resolves them no matter where it is run from.
    yaml_path = task.data_dir / "data.yaml"
    text = yaml_path.read_text()
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        text = text.replace(f"{split}: ../{folder}/images", f"{split}: {(task.data_dir / folder / 'images').as_posix()}")
    yaml_path.write_text(text)
    return yaml_path


def train(task: Task, data_yaml: Path) -> dict:
    last = ROOT / "runs" / task.key / "weights" / "last.pt"
    if os.environ.get("RESUME") == "1" and last.exists():
        model = YOLO(str(last))
        model.train(resume=True, workers=TRAIN_ARGS["workers"])
    else:
        model = YOLO(BASE_MODEL)
        model.train(data=str(data_yaml), project=str(ROOT / "runs"), name=task.key, exist_ok=True, **TRAIN_ARGS)
    best = Path(model.trainer.best)
    task.weights.parent.mkdir(exist_ok=True)
    # Ultralytics downloads pretrained checkpoints (e.g. for its AMP check)
    # into weights/. Move them out so weights/ holds only the models we ship.
    shipped = {t.weights for t in TASKS}
    for stray in task.weights.parent.glob("*.pt"):
        if stray not in shipped:
            shutil.move(stray, ROOT / "runs" / stray.name)
    shutil.copy(best, task.weights)
    print(f"Saved {task.weights}")
    return {"base_model": BASE_MODEL, **TRAIN_ARGS, "epochs_run": int(model.trainer.epoch) + 1}


# --------------------------------------------------------------------------- #
# Ground truth and predictions
# --------------------------------------------------------------------------- #

def xywhn_to_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    x, y, bw, bh = boxes.T
    return np.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], axis=1)


def load_split(task: Task, split: str) -> list[dict]:
    images = sorted((task.data_dir / split / "images").glob("*.jpg"))
    out = []
    for path in images:
        label = task.data_dir / split / "labels" / f"{path.stem}.txt"
        rows = np.loadtxt(label, ndmin=2) if label.exists() and label.stat().st_size else np.zeros((0, 5))
        w, h = Image.open(path).size
        out.append({
            "path": path,
            "cls": rows[:, 0].astype(int),
            "boxes": xywhn_to_xyxy(rows[:, 1:5], w, h),
            "area": rows[:, 3] * rows[:, 4],
        })
    return out


def predict(model: YOLO, gts: list[dict], class_map: dict[int, list[int]] | None = None) -> list[dict]:
    """Run the model over every image. class_map renames model classes into
    dataset classes (and drops the rest), which is how COCO is scored."""
    preds = []
    for gt in gts:
        r = model.predict(str(gt["path"]), conf=PRED_CONF, max_det=300, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        if class_map is not None:
            keep_b, keep_c, keep_k = [], [], []
            for src, targets in class_map.items():
                for target in targets:
                    m = cls == src
                    keep_b.append(boxes[m]); keep_c.append(conf[m]); keep_k.append(np.full(m.sum(), target))
            boxes, conf, cls = np.concatenate(keep_b), np.concatenate(keep_c), np.concatenate(keep_k)
        preds.append({"boxes": boxes, "conf": conf, "cls": cls})
    return preds


def no_predictions(gts: list[dict]) -> list[dict]:
    return [{"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "cls": np.zeros(0, int)} for _ in gts]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(gt: dict, pred: dict) -> np.ndarray:
    """(n_pred, 10) bool: is each prediction a true positive at each IoU
    threshold. Same greedy rule Ultralytics' validator uses."""
    correct = np.zeros((len(pred["cls"]), len(IOU_THRESHOLDS)), dtype=bool)
    iou = box_iou(gt["boxes"], pred["boxes"]) * (gt["cls"][:, None] == pred["cls"][None, :])
    for i, t in enumerate(IOU_THRESHOLDS):
        pairs = np.argwhere(iou >= t)
        if len(pairs):
            pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
            correct[pairs[:, 1], i] = True
    return correct


# --------------------------------------------------------------------------- #
# Detection metrics
# --------------------------------------------------------------------------- #

def detection_metrics(task: Task, gts: list[dict], preds: list[dict], op_conf: float) -> dict:
    """mAP plus per-class AP50, AP50-95, and precision/recall at op_conf."""
    tp = np.concatenate([match(g, p) for g, p in zip(gts, preds)] or [np.zeros((0, 10), bool)])
    conf = np.concatenate([p["conf"] for p in preds])
    pred_cls = np.concatenate([p["cls"] for p in preds])
    target_cls = np.concatenate([g["cls"] for g in gts])

    per_class = {}
    if len(conf):
        out = ap_per_class(tp, conf, pred_cls, target_cls, names=task.names)
        ap, classes = out[5], out[6]
        ap_by_class = {int(c): ap[i] for i, c in enumerate(classes)}
    else:
        ap_by_class = {}
    for c, name in task.names.items():
        n_gt = int((target_cls == c).sum())
        at_op = (pred_cls == c) & (conf >= op_conf)
        hits = int(tp[at_op, 0].sum())
        ap_c = ap_by_class.get(c, np.zeros(10))
        per_class[name] = {
            "instances": n_gt,
            "AP50": float(ap_c[0]),
            "AP50_95": float(ap_c.mean()),
            "precision": hits / at_op.sum() if at_op.sum() else 0.0,
            "recall": hits / n_gt if n_gt else 0.0,
        }
    return {
        "mAP50": float(np.mean([v["AP50"] for v in per_class.values()])),
        "mAP50_95": float(np.mean([v["AP50_95"] for v in per_class.values()])),
        "per_class": per_class,
    }


def confusion_matrix(task: Task, gts: list[dict], preds: list[dict], conf_thres: float,
                     iou_thres: float = 0.5) -> list[list[int]]:
    """Detection confusion matrix, rows = true class, columns = predicted
    class, last row/column = background (missed object / false box)."""
    n = len(task.names)
    m = np.zeros((n + 1, n + 1), dtype=int)
    for gt, pred in zip(gts, preds):
        keep = pred["conf"] >= conf_thres
        pb, pc = pred["boxes"][keep], pred["cls"][keep]
        iou = box_iou(gt["boxes"], pb)
        used_g, used_p = set(), set()
        for gi, pi in sorted(np.argwhere(iou >= iou_thres), key=lambda ij: -iou[ij[0], ij[1]]):
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi)
            m[gt["cls"][gi], pc[pi]] += 1
        for gi in set(range(len(gt["cls"]))) - used_g:
            m[gt["cls"][gi], n] += 1
        for pi in set(range(len(pc))) - used_p:
            m[n, pc[pi]] += 1
    return m.tolist()


def recall_by_size(task: Task, gts: list[dict], preds: list[dict], conf_thres: float) -> list[dict]:
    """Recall of the positive class split by box area as a share of the image."""
    bins = task.size_bins
    found, total = np.zeros(len(bins)), np.zeros(len(bins))
    for gt, pred in zip(gts, preds):
        keep = (pred["cls"] == task.positive) & (pred["conf"] >= conf_thres)
        iou = box_iou(gt["boxes"], pred["boxes"][keep])
        for gi in np.where(gt["cls"] == task.positive)[0]:
            b = next(i for i, (lo, hi, _) in enumerate(bins) if lo <= gt["area"][gi] < hi)
            total[b] += 1
            found[b] += bool(keep.sum()) and iou[gi].max() >= 0.5
    return [{"size": name, "instances": int(t), "recall": float(f / t) if t else 0.0}
            for (_, _, name), f, t in zip(bins, found, total)]


# --------------------------------------------------------------------------- #
# Snapshot-level scores and costing
# --------------------------------------------------------------------------- #

def snapshot_scores(task: Task, gts: list[dict], preds: list[dict], cls: int) -> pd.DataFrame:
    return pd.DataFrame({
        "image": [g["path"].name for g in gts],
        "label": [int((g["cls"] == task.positive).any()) for g in gts],
        "score": [float(p["conf"][p["cls"] == cls].max()) if (p["cls"] == cls).any() else 0.0 for p in preds],
    })


def sensitivity(task: Task, valid: pd.DataFrame) -> list[dict]:
    """How the cheapest single threshold moves if the two shakiest figures,
    the cost of a miss and the base rate, are wrong."""
    base_costs, _ = task.costs
    rows = []
    for fn in (25, 50, 100, 300, 1000, 3000):
        for base in (0.01, 0.05, 0.10, 0.20):
            costs = {**base_costs, "fn": fn}
            best = costing.best_single_threshold(valid["score"], valid["label"], costs, base)
            rows.append({"fn_cost": fn, "base_rate": base, "best_threshold": best["threshold"],
                         "cost_per_1000": best["cost_per_1000"]})
    return rows


def save_examples(task: Task, test: pd.DataFrame, low: float, high: float, k: int = 6) -> dict:
    """Copy the worst misses and false alarms on the test split for the dashboard."""
    out_dir = task.artifacts / "examples"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    picks = {
        "missed_violations": test[(test.label == 1) & (test.score < low)].nsmallest(k, "score"),
        "false_alarms": test[(test.label == 0) & (test.score >= high)].nlargest(k, "score"),
    }
    listing = {}
    for kind, df in picks.items():
        listing[kind] = []
        for row in df.itertuples():
            name = f"{kind}_{row.Index}.jpg"
            shutil.copy(task.data_dir / "test" / "images" / row.image, out_dir / name)
            listing[kind].append({"file": name, "score": row.score})
    return listing


def run_task(task: Task, skip_train: bool) -> None:
    print(f"\n{'=' * 30} {task.key} {'=' * 30}")
    data_yaml = download_dataset(task)
    task.artifacts.mkdir(parents=True, exist_ok=True)
    metrics_path = task.artifacts / "metrics.json"
    if skip_train and task.weights.exists():
        print(f"Skipping training for {task.key}: reusing {task.weights}")
        settings = json.loads(metrics_path.read_text())["training"] if metrics_path.exists() else {"base_model": BASE_MODEL, **TRAIN_ARGS}
    else:
        settings = train(task, data_yaml)

    costs, base_rate = task.costs
    finetuned = YOLO(str(task.weights))
    coco = YOLO(BASE_MODEL)
    # COCO has `person` but none of the PPE classes. Scoring its person boxes
    # as each dataset class in turn gives COCO the most generous reading.
    coco_map = {COCO_PERSON: list(task.names)}

    splits = {}
    for split in ("valid", "test"):
        gts = load_split(task, split)
        splits[split] = {
            "gts": gts,
            "finetuned": predict(finetuned, gts),
            "coco_zero_shot": predict(coco, gts, coco_map),
            "predict_nothing": no_predictions(gts),
        }
        print(f"Predicted {split}: {len(gts)} images")

    # Snapshot scores. COCO's best guess at a violation is "there is a person".
    scores = {
        split: {name: snapshot_scores(task, d["gts"], d[name], task.positive)
                for name in ("finetuned", "coco_zero_shot", "predict_nothing")}
        for split, d in splits.items()
    }

    # Thresholds are chosen on validation only; test is for reporting.
    valid_ft, test_ft = scores["valid"]["finetuned"], scores["test"]["finetuned"]
    single = costing.best_single_threshold(valid_ft["score"], valid_ft["label"], costs, base_rate)
    band = costing.best_band(valid_ft["score"], valid_ft["label"], costs, base_rate)
    op = single["threshold"]
    print(f"Chosen single threshold {op:.2f}, band [{band['low']:.2f}, {band['high']:.2f})")

    test = splits["test"]
    models = {}
    for name in ("finetuned", "coco_zero_shot", "predict_nothing"):
        s, v = scores["test"][name], scores["valid"][name]
        t_single = costing.best_single_threshold(v["score"], v["label"], costs, base_rate)["threshold"]
        models[name] = {
            **detection_metrics(task, test["gts"], test[name], op),
            "snapshot_alone": costing.evaluate(s["score"], s["label"], t_single, t_single, costs, base_rate),
        }

    test_ft.to_csv(task.artifacts / "scores_test.csv", index=False)
    valid_ft.to_csv(task.artifacts / "scores_valid.csv", index=False)
    for split, df in (("test", test_ft), ("valid", valid_ft)):
        pd.DataFrame([
            {"threshold": r["threshold"], "cost_per_1000": r["cost_per_1000"]}
            for r in costing.single_threshold_sweep(df["score"], df["label"], costs, base_rate)
        ]).to_csv(task.artifacts / f"cost_sweep_{split}.csv", index=False)

    metrics = {
        "dataset": {"workspace": task.workspace, "project": task.project, "version": task.version,
                    "url": f"https://universe.roboflow.com/{task.workspace}/{task.project}/dataset/{task.version}"},
        "training": settings,
        "class_names": task.names,
        "positive_class": task.names[task.positive],
        "test_snapshots": int(len(test_ft)),
        "test_positive_share": float(test_ft["label"].mean()),
        "valid_positive_share": float(valid_ft["label"].mean()),
        "costs": costs,
        "base_rate": base_rate,
        "human_miss": costing.HUMAN_MISS,
        "human_miss_check_all": costing.HUMAN_MISS_CHECK_ALL,
        "operating_point": {"single_threshold": op, "low": band["low"], "high": band["high"]},
        "models": models,
        "confusion_matrix_test": confusion_matrix(task, test["gts"], test["finetuned"], op),
        "positive_recall_by_size_test": recall_by_size(task, test["gts"], test["finetuned"], op),
        "policies_test": {
            **costing.alternatives(costs, base_rate),
            "model_alone": costing.evaluate(test_ft["score"], test_ft["label"], op, op, costs, base_rate),
            "model_plus_review": costing.evaluate(test_ft["score"], test_ft["label"], band["low"], band["high"],
                                                  costs, base_rate),
        },
        "sensitivity_valid": sensitivity(task, valid_ft),
        "examples": save_examples(task, test_ft, band["low"], band["high"]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: metrics[k] for k in ("operating_point", "policies_test")}, indent=2, default=float))
    for name, m in models.items():
        print(f"{task.key} {name:16s} mAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP50_95']:.3f}  "
              f"cost/1000 alone {m['snapshot_alone']['cost_per_1000']:.0f}")


def main() -> None:
    skip = {s.strip() for s in os.environ.get("SKIP_TRAIN", "").split(",") if s.strip()}
    for task in TASKS:
        run_task(task, skip_train=task.key in skip or "1" in skip)


if __name__ == "__main__":
    main()
