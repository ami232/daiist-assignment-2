"""
Assignment 2: training pipeline for garment quality control.

    uv run python main.py train               # everything, start to finish
    uv run python train.py --skip-training    # reuse weights/best.pt, redo evaluation + costing

Stages, in order, with no manual steps:
1. Download the Roboflow Universe dataset named in config.yaml (key via api_key.py).
2. Describe it: images/boxes per class per split, box sizes, defect-free images,
   and train/test leakage (near-identical images across splits).
3. Finetune yolo11n from the COCO checkpoint and copy the best weights to weights/.
4. Evaluate on valid and test: Ultralytics per-class P/R/AP50/mAP50-95, my own
   box confusion matrix, and the two baselines (COCO zero-shot, predict nothing).
5. Build defect-free negatives (crops that touch no labelled box), score them,
   sweep the image-level threshold, choose the cheapest single threshold and the
   cheapest pass/review/reject pair on valid, and report both on test.
6. Sensitivity analysis on every assumption in config.yaml.
Everything app.py needs is written to results/.
"""

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

import costing  # noqa: E402
from api_key import load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
WEIGHTS = ROOT / "weights" / "best.pt"
RUNS = ROOT / "runs"
SPLITS = ["train", "valid", "test"]


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=1, default=default))


def source_name(image: Path) -> str:
    """Roboflow appends `_jpg.rf.<hash>` to each augmented copy; strip it to get the source image."""
    return image.name.split(".rf.")[0]


# ---------------------------------------------------------------- 1. data


def download_dataset(cfg: dict) -> Path:
    from roboflow import Roboflow

    d = cfg["dataset"]
    target = ROOT / d["local_dir"]
    if (target / "data.yaml").exists():
        print(f"Dataset already at {target.relative_to(ROOT)}, not downloading again.")
        return target
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(d["workspace"]).project(d["project"]).version(d["version"])
    version.download(d["export_format"], location=str(target), overwrite=True)
    return target


def read_split(ds: Path, split: str) -> list[dict]:
    """Images of a split with their labelled boxes in pixel xyxy."""
    items = []
    for img in sorted((ds / split / "images").iterdir()):
        w, h = Image.open(img).size
        boxes = []
        label = ds / split / "labels" / f"{img.stem}.txt"
        if label.exists():
            for line in label.read_text().splitlines():
                v = line.split()
                if len(v) != 5:
                    continue
                c, xc, yc, bw, bh = int(v[0]), *map(float, v[1:])
                boxes.append([c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        items.append({"path": img, "w": w, "h": h, "gt": boxes})
    return items


# ---------------------------------------------------------------- 2. describe


def leakage(ds: Path) -> dict:
    """For each valid/test image, distance to its nearest train image (mean abs grey-level diff at 64x64)."""
    def vec(p):
        return np.asarray(Image.open(p).convert("L").resize((64, 64)), np.float32).reshape(-1)

    train = np.stack([vec(p) for p in sorted((ds / "train" / "images").iterdir())])
    out = {}
    for split in ["valid", "test"]:
        dists = np.array([np.abs(train - vec(p)).mean(axis=1).min()
                          for p in sorted((ds / split / "images").iterdir())])
        out[split] = {"images": len(dists), "identical_in_train": int((dists < 2).sum()),
                      "median_nearest_distance": float(np.median(dists))}
    return out


def describe_dataset(ds: Path, names: list[str], data: dict) -> dict:
    stats = {"classes": names, "splits": {}}
    for split in SPLITS:
        items = data[split]
        boxes = [b for it in items for b in it["gt"]]
        per_class = {}
        for c, name in enumerate(names):
            areas = np.array([(b[3] - b[1]) * (b[4] - b[2]) / (it["w"] * it["h"])
                              for it in items for b in it["gt"] if b[0] == c])
            per_class[name] = {
                "boxes": len(areas),
                "images": sum(any(b[0] == c for b in it["gt"]) for it in items),
                "area_pct_median": float(np.median(areas) * 100) if len(areas) else None,
                "area_pct_p10": float(np.percentile(areas, 10) * 100) if len(areas) else None,
                "area_pct_p90": float(np.percentile(areas, 90) * 100) if len(areas) else None,
            }
        stats["splits"][split] = {
            "images": len(items),
            "source_images": len({source_name(it["path"]) for it in items}),
            "boxes": len(boxes),
            "defect_free_images": sum(not it["gt"] for it in items),
            "boxes_per_image": len(boxes) / max(len(items), 1),
            "image_sizes": sorted({f"{it['w']}x{it['h']}" for it in items}),
            "per_class": per_class,
        }
    stats["leakage"] = leakage(ds)
    save_json(stats, RESULTS / "dataset_stats.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(names))
    for k, split in enumerate(SPLITS):
        counts = [stats["splits"][split]["per_class"][n]["boxes"] for n in names]
        axes[0].bar(x + (k - 1) * 0.27, counts, 0.27, label=split)
        for xi, cnt in zip(x + (k - 1) * 0.27, counts):
            axes[0].text(xi, cnt, str(cnt), ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("labelled boxes")
    axes[0].set_title("Boxes per class and split")
    axes[0].legend()
    areas = [[(b[3] - b[1]) * (b[4] - b[2]) / (it["w"] * it["h"]) * 100
              for split in SPLITS for it in data[split] for b in it["gt"] if b[0] == c] for c in range(len(names))]
    axes[1].boxplot(areas, tick_labels=names)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("box area, % of image (log)")
    axes[1].set_title("Object size relative to the image (all splits)")
    fig.tight_layout()
    fig.savefig(PLOTS / "dataset.png", dpi=120)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------- 3. finetune


def finetune(cfg: dict, data_yaml: Path) -> dict:
    from ultralytics import YOLO

    t = cfg["training"]
    started = time.time()
    model = YOLO(t["model"])
    model.train(data=str(data_yaml), epochs=t["epochs"], patience=t["patience"], imgsz=t["imgsz"],
                batch=t["batch"], device=t["device"], workers=t["workers"], seed=t["seed"],
                fraction=t.get("fraction", 1.0), deterministic=True, project=str(RUNS), name="finetune",
                exist_ok=True, plots=True)
    save_dir = Path(model.trainer.save_dir)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(save_dir / "weights" / "best.pt", WEIGHTS)
    shutil.copy(save_dir / "results.csv", RESULTS / "training_log.csv")
    if (save_dir / "results.png").exists():
        shutil.copy(save_dir / "results.png", PLOTS / "training_curves.png")

    import pandas as pd
    import torch

    log = pd.read_csv(save_dir / "results.csv")
    log.columns = [c.strip() for c in log.columns]
    best = log.loc[log["metrics/mAP50-95(B)"].idxmax()]
    info = {
        "settings": t,
        "where": f"{platform.node()} | {platform.platform()} | CPU, {os.cpu_count()} cores | "
                 f"torch {torch.__version__}",
        "minutes": round((time.time() - started) / 60, 1),
        "epochs_run": int(len(log)),
        "best_epoch": int(best["epoch"]),
        "best_val_mAP50": float(best["metrics/mAP50(B)"]),
        "best_val_mAP50_95": float(best["metrics/mAP50-95(B)"]),
    }
    save_json(info, RESULTS / "training.json")
    return info


# ---------------------------------------------------------------- 4. predict + evaluate


def predict(model, paths: list[Path], cfg: dict, imgsz: int) -> list[list]:
    ev = cfg["evaluation"]
    out = []
    for r in model.predict([str(p) for p in paths], imgsz=imgsz, conf=ev["conf_floor"], max_det=ev["max_det"],
                           device=cfg["training"]["device"], verbose=False, stream=True):
        b = r.boxes
        out.append([[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                    for c, s, xyxy in zip(b.cls, b.conf, b.xyxy)])
    return out


def negative_crops(items: list[dict], split: str, ds: Path, cfg: dict) -> list[Path]:
    """Cut windows that touch no labelled box: stand-ins for photos of good fabric."""
    ev = cfg["evaluation"]
    s, margin, per_image = ev["negative_crop_px"], ev["negative_crop_margin_px"], ev["negative_crops_per_image"]
    out_dir = ds / "negatives" / split
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    crops = []
    for it in items:
        gt = np.array([b[1:] for b in it["gt"]], dtype=float).reshape(-1, 4)
        gt[:, :2] -= margin
        gt[:, 2:] += margin
        cands = [(x, y) for x in range(0, it["w"] - s + 1, 32) for y in range(0, it["h"] - s + 1, 32)]
        cands = [(x, y) for x, y in cands
                 if not np.any((gt[:, 0] < x + s) & (gt[:, 2] > x) & (gt[:, 1] < y + s) & (gt[:, 3] > y))]
        chosen = []  # raster order packs the most non-overlapping windows
        for x, y in cands:
            if all(abs(x - cx) >= s or abs(y - cy) >= s for cx, cy in chosen):
                chosen.append((x, y))
            if len(chosen) == per_image:
                break
        if not chosen:
            continue
        img = Image.open(it["path"]).convert("RGB")
        for k, (x, y) in enumerate(chosen):
            path = out_dir / f"{it['path'].stem}_neg{k}.jpg"
            img.crop((x, y, x + s, y + s)).save(path, quality=95)
            crops.append(path)
    return crops


def ultralytics_metrics(weights: Path, data_yaml: Path, split: str, names: list[str], cfg: dict) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split="val" if split == "valid" else split, imgsz=cfg["training"]["imgsz"], batch=16,
                               conf=0.001, iou=0.7, device=cfg["training"]["device"], plots=True,
                               project=str(RUNS), name=f"val_{split}", exist_ok=True, verbose=False)
    box = m.box
    per_class = {n: {"precision": None, "recall": None, "ap50": None, "ap50_95": None} for n in names}
    for k, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {"precision": float(box.p[k]), "recall": float(box.r[k]),
                                    "ap50": float(box.ap50[k]), "ap50_95": float(box.ap[k])}
    src = Path(m.save_dir) / "confusion_matrix_normalized.png"
    if src.exists():
        shutil.copy(src, PLOTS / f"ultralytics_confusion_{split}.png")
    return {"precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
            "map50_95": float(box.map), "per_class": per_class}


def recall_by_size(images: list[dict], conf: float, iou_thr: float, names: list[str]) -> dict:
    """Which labelled defects are found (right class, IoU >= thr) by class and size bucket."""
    buckets = [("small (<0.5% of image)", 0, 0.005), ("medium (0.5-2%)", 0.005, 0.02), ("large (>2%)", 0.02, 1.1)]
    tally = {n: {b[0]: [0, 0] for b in buckets} for n in names}
    for im in images:
        gt = np.array(im["gt"], dtype=float).reshape(-1, 5)
        pr = np.array(im["pred"], dtype=float).reshape(-1, 6)
        pr = pr[pr[:, 1] >= conf]
        pairs, _, _ = costing.greedy_pairs(costing.box_iou(gt[:, 1:], pr[:, 2:]), iou_thr)
        found = {g for g, p in pairs if gt[g, 0] == pr[p, 0]}
        for g in range(len(gt)):
            area = (gt[g, 3] - gt[g, 1]) * (gt[g, 4] - gt[g, 2]) / (im["w"] * im["h"])
            label = next(b[0] for b in buckets if b[1] <= area < b[2])
            tally[names[int(gt[g, 0])]][label][0] += g in found
            tally[names[int(gt[g, 0])]][label][1] += 1
    return {n: {b: {"found": v[0], "total": v[1]} for b, v in d.items()} for n, d in tally.items()}


# ---------------------------------------------------------------- 5. costing


def choose_thresholds(pos, neg, cfg: dict) -> dict:
    th = cfg["thresholds"]
    sweep = costing.single_sweep(pos, neg, cfg)
    i = int(np.argmin(sweep["cost"]))
    grid = costing.two_threshold_grid(pos, neg, cfg)
    pair = costing.best_two_thresholds(grid, cfg)
    return {
        "single": th["single"] if th["single"] is not None else float(sweep["thresholds"][i]),
        "low": th["review_low"] if th["review_low"] is not None else pair["constrained"]["low"],
        "high": th["review_high"] if th["review_high"] is not None else pair["constrained"]["high"],
        "single_auto": float(sweep["thresholds"][i]),
        "pair_unconstrained": pair["unconstrained"],
        "pair_constrained": pair["constrained"],
        "forced_by_config": {k: th[k] for k in th if th[k] is not None},
    }


def bootstrap(pos, neg, cfg: dict, single: float, low: float, high: float) -> dict:
    rng = np.random.default_rng(cfg["training"]["seed"])
    keys = ["Model alone", "Model + human review", "Person checks every garment"]
    draws = {k: [] for k in keys}
    wins = 0
    for _ in range(cfg["evaluation"]["bootstrap_resamples"]):
        p = rng.choice(pos, len(pos))
        n = rng.choice(neg, len(neg))
        s = costing.strategy_costs(p, n, cfg, single, low, high)
        for k in keys:
            draws[k].append(s[k]["cost_per_1000"])
        wins += s["Model + human review"]["cost_per_1000"] < s["Person checks every garment"]["cost_per_1000"]
    out = {k: {"p2_5": float(np.percentile(v, 2.5)), "p97_5": float(np.percentile(v, 97.5))}
           for k, v in draws.items()}
    out["share_model_review_beats_manual"] = wins / cfg["evaluation"]["bootstrap_resamples"]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


SENSITIVITY = [
    ("business.base_rate", "Defect base rate", [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]),
    ("costs_eur.false_negative", "Cost of a missed defect (EUR)", [2, 5, 10, 15, 20, 30, 45, 60, 100]),
    ("costs_eur.false_positive", "Cost of a false reject (EUR)", [0.5, 1, 2, 3, 5, 8, 12, 20]),
    ("costs_eur.true_positive", "Cost of fixing a caught defect (EUR)", [0, 2, 5, 10, 15, 20, 25]),
    ("human.wage_eur_per_hour", "Reviewer wage (EUR/h)", [5, 8, 12, 16, 20, 30, 40]),
    ("human.review_minutes", "Minutes per review", [0.1, 0.25, 0.5, 1, 2, 3, 5]),
    ("human.review_recall", "Reviewer recall", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]),
    ("human.full_inspection_minutes", "Minutes per full manual inspection", [0.25, 0.5, 1, 1.5, 2, 3, 5]),
    ("evaluation.negative_area_factor", "False-alarm area factor", [1, 2, 4, 6.25, 9, 12]),
]


def with_value(cfg: dict, dotted: str, value) -> dict:
    new = json.loads(json.dumps(cfg))
    section, key = dotted.split(".")
    new[section][key] = value
    return new


def sensitivity(sel_pos, sel_neg, test_pos, test_neg, cfg: dict) -> dict:
    """Re-choose thresholds on valid and re-price every strategy on test for each assumption."""
    out = {}
    for dotted, label, values in SENSITIVITY:
        rows = []
        for v in values:
            c = with_value(cfg, dotted, v)
            ch = choose_thresholds(sel_pos, sel_neg, c)
            s = costing.strategy_costs(test_pos, test_neg, c, ch["single"], ch["low"], ch["high"])
            real = {k: s[k]["cost_per_1000"] for k in s if "reference" not in k}
            rows.append({"value": v, "single": ch["single"], "low": ch["low"], "high": ch["high"],
                         "costs": real, "best": min(real, key=real.get),
                         "reviewers": s["Model + human review"]["people"]})
        section, key = dotted.split(".")
        base = cfg[section][key]
        base_best = min((r for r in rows), key=lambda r: abs(r["value"] - base))["best"]
        switches = [f"{a['best']} -> {b['best']} between {a['value']} and {b['value']}"
                    for a, b in zip(rows, rows[1:]) if a["best"] != b["best"]]
        out[dotted] = {"label": label, "baseline": base, "baseline_best": base_best, "rows": rows,
                       "switches": switches}
    save_json(out, RESULTS / "sensitivity.json")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (dotted, d) in zip(axes.ravel(), out.items()):
        xs = [r["value"] for r in d["rows"]]
        for name in d["rows"][0]["costs"]:
            ax.plot(xs, [r["costs"][name] for r in d["rows"]], marker="o", ms=3, label=name)
        ax.axvline(d["baseline"], color="k", ls=":", lw=1)
        ax.set_title(d["label"], fontsize=10)
        ax.set_ylabel("EUR per 1,000 garments")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Sensitivity: cost per 1,000 garments of each strategy (dotted line = my assumption)")
    fig.tight_layout()
    fig.savefig(PLOTS / "sensitivity.png", dpi=110)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- plots


def plot_cost_curves(curves: dict, chosen: dict, cfg: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for split, sw in curves.items():
        axes[0].plot(sw["thresholds"], 1000 * np.asarray(sw["cost"]), label=f"model alone, {split}")
    axes[0].axvline(chosen["single"], color="red", ls="--", label=f"chosen single threshold {chosen['single']:.2f}")
    axes[0].axvspan(chosen["low"], chosen["high"], color="orange", alpha=0.2,
                    label=f"review band {chosen['low']:.2f}-{chosen['high']:.2f}")
    p, c = cfg["business"]["base_rate"], cfg["costs_eur"]
    axes[0].axhline(1000 * p * c["false_negative"], color="grey", ls=":", label="do nothing")
    axes[0].set_xlabel("confidence threshold (reject if max box confidence >= t)")
    axes[0].set_ylabel("expected EUR per 1,000 garments")
    axes[0].set_title(f"Cost against threshold (base rate {p:.0%})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for split, sw in curves.items():
        axes[1].plot(sw["thresholds"], sw["tpr"], label=f"defects caught (recall), {split}")
        axes[1].plot(sw["thresholds"], sw["fpr"], ls="--", label=f"good garments rejected, {split}")
    axes[1].axvline(chosen["single"], color="red", ls="--")
    axes[1].set_xlabel("confidence threshold")
    axes[1].set_ylabel("rate")
    axes[1].set_title("What the threshold trades")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "cost_vs_threshold.png", dpi=120)
    plt.close(fig)


def plot_scores(pos, neg, cfg: dict, chosen: dict) -> None:
    af = cfg["evaluation"]["negative_area_factor"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(pos, bins, alpha=0.6, label=f"defective photos (n={len(pos)})")
    ax.hist(neg, bins, alpha=0.6, label=f"defect-free crops (n={len(neg)}, x{af} area per photo)")
    for t, name in [(chosen["low"], "pass | review"), (chosen["high"], "review | reject")]:
        ax.axvline(t, color="k", ls="--")
        ax.text(t, ax.get_ylim()[1] * 0.9, f" {name}", fontsize=8)
    ax.set_xlabel("highest box confidence in the image")
    ax.set_ylabel("images")
    ax.set_title("Test-split score distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "score_distributions.png", dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, names: list[str], conf: float, split: str) -> None:
    labels = names + ["background"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Box confusion matrix, {split}, conf >= {conf:.2f}, IoU >= 0.5")
    fig.tight_layout()
    fig.savefig(PLOTS / f"confusion_{split}.png", dpi=120)
    plt.close(fig)


def plot_strategies(strategies: dict) -> None:
    names = list(strategies)
    costs = [strategies[n]["cost_per_1000"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, costs, color=["grey", "tab:blue", "tab:orange", "tab:green", "lightgrey"])
    ax.bar_label(bars, fmt="EUR %.0f", padding=3)
    ax.invert_yaxis()
    ax.set_xlabel("expected EUR per 1,000 garments (test split, my base rate)")
    ax.set_title("Four ways to run inspection")
    fig.tight_layout()
    fig.savefig(PLOTS / "strategies.png", dpi=120)
    plt.close(fig)


def plot_failures(images: list[dict], names: list[str], conf: float) -> None:
    """Test images where the model's best box is weakest: where it misses defects."""
    worst = sorted(images, key=lambda im: max((p[1] for p in im["pred"]), default=0))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, im in zip(axes.ravel(), worst):
        ax.imshow(Image.open(ROOT / im["path"]))
        for c, *xyxy in im["gt"]:
            ax.add_patch(plt.Rectangle(xyxy[:2], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1], fill=False, color="lime", lw=2))
            ax.text(xyxy[0], xyxy[1] - 4, names[int(c)], color="lime", fontsize=8)
        for c, s, *xyxy in im["pred"]:
            if s >= conf:
                ax.add_patch(plt.Rectangle(xyxy[:2], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1], fill=False,
                                           color="red", lw=1.5))
                ax.text(xyxy[0], xyxy[3] + 12, f"{names[int(c)]} {s:.2f}", color="red", fontsize=8)
        top = max((p[1] for p in im["pred"]), default=0)
        ax.set_title(f"max confidence {top:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Hardest test images. Green = labelled defect, red = model box (conf >= {conf:.2f})")
    fig.tight_layout()
    fig.savefig(PLOTS / "failures.png", dpi=100)
    plt.close(fig)


def plot_per_class(metrics: dict, names: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(names))
    for ax, split in zip(axes, ["valid", "test"]):
        pc = metrics[split]["per_class"]
        for k, key in enumerate(["precision", "recall", "ap50"]):
            vals = [pc[n][key] if pc[n][key] is not None else 0 for n in names]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=key)
        for xi, n in zip(x, names):
            if pc[n]["ap50"] is None:
                ax.text(xi, 0.02, "no test\ninstances", ha="center", fontsize=8)
        ax.set_xticks(x, names)
        ax.set_title(f"Per-class, {split} split (mAP50 {metrics[split]['map50']:.2f})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS / "per_class.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main


def evaluate_and_cost(cfg: dict, ds: Path, data_yaml: Path, names: list[str], data: dict) -> None:
    from ultralytics import YOLO

    ev = cfg["evaluation"]
    imgsz = cfg["training"]["imgsz"]
    finetuned, coco = YOLO(str(WEIGHTS)), YOLO(cfg["training"]["model"])

    images, crops_scores, zs = {}, {}, {}
    for split in ["valid", "test"]:
        items = data[split]
        preds = predict(finetuned, [it["path"] for it in items], cfg, imgsz)
        images[split] = [{"path": str(it["path"].relative_to(ROOT)), "w": it["w"], "h": it["h"],
                          "gt": [[b[0], *[round(v, 1) for v in b[1:]]] for b in it["gt"]], "pred": p}
                         for it, p in zip(items, preds)]
        crops = negative_crops(items, split, ds, cfg)
        crops_scores[split] = costing.image_scores(
            [{"pred": p} for p in predict(finetuned, crops, cfg, ev["negative_crop_px"])])
        zs_preds = predict(coco, [it["path"] for it in items], cfg, imgsz)
        zs_images = [{"gt": im["gt"], "pred": p} for im, p in zip(images[split], zs_preds)]
        zs[split] = {
            "agnostic_ap50": costing.agnostic_ap50(zs_images, ev["iou_match"]),
            "pos": costing.image_scores([{"pred": p} for p in zs_preds]),
            "neg": costing.image_scores(
                [{"pred": p} for p in predict(coco, crops, cfg, ev["negative_crop_px"])]),
            "boxes_drawn": sum(len(p) for p in zs_preds),
            "top_coco_classes": sorted({coco.names[c] for p in zs_preds for c, *_ in p})[:10],
        }
        save_json({"images": images[split], "negative_crop_scores": crops_scores[split]},
                  RESULTS / f"predictions_{split}.json")

    pos = {s: costing.image_scores([im for im in images[s] if im["gt"]]) for s in images}
    neg = crops_scores
    sel = ev["select_on_split"]
    chosen = choose_thresholds(pos[sel], neg[sel], cfg)
    single, low, high = chosen["single"], chosen["low"], chosen["high"]

    # Ultralytics per-class metrics (its own P/R are at the confidence that maximises F1)
    metrics = {s: ultralytics_metrics(WEIGHTS, data_yaml, s, names, cfg) for s in ["valid", "test"]}
    for s in ["valid", "test"]:
        cm = costing.class_confusion(images[s], single, ev["iou_match"], len(names))
        metrics[s]["confusion_at_single"] = cm
        metrics[s]["per_class_at_single"] = dict(zip(names, costing.per_class_pr(cm)))
        metrics[s]["recall_by_size_at_single"] = recall_by_size(images[s], single, ev["iou_match"], names)
        metrics[s]["agnostic_ap50"] = costing.agnostic_ap50(images[s], ev["iou_match"])
        plot_confusion(cm, names, single, s)
    plot_per_class(metrics, names)

    # Baselines on the same test split
    p = cfg["business"]["base_rate"]
    baselines = {}
    zs_sel = costing.single_sweep(zs[sel]["pos"], zs[sel]["neg"], cfg)
    zs_t = float(zs_sel["thresholds"][int(np.argmin(zs_sel["cost"]))])  # its own best threshold, chosen like mine
    for s in ["valid", "test"]:
        zs_sweep = costing.single_sweep(zs[s]["pos"], zs[s]["neg"], cfg)
        zs_rates = costing.outcome_rates(zs[s]["pos"], zs[s]["neg"], zs_t, zs_t, ev["negative_area_factor"])
        ft_rates = costing.outcome_rates(pos[s], neg[s], single, single, ev["negative_area_factor"])
        n_pos = len(pos[s])
        baselines[s] = {
            "Finetuned yolo11n": {
                "map50": metrics[s]["map50"], "map50_95": metrics[s]["map50_95"],
                "agnostic_ap50": metrics[s]["agnostic_ap50"], "threshold": single,
                "image_recall": ft_rates["pos"]["reject"], "garment_false_reject_rate": ft_rates["neg"]["reject"],
                "image_accuracy_on_split": ft_rates["pos"]["reject"],
                "cost_per_1000": 1000 * float(costing.policy_cost(
                    ft_rates["pos"]["pass"], 0, ft_rates["pos"]["reject"], ft_rates["neg"]["pass"], 0,
                    ft_rates["neg"]["reject"], cfg))},
            "COCO yolo11n zero-shot": {
                "map50": 0.0, "map50_95": 0.0, "agnostic_ap50": zs[s]["agnostic_ap50"], "threshold": zs_t,
                "image_recall": zs_rates["pos"]["reject"], "garment_false_reject_rate": zs_rates["neg"]["reject"],
                "image_accuracy_on_split": zs_rates["pos"]["reject"],
                "cost_per_1000": 1000 * float(zs_sweep["cost"][int(np.argmin(np.abs(zs_sweep["thresholds"] - zs_t)))]),
                "boxes_drawn": zs[s]["boxes_drawn"], "coco_classes_seen": zs[s]["top_coco_classes"]},
            "Trivial: predict nothing (always pass)": {
                "map50": 0.0, "map50_95": 0.0, "agnostic_ap50": 0.0, "threshold": None, "image_recall": 0.0,
                "garment_false_reject_rate": 0.0, "image_accuracy_on_split": 0.0,
                "cost_per_1000": 1000 * (p * cfg["costs_eur"]["false_negative"]
                                         + (1 - p) * cfg["costs_eur"]["true_negative"])},
            "Trivial: majority class on this split (always reject)": {
                "map50": 0.0, "map50_95": 0.0, "agnostic_ap50": 0.0, "threshold": None, "image_recall": 1.0,
                "garment_false_reject_rate": 1.0, "image_accuracy_on_split": 1.0,
                "cost_per_1000": 1000 * (p * cfg["costs_eur"]["true_positive"]
                                         + (1 - p) * cfg["costs_eur"]["false_positive"])},
        }
        baselines[s]["n_defective_images"] = n_pos
    metrics["baselines"] = baselines
    save_json(metrics, RESULTS / "metrics.json")

    # Costing on test at the thresholds chosen on valid
    curves = {s: costing.single_sweep(pos[s], neg[s], cfg) for s in ["valid", "test"]}
    plot_cost_curves(curves, chosen, cfg)
    plot_scores(pos["test"], neg["test"], cfg, chosen)
    plot_failures(images["test"], names, single)
    strategies = costing.strategy_costs(pos["test"], neg["test"], cfg, single, low, high)
    plot_strategies(strategies)
    grid = costing.two_threshold_grid(pos["test"], neg["test"], cfg)
    rates = costing.outcome_rates(pos["test"], neg["test"], low, high, ev["negative_area_factor"])
    share = costing.review_share(rates, p)
    k_single = int((pos["test"] >= single).sum())
    q_single = int((neg["test"] >= single).sum())
    test_best = int(np.argmin(curves["test"]["cost"]))
    operating = {
        "chosen": chosen,
        "selected_on": sel,
        "test_optimal_single": float(curves["test"]["thresholds"][test_best]),
        "test_optimal_pair": costing.best_two_thresholds(grid, cfg),
        "test": {
            "n_defective_images": len(pos["test"]), "n_negative_crops": len(neg["test"]),
            "recall_at_single": k_single / len(pos["test"]),
            "recall_at_single_ci95": wilson(k_single, len(pos["test"])),
            "crop_false_alarm_at_single": q_single / len(neg["test"]),
            "crop_false_alarm_at_single_ci95": wilson(q_single, len(neg["test"])),
            "garment_false_reject_at_single": float(curves["test"]["fpr"][int(np.argmin(np.abs(
                curves["test"]["thresholds"] - single)))]),
            "rates_two_thresholds": rates,
            "review_share": share,
            "review_per_day": share * cfg["business"]["garments_per_day"],
            "review_minutes_per_day": share * cfg["business"]["garments_per_day"] * cfg["human"]["review_minutes"],
            "reviewers_needed": costing.reviewers_needed(share, cfg),
            "confusion_per_1000": costing.expected_confusion(rates, p),
            "raw_counts": {
                "defective": {k: int(round(v * len(pos["test"]))) for k, v in rates["pos"].items()},
                "crops": {"pass": int((neg["test"] < low).sum()),
                          "review": int(((neg["test"] >= low) & (neg["test"] < high)).sum()),
                          "reject": int((neg["test"] >= high).sum())},
            },
            "strategies": strategies,
            "bootstrap_ci95": bootstrap(pos["test"], neg["test"], cfg, single, low, high),
        },
        "curves": {s: {k: np.asarray(v) for k, v in c.items()} for s, c in curves.items()},
        "zero_shot_scores": {s: {"pos": zs[s]["pos"], "neg": zs[s]["neg"]} for s in zs},
        "config_used": cfg,
    }
    save_json(operating, RESULTS / "operating_point.json")
    sensitivity(pos[sel], neg[sel], pos["test"], neg["test"], cfg)

    print("\n=== Summary (test split) ===")
    print(f"mAP50 {metrics['test']['map50']:.3f}  mAP50-95 {metrics['test']['map50_95']:.3f}")
    print(f"Thresholds (chosen on {sel}): single {single:.2f}, pass<{low:.2f}<=review<{high:.2f}<=reject")
    for k, v in strategies.items():
        print(f"  {k:32s} EUR {v['cost_per_1000']:8.1f} per 1,000   review share {v['review_share']:.1%}")
    print(f"Review: {operating['test']['review_per_day']:.0f} garments/day, "
          f"{operating['test']['reviewers_needed']:.2f} reviewers")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true", help="reuse weights/best.pt")
    parser.add_argument("--config", default=str(costing.CONFIG_PATH))
    args = parser.parse_args()

    cfg = costing.load_config(Path(args.config))
    PLOTS.mkdir(parents=True, exist_ok=True)
    ds = download_dataset(cfg)
    data_yaml = ds / "data.yaml"
    names = yaml.safe_load(data_yaml.read_text())["names"]
    data = {s: read_split(ds, s) for s in SPLITS}
    stats = describe_dataset(ds, names, data)
    for s in SPLITS:
        st = stats["splits"][s]
        print(f"{s:5s}: {st['images']} images, {st['boxes']} boxes, {st['defect_free_images']} defect-free, "
              + ", ".join(f"{n} {st['per_class'][n]['boxes']}" for n in names))

    if args.skip_training:
        if not WEIGHTS.exists():
            raise SystemExit(f"--skip-training needs {WEIGHTS.relative_to(ROOT)}; run without it first.")
    else:
        info = finetune(cfg, data_yaml)
        print(f"Trained {info['epochs_run']} epochs in {info['minutes']} min; best epoch {info['best_epoch']}.")

    evaluate_and_cost(cfg, ds, data_yaml, names, data)
    print(f"\nDone. Weights: {WEIGHTS.relative_to(ROOT)}   Results: {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
