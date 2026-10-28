"""
Assignment 2 training pipeline: early fire & smoke detection from fixed cameras.

Run with `uv run python main.py train`. Stages, top to bottom, no manual steps:

  1. setup      load the Roboflow key, pick a device, seed everything
  2. download   the pinned Roboflow Universe version into datasets/
     group      GROUPED=1 (default): local rebuild with a source-grouped re-split and
                same-source negative crops -> datasets/<version>-grouped/
  3. describe   split sizes, class balance, box sizes  -> weights/dataset_stats.json
  4. finetune   MODEL on train/val                     -> weights/best.pt, run_meta.json
  5. evaluate   test-split detection metrics, scores   -> weights/eval.json, test_scores.csv
  6. baselines  COCO zero-shot and trivial policies    -> weights/baselines.json
  7. decide     cost sweep, review band, sensitivity   -> weights/decision.json
  8. report     refresh the AUTO blocks in REPORT.md

The unit of decision is one camera frame. Its score is the highest confidence
of any fire or smoke box (0.0 if there is none). A frame is positive if it has
at least one ground-truth fire or smoke box.

Set EVAL_ONLY=1 to skip training (weights/best.pt must exist) and redo
everything downstream, e.g. after changing DECISIONS.

Set GROUPED=0 to train on Roboflow's original split instead (its train and test
share source photos, and each source family has a single label; see section 7b).

Set RESUME=1 to continue an interrupted training run from its last finished
epoch (runs/finetune/weights/last.pt, saved by Ultralytics after every epoch)
instead of starting again from epoch 1. Everything downstream then runs as usual.
"""

import json
import os
import platform
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from api_key import load_api_key

# =====================================================================
# >>> STUDENT DECISIONS — replace every value with your own, justified
#     in REPORT.md §6–7. Placeholders only; the code never hardcodes
#     these anywhere else.
# =====================================================================
DECISIONS = {
    # Costs in EUR per frame outcome (one frame = one minute of one camera)
    "cost_fn": 50_000.0,     # one more minute of pre-flashover growth: (~€250k full loss - ~€10k contained) / ~5 min
    "cost_fp": 500.0,        # alarm protocol run for nothing: brigade turn-out fee, guard call-out, keyholder, disruption
    "cost_tp": 500.0,        # the same response is paid when the fire is real
    "cost_tn": 0.0,
    # Production base rate: share of frames showing an unhandled fire (deliberately high, see REPORT.md)
    "base_rate": 0.0001,
    # Human review
    "review_seconds": 20,    # look at one still, compare with the previous minute, decide
    "reviewer_cost_per_hour": 25.0,  # loaded cost of an alarm-centre operator
    "human_recall": 0.95,    # P(operator escalates | real fire): faint or distant smoke is still missed
    "human_fpr": 0.02,       # P(operator escalates | no fire): steam/dust/haze escalated to be safe
    # Volume and staffing
    "n_cameras": 200,
    "frames_per_camera_per_day": 1440,   # one still per minute
    "productive_hours_per_fte_day": 6.5, # 8 paid hours minus breaks, handovers, other alarms
    "paid_hours_per_fte_day": 8.0,
    "min_on_duty_fte_24_7": 4.2,         # one seat 24/7: 8,760 h / ~2,080 h per FTE-year
    "available_reviewer_fte": 5.0,       # the centre's current operator headcount for this service
}
# =====================================================================

# --- Dataset (Roboflow Universe, public, downloaded as-is; never forked) ---
# Primary choice was D-Fire (uweeaai/d-fire-ehbj2), but on Universe it has no
# generated version, so nothing is downloadable. This is the fallback, pinned
# to its latest public version (v5: yolov11 export, test split with backgrounds).
RF_WORKSPACE = "dangerzone-6iqvs"
RF_PROJECT = "smoke_fire-idefg"
RF_VERSION = 5
RF_FORMAT = "yolov11"


# --- Training settings, overridable by environment variables ---
def _env(name, default, cast):
    value = os.environ.get(name, "").strip()
    return cast(value) if value else default


MODEL = _env("MODEL", "yolo11n.pt", str)
EPOCHS = _env("EPOCHS", 30, int)
IMGSZ = _env("IMGSZ", 640, int)
PATIENCE = _env("PATIENCE", 10, int)
BATCH = _env("BATCH", 16, int)
FRACTION = _env("FRACTION", 1.0, float)
SEED = _env("SEED", 0, int)
WORKERS = _env("WORKERS", 8, int)  # dataloader processes; lower it if RAM/swap runs out
GROUPED = os.environ.get("GROUPED", "1").strip() == "1"  # group split + same-source negatives (see prepare_grouped)
EVAL_ONLY = os.environ.get("EVAL_ONLY", "0").strip() == "1"
RESUME = os.environ.get("RESUME", "0").strip() == "1"

# --- Fixed technical settings (not business decisions) ---
PRED_CONF = 0.001        # keep (almost) every box so the full score distribution is swept
NMS_IOU = 0.7            # Ultralytics' default NMS IoU for val/predict
CM_CONF = 0.25           # Ultralytics' standard conf for the detection confusion matrix
COCO_LOG_CONF = 0.25     # conf used to log what the COCO model "sees" on test frames
LOW_COUNT = 100          # fewer test instances than this -> per-class numbers flagged
SMALL_BOX = 0.01         # box area under 1% of the image counts as "small"
SPLIT_SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}  # of source images, for GROUPED=1
NEG_TAG = "_negcrop"     # filename marker for same-source negative crops
NEG_MARGIN = 0.05        # keep crops this far (share of the image) from any box, since smoke spills past its box
NEG_MIN_SIDE = 0.3       # a crop must cover at least this share of both image sides
NEG_MAX_BLACK = 0.3      # reject crops that are mostly black rotation padding
N_SWEEP = 201            # threshold grid points for the model-alone sweep (plus every unique score)
N_POLICY_GRID = 101      # lo/hi grid points for the three-way search (step 0.01)
N_BOOT = 1000            # stratified bootstrap resamples
EPS = 1e-12

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
DATA_ROOT = ROOT / "datasets"
RUNS = ROOT / "runs"
LAST_CKPT = RUNS / "finetune" / "weights" / "last.pt"  # written by Ultralytics after every epoch
REPORT = ROOT / "REPORT.md"
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# One palette shared with app.py
PASS_C, REVIEW_C, ALARM_C = "#2F7F72", "#C98A0B", "#D2491B"
ACCENT, INK, MUTED = "#4F46E5", "#1E293B", "#94A3B8"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(stage, message):
    print(f"[{datetime.now():%H:%M:%S}] {stage:<10} {message}", flush=True)


def save_json(name, payload):
    path = WEIGHTS / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    log("save", f"{path.relative_to(ROOT)}")


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def fmt_eur(x):
    return f"€{x:,.2f}"


def derived(D):
    """Quantities derived from DECISIONS, never inputs themselves."""
    return {
        "cost_review": D["review_seconds"] / 3600 * D["reviewer_cost_per_hour"],
        "frames_per_day": D["n_cameras"] * D["frames_per_camera_per_day"],
    }


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
def cpu_name():
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def setup():
    import torch
    import ultralytics

    load_api_key()  # puts the key in os.environ for the roboflow SDK; never printed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        device, device_name = "0", torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device, device_name = "mps", f"Apple MPS ({cpu_name()})"
    else:
        device, device_name = "cpu", cpu_name()
    env = {
        "device": device,
        "device_name": device_name,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    WEIGHTS.mkdir(exist_ok=True)
    log("setup", f"device={device} ({device_name}), torch {env['torch']}, "
                 f"ultralytics {env['ultralytics']}, eval_only={EVAL_ONLY}, resume={RESUME}")
    return env


# ---------------------------------------------------------------------------
# 2. Download
# ---------------------------------------------------------------------------
def dataset_dir():
    return DATA_ROOT / f"{RF_PROJECT}-{RF_VERSION}"


def download_dataset():
    target = dataset_dir()
    if (target / "data.yaml").is_file() and (target / "test" / "images").is_dir():
        log("download", f"already present at {target.relative_to(ROOT)}, skipping")
        return target
    from roboflow import Roboflow

    log("download", f"{RF_WORKSPACE}/{RF_PROJECT} v{RF_VERSION} as {RF_FORMAT}")
    DATA_ROOT.mkdir(exist_ok=True)
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION)
    version.download(RF_FORMAT, location=str(target), overwrite=True)
    return target


def dataset_config(ds):
    """Read class names and write a data YAML with absolute split paths."""
    raw = yaml.safe_load((ds / "data.yaml").read_text())
    names = raw["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    data_yaml = ds / "data_abs.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(ds),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }, sort_keys=False))
    licence = None
    for readme in ("README.roboflow.txt", "README.dataset.txt"):
        path = ds / readme
        if path.is_file():
            match = re.search(r"License:\s*(.+)", path.read_text(errors="replace"))
            if match:
                licence = match.group(1).strip()
                break
    source = {
        "workspace": RF_WORKSPACE,
        "project": RF_PROJECT,
        "version": RF_VERSION,
        "format": RF_FORMAT,
        "url": f"https://universe.roboflow.com/{RF_WORKSPACE}/{RF_PROJECT}/dataset/{RF_VERSION}",
        "licence": licence or "see Roboflow Universe page",
    }
    return names, data_yaml, source


def prepare_grouped(ds):
    """Local, deterministic rebuild of the downloaded version (the Roboflow version is untouched):
    1. Group split: Roboflow's split puts augmented copies of one source photo in train and test.
       Re-split by source image (SPLIT_SHARES), so no source is seen in training and scored in test.
    2. Same-source negatives: for each positive frame, crop its largest region with no box
       (margin NEG_MARGIN) and add it as a background frame in the same split. Every source family
       then has both labels, so recognising the source no longer predicts the label."""
    from PIL import Image

    out = DATA_ROOT / f"{ds.name}-grouped"
    if (out / "grouping.json").is_file():
        log("group", f"already built at {out.relative_to(ROOT)}, skipping")
        return out, json.loads((out / "grouping.json").read_text())
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in ("data.yaml", "README.roboflow.txt", "README.dataset.txt"):
        if (ds / name).is_file():
            shutil.copy2(ds / name, tmp / name)

    # Group every copy by source image, then assign whole sources to splits, stratified by
    # (family, label) so each split keeps the same mix. Val/test keep one copy per source,
    # preferring one Roboflow did not augment (originally in test, then val).
    groups = {}
    for split in ("test", "val", "train"):
        for image in list_images(ds, split):
            groups.setdefault(source_base(image.name), []).append(image)
    strata = {}
    for base, copies in groups.items():
        key = (source_family(copies[0].name), int(bool(read_boxes(copies[0]))))
        strata.setdefault(key, []).append(base)
    rng = random.Random(SEED)
    targets = {split: [] for split in SPLIT_DIRS}
    for key in sorted(strata):
        bases = sorted(strata[key])
        rng.shuffle(bases)
        n_train = round(len(bases) * SPLIT_SHARES["train"])
        n_val = round(len(bases) * SPLIT_SHARES["val"])
        for i, base in enumerate(bases):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            targets[split] += groups[base] if split == "train" else groups[base][:1]
    n_frames = sum(len(c) for c in groups.values())

    summary = {"sources": len(groups), "frames_in_download": n_frames, "kept": {}, "negative_crops": {},
               "negative_crops_by_family": {}}
    for split, split_images in targets.items():
        img_dir, lab_dir = tmp / SPLIT_DIRS[split] / "images", tmp / SPLIT_DIRS[split] / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        kept, crops, by_family = 0, 0, Counter()
        for image in sorted(split_images, key=lambda p: p.name):
            kept += 1
            os.link(image, img_dir / image.name)
            lab = label_path(image)
            (lab_dir / lab.name).write_text(lab.read_text() if lab.is_file() else "")
            boxes = read_boxes(image)
            if not boxes:
                continue
            crop = negative_crop(image, boxes, Image)
            if crop is None:
                continue
            stem, _, suffix = image.name.partition(".rf.")
            name = f"{stem}{NEG_TAG}.rf.{suffix}" if suffix else f"{image.stem}{NEG_TAG}.jpg"
            crop.save(img_dir / name, quality=95)
            (lab_dir / f"{Path(name).stem}.txt").write_text("")
            crops += 1
            by_family[source_family(image.name)] += 1
        summary["kept"][split] = kept
        summary["negative_crops"][split] = crops
        summary["negative_crops_by_family"][split] = dict(by_family)
        log("group", f"{split:<5} {kept:>6} frames, {crops:>5} same-source negative crops added")
    summary["dropped_copies"] = n_frames - sum(summary["kept"].values())
    shares = "/".join(f"{v:.0%}" for v in SPLIT_SHARES.values())
    summary["rule"] = (f"source images (filename before '.rf.') re-split {shares} train/val/test, stratified "
                       f"by source family and label, one copy per source in val and test; negative crop = "
                       f"largest box-free strip at least {NEG_MIN_SIDE:.0%} of each side, {NEG_MARGIN:.0%} "
                       f"margin around every box, rejected if over {NEG_MAX_BLACK:.0%} black padding")
    (tmp / "grouping.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out, summary


def negative_crop(image_path, boxes, Image):
    """Largest left/right/top/bottom strip that stays NEG_MARGIN away from every box, or None."""
    x0 = max(min(b[1] for b in boxes) - NEG_MARGIN, 0)
    y0 = max(min(b[2] for b in boxes) - NEG_MARGIN, 0)
    x1 = min(max(b[3] for b in boxes) + NEG_MARGIN, 1)
    y1 = min(max(b[4] for b in boxes) + NEG_MARGIN, 1)
    strips = [(0, 0, x0, 1), (x1, 0, 1, 1), (0, 0, 1, y0), (0, y1, 1, 1)]
    strips = [s for s in strips if s[2] - s[0] >= NEG_MIN_SIDE and s[3] - s[1] >= NEG_MIN_SIDE]
    if not strips:
        return None
    sx0, sy0, sx1, sy1 = max(strips, key=lambda s: (s[2] - s[0]) * (s[3] - s[1]))
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        crop = im.crop((round(sx0 * w), round(sy0 * h), round(sx1 * w), round(sy1 * h)))
    if (np.asarray(crop).max(axis=2) < 12).mean() > NEG_MAX_BLACK:  # rotation padding, not a scene
        return None
    return crop


def class_roles(names):
    """Which class ids count as fire and as smoke; anything else is excluded from the score."""
    fire = [i for i, n in enumerate(names) if re.search(r"fire|flame", n, re.I)]
    smoke = [i for i, n in enumerate(names) if re.search(r"smoke", n, re.I)]
    other = [i for i in range(len(names)) if i not in fire and i not in smoke]
    return {"fire": fire, "smoke": smoke, "other": other}


# ---------------------------------------------------------------------------
# 3. Describe the dataset (from the YOLO label files)
# ---------------------------------------------------------------------------
def list_images(ds, split):
    folder = ds / SPLIT_DIRS[split] / "images"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def label_path(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_boxes(image_path):
    """Return [(class_id, x0, y0, x1, y1)] in relative coordinates. Polygons become their bbox."""
    path = label_path(image_path)
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, nums = int(float(values[0])), [float(v) for v in values[1:]]
        if len(nums) == 4:
            x, y, w, h = nums
            rows.append((cls, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        else:
            xs, ys = nums[0::2], nums[1::2]
            rows.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return rows


def read_labels(image_path):
    """Return [(class_id, relative_box_area)] for one image."""
    return [(c, (x1 - x0) * (y1 - y0)) for c, x0, y0, x1, y1 in read_boxes(image_path)]


def scan_labels(ds, roles):
    """One row per image and one row per box, across all splits."""
    frames, boxes = [], []
    for split in SPLIT_DIRS:
        for image in list_images(ds, split):
            labels = read_labels(image)
            classes = {c for c, _ in labels}
            frames.append({
                "split": split,
                "image_id": image.name,
                "path": str(image),
                "n_boxes": len(labels),
                "y_fire": int(bool(classes & set(roles["fire"]))),
                "y_smoke": int(bool(classes & set(roles["smoke"]))),
                "y_other": int(bool(classes & set(roles["other"]))),
            })
            boxes.extend({"split": split, "cls": c, "area": a} for c, a in labels)
    frames = pd.DataFrame(frames)
    frames["y_true"] = (frames.y_fire | frames.y_smoke).astype(int)
    return frames, pd.DataFrame(boxes, columns=["split", "cls", "area"])


def describe_dataset(frames, boxes, names, roles, source):
    splits = {}
    for split in SPLIT_DIRS:
        f = frames[frames.split == split]
        b = boxes[boxes.split == split]
        n = len(f)
        counts = b.cls.value_counts()
        splits[split] = {
            "images": n,
            "background_images": int((f.n_boxes == 0).sum()),
            "background_pct": r6(100 * (f.n_boxes == 0).mean()) if n else None,
            "instances": {names[c]: int(counts.get(c, 0)) for c in range(len(names))},
            "frames": {
                "fire_only": int(((f.y_fire == 1) & (f.y_smoke == 0)).sum()),
                "smoke_only": int(((f.y_fire == 0) & (f.y_smoke == 1)).sum()),
                "both": int(((f.y_fire == 1) & (f.y_smoke == 1)).sum()),
                "neither": int((f.y_true == 0).sum()),
            },
            "positive_prevalence": r6(f.y_true.mean()) if n else None,
        }
    box_area = {}
    for c, name in enumerate(names):
        for scope, sub in (("all", boxes), ("test", boxes[boxes.split == "test"])):
            a = sub[sub.cls == c].area.to_numpy()
            entry = {"n": int(a.size)}
            if a.size:
                q1, med, q3 = np.percentile(a, [25, 50, 75])
                entry.update(median=r6(med), q1=r6(q1), q3=r6(q3), iqr=r6(q3 - q1),
                             small_share=r6((a < SMALL_BOX).mean()))
            box_area.setdefault(name, {})[scope] = entry
    test_inst = splits["test"]["instances"]
    stats = {
        "source": source,
        "class_names": names,
        "positive_classes": {"fire": [names[i] for i in roles["fire"]],
                             "smoke": [names[i] for i in roles["smoke"]]},
        "excluded_from_score": [names[i] for i in roles["other"]],
        "positive_definition": "frame contains at least one fire or smoke ground-truth box",
        "splits": splits,
        "box_area_relative": box_area,
        "small_box_threshold": SMALL_BOX,
        "low_count_threshold": LOW_COUNT,
        "low_count_classes_test": [k for k, v in test_inst.items() if v < LOW_COUNT],
        "test_prevalence": splits["test"]["positive_prevalence"],
    }
    for split, s in splits.items():
        log("describe", f"{split:<5} {s['images']:>6} images, {s['background_images']:>5} background "
                        f"({s['background_pct']:.1f}%), instances {s['instances']}")
    log("describe", f"test prevalence of fire-or-smoke frames: {stats['test_prevalence']:.3f}")
    return stats


def verify_dataset(stats, roles):
    """The checks from the brief. Fail loudly so the fallback dataset can be pinned instead."""
    problems = []
    if len(stats["class_names"]) < 2:
        problems.append("fewer than two classes")
    if not roles["fire"] or not roles["smoke"]:
        problems.append(f"no fire and smoke classes among {stats['class_names']}")
    test = stats["splits"]["test"]
    if not test["images"]:
        problems.append("no test split")
    elif not test["background_images"]:
        problems.append("test split has no background images (no true negatives)")
    if problems:
        raise SystemExit("Dataset fails the checks: " + "; ".join(problems) +
                         ". Pin another public version/dataset at the top of train.py.")
    log("describe", "dataset passes checks (>=2 classes incl. fire+smoke, test split with backgrounds)")


def plot_dataset(stats, boxes):
    import matplotlib.pyplot as plt

    names = stats["class_names"]
    splits = list(SPLIT_DIRS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.8 / max(len(names), 1)
    colours = [_class_colour(n) for n in names]
    for k, name in enumerate(names):
        vals = [stats["splits"][s]["instances"][name] for s in splits]
        xs = np.arange(len(splits)) + (k - (len(names) - 1) / 2) * width
        ax1.bar(xs, vals, width, label=name, color=colours[k])
        for x, v in zip(xs, vals):
            ax1.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=7, color=INK)
    ax1.set_xticks(range(len(splits)), splits)
    ax1.set_title("Instances per class per split", loc="left", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=8)
    bins = np.logspace(-5, 0, 41)
    for k, name in enumerate(names):
        a = boxes[boxes.cls == k].area.to_numpy()
        if a.size:
            ax2.hist(a, bins=bins, histtype="step", lw=1.6, color=colours[k], label=name)
    ax2.axvline(SMALL_BOX, color=MUTED, ls="--", lw=1)
    ax2.text(SMALL_BOX, ax2.get_ylim()[1] * 0.95, " 1% of image", color=MUTED, fontsize=7, va="top")
    ax2.set_xscale("log")
    ax2.set_xlabel("box area / image area (all splits)", fontsize=8)
    ax2.set_title("Box-size distribution", loc="left", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(WEIGHTS / "dataset_overview.png", dpi=150)
    plt.close(fig)
    log("save", "weights/dataset_overview.png")


def _class_colour(name):
    if re.search(r"fire|flame", name, re.I):
        return ALARM_C
    if re.search(r"smoke", name, re.I):
        return "#64748B"
    return MUTED


def _clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD5E1")
    ax.tick_params(colors=INK, labelsize=8)


# ---------------------------------------------------------------------------
# 4. Finetune
# ---------------------------------------------------------------------------
def finetune(data_yaml, env, source):
    from ultralytics import YOLO

    if MODEL not in ("yolo11n.pt", "yolo11s.pt"):
        raise SystemExit(f"MODEL must be yolo11n.pt or yolo11s.pt, got {MODEL}")
    started = time.time()
    if RESUME:
        if not LAST_CKPT.is_file():
            raise SystemExit(f"RESUME=1 but {LAST_CKPT.relative_to(ROOT)} does not exist. "
                             "Start a normal run (without RESUME).")
        log("finetune", f"resuming from {LAST_CKPT.relative_to(ROOT)} (settings come from the checkpoint)")
        model = YOLO(str(LAST_CKPT))
        try:
            model.train(resume=True, device=env["device"], workers=WORKERS)
        except AssertionError as error:  # Ultralytics asserts when the run already finished
            raise SystemExit(f"Nothing to resume: {error}\nThat run finished. Use EVAL_ONLY=1 to redo the "
                             "evaluation, or drop RESUME to train again from scratch.") from error
    else:
        log("finetune", f"{MODEL}, epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, "
                        f"patience={PATIENCE}, fraction={FRACTION}, seed={SEED}")
        model = YOLO(MODEL)
        model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, patience=PATIENCE, batch=BATCH,
                    fraction=FRACTION, seed=SEED, deterministic=True, device=env["device"], workers=WORKERS,
                    project=str(RUNS), name="finetune", exist_ok=True, plots=True)
    wall = time.time() - started
    trainer = model.trainer
    targs = trainer.args
    shutil.copy2(trainer.best, WEIGHTS / "best.pt")
    log("save", "weights/best.pt")

    results = pd.read_csv(Path(trainer.save_dir) / "results.csv")
    results.columns = [c.strip() for c in results.columns]
    epochs_done = int(results["epoch"].max()) if "epoch" in results else trainer.epoch + 1
    if RESUME and "time" in results:  # the resumed process only saw part of the training time
        wall = training_seconds(results)
    best_row = results.loc[results["metrics/mAP50-95(B)"].idxmax()]
    args = {k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
            for k, v in vars(trainer.args).items()}
    meta = {
        "model": Path(str(targs.model)).name if not RESUME else MODEL,
        "settings": {"epochs": targs.epochs, "imgsz": targs.imgsz, "patience": targs.patience,
                     "batch": targs.batch, "fraction": targs.fraction, "seed": targs.seed},
        "resumed": RESUME,
        "epochs_completed": epochs_done,
        "stopped_early": epochs_done < targs.epochs,
        "best_epoch": int(best_row["epoch"]),
        "best_val": {
            "precision": r6(best_row["metrics/precision(B)"]),
            "recall": r6(best_row["metrics/recall(B)"]),
            "mAP50": r6(best_row["metrics/mAP50(B)"]),
            "mAP50-95": r6(best_row["metrics/mAP50-95(B)"]),
        },
        "train_images_used": len(trainer.train_loader.dataset),
        "wall_clock_seconds": round(wall, 1),
        "seconds_per_epoch": round(wall / max(epochs_done, 1), 1),
        "environment": env,
        "dataset": source,
        "split": "grouped" if GROUPED else "original",
        "ultralytics_args": args,
    }
    save_json("run_meta.json", meta)
    return meta


def training_seconds(results):
    """Total training time from results.csv. Its `time` column restarts at 0 in each resumed
    session, so add up the last value of every session."""
    t = results["time"].to_numpy(dtype=float)
    session_ends = np.r_[t[1:] < t[:-1], True]
    return float(t[session_ends].sum())


# ---------------------------------------------------------------------------
# 5. Evaluate on test
# ---------------------------------------------------------------------------
def frame_scores(model, paths, roles, device, conf):
    """Per-frame max confidence for fire, smoke and either, plus every predicted class id."""
    fire, smoke = set(roles["fire"]), set(roles["smoke"])
    out = []
    for start in range(0, len(paths), 64):
        chunk = [str(p) for p in paths[start:start + 64]]
        results = model.predict(chunk, conf=conf, iou=NMS_IOU, imgsz=IMGSZ, device=device,
                                 verbose=False, stream=True)
        for result in results:
            cls = result.boxes.cls.cpu().numpy().astype(int)
            cf = result.boxes.conf.cpu().numpy()
            s_fire = float(cf[np.isin(cls, list(fire))].max(initial=0.0))
            s_smoke = float(cf[np.isin(cls, list(smoke))].max(initial=0.0))
            out.append({"s": max(s_fire, s_smoke), "s_fire": s_fire, "s_smoke": s_smoke,
                        "classes": cls.tolist()})
    return out


def evaluate(data_yaml, names, roles, test_frames, env):
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS / "best.pt"))
    log("evaluate", "model.val on the test split (conf=0.001)")
    m = model.val(data=str(data_yaml), split="test", conf=PRED_CONF, iou=NMS_IOU, imgsz=IMGSZ,
                  batch=BATCH, device=env["device"], project=str(RUNS), name="test_eval",
                  exist_ok=True, plots=False, verbose=False)
    nt = getattr(m, "nt_per_class", None)
    per_class = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class.append({"class": names[int(c)], "precision": r6(p), "recall": r6(r),
                          "AP50": r6(ap50), "AP50-95": r6(ap),
                          "test_instances": int(nt[int(c)]) if nt is not None else None})

    log("evaluate", f"detection confusion matrix (conf={CM_CONF})")
    cm = model.val(data=str(data_yaml), split="test", conf=CM_CONF, iou=NMS_IOU, imgsz=IMGSZ,
                   batch=BATCH, device=env["device"], project=str(RUNS), name="test_cm",
                   exist_ok=True, plots=True, verbose=False).confusion_matrix.matrix
    labels = list(names) + ["background"]
    matrix = np.asarray(cm, dtype=int).T  # Ultralytics is [pred, true]; store [true, pred]
    ev = {
        "class_names": names,
        "positive_class": "frame contains fire or smoke (any ground-truth box of either class)",
        "excluded_from_score": [names[i] for i in roles["other"]],
        "overall": {"mAP50": r6(m.box.map50), "mAP50-95": r6(m.box.map),
                    "precision": r6(m.box.mp), "recall": r6(m.box.mr)},
        "per_class": per_class,
        "per_class_note": "precision/recall at the single confidence that maximises mean F1 "
                          "(Ultralytics convention); AP is threshold-free",
        "val_settings": {"conf": PRED_CONF, "iou": NMS_IOU, "imgsz": IMGSZ, "split": "test"},
        "confusion_matrix": {"labels": labels, "rows": "true", "cols": "predicted",
                             "conf": CM_CONF, "matrix": matrix.tolist()},
    }
    plot_detection_cm(matrix, labels)

    log("evaluate", f"frame-level inference on {len(test_frames)} test images (conf={PRED_CONF})")
    scores = frame_scores(model, test_frames.path.tolist(), roles, env["device"], PRED_CONF)
    df = pd.DataFrame({
        "image_id": test_frames.image_id.to_numpy(),
        "y_true": test_frames.y_true.to_numpy(),
        "y_fire": test_frames.y_fire.to_numpy(),
        "y_smoke": test_frames.y_smoke.to_numpy(),
        "s": [r["s"] for r in scores],
        "s_fire": [r["s_fire"] for r in scores],
        "s_smoke": [r["s_smoke"] for r in scores],
    })
    df.round(6).to_csv(WEIGHTS / "test_scores.csv", index=False)
    log("save", "weights/test_scores.csv")
    log("evaluate", f"mAP50={ev['overall']['mAP50']:.3f}  mAP50-95={ev['overall']['mAP50-95']:.3f}")
    return ev, df


def plot_detection_cm(matrix, labels):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(np.log1p(matrix), cmap="Greys", vmin=0, vmax=np.log1p(matrix.max()) * 1.6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            dark = np.log1p(matrix[i, j]) > 0.45 * np.log1p(matrix.max()) * 1.6
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=9,
                    color="white" if dark else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(f"Detection confusion matrix (test, conf ≥ {CM_CONF})", loc="left", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(WEIGHTS / "confusion_detection.png", dpi=150)
    plt.close(fig)
    log("save", "weights/confusion_detection.png")


# ---------------------------------------------------------------------------
# 6. Baselines (inference part; costed after the decision stage)
# ---------------------------------------------------------------------------
def coco_zero_shot(test_frames, env):
    from ultralytics import YOLO

    coco = YOLO(MODEL)  # the pretrained COCO checkpoint of the same family
    names = coco.names
    log("baseline", f"COCO zero-shot {MODEL} on {len(test_frames)} test images (logging conf={COCO_LOG_CONF})")
    preds = frame_scores(coco, test_frames.path.tolist(), {"fire": [], "smoke": []},
                         env["device"], COCO_LOG_CONF)
    pos_counts, neg_counts = Counter(), Counter()
    for y, p in zip(test_frames.y_true.to_numpy(), preds):
        (pos_counts if y else neg_counts).update({names[c] for c in p["classes"]})
    fire_like = [n for n in names.values() if re.fullmatch(r"fire|smoke|flame", n, re.I)]  # not "fire hydrant"
    return {
        "weights": MODEL,
        "coco_has_fire_or_smoke_class": bool(fire_like),
        "logging_conf": COCO_LOG_CONF,
        "frames_with_any_detection": {
            "positive": int(sum(bool(p["classes"]) for y, p in zip(test_frames.y_true, preds) if y)),
            "negative": int(sum(bool(p["classes"]) for y, p in zip(test_frames.y_true, preds) if not y)),
        },
        "top_classes_positive_frames": pos_counts.most_common(5),
        "top_classes_negative_frames": neg_counts.most_common(5),
        "note": "COCO has no fire or smoke class, so its decision score is 0.0 for every frame and "
                "its fire/smoke AP is 0. Counts are frames in which the class appears at least once.",
    }


# ---------------------------------------------------------------------------
# 7. Cost analysis (pure functions; app.py mirrors these)
# ---------------------------------------------------------------------------
def share_at_or_above(sorted_scores, t):
    """P(s >= t) for an ascending score array, vectorised over t."""
    t = np.asarray(t, dtype=float)
    return 1.0 - np.searchsorted(sorted_scores, t, side="left") / sorted_scores.size


def three_way_rates(pos_sorted, neg_sorted, lo, hi):
    """(p_alarm|pos, p_review|pos, p_alarm|neg, p_review|neg) for pass < lo <= review < hi <= alarm."""
    pa, pl = share_at_or_above(pos_sorted, hi), share_at_or_above(pos_sorted, lo)
    na, nl = share_at_or_above(neg_sorted, hi), share_at_or_above(neg_sorted, lo)
    return pa, np.clip(pl - pa, 0, None), na, np.clip(nl - na, 0, None)


def outcomes(pa, pr, na, nr, D):
    """Final outcome probabilities after human review, per true class."""
    tpr = pa + pr * D["human_recall"]
    fpr = na + nr * D["human_fpr"]
    return tpr, 1 - tpr, fpr, 1 - fpr


def staffing(review_share, D):
    """Review volume and the FTE it needs. Salary is paid on FTE, not on reviewed frames."""
    d = derived(D)
    review_share = np.asarray(review_share, dtype=float)
    frames = review_share * d["frames_per_day"]
    hours = frames * D["review_seconds"] / 3600
    workload_fte = hours / D["productive_hours_per_fte_day"]
    fte = np.where(review_share > EPS, np.maximum(workload_fte, D["min_on_duty_fte_24_7"]), 0.0)
    cost_day = fte * D["paid_hours_per_fte_day"] * D["reviewer_cost_per_hour"]
    return {
        "review_frames_per_day": frames,
        "reviewer_hours_per_day": hours,
        "workload_fte": workload_fte,
        "fte_needed": fte,
        "feasible": fte <= D["available_reviewer_fte"] + 1e-9,
        "staffing_cost_per_frame": cost_day / d["frames_per_day"],
        "review_labour_per_frame": review_share * d["cost_review"],
    }


def policy_cost(pa, pr, na, nr, D):
    """Expected cost per frame of a pass/review/alarm policy at the production base rate.

    total = outcome cost + staffing cost, where staffing = FTE x paid hours x hourly cost
    spread over all frames. Staffing contains the per-frame review labour
    (review share x cost_review) plus coverage/idle overhead.
    """
    pi = D["base_rate"]
    tpr, fnr, fpr, tnr = outcomes(pa, pr, na, nr, D)
    outcome = (pi * (tpr * D["cost_tp"] + fnr * D["cost_fn"])
               + (1 - pi) * (fpr * D["cost_fp"] + tnr * D["cost_tn"]))
    share = pi * np.asarray(pr) + (1 - pi) * np.asarray(nr)
    st = staffing(share, D)
    return {"outcome_cost_per_frame": outcome, "review_share": share, **st,
            "total_cost_per_frame": outcome + st["staffing_cost_per_frame"]}


def model_alone_cost(tpr, fpr, D):
    return policy_cost(tpr, 0.0, fpr, 0.0, D)["total_cost_per_frame"]


def sweep_thresholds(scores):
    t = np.unique(np.concatenate([np.linspace(0, 1, N_SWEEP), scores]))
    return t[(t >= 0) & (t <= 1)]


def policy_grid():
    g = np.round(np.linspace(0, 1, N_POLICY_GRID), 2)
    lo, hi = np.meshgrid(g, g, indexing="ij")
    keep = lo <= hi
    return lo[keep], hi[keep]


def behaviour(pa, pr, na, nr, D):
    """Name what a policy actually does, so break-evens can be stated in words."""
    pi = D["base_rate"]
    if pi * pr + (1 - pi) * nr > EPS:
        return "review everything" if pr > 1 - 1e-9 and nr > 1 - 1e-9 else "three-way"
    if pi * pa + (1 - pi) * na <= EPS:
        return "do nothing"
    return "always alarm" if pa > 1 - 1e-9 and na > 1 - 1e-9 else "model alone"


def optimise(pos_sorted, neg_sorted, D, curve=None):
    """Best model-alone threshold and best (lo, hi), unconstrained and staff-feasible."""
    t = curve if curve is not None else sweep_thresholds(np.concatenate([pos_sorted, neg_sorted]))
    tpr, fpr = share_at_or_above(pos_sorted, t), share_at_or_above(neg_sorted, t)
    cost = model_alone_cost(tpr, fpr, D)
    k = int(np.argmin(cost))
    lo, hi = policy_grid()
    rates = three_way_rates(pos_sorted, neg_sorted, lo, hi)
    pc = policy_cost(*rates, D)
    total = pc["total_cost_per_frame"]
    u = int(np.argmin(total))
    feasible_total = np.where(pc["feasible"], total, np.inf)
    f = int(np.argmin(feasible_total))
    candidates = {
        "do nothing": (0.0, 0.0, 0.0, 0.0),
        "review everything": (0.0, 1.0, 0.0, 1.0),
        "model alone": (tpr[k], 0.0, fpr[k], 0.0),
        "three-way": tuple(r[f] for r in rates),
    }
    priced = {}
    for name, r in candidates.items():
        p = policy_cost(*r, D)
        if bool(p["feasible"]):
            priced[name] = float(p["total_cost_per_frame"])
    best = min(priced, key=lambda n: priced[n] - (1e-9 if n != "three-way" else 0))
    best_label = behaviour(*candidates[best], D)
    return {
        "t": t, "tpr": tpr, "fpr": fpr, "cost": cost,
        "t_star": float(t[k]), "model_alone_cost": float(cost[k]),
        "grid_lo": lo, "grid_hi": hi, "grid_rates": rates, "grid": pc,
        "unconstrained_idx": u, "feasible_idx": f,
        "n_feasible": int(np.isfinite(feasible_total).sum()),
        "recommended": best_label,
        "feasible_three_way_label": behaviour(*(r[f] for r in rates), D),
    }


def describe_policy(name, pa, pr, na, nr, D, lo=None, hi=None, n_pos=None, n_neg=None):
    """JSON-friendly record of one policy, per 1,000 frames and per day."""
    pi = D["base_rate"]
    d = derived(D)
    pc = {k: float(np.asarray(v)) for k, v in policy_cost(pa, pr, na, nr, D).items()}
    tpr, fnr, fpr, tnr = (float(x) for x in outcomes(pa, pr, na, nr, D))
    pa, pr, na, nr = (float(x) for x in (pa, pr, na, nr))
    rec = {
        "policy": name,
        "lo": r6(lo) if lo is not None else None,
        "hi": r6(hi) if hi is not None else None,
        "behaviour": behaviour(pa, pr, na, nr, D),
        "cost_per_1000": r6(1000 * pc["total_cost_per_frame"]),
        "cost_per_day": r6(d["frames_per_day"] * pc["total_cost_per_frame"]),
        "outcome_cost_per_1000": r6(1000 * pc["outcome_cost_per_frame"]),
        "staffing_cost_per_1000": r6(1000 * pc["staffing_cost_per_frame"]),
        "review_labour_per_1000": r6(1000 * pc["review_labour_per_frame"]),
        "review_share": r6(pc["review_share"]),
        "review_frames_per_day": r6(pc["review_frames_per_day"]),
        "reviewer_hours_per_day": r6(pc["reviewer_hours_per_day"]),
        "workload_fte": r6(pc["workload_fte"]),
        "fte_needed": r6(pc["fte_needed"]),
        "fte_available": D["available_reviewer_fte"],
        "feasible": bool(pc["feasible"]),
        "rates": {"pos": {"pass": r6(1 - pa - pr), "review": r6(pr), "alarm": r6(pa)},
                  "neg": {"pass": r6(1 - na - nr), "review": r6(nr), "alarm": r6(na)}},
        "matrix_per_1000": {  # rows actual, cols pass/review/alarm, at the production base rate
            "fire_smoke": [r6(1000 * pi * x) for x in (1 - pa - pr, pr, pa)],
            "clear": [r6(1000 * (1 - pi) * x) for x in (1 - na - nr, nr, na)],
        },
        "final_per_1000": {"TP": r6(1000 * pi * tpr), "FN": r6(1000 * pi * fnr),
                           "FP": r6(1000 * (1 - pi) * fpr), "TN": r6(1000 * (1 - pi) * tnr)},
        "missed_per_1000_true_frames": r6(1000 * fnr),
    }
    if n_pos is not None:
        rec["matrix_test_counts"] = {
            "fire_smoke": [round(n_pos * x) for x in (1 - pa - pr, pr, pa)],
            "clear": [round(n_neg * x) for x in (1 - na - nr, nr, na)],
        }
    return rec


def frame_pr(y, s, t):
    pred = s >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"threshold": r6(t), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": r6(tp / (tp + fp)) if tp + fp else None,
            "recall": r6(tp / (tp + fn)) if tp + fn else None,
            "fpr": r6(fp / (fp + tn)) if fp + tn else None}


def scan_param(param, multipliers, pos_sorted, neg_sorted, D, curve):
    rows = []
    for m in multipliers:
        D2 = dict(D)
        D2[param] = min(D[param] * m, 0.999) if param == "base_rate" else D[param] * m
        o = optimise(pos_sorted, neg_sorted, D2, curve)
        u, f = o["unconstrained_idx"], o["feasible_idx"]
        rows.append({
            "multiplier": r6(m), "value": r6(D2[param]),
            "t_star": r6(o["t_star"]),
            "model_alone_cost_per_1000": r6(1000 * o["model_alone_cost"]),
            "unconstrained_lo": r6(o["grid_lo"][u]), "unconstrained_hi": r6(o["grid_hi"][u]),
            "unconstrained_cost_per_1000": r6(1000 * o["grid"]["total_cost_per_frame"][u]),
            "feasible_lo": r6(o["grid_lo"][f]), "feasible_hi": r6(o["grid_hi"][f]),
            "feasible_cost_per_1000": r6(1000 * o["grid"]["total_cost_per_frame"][f]),
            "feasible_review_share": r6(o["grid"]["review_share"][f]),
            "recommended": o["recommended"],
            "feasible_three_way_uses_review": o["feasible_three_way_label"] in ("three-way", "review everything"),
        })
    return rows


def break_evens(param, pos_sorted, neg_sorted, D, curve, lo_m=0.01, hi_m=100.0, key="recommended"):
    """Where does the recommended policy (or review usage) switch as one input scales?"""
    ms = np.logspace(np.log10(lo_m), np.log10(hi_m), 41)

    def label(m):
        D2 = dict(D)
        D2[param] = min(D[param] * m, 0.999) if param == "base_rate" else D[param] * m
        o = optimise(pos_sorted, neg_sorted, D2, curve)
        return o["recommended"] if key == "recommended" else (
            o["feasible_three_way_label"] in ("three-way", "review everything"))

    labels = [label(m) for m in ms]
    switches = []
    for i in range(len(ms) - 1):
        if labels[i] == labels[i + 1]:
            continue
        a, b = np.log10(ms[i]), np.log10(ms[i + 1])
        for _ in range(20):
            mid = (a + b) / 2
            if label(10 ** mid) == labels[i]:
                a = mid
            else:
                b = mid
        m = 10 ** ((a + b) / 2)
        value = min(D[param] * m, 0.999) if param == "base_rate" else D[param] * m
        switches.append({"param": param, "value": r6(value), "multiplier": r6(m),
                         "below": labels[i], "above": labels[i + 1]})
    return switches


def bootstrap(pos, neg, t_star, lo, hi, D):
    rng = np.random.default_rng(SEED)
    pos_b = pos[rng.integers(0, pos.size, (N_BOOT, pos.size))]
    neg_b = neg[rng.integers(0, neg.size, (N_BOOT, neg.size))]
    alone = 1000 * model_alone_cost((pos_b >= t_star).mean(1), (neg_b >= t_star).mean(1), D)
    pa, na = (pos_b >= hi).mean(1), (neg_b >= hi).mean(1)
    pr, nr = (pos_b >= lo).mean(1) - pa, (neg_b >= lo).mean(1) - na
    three = 1000 * policy_cost(pa, pr, na, nr, D)["total_cost_per_frame"]
    tpr_b = (pos_b >= t_star).mean(1)

    def ci(x):
        return {"mean": r6(x.mean()), "lo95": r6(np.percentile(x, 2.5)), "hi95": r6(np.percentile(x, 97.5))}

    return {"n_resamples": N_BOOT, "stratified": True, "seed": SEED,
            "model_alone_cost_per_1000": ci(alone), "three_way_cost_per_1000": ci(three),
            "model_alone_recall": ci(tpr_b)}


def decide(scores, D):
    d = derived(D)
    y = scores.y_true.to_numpy()
    s = scores.s.to_numpy()
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    if not pos.size or not neg.size:
        raise SystemExit("Test split needs both positive and negative frames for the cost analysis.")
    curve = sweep_thresholds(s)
    o = optimise(pos, neg, D, curve)
    u, f = o["unconstrained_idx"], o["feasible_idx"]
    lo_f, hi_f = float(o["grid_lo"][f]), float(o["grid_hi"][f])
    lo_u, hi_u = float(o["grid_lo"][u]), float(o["grid_hi"][u])
    t_star = o["t_star"]
    tpr_s, fpr_s = float(share_at_or_above(pos, t_star)), float(share_at_or_above(neg, t_star))
    n_pos, n_neg = pos.size, neg.size
    kw = {"n_pos": n_pos, "n_neg": n_neg}

    policies = {
        "do_nothing": describe_policy("(a) Do nothing / never alarm", 0, 0, 0, 0, D, **kw),
        "review_all": describe_policy("(b) Human reviews every frame", 0, 1, 0, 1, D, lo=0.0, hi=1.01, **kw),
        "model_alone": describe_policy(f"(c) Model alone at t*={t_star:.3f}", tpr_s, 0, fpr_s, 0, D,
                                       lo=t_star, hi=t_star, **kw),
        "three_way": describe_policy(f"(d) Three-way at feasible optimum ({lo_f:.2f}, {hi_f:.2f})",
                                     *three_way_rates(pos, neg, lo_f, hi_f), D, lo=lo_f, hi=hi_f, **kw),
    }
    unconstrained = describe_policy(f"Three-way unconstrained ({lo_u:.2f}, {hi_u:.2f})",
                                    *three_way_rates(pos, neg, lo_u, hi_u), D, lo=lo_u, hi=hi_u, **kw)
    always = describe_policy("Always alarm", 1, 0, 1, 0, D, **kw)

    # Capacity frontier: the largest review share the available staff can absorb
    cap_share = 0.0
    if D["available_reviewer_fte"] >= D["min_on_duty_fte_24_7"]:
        cap_share = min(1.0, D["available_reviewer_fte"] * D["productive_hours_per_fte_day"] * 3600
                        / D["review_seconds"] / d["frames_per_day"])
    feasibility = {
        "available_reviewer_fte": D["available_reviewer_fte"],
        "min_on_duty_fte_24_7": D["min_on_duty_fte_24_7"],
        "max_feasible_review_share": r6(cap_share),
        "max_feasible_review_frames_per_day": r6(cap_share * d["frames_per_day"]),
        "grid_pairs": int(o["grid_lo"].size),
        "feasible_pairs": o["n_feasible"],
        "unconstrained_is_feasible": bool(unconstrained["feasible"]),
    }

    # Sensitivity and break-evens
    mults = np.logspace(-1, 1, 9)
    log("decide", "sensitivity scans and break-evens")
    sensitivity = {p: scan_param(p, mults, pos, neg, D, curve) for p in ("cost_fn", "base_rate")}
    bes = {
        "recommended_policy_vs_cost_fn": break_evens("cost_fn", pos, neg, D, curve),
        "recommended_policy_vs_cost_fp": break_evens("cost_fp", pos, neg, D, curve),
        "recommended_policy_vs_base_rate": break_evens("base_rate", pos, neg, D, curve),
        "recommended_policy_vs_review_seconds": break_evens("review_seconds", pos, neg, D, curve),
        "review_band_used_vs_review_seconds": break_evens("review_seconds", pos, neg, D, curve,
                                                          key="review_used"),
    }

    boot = bootstrap(pos, neg, t_star, lo_f, hi_f, D)

    per_class = []
    for name, ycol, scol in (("fire", "y_fire", "s_fire"), ("smoke", "y_smoke", "s_smoke"),
                             ("fire or smoke", "y_true", "s")):
        yy, ss = scores[ycol].to_numpy(), scores[scol].to_numpy()
        per_class.append({"class": name, "support_pos": int(yy.sum()), "support_neg": int((1 - yy).sum()),
                          "at_t_star": frame_pr(yy, ss, t_star),
                          "flagged_at_lo": frame_pr(yy, ss, lo_f),
                          "alarm_at_hi": frame_pr(yy, ss, hi_f)})

    always_c = always["cost_per_1000"]
    never_c = policies["do_nothing"]["cost_per_1000"]
    curve_cost = 1000 * o["cost"]
    sanity = {
        "curve_at_t0_equals_always_alarm": bool(np.isclose(curve_cost[0], always_c, rtol=1e-6)),
        "curve_at_t1_equals_never_alarm": bool(s.max() < 1.0 and np.isclose(curve_cost[-1], never_c, rtol=1e-6)),
        "lo_equals_hi_equals_model_alone": bool(np.isclose(
            1000 * policy_cost(*three_way_rates(pos, neg, t_star, t_star), D)["total_cost_per_frame"],
            policies["model_alone"]["cost_per_1000"], rtol=1e-9)),
        "lo0_hi_above_1_equals_review_all": bool(np.isclose(
            1000 * policy_cost(*three_way_rates(pos, neg, 0.0, 1.01), D)["total_cost_per_frame"],
            policies["review_all"]["cost_per_1000"], rtol=1e-9)),
    }
    for k, v in sanity.items():
        log("decide", f"sanity {k}: {'ok' if v else 'FAILED'}")

    decision = {
        "decisions": D,
        "derived": {"cost_review_per_frame": r6(d["cost_review"]), "frames_per_day": d["frames_per_day"]},
        "score_definition": "s = max confidence over predicted fire/smoke boxes (0.0 if none); "
                            f"inference conf={PRED_CONF}",
        "cost_model": "per frame: base_rate*[TPR*c_tp+(1-TPR)*c_fn] + (1-base_rate)*[FPR*c_fp+(1-FPR)*c_tn] "
                      "+ staffing; reviewed positives caught with human_recall, reviewed negatives "
                      "escalated with human_fpr; staffing = FTE*paid_hours*hourly/frames_per_day, "
                      "FTE = max(workload, coverage floor) if any review",
        "test_counts": {"n_pos": n_pos, "n_neg": n_neg, "prevalence": r6(n_pos / (n_pos + n_neg))},
        "curve": {"t": [r6(x) for x in o["t"]], "tpr": [r6(x) for x in o["tpr"]],
                  "fpr": [r6(x) for x in o["fpr"]], "cost_per_1000": [r6(x) for x in curve_cost]},
        "t_star": r6(t_star),
        "model_alone": policies["model_alone"],
        "three_way_unconstrained": unconstrained,
        "three_way_feasible": policies["three_way"],
        "recommended_operating_point": {"lo": r6(lo_f), "hi": r6(hi_f)},
        "recommended_policy": o["recommended"],
        "feasibility": feasibility,
        "policies": policies,
        "always_alarm": always,
        "sensitivity": sensitivity,
        "break_evens": bes,
        "bootstrap": boot,
        "frame_level_per_class": per_class,
        "frame_level_note": "raw test-set counts; precision depends on test prevalence, not the base rate",
        "sanity_checks": sanity,
    }
    plot_cost_curve(decision)
    log("decide", f"t*={t_star:.3f} -> {fmt_eur(policies['model_alone']['cost_per_1000'])}/1000; "
                  f"feasible (lo,hi)=({lo_f:.2f},{hi_f:.2f}) -> "
                  f"{fmt_eur(policies['three_way']['cost_per_1000'])}/1000; "
                  f"recommended: {o['recommended']}")
    return decision


def plot_cost_curve(dec):
    import matplotlib.pyplot as plt

    t = np.array(dec["curve"]["t"], dtype=float)
    c = np.array(dec["curve"]["cost_per_1000"], dtype=float)
    pol = dec["policies"]
    lo, hi = dec["recommended_operating_point"]["lo"], dec["recommended_operating_point"]["hi"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if hi > lo:
        ax.axvspan(lo, hi, color=REVIEW_C, alpha=0.12, lw=0)
        ax.text((lo + hi) / 2, 0.98, "review band", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=REVIEW_C)
    ax.plot(t, c, color=ACCENT, lw=1.8, label="model alone")
    for key, style, label in (("do_nothing", "--", "do nothing"), ("review_all", ":", "review everything")):
        ax.axhline(pol[key]["cost_per_1000"], color=MUTED, ls=style, lw=1)
        ax.text(1.0, pol[key]["cost_per_1000"], f" {label}", color=INK, fontsize=8, va="center",
                transform=ax.get_yaxis_transform())
    ts, cs = dec["t_star"], pol["model_alone"]["cost_per_1000"]
    ax.axvline(ts, color=ACCENT, lw=0.8, alpha=0.5)
    ax.plot([ts], [cs], "o", color=ACCENT)
    ax.annotate(f"t* = {ts:.3f}\n{fmt_eur(cs)} / 1,000", (ts, cs), xytext=(8, 22),
                textcoords="offset points", fontsize=8, color=INK)
    tw = pol["three_way"]["cost_per_1000"]
    ax.plot([(lo + hi) / 2], [tw], "D", color=REVIEW_C, ms=6)
    ax.annotate(f"three-way ({lo:.2f}, {hi:.2f})\n{fmt_eur(tw)} / 1,000", ((lo + hi) / 2, tw),
                xytext=(10, -4), textcoords="offset points", fontsize=8, color=INK, va="top")
    ax.set_yscale("log")
    shown = [c.min(), tw, pol["review_all"]["cost_per_1000"]]
    ax.set_ylim(min(v for v in shown if v > 0) / 2, max(c.max(), pol["do_nothing"]["cost_per_1000"]) * 1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("threshold t on frame score s", fontsize=9)
    ax.set_ylabel("expected cost per 1,000 frames (EUR, log)", fontsize=9)
    ax.set_title(f"Expected cost vs threshold at base rate {dec['decisions']['base_rate']:g}",
                 loc="left", fontsize=10, color=INK)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(WEIGHTS / "cost_curve.png", dpi=150)
    plt.close(fig)
    log("save", "weights/cost_curve.png")


def assemble_baselines(ev, coco, dec, scores, D):
    y = scores.y_true.to_numpy()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    def frame_cm(pa, na):
        return {"TP": round(n_pos * pa), "FN": n_pos - round(n_pos * pa),
                "FP": round(n_neg * na), "TN": n_neg - round(n_neg * na)}

    pol = dec["policies"]
    ma = pol["model_alone"]
    out = {
        "positive_class": ev["positive_class"],
        "test_frames": {"positive": n_pos, "negative": n_neg},
        "finetuned": {
            "mAP50": ev["overall"]["mAP50"], "mAP50-95": ev["overall"]["mAP50-95"],
            "operating_point": f"model alone at t*={dec['t_star']:.3f}",
            "frame_confusion": frame_cm(ma["rates"]["pos"]["alarm"], ma["rates"]["neg"]["alarm"]),
            "frame_recall": ma["rates"]["pos"]["alarm"], "frame_fpr": ma["rates"]["neg"]["alarm"],
            "cost_per_1000": ma["cost_per_1000"],
        },
        "coco_zero_shot": {
            "mAP50": 0.0, "mAP50-95": 0.0,
            "frame_confusion": frame_cm(0.0, 0.0), "frame_recall": 0.0, "frame_fpr": 0.0,
            "cost_per_1000": pol["do_nothing"]["cost_per_1000"],
            **coco,
        },
        "predict_nothing": {
            "mAP50": 0.0, "mAP50-95": 0.0,
            "frame_confusion": frame_cm(0.0, 0.0), "frame_recall": 0.0, "frame_fpr": 0.0,
            "cost_per_1000": pol["do_nothing"]["cost_per_1000"],
        },
        "majority_class_at_frame_level": "positive" if n_pos > n_neg else "negative",
    }
    if n_pos > n_neg:
        out["always_alarm"] = {
            "mAP50": None, "mAP50-95": None,
            "frame_confusion": frame_cm(1.0, 1.0), "frame_recall": 1.0, "frame_fpr": 1.0,
            "cost_per_1000": dec["always_alarm"]["cost_per_1000"],
        }
    return out


# ---------------------------------------------------------------------------
# 7b. Source audit: train/test overlap and source-label confounding
# ---------------------------------------------------------------------------
FAMILIES = [  # (pattern on the original filename, label for the report)
    (r"^[HML]_\d+", "H_/M_/L_ (indoor rooms, composited smoke)"),
    (r"^image\d+", "imageNNNN (web photos)"),
    (r"mp4", "video frames (*_mp4-NNNN)"),
    (r"^\d", "numeric names (outdoor/news photos)"),
]


def source_base(image_id):
    """Original filename before Roboflow's '.rf.<hash>' suffix: augmented copies share it."""
    return re.sub(r"\.rf\.[0-9a-f]+\.\w+$", "", image_id)


def source_family(image_id):
    for pattern, label in FAMILIES:
        if re.search(pattern, image_id):
            return label
    return "other web images"


def roc_auc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def audit_sources(frames, scores, dec, D, grouping=None):
    f = frames.assign(base=frames.image_id.map(source_base), family=frames.image_id.map(source_family),
                      crop=frames.image_id.str.contains(NEG_TAG, regex=False))
    seen = set(f.loc[f.split != "test", "base"])
    test = f[f.split == "test"].reset_index(drop=True)
    test = test.assign(leaked=test.base.isin(seen).to_numpy(),
                       s=test.image_id.map(scores.set_index("image_id").s).to_numpy())
    t_star = dec["t_star"]
    lo, hi = dec["recommended_operating_point"]["lo"], dec["recommended_operating_point"]["hi"]

    def rates(df):
        pos, neg = np.sort(df.s[df.y_true == 1].to_numpy()), np.sort(df.s[df.y_true == 0].to_numpy())
        row = {"n_pos": int(pos.size), "n_neg": int(neg.size), "auc": r6(roc_auc(pos, neg))}
        row["recall_at_t_star"] = r6(float(share_at_or_above(pos, t_star))) if pos.size else None
        row["fpr_at_t_star"] = r6(float(share_at_or_above(neg, t_star))) if neg.size else None
        row["pos_passed_at_band"] = r6(1 - float(share_at_or_above(pos, lo))) if pos.size else None
        row["neg_reviewed_or_alarmed_at_band"] = r6(float(share_at_or_above(neg, lo))) if neg.size else None
        if pos.size and neg.size:
            pa, pr, na, nr = three_way_rates(pos, neg, lo, hi)
            row["band_cost_per_1000"] = r6(1000 * float(policy_cost(pa, pr, na, nr, D)["total_cost_per_frame"]))
        return row

    subsets = {"all test frames": test}
    if test.leaked.any():
        subsets["clean (source not in train/val)"] = test[~test.leaked]
        subsets["leaked (source also in train/val)"] = test[test.leaked]
    if test.crop.any():  # the shortcut test: same source, no smoke, so only the pixels can say "clear"
        subsets["original frames only (no crops)"] = test[~test.crop]
        subsets["positives + same-source negative crops"] = test[test.crop | (test.y_true == 1)]
    by_family = {}
    for fam, df in test.groupby("family"):
        by_family[fam] = {**rates(df), "n_frames": int(len(df)), "leaked_frames": int(df.leaked.sum()),
                          "fire_frames": int(df.y_fire.sum()), "smoke_frames": int(df.y_smoke.sum()),
                          "background_frames": int((df.y_true == 0).sum()), "negative_crops": int(df.crop.sum())}

    # Pixel-free baseline: each family predicts its majority frame label in train.
    train = f[f.split == "train"]
    majority = train.groupby("family").y_true.mean().ge(0.5).astype(int).to_dict()
    pred = test.family.map(majority).fillna(int(train.y_true.mean() >= 0.5)).astype(int)
    tp = int(((pred == 1) & (test.y_true == 1)).sum())
    fp = int(((pred == 1) & (test.y_true == 0)).sum())
    n_pos, n_neg = int(test.y_true.sum()), int((test.y_true == 0).sum())
    tpr, fpr = tp / n_pos, fp / n_neg
    source_only = {
        "rule": {fam: ("fire/smoke" if v else "clear") for fam, v in majority.items()},
        "frame_confusion": {"TP": tp, "FN": n_pos - tp, "FP": fp, "TN": n_neg - fp},
        "accuracy": r6((tp + n_neg - fp) / len(test)), "frame_recall": r6(tpr), "frame_fpr": r6(fpr),
        "cost_per_1000": r6(1000 * float(model_alone_cost(tpr, fpr, D))),
    }

    overlap = {}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        shared = set(f.base[f.split == a]) & set(f.base[f.split == b])
        overlap[f"{a}-{b}"] = {"shared_sources": len(shared),
                               f"{b}_frames_affected": int(f[(f.split == b) & f.base.isin(shared)].shape[0])}

    leaked = test[test.leaked]
    return {
        "note": "source = filename before Roboflow's '.rf.<hash>' suffix; family = filename pattern. "
                "Leaked = a test frame whose source image also appears (augmented) in train or val. "
                "The source-only baseline reads no pixels and is not deployable; it shows how much of the "
                "test split can be solved from where an image came from. Negative crops are box-free regions "
                "of positive frames from the same source, added as backgrounds by prepare_grouped.",
        "leaked_family_mix": leaked.family.value_counts().to_dict(),
        "t_star": t_star, "band": {"lo": lo, "hi": hi},
        "unique_sources": {s: int(f.base[f.split == s].nunique()) for s in SPLIT_DIRS},
        "frames": {s: int((f.split == s).sum()) for s in SPLIT_DIRS},
        "split_overlap": overlap,
        "subsets": {k: rates(v) for k, v in subsets.items()},
        "by_family": by_family,
        "source_only_baseline": source_only,
        "grouping": grouping,
    }


# ---------------------------------------------------------------------------
# 8. REPORT.md auto blocks
# ---------------------------------------------------------------------------
def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join("—" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def block_dataset(stats):
    src = stats["source"]
    names = stats["class_names"]
    sp = stats["splits"]
    out = [
        f"**Source:** Roboflow Universe, workspace `{src['workspace']}`, project `{src['project']}`, "
        f"version `{src['version']}` ([link]({src['url']})). Export format `{src['format']}`. "
        f"Licence: {src['licence']}.",
        "",
        f"**Classes:** {', '.join(f'`{n}`' for n in names)}. Positive class at decision level: "
        f"a frame with at least one `fire` or `smoke` box. "
        + (f"Excluded from the decision score (kept in training): "
           f"{', '.join(f'`{n}`' for n in stats['excluded_from_score'])}."
           if stats["excluded_from_score"] else ""),
    ]
    tables = [
        md_table(["Split", "Images", "Background (no labels)", "Fire only", "Smoke only", "Both",
                  "Neither fire nor smoke", "Fire-or-smoke prevalence"],
                 [[s, f"{v['images']:,}", f"{v['background_images']:,} ({v['background_pct']:.1f}%)",
                   f"{v['frames']['fire_only']:,}", f"{v['frames']['smoke_only']:,}",
                   f"{v['frames']['both']:,}", f"{v['frames']['neither']:,}",
                   pct(v["positive_prevalence"])] for s, v in sp.items()]),
        "",
        md_table(["Instances"] + list(sp), [[n] + [f"{sp[s]['instances'][n]:,}" for s in sp] for n in names]),
        "",
        md_table(["Box area / image area", "Boxes (all)", "Median", "IQR (Q1–Q3)", f"Share < {SMALL_BOX:.0%}",
                  "Test median", f"Test share < {SMALL_BOX:.0%}"],
                 [[n, f"{a['all']['n']:,}", pct(a["all"].get("median"), 2),
                   f"{pct(a['all'].get('q1'), 2)}–{pct(a['all'].get('q3'), 2)}",
                   pct(a["all"].get("small_share")), pct(a["test"].get("median"), 2),
                   pct(a["test"].get("small_share"))]
                  for n, a in stats["box_area_relative"].items()]),
        "",
        f"**Test-set prevalence** of fire-or-smoke frames: {pct(stats['test_prevalence'])}.",
        "",
        "**Low-count flag** (< " + str(LOW_COUNT) + " test instances): "
        + (", ".join(f"`{c}`" for c in stats["low_count_classes_test"]) or "none") + ".",
        "",
        "![Dataset overview](weights/dataset_overview.png)",
    ]
    return "\n".join(out + [""] + tables), "\n".join(tables)


def block_training(meta):
    env = meta.get("environment", {})
    s = meta.get("settings", {})
    a = meta.get("ultralytics_args", {})
    rows = [
        ["Model", f"`{meta.get('model')}` (COCO-pretrained, finetuned)"],
        ["Device", f"{env.get('device_name')} (`{env.get('device')}`)"],
        ["Platform", env.get("platform")],
        ["Versions", f"python {env.get('python')}, torch {env.get('torch')}, ultralytics {env.get('ultralytics')}"],
        ["Epochs (requested / completed)", f"{s.get('epochs')} / {meta.get('epochs_completed')}"
                                           + (" (early stop)" if meta.get("stopped_early") else "")],
        ["Best epoch", meta.get("best_epoch")],
        ["Image size", s.get("imgsz")],
        ["Batch", s.get("batch")],
        ["Patience", s.get("patience")],
        ["Train fraction", s.get("fraction")],
        ["Train images used", meta.get("train_images_used")],
        ["Seed", s.get("seed")],
        ["Optimizer / lr0 / momentum", f"{a.get('optimizer')} / {a.get('lr0')} / {a.get('momentum')}"],
        ["Augmentation (mosaic / fliplr / hsv_v)", f"{a.get('mosaic')} / {a.get('fliplr')} / {a.get('hsv_v')}"],
        ["Training time", f"{meta.get('wall_clock_seconds', 0) / 3600:.1f} h "
                          f"({meta.get('seconds_per_epoch', 0) / 60:.1f} min/epoch"
                          + (f", {meta['training_sessions']} sessions: interrupted and resumed"
                             if meta.get("training_sessions", 1) > 1 else "") + ")"],
        ["Best val (P / R / mAP50 / mAP50-95)",
         " / ".join(num(meta.get("best_val", {}).get(k)) for k in ("precision", "recall", "mAP50", "mAP50-95"))],
        ["Started", env.get("started_at")],
    ]
    return md_table(["Setting", "Value"], rows) + "\n\nFull settings: `weights/run_meta.json`."


def block_results(ev, base, stats, audit=None):
    inst = stats["splits"]["test"]["instances"]
    rows = [[c["class"], num(c["precision"]), num(c["recall"]), num(c["AP50"]), num(c["AP50-95"]),
             f"{inst.get(c['class'], c['test_instances']):,}"] for c in ev["per_class"]]
    bl = [("Finetuned (model alone at t*)", base["finetuned"]),
          ("COCO zero-shot", base["coco_zero_shot"]),
          ("Predict nothing", base["predict_nothing"])]
    if "always_alarm" in base:
        bl.append(("Always alarm", base["always_alarm"]))
    if audit is not None:
        bl.append(("Source-only rule (filename, no pixels)", {"mAP50": None, **audit["source_only_baseline"]}))
    brow = [[n, num(b["mAP50"]), pct(b["frame_recall"]), pct(b["frame_fpr"]),
             fmt_eur(b["cost_per_1000"]),
             f"{b['frame_confusion']['TP']}/{b['frame_confusion']['FN']}/"
             f"{b['frame_confusion']['FP']}/{b['frame_confusion']['TN']}"] for n, b in bl]
    coco = base["coco_zero_shot"]
    top = lambda lst: ", ".join(f"`{n}` ({c})" for n, c in lst) or "none"
    per_class = [
        f"**Positive class:** {ev['positive_class']}.",
        "",
        f"**Overall (test):** mAP50 = {num(ev['overall']['mAP50'])}, "
        f"mAP50-95 = {num(ev['overall']['mAP50-95'])}.",
        "",
        md_table(["Class", "Precision", "Recall", "AP50", "AP50-95", "Test instances"], rows),
        "",
        f"*{ev['per_class_note']}.*",
        "",
        "![Detection confusion matrix](weights/confusion_detection.png)",
    ]
    baselines = [
        "**Baselines on the same test split** (frame level; costs use the DECISIONS block at the "
        "production base rate):",
        "",
        md_table(["Model / policy", "mAP50", "Frame recall", "Frame FPR", "Cost / 1,000 frames",
                  "TP/FN/FP/TN (test)"], brow),
        "",
        f"**COCO zero-shot** (`{coco['weights']}`, conf ≥ {coco['logging_conf']}): COCO has "
        f"{'a' if coco['coco_has_fire_or_smoke_class'] else 'no'} fire/smoke class. "
        f"It detected something in {coco['frames_with_any_detection']['positive']:,} positive and "
        f"{coco['frames_with_any_detection']['negative']:,} negative test frames. Top classes on positive "
        f"frames: {top(coco['top_classes_positive_frames'])}. On negative frames: "
        f"{top(coco['top_classes_negative_frames'])}.",
    ]
    return "\n".join(per_class), "\n".join(baselines)


def block_costing(dec):
    D = dec["decisions"]
    rows = [[f"`{k}`", f"{v:,}" if isinstance(v, (int, float)) else v] for k, v in D.items()]
    rows.append(["`cost_review` (derived)", f"€{dec['derived']['cost_review_per_frame']:.4f} per reviewed frame"])
    rows.append(["`frames_per_day` (derived)", f"{dec['derived']['frames_per_day']:,}"])
    b = dec["bootstrap"]
    ma = dec["model_alone"]

    def sens(rows_, label):
        return md_table([label, "t*", "Model alone / 1k", "Unconstrained (lo, hi)", "Feasible (lo, hi)",
                         "Feasible / 1k", "Recommended"],
                        [[f"{r['value']:g} (×{r['multiplier']:.2f})", num(r["t_star"]),
                          fmt_eur(r["model_alone_cost_per_1000"]),
                          f"({r['unconstrained_lo']:.2f}, {r['unconstrained_hi']:.2f})",
                          f"({r['feasible_lo']:.2f}, {r['feasible_hi']:.2f})",
                          fmt_eur(r["feasible_cost_per_1000"]), r["recommended"]] for r in rows_])

    be_lines = []
    titles = {
        "recommended_policy_vs_cost_fn": "Recommended policy vs `cost_fn`",
        "recommended_policy_vs_cost_fp": "Recommended policy vs `cost_fp`",
        "recommended_policy_vs_base_rate": "Recommended policy vs `base_rate`",
        "recommended_policy_vs_review_seconds": "Recommended policy vs `review_seconds`",
        "review_band_used_vs_review_seconds": "Feasible review band in use vs `review_seconds`",
    }
    for key, title in titles.items():
        sw = dec["break_evens"][key]
        if not sw:
            be_lines.append(f"- {title}: no switch between ×0.01 and ×100 of the current value.")
        for s_ in sw:
            below, above = s_["below"], s_["above"]
            if isinstance(below, bool):
                below, above = ("review band used" if below else "no review band",
                                "review band used" if above else "no review band")
            be_lines.append(f"- {title}: switches at **{s_['value']:,.4g}** (×{s_['multiplier']:.3g}); "
                            f"below → {below}, above → {above}.")
    out = [
        md_table(["Decision input", "Value used"], rows),
        "",
        "![Cost vs threshold](weights/cost_curve.png)",
        "",
        f"**Model alone:** t* = {num(dec['t_star'])}, expected cost {fmt_eur(ma['cost_per_1000'])} per 1,000 "
        f"frames (bootstrap 95% CI {fmt_eur(b['model_alone_cost_per_1000']['lo95'])}–"
        f"{fmt_eur(b['model_alone_cost_per_1000']['hi95'])}); frame recall at t* {pct(ma['rates']['pos']['alarm'])} "
        f"(95% CI {pct(b['model_alone_recall']['lo95'])}–{pct(b['model_alone_recall']['hi95'])}), "
        f"FPR {pct(ma['rates']['neg']['alarm'], 2)}.",
        "",
        f"**Three-way at feasible (lo, hi):** {fmt_eur(dec['three_way_feasible']['cost_per_1000'])} per 1,000 "
        f"(95% CI {fmt_eur(b['three_way_cost_per_1000']['lo95'])}–{fmt_eur(b['three_way_cost_per_1000']['hi95'])}). "
        f"Bootstrap: {b['n_resamples']} stratified resamples of the test set, thresholds held fixed.",
        "",
        f"Test prevalence {pct(dec['test_counts']['prevalence'])} vs assumed base rate "
        f"{D['base_rate']:g}: all costs reweight the test-measured TPR/FPR to the base rate.",
        "",
        "**Sensitivity to `cost_fn`:**",
        "",
        sens(dec["sensitivity"]["cost_fn"], "cost_fn (EUR)"),
        "",
        "**Sensitivity to `base_rate`:**",
        "",
        sens(dec["sensitivity"]["base_rate"], "base_rate"),
        "",
        "**Break-even points** (one input scaled, all others held at the values above):",
        "",
        *be_lines,
    ]
    return "\n".join(out)


def block_review(dec):
    f = dec["three_way_feasible"]
    u = dec["three_way_unconstrained"]
    fe = dec["feasibility"]
    pol = dec["policies"]

    def row(p):
        return [p["policy"], fmt_eur(p["cost_per_1000"]), fmt_eur(p["cost_per_day"]), pct(p["review_share"], 2),
                f"{p['review_frames_per_day']:,.0f}", num(p["fte_needed"], 1),
                "yes" if p["feasible"] else "**no**", num(p["missed_per_1000_true_frames"], 1)]

    out = [
        f"**Recommended operating point (feasible optimum):** lo = {f['lo']:.2f}, hi = {f['hi']:.2f} "
        f"(pass if s < lo, review if lo ≤ s < hi, alarm if s ≥ hi). Recommended policy overall: "
        f"**{dec['recommended_policy']}**.",
        "",
        md_table(["", "Feasible optimum", "Unconstrained optimum"], [
            ["(lo, hi)", f"({f['lo']:.2f}, {f['hi']:.2f})", f"({u['lo']:.2f}, {u['hi']:.2f})"],
            ["Cost / 1,000 frames", fmt_eur(f["cost_per_1000"]), fmt_eur(u["cost_per_1000"])],
            ["of which outcomes / staffing", f"{fmt_eur(f['outcome_cost_per_1000'])} / "
                                             f"{fmt_eur(f['staffing_cost_per_1000'])}",
             f"{fmt_eur(u['outcome_cost_per_1000'])} / {fmt_eur(u['staffing_cost_per_1000'])}"],
            ["Share sent to review", pct(f["review_share"], 3), pct(u["review_share"], 3)],
            ["Frames reviewed per day", f"{f['review_frames_per_day']:,.0f}", f"{u['review_frames_per_day']:,.0f}"],
            ["Reviewer hours per day", f"{f['reviewer_hours_per_day']:,.1f}", f"{u['reviewer_hours_per_day']:,.1f}"],
            ["Workload FTE", num(f["workload_fte"], 2), num(u["workload_fte"], 2)],
            ["FTE needed (incl. 24/7 floor)", num(f["fte_needed"], 2), num(u["fte_needed"], 2)],
            ["FTE available", f"{fe['available_reviewer_fte']:g}", f"{fe['available_reviewer_fte']:g}"],
            ["Feasible", "yes" if f["feasible"] else "no", "yes" if u["feasible"] else "no"],
            ["Missed per 1,000 true fire/smoke frames", num(f["missed_per_1000_true_frames"], 1),
             num(u["missed_per_1000_true_frames"], 1)],
        ]),
        "",
        f"Staff capacity: at most {pct(fe['max_feasible_review_share'], 3)} of frames "
        f"({fe['max_feasible_review_frames_per_day']:,.0f} per day) can be reviewed with "
        f"{fe['available_reviewer_fte']:g} FTE; {fe['feasible_pairs']:,} of {fe['grid_pairs']:,} (lo, hi) "
        f"pairs on the 0.01 grid are feasible.",
        "",
        "**Policy comparison** (same base rate and costs; per 1,000 frames and per day):",
        "",
        md_table(["Policy", "Cost / 1,000", "Cost / day", "Review share", "Reviews / day", "FTE needed",
                  "Feasible", "Missed / 1,000 true"],
                 [row(pol[k]) for k in ("do_nothing", "review_all", "model_alone", "three_way")]),
        "",
        "Staffing cost = FTE needed × paid hours per FTE-day × hourly cost. It includes the review time itself "
        "(share × `cost_review`) plus coverage/idle time.",
    ]
    return "\n".join(out)


def block_audit(a):
    ov = a["split_overlap"]
    sub_rows = [[k, f"{v['n_pos']:,}", f"{v['n_neg']:,}", num(v["auc"]), pct(v["recall_at_t_star"]),
                 pct(v["fpr_at_t_star"]), pct(v["pos_passed_at_band"]), pct(v["neg_reviewed_or_alarmed_at_band"]),
                 fmt_eur(v["band_cost_per_1000"]) if v.get("band_cost_per_1000") is not None else None]
                for k, v in a["subsets"].items()]
    fam_rows = [[k, f"{v['n_frames']:,}", f"{v['smoke_frames']:,}", f"{v['fire_frames']:,}",
                 f"{v['background_frames']:,}", f"{v.get('negative_crops', 0):,}", f"{v['leaked_frames']:,}",
                 num(v["auc"]), pct(v["recall_at_t_star"]), pct(v["fpr_at_t_star"])]
                for k, v in sorted(a["by_family"].items(), key=lambda kv: -kv[1]["n_frames"])]
    so = a["source_only_baseline"]
    rule = "; ".join(f"{fam} → {lab}" for fam, lab in so["rule"].items())
    cm = so["frame_confusion"]
    g = a.get("grouping")
    grouping = [] if not g else [
        f"**Fix applied before training (GROUPED=1):** {g['rule']}. {g['sources']:,} source images; "
        f"{g['dropped_copies']:,} extra augmented copies dropped from val/test. Frames kept: "
        + ", ".join(f"{s} {n:,}" for s, n in g["kept"].items())
        + ". Same-source negative crops added: "
        + ", ".join(f"{s} {n:,}" for s, n in g["negative_crops"].items()) + ".",
        "",
    ]
    return "\n".join(grouping + [
        f"**Unique source images** (augmented copies share one): train {a['unique_sources']['train']:,} of "
        f"{a['frames']['train']:,} frames, val {a['unique_sources']['val']:,} of {a['frames']['val']:,}, "
        f"test {a['unique_sources']['test']:,} of {a['frames']['test']:,}.",
        "",
        f"**Split overlap:** {ov['train-test']['shared_sources']:,} source images appear in both train and test "
        f"({ov['train-test']['test_frames_affected']:,} test frames); train–val {ov['train-val']['shared_sources']:,}; "
        f"val–test {ov['val-test']['shared_sources']:,}.",
        "",
        "**Test split by source family** (rates at the model-alone t* = "
        f"{a['t_star']:.3f}):",
        "",
        md_table(["Family", "Frames", "Smoke frames", "Fire frames", "Background", "of which negative crops",
                  "Leaked", "ROC AUC within family", "Recall at t*", "FPR at t*"], fam_rows),
        "",
        f"**Test subsets** (band = ({a['band']['lo']:.2f}, {a['band']['hi']:.2f})):",
        "",
        md_table(["Subset", "Positives", "Negatives", "ROC AUC", "Recall at t*", "FPR at t*",
                  "Positives passed (s < lo)", "Negatives reviewed or alarmed (s ≥ lo)", "Cost / 1,000 at band"],
                 sub_rows),
        "",
        ("Leaked test frames by family: " + ", ".join(f"{k} {v:,}" for k, v in a["leaked_family_mix"].items())
         + ". The leaked and clean rows therefore differ in family mix as well as in leakage."
         if a["leaked_family_mix"] else "No test frame shares a source photo with train or val."),
        "",
        f"**Pixel-free source-only baseline** (each family predicts its majority train label: {rule}): "
        f"accuracy {pct(so['accuracy'])}, frame recall {pct(so['frame_recall'])}, frame FPR {pct(so['frame_fpr'])}, "
        f"TP/FN/FP/TN {cm['TP']}/{cm['FN']}/{cm['FP']}/{cm['TN']}, cost {fmt_eur(so['cost_per_1000'])} per 1,000 frames.",
        "",
        f"*{a['note']}*",
    ])


def block_dashboard():
    return "\n".join([
        "```bash",
        "uv run python main.py app",
        "```",
        "",
        "Opens on http://127.0.0.1:7860. It loads only `weights/` (never trains, never needs the dataset).",
        "",
        "- **Header:** live KPIs: expected cost / 1,000 frames, share sent to review, reviewers needed "
        "(FTE, flagged when over capacity), missed fires per 1,000 true fire frames.",
        "- **Decide:** lo/hi sliders (start at the feasible optimum), reset and model-only controls; "
        "2×3 pass/review/alarm matrix (per 1,000 at the base rate, or raw test counts), final outcome after "
        "review, cost-vs-threshold curve with t* and the current band, and the policy comparison.",
        "- **Try a frame:** upload an image; boxes ≥ lo are drawn and the frame gets a Pass / Review / Alarm "
        "badge from the current sliders.",
        "- **Model:** overall and per-class detection metrics, frame-level per-class P/R at the operating "
        "point, detection confusion matrix, baselines, dataset summary.",
        "- **Assumptions:** the DECISIONS used, derived review cost, bootstrap intervals, and a what-if "
        "panel that never overwrites the report numbers.",
    ])


def refresh_report(blocks):
    if not REPORT.is_file():
        log("report", "REPORT.md missing, skipping")
        return
    text = REPORT.read_text()
    for name, body in blocks.items():
        pattern = re.compile(rf"(<!-- AUTO:{name} -->)(.*?)(<!-- /AUTO:{name} -->)", re.S)
        if not pattern.search(text):
            log("report", f"markers for AUTO:{name} not found, skipped")
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text, count=1)
    REPORT.write_text(text)
    log("report", f"refreshed AUTO blocks: {', '.join(blocks)}")


# ---------------------------------------------------------------------------
def main():
    import matplotlib

    matplotlib.use("Agg")
    started = time.time()
    env = setup()

    ds = download_dataset()
    grouping = None
    if GROUPED:
        ds, grouping = prepare_grouped(ds)
    names, data_yaml, source = dataset_config(ds)
    roles = class_roles(names)
    log("describe", f"classes {names}; fire={roles['fire']} smoke={roles['smoke']} excluded={roles['other']}")
    frames, boxes = scan_labels(ds, roles)
    stats = describe_dataset(frames, boxes, names, roles, source)
    verify_dataset(stats, roles)
    save_json("dataset_stats.json", stats)
    plot_dataset(stats, boxes)

    if EVAL_ONLY:
        if not (WEIGHTS / "best.pt").is_file():
            raise SystemExit("EVAL_ONLY=1 but weights/best.pt does not exist. Train first.")
        meta_path = WEIGHTS / "run_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"model": MODEL}
        trained_on = meta.get("split", "original")
        if trained_on != ("grouped" if GROUPED else "original"):
            raise SystemExit(f"weights/best.pt was trained on the {trained_on} split, but GROUPED={int(GROUPED)}. "
                             "Its test frames leak into that training set; retrain, or set GROUPED to match.")
        meta["last_eval_only_run"] = env["started_at"]
        save_json("run_meta.json", meta)
        log("finetune", "EVAL_ONLY=1, reusing weights/best.pt")
    else:
        meta = finetune(data_yaml, env, source)

    test_frames = frames[frames.split == "test"].reset_index(drop=True)
    ev, scores = evaluate(data_yaml, names, roles, test_frames, env)
    save_json("eval.json", ev)

    coco = coco_zero_shot(test_frames, env)
    dec = decide(scores, DECISIONS)
    save_json("decision.json", dec)
    base = assemble_baselines(ev, coco, dec, scores, DECISIONS)
    save_json("baselines.json", base)
    audit = audit_sources(frames, scores, dec, DECISIONS, grouping)
    save_json("source_audit.json", audit)

    per_class, baselines = block_results(ev, base, stats, audit)
    dataset_block, dataset_tables = block_dataset(stats)
    refresh_report({
        "dataset": dataset_block,
        "dataset_tables": dataset_tables,  # tables only, for a hand-written dataset intro
        "audit": block_audit(audit),
        "training": block_training(meta),
        "baselines": baselines,
        "per_class": per_class,
        "costing": block_costing(dec),
        "review": block_review(dec),
        "dashboard": block_dashboard(),
    })
    log("done", f"pipeline finished in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
