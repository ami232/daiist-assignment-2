import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
import costing
from api_key import load_api_key

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet-1k has no "rotten" class. Its fruit classes sit at indices 948-957
# (Granny Smith, strawberry, orange, lemon, fig, pineapple, banana, jackfruit,
# custard apple, pomegranate). Zero-shot, we read "the pretrained model does
# not see a recognisable fruit" as a sign of rot: 1 - P(any fruit class).
IMAGENET_FRUIT_IDS = list(range(948, 958))


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_dataset() -> Path:
    raw = config.DATASET_RAW
    if (raw / "train").is_dir():
        print(f"Dataset already downloaded at {raw.relative_to(config.ROOT)}")
        return raw
    from roboflow import Roboflow

    print(f"Downloading {config.RF_WORKSPACE}/{config.RF_PROJECT} "
          f"version {config.RF_VERSION} from Roboflow Universe ...")
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(config.RF_WORKSPACE).project(config.RF_PROJECT).version(
        config.RF_VERSION)
    version.download("folder", location=str(raw), overwrite=True)
    return raw


# ---------------------------------------------------------------------------
# 2. Our own splits
# ---------------------------------------------------------------------------
def prepare_splits(raw: Path) -> dict:
    found = sorted(p.name for p in (raw / "train").iterdir() if p.is_dir())
    missing = [c for c in config.KEEP_CLASSES if c not in found]
    if missing:
        raise SystemExit(f"Classes {missing} not found in the download. "
                         f"Folders present: {found}")

    # Count what we drop, so the report can say exactly what was removed
    dropped = {}
    for split in ("train", "valid", "test"):
        if not (raw / split).is_dir():
            continue
        for folder in (raw / split).iterdir():
            if folder.is_dir() and folder.name not in config.KEEP_CLASSES:
                dropped[f"{split}/{folder.name}"] = len(list_images(folder))

    out = config.DATASET_PREPARED
    if out.exists():
        shutil.rmtree(out)

    rng = random.Random(config.SEED)
    counts = {"train": {}, "val": {}, "test": {}}
    for cls in config.KEEP_CLASSES:
        # Shuffle within each class, so val keeps the same class balance
        train_imgs = list_images(raw / "train" / cls)
        rng.shuffle(train_imgs)
        n_val = round(len(train_imgs) * config.VAL_FRACTION_OF_TRAIN)
        # Roboflow's "valid" (plus "test", if a version ever has one) = our test
        test_imgs = list_images(raw / "valid" / cls) + list_images(raw / "test" / cls)
        assignment = {"val": train_imgs[:n_val], "train": train_imgs[n_val:],
                      "test": test_imgs}
        for split, imgs in assignment.items():
            dest = out / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)
            counts[split][cls] = len(imgs)

    print("\nImages per class after our split:")
    print(pd.DataFrame(counts).to_string())
    if dropped:
        print(f"\nDropped (not in KEEP_CLASSES): {dropped}")
    return {"counts": counts, "dropped": dropped, "classes_in_download": found}


# ---------------------------------------------------------------------------
# 3. Describe the dataset
# ---------------------------------------------------------------------------
def _hash_variants(path: Path, size: int = 16) -> np.ndarray:
    """Average-hash of the image under all 8 rotations/flips, shape (8, size*size).

    Two photos of the same fruit, one rotated, end up with near-identical bits
    in at least one of the 8 variants.
    """
    g = np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=float)
    bits = g > g.mean()
    variants = []
    for k in range(4):
        r = np.rot90(bits, k)
        variants += [r.ravel(), np.fliplr(r).ravel()]
    return np.array(variants)


def find_near_duplicates(max_bits_different: int = 20) -> dict:
    """For each test image: is there a train image that looks almost the same?"""
    root = config.DATASET_PREPARED
    train = [p for c in config.KEEP_CLASSES for p in list_images(root / "train" / c)]
    test = [p for c in config.KEEP_CLASSES for p in list_images(root / "test" / c)]

    train_bits = np.array([_hash_variants(p)[0] for p in train], dtype=np.float32)
    matches, examples = 0, []
    for p in test:
        v = _hash_variants(p).astype(np.float32)               # (8, 256)
        # bits that differ = a(1-b) + (1-a)b, for every variant x train image
        diff = v @ (1 - train_bits).T + (1 - v) @ train_bits.T  # (8, n_train)
        best = diff.min(axis=0)
        j = int(best.argmin())
        if best[j] <= max_bits_different:
            matches += 1
            if len(examples) < 6:
                examples.append((p, train[j], int(best[j])))

    _plot_duplicate_examples(examples)
    return {"test_images": len(test), "test_with_near_duplicate_in_train": matches,
            "share": matches / max(len(test), 1),
            "hash": "16x16 average hash, rotation/flip invariant",
            "max_bits_different": max_bits_different}


def _plot_duplicate_examples(examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not examples:
        return
    fig, axes = plt.subplots(2, len(examples), figsize=(2.2 * len(examples), 4.6),
                             squeeze=False)
    for i, (test_p, train_p, d) in enumerate(examples):
        for row, (p, label) in enumerate(((test_p, "test"), (train_p, "train"))):
            ax = axes[row][i]
            ax.imshow(Image.open(p).convert("RGB"))
            title = f"{label}: {p.parent.name}"
            if row == 1:
                title += f"\n{d} bits differ"
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    fig.suptitle("Flagged near-duplicates: test image (top) vs closest train image")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "near_duplicate_examples.png", dpi=120)
    plt.close(fig)


def describe_dataset(split_info: dict) -> dict:
    sizes = []
    for split in ("train", "val", "test"):
        for cls in config.KEEP_CLASSES:
            for p in list_images(config.DATASET_PREPARED / split / cls):
                with Image.open(p) as im:
                    sizes.append(im.size)
    w, h = np.array(sizes).T
    print("\nChecking for near-duplicate photos between train and test ...")
    dups = find_near_duplicates()
    print(f"  {dups['test_with_near_duplicate_in_train']} of {dups['test_images']} "
          f"test images ({100 * dups['share']:.1f}%) look almost identical to a "
          f"train image. Check results/near_duplicate_examples.png by eye.")

    summary = {
        "roboflow": f"{config.RF_WORKSPACE}/{config.RF_PROJECT}/{config.RF_VERSION}",
        **split_info,
        "image_size": {"width_median": float(np.median(w)),
                       "height_median": float(np.median(h)),
                       "width_min": int(w.min()), "width_max": int(w.max()),
                       "height_min": int(h.min()), "height_max": int(h.max()),
                       "distinct_sizes": len(set(sizes))},
        "near_duplicates": dups,
    }
    (config.RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# 4. Finetune
# ---------------------------------------------------------------------------
def finetune(device: str) -> Path:
    from ultralytics import YOLO

    print(f"\nFinetuning {config.BASE_MODEL} for {config.EPOCHS} epochs on {device} ...")
    model = YOLO(config.BASE_MODEL)
    model.train(
        data=str(config.DATASET_PREPARED),
        epochs=config.EPOCHS,
        imgsz=config.IMGSZ,
        batch=config.BATCH,
        seed=config.SEED,
        device=device,
        project=str(config.RUNS_DIR),
        name="finetune",
        exist_ok=True,
    )
    save_dir = Path(model.trainer.save_dir)
    config.WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    shutil.copy2(save_dir / "weights" / "best.pt", config.WEIGHTS_PATH)
    if (save_dir / "results.csv").exists():
        shutil.copy2(save_dir / "results.csv", config.RESULTS_DIR / "training_log.csv")
    print(f"Saved best weights to {config.WEIGHTS_PATH.relative_to(config.ROOT)}")
    return config.WEIGHTS_PATH


# ---------------------------------------------------------------------------
# 5. Score images and evaluate against the baselines
# ---------------------------------------------------------------------------
def _predict_probs(model, paths, device):
    probs = []
    for r in model.predict(source=[str(p) for p in paths], imgsz=config.IMGSZ,
                           batch=32, device=device, stream=True, verbose=False):
        probs.append(r.probs.data.cpu().numpy())
    return np.array(probs)


def score_images(device: str) -> pd.DataFrame:
    from ultralytics import YOLO

    finetuned = YOLO(str(config.WEIGHTS_PATH))
    zero_shot = YOLO(config.BASE_MODEL)  # the same checkpoint, before finetuning
    names = [finetuned.names[i] for i in range(len(finetuned.names))]
    pos_idx = [names.index(c) for c in config.POSITIVE_CLASSES]

    rows = []
    for split in ("val", "test"):
        paths, labels = [], []
        for cls in config.KEEP_CLASSES:
            imgs = list_images(config.DATASET_PREPARED / split / cls)
            paths += imgs
            labels += [cls] * len(imgs)
        print(f"Scoring {len(paths)} {split} images ...")
        p_ft = _predict_probs(finetuned, paths, device)
        p_zs = _predict_probs(zero_shot, paths, device)
        for k, path in enumerate(paths):
            rows.append({
                "split": split,
                "image": path.relative_to(config.ROOT).as_posix(),
                "true_class": labels[k],
                "is_rotten": int(labels[k] in config.POSITIVE_CLASSES),
                "pred_class": names[int(p_ft[k].argmax())],
                **{f"p_{c}": float(p_ft[k][i]) for i, c in enumerate(names)},
                "p_rotten": float(p_ft[k][pos_idx].sum()),
                "zs_top1": zero_shot.names[int(p_zs[k].argmax())],
                "zs_p_rotten": float(1 - p_zs[k][IMAGENET_FRUIT_IDS].sum()),
            })
    scores = pd.DataFrame(rows)
    scores.to_csv(config.RESULTS_DIR / "scores.csv", index=False)
    return scores


def _auc(scores, is_rotten) -> float:
    """Chance that a random rotten fruit scores above a random fresh one."""
    s = pd.Series(np.asarray(scores, dtype=float))
    y = np.asarray(is_rotten, dtype=bool)
    ranks = s.rank().to_numpy()
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _binary_metrics(scores, is_rotten, threshold=0.5) -> dict:
    pred = np.asarray(scores) >= threshold
    y = np.asarray(is_rotten, dtype=bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_rotten": tp / max(tp + fp, 1),
            "recall_rotten": tp / max(tp + fn, 1),
            "accuracy": (tp + tn) / len(y), "auc": _auc(scores, y)}


def evaluate(scores: pd.DataFrame, split_info: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = scores[scores["split"] == "test"]
    classes = config.KEEP_CLASSES

    # Finetuned, all classes
    cm = pd.crosstab(test["true_class"], test["pred_class"]).reindex(
        index=classes, columns=classes, fill_value=0)
    per_class = {}
    for c in classes:
        tp = cm.loc[c, c]
        per_class[c] = {"support": int(cm.loc[c].sum()),
                        "precision": float(tp / max(cm[c].sum(), 1)),
                        "recall": float(tp / max(cm.loc[c].sum(), 1))}
    accuracy = float(np.trace(cm.to_numpy()) / cm.to_numpy().sum())

    # Trivial baseline: always predict the most common TRAINING class
    train_counts = split_info["counts"]["train"]
    majority = max(train_counts, key=train_counts.get)
    majority_acc = float((test["true_class"] == majority).mean())

    # What the pretrained model thinks our fruit is, per class
    zs_top = {c: dict(Counter(test.loc[test.true_class == c, "zs_top1"]).most_common(5))
              for c in classes}

    metrics = {
        "test_images": len(test),
        "positive_class": "rotten = " + " + ".join(config.POSITIVE_CLASSES),
        "finetuned": {"accuracy_4class": accuracy, "per_class": per_class,
                      "confusion_matrix": {"labels": classes, "rows_true_cols_pred":
                                           cm.to_numpy().tolist()},
                      "binary_at_0.5": _binary_metrics(test["p_rotten"], test["is_rotten"])},
        "zero_shot": {"accuracy_4class": None,
                      "note": "ImageNet has no fresh/rotten classes, so no 4-class "
                              "prediction exists. Rotten score = 1 - P(ImageNet fruit).",
                      "binary_at_0.5": _binary_metrics(test["zs_p_rotten"], test["is_rotten"]),
                      "top_imagenet_predictions_per_class": zs_top},
        "trivial": {"majority_class": majority, "accuracy_4class": majority_acc,
                    "binary_pass_everything": _binary_metrics(
                        np.zeros(len(test)), test["is_rotten"])},
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix figure for the report
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm.iloc[i, j], ha="center", va="center",
                    color="white" if cm.iloc[i, j] > cm.to_numpy().max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Finetuned model, test split (accuracy {accuracy:.1%})")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nTest accuracy ({len(classes)} classes): finetuned {accuracy:.1%}, "
          f"majority-class baseline {majority_acc:.1%}")
    for c, m in per_class.items():
        print(f"  {c:<15} precision {m['precision']:.1%}  recall {m['recall']:.1%}  "
              f"(n={m['support']})")
    print(f"Rotten-vs-fresh AUC on test: finetuned "
          f"{metrics['finetuned']['binary_at_0.5']['auc']:.3f}, zero-shot "
          f"{metrics['zero_shot']['binary_at_0.5']['auc']:.3f}, trivial 0.500")
    return metrics


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    device = pick_device()
    raw = download_dataset()
    split_info = prepare_splits(raw)
    describe_dataset(split_info)
    finetune(device)
    scores = score_images(device)
    evaluate(scores, split_info)
    print("\nCost analysis:")
    costing.run_analysis()
    print("\nDone. Commit weights/best.pt and the results/ folder.")


if __name__ == "__main__":
    main()