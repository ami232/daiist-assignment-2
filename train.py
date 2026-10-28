"""
Assignment 2: training pipeline. Run with `uv run python main.py train`.

1. Make sure the pinned dataset is on disk (src.data.ensure_dataset).
2. Finetune yolo11n.pt on it. GPU if there is one, else CPU.
3. Copy the best checkpoint to weights/best.pt, the file the app serves.
4. Evaluate on the test split (src.evaluate), writing everything the
   dashboard needs into outputs/.

Device, library versions, settings and wall-clock go to
outputs/train_log.json and are appended to NOTES.md.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import DATASET_ID, ensure_dataset  # noqa: E402

SEED = 42
BASE_WEIGHTS = "yolo11n.pt"
WEIGHTS_OUT = ROOT / "weights" / "best.pt"
# Ultralytics' AMP check downloads a small checkpoint into weights/ on GPU
# machines. check.py loads every .pt in weights/, so these must not stay.
AMP_CHECK_WEIGHTS = ("yolo26n.pt", "yolo11n.pt")
OUT_DIR = ROOT / "outputs"
TRAIN_LOG = OUT_DIR / "train_log.json"
NOTES = ROOT / "NOTES.md"

TRAIN_ARGS = dict(
    imgsz=640,
    epochs=50,
    batch=16,
    seed=SEED,
    deterministic=True,
    patience=10,
    project="runs",
    name="finetune",
    exist_ok=True,
)


def pick_device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def device_name(device: str) -> str:
    import torch

    if device == "cpu":
        return platform.processor() or "cpu"
    return torch.cuda.get_device_name(0)


def append_notes(log: dict) -> None:
    lines = [
        "",
        f"## Training run ({log['started_utc'][:10]})",
        "",
        f"- Dataset `{log['dataset']}`, base weights `{log['base_weights']}`.",
        f"- Device: {log['device_name']} ({log['device']}). "
        f"ultralytics {log['ultralytics']}, torch {log['torch']}, python {log['python']}.",
        "- Settings: " + ", ".join(f"{k}={v}" for k, v in log["train_args"].items()) + ".",
        f"- Epochs run: {log['epochs_completed']} of {log['train_args']['epochs']} "
        f"(patience {log['train_args']['patience']}). Best epoch: {log['best_epoch']}.",
        f"- Wall-clock: training {log['train_seconds']:.0f}s, evaluation "
        f"{log['eval_seconds']:.0f}s, total {log['total_seconds']:.0f}s.",
        f"- Run directory: `{log['save_dir']}`. Best checkpoint copied to "
        f"`weights/best.pt` ({log['best_pt_bytes'] / 1e6:.1f} MB).",
    ]
    if log.get("eval_error"):
        lines.append(f"- Evaluation FAILED: {log['eval_error']}. Weights were still saved.")
    with NOTES.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def training_save_dir(results, model=None) -> Path:
    """Where Ultralytics actually wrote this run.

    Never a hard-coded path: with project="runs" Ultralytics nests the run
    under runs/detect/runs/finetune/, not runs/finetune/. The trainer's
    return value carries the real location.
    """
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and model is not None:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        raise RuntimeError("training returned no save_dir; cannot locate best.pt")
    return Path(save_dir)


def copy_best_checkpoint(results, dest: Path = WEIGHTS_OUT, model=None) -> Path:
    """Copy <save_dir>/weights/best.pt to `dest` and return the source path."""
    best = training_save_dir(results, model) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished but {best} is missing")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    return best


def remove_amp_check_weights(weights_dir: Path = WEIGHTS_OUT.parent) -> list[str]:
    """Delete checkpoints the AMP check dropped into weights/. Returns names removed."""
    removed = []
    for name in AMP_CHECK_WEIGHTS:
        path = weights_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def best_epoch_from_results(run_dir: Path) -> tuple[int | None, int | None]:
    """(best epoch by fitness, epochs completed) from results.csv, if present."""
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return None, None
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    completed = int(len(df))
    cols = [c for c in df.columns if "mAP50-95" in c]
    if not cols:
        return None, completed
    best = int(df[cols[0]].idxmax())
    epoch_col = "epoch" if "epoch" in df.columns else None
    return (int(df.loc[best, epoch_col]) if epoch_col else best + 1), completed


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO

    started = time.time()
    torch.manual_seed(SEED)
    device = pick_device()
    log = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET_ID,
        "base_weights": BASE_WEIGHTS,
        "device": device,
        "device_name": device_name(device),
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device != "cpu" else None,
        "train_args": TRAIN_ARGS,
    }
    print(f"device: {log['device_name']} | ultralytics {log['ultralytics']} | torch {log['torch']}")

    data_yaml = ensure_dataset()
    log["data_yaml"] = str(data_yaml)

    t0 = time.time()
    model = YOLO(BASE_WEIGHTS)
    results = model.train(data=str(data_yaml), device=device, **TRAIN_ARGS)
    log["train_seconds"] = time.time() - t0

    run_dir = training_save_dir(results, model)
    best = copy_best_checkpoint(results, WEIGHTS_OUT, model)
    log["save_dir"] = str(run_dir)
    log["best_pt_bytes"] = WEIGHTS_OUT.stat().st_size
    log["best_epoch"], log["epochs_completed"] = best_epoch_from_results(run_dir)
    log["amp_check_weights_removed"] = remove_amp_check_weights()
    print(f"copied {best} -> {WEIGHTS_OUT}")
    if log["amp_check_weights_removed"]:
        print(f"removed from weights/: {log['amp_check_weights_removed']}")

    # Write the log before evaluating, so a failure there cannot lose it.
    OUT_DIR.mkdir(exist_ok=True)
    log.update(eval_seconds=0.0, total_seconds=time.time() - started, eval_error=None)
    TRAIN_LOG.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    t1 = time.time()
    try:
        from src.evaluate import main as evaluate

        evaluate(weights=WEIGHTS_OUT, out_dir=OUT_DIR)
    except Exception as error:  # noqa: BLE001 - record, then re-raise
        log["eval_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        log["eval_seconds"] = time.time() - t1
        log["total_seconds"] = time.time() - started
        TRAIN_LOG.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        append_notes(log)
        print(f"wrote {TRAIN_LOG} and appended NOTES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
