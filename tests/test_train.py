"""Tests for the checkpoint-handling helpers in train.py. No training."""

from types import SimpleNamespace

import pytest

import train


def _fake_run(tmp_path, nested="runs/detect/runs/finetune"):
    """A fake Ultralytics save_dir with a weights/best.pt inside it."""
    save_dir = tmp_path / nested
    (save_dir / "weights").mkdir(parents=True)
    best = save_dir / "weights" / "best.pt"
    best.write_bytes(b"fake-checkpoint")
    return save_dir, best


def test_copy_best_uses_trainer_save_dir_not_hardcoded_path(tmp_path):
    save_dir, best = _fake_run(tmp_path)           # the nested path Colab produced
    results = SimpleNamespace(save_dir=save_dir)
    dest = tmp_path / "weights" / "best.pt"
    src = train.copy_best_checkpoint(results, dest)
    assert src == best
    assert dest.read_bytes() == b"fake-checkpoint"
    assert not (tmp_path / "runs" / "finetune").exists()   # nothing assumed at the old path


def test_copy_best_accepts_string_save_dir(tmp_path):
    save_dir, _ = _fake_run(tmp_path, nested="elsewhere/run1")
    dest = tmp_path / "out" / "best.pt"
    train.copy_best_checkpoint(SimpleNamespace(save_dir=str(save_dir)), dest)
    assert dest.is_file()


def test_copy_best_falls_back_to_model_trainer(tmp_path):
    save_dir, _ = _fake_run(tmp_path)
    model = SimpleNamespace(trainer=SimpleNamespace(save_dir=save_dir))
    dest = tmp_path / "best.pt"
    train.copy_best_checkpoint(results=None, dest=dest, model=model)
    assert dest.is_file()


def test_copy_best_missing_file_raises(tmp_path):
    save_dir = tmp_path / "run"
    (save_dir / "weights").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        train.copy_best_checkpoint(SimpleNamespace(save_dir=save_dir), tmp_path / "best.pt")


def test_no_save_dir_raises():
    with pytest.raises(RuntimeError):
        train.training_save_dir(SimpleNamespace(), model=None)


def test_remove_amp_check_weights_keeps_best(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"keep")
    (tmp_path / "yolo26n.pt").write_bytes(b"amp")
    removed = train.remove_amp_check_weights(tmp_path)
    assert removed == ["yolo26n.pt"]
    assert (tmp_path / "best.pt").is_file()
    assert not (tmp_path / "yolo26n.pt").exists()
    assert train.remove_amp_check_weights(tmp_path) == []
