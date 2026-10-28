"""Tests for src/scoring.py. Stand-in objects mimic Ultralytics Results."""

from types import SimpleNamespace

import pytest

from src.config import NEGATIVE_CLASSES, POSITIVE_CLASSES
from src.scoring import decide, image_label, image_score

NAMES = {0: "Box", 1: "Box_broken", 2: "Open_package", 3: "Package"}
IDX = {v: k for k, v in NAMES.items()}


def fake_results(detections):
    """detections: list of (class_name, confidence)."""
    boxes = SimpleNamespace(
        cls=[IDX[name] for name, _ in detections],
        conf=[conf for _, conf in detections],
    )
    return SimpleNamespace(boxes=boxes, names=NAMES)


def test_no_detections_scores_zero():
    assert image_score(fake_results([])) == 0.0


def test_none_results_scores_zero():
    assert image_score(None) == 0.0


def test_negative_class_ignored_positive_taken():
    res = fake_results([("Box", 0.9), ("Box_broken", 0.4)])
    assert image_score(res) == pytest.approx(0.4)


def test_two_positives_take_max():
    res = fake_results([("Open_package", 0.35), ("Box_broken", 0.7), ("Package", 0.99)])
    assert image_score(res) == pytest.approx(0.7)


def test_only_negatives_scores_zero():
    res = fake_results([("Box", 0.95), ("Package", 0.8)])
    assert image_score(res) == 0.0


def test_list_of_results_is_accepted():
    res = [fake_results([("Box", 0.9)]), fake_results([("Open_package", 0.6)])]
    assert image_score(res) == pytest.approx(0.6)


def test_names_resolved_by_name_not_index():
    # Same detections, shuffled index map: result must not change.
    names = {0: "Open_package", 1: "Package", 2: "Box", 3: "Box_broken"}
    idx = {v: k for k, v in names.items()}
    boxes = SimpleNamespace(cls=[idx["Package"], idx["Box_broken"]], conf=[0.9, 0.3])
    assert image_score(SimpleNamespace(boxes=boxes, names=names)) == pytest.approx(0.3)


def test_decide_bands():
    assert decide(0.10, 0.3, 0.7) == "pass"
    assert decide(0.50, 0.3, 0.7) == "review"
    assert decide(0.90, 0.3, 0.7) == "fail"


def test_score_on_threshold_goes_to_stricter_outcome():
    assert decide(0.3, 0.3, 0.7) == "review"  # not pass
    assert decide(0.7, 0.3, 0.7) == "fail"    # not review


def test_equal_thresholds_never_review():
    t = 0.5
    for score in (0.0, 0.25, 0.4999, 0.5, 0.5001, 0.75, 1.0):
        assert decide(score, t, t) in ("pass", "fail")
    assert decide(0.5, t, t) == "fail"
    assert decide(0.4999, t, t) == "pass"


def test_decide_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        decide(0.5, 0.7, 0.3)
    with pytest.raises(ValueError):
        decide(0.5, -0.1, 0.5)
    with pytest.raises(ValueError):
        decide(0.5, 0.5, 1.5)


def test_image_label_positive_and_negative(tmp_path):
    names = [NAMES[i] for i in range(4)]
    pos = tmp_path / "pos.txt"
    pos.write_text("0 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n")
    neg = tmp_path / "neg.txt"
    neg.write_text("0 0.5 0.5 0.2 0.2\n3 0.4 0.4 0.1 0.1\n")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert image_label(pos, names) is True
    assert image_label(neg, names) is False
    assert image_label(empty, names) is False
    assert image_label(tmp_path / "missing.txt", names) is False


def test_image_label_default_names_match_pinned_dataset(tmp_path):
    from src.data import CLASS_NAMES

    broken = tmp_path / "b.txt"
    broken.write_text(f"{CLASS_NAMES.index('Box_broken')} 0.5 0.5 0.5 0.5\n")
    assert image_label(broken) is True


def test_config_classes_are_disjoint_and_cover_dataset():
    from src.data import CLASS_NAMES

    assert not set(POSITIVE_CLASSES) & set(NEGATIVE_CLASSES)
    assert set(POSITIVE_CLASSES) | set(NEGATIVE_CLASSES) == set(CLASS_NAMES)
