"""
Turn detector output into one score per image and one of three decisions.

Score rule (src/config.py, SCORE_RULE): the maximum confidence over detections
whose class name is in POSITIVE_CLASSES; 0.0 when there is none.

Decision rule, for thresholds 0 <= t_low <= t_high <= 1:

    score <  t_low             -> "pass"    (released)
    t_low <= score < t_high    -> "review"  (full unpack by a person)
    score >= t_high            -> "fail"    (set aside, photo as claim evidence)

A score exactly on a threshold goes to the stricter outcome. With
t_low == t_high there is no review band.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import POSITIVE_CLASSES  # noqa: E402

Decision = str  # "pass" | "review" | "fail"
DECISIONS = ("pass", "review", "fail")


def _as_list(values) -> list:
    """Tensor, numpy array or plain sequence -> python list."""
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [values]
    return list(values)


def _iter_results(results) -> Iterable:
    """Accept one Ultralytics Results object or a list of them."""
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return results
    return [results]


def image_score(results) -> float:
    """Max confidence over positive-class detections; 0.0 if there are none.

    `results` is an Ultralytics `Results` object (or a list of them for the
    same image). Positive classes are resolved by name through `results.names`,
    so the rule does not depend on class index order.
    """
    best = 0.0
    for res in _iter_results(results):
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        names = getattr(res, "names", None) or {}
        classes = _as_list(getattr(boxes, "cls", None))
        confs = _as_list(getattr(boxes, "conf", None))
        for cls, conf in zip(classes, confs):
            name = names.get(int(cls), names.get(str(int(cls)), None))
            if name in POSITIVE_CLASSES:
                best = max(best, float(conf))
    return best


def decide(score: float, t_low: float, t_high: float) -> Decision:
    """Three-way decision. Ties go to the stricter outcome."""
    if not 0.0 <= t_low <= t_high <= 1.0:
        raise ValueError(f"need 0 <= t_low <= t_high <= 1, got t_low={t_low}, t_high={t_high}")
    if score >= t_high:
        return "fail"
    if score >= t_low:
        return "review"
    return "pass"


def image_label(label_file, names: list[str] | None = None) -> bool:
    """True if any box in a YOLO label file belongs to POSITIVE_CLASSES.

    `names` maps class index to class name, in data.yaml order. Defaults to
    the pinned dataset's names from src/data.py. A missing or empty file is
    negative.
    """
    if names is None:
        from src.data import CLASS_NAMES

        names = CLASS_NAMES
    path = Path(label_file)
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if 0 <= cls < len(names) and names[cls] in POSITIVE_CLASSES:
            return True
    return False
