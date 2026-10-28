"""The dashboard must load committed weights once, never train, never write runs/."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _snapshot(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*")} if path.is_dir() else set()


def test_import_app_trains_nothing_and_writes_no_runs():
    runs_before = _snapshot(ROOT / "runs")
    weights_before = _snapshot(ROOT / "weights")
    import app  # noqa: F401  (module-level load happens here)

    assert _snapshot(ROOT / "runs") == runs_before
    assert _snapshot(ROOT / "weights") == weights_before
    assert app.MODEL_LOADS == 1
    assert app.MODEL is not None
    import gradio as gr
    assert isinstance(app.demo, gr.Blocks)


def test_app_source_has_no_training_or_download_calls():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.search(r"\.train\(", src)
    # Call-like forms only; prose in docstrings may say "downloaded".
    assert not re.search(r"\.hub\b|download\w*\(|urlopen\(|requests\.(get|post)\(|attempt_download", src)
    assert "weights/best.pt" in src.replace('"weights" / "best.pt"', "weights/best.pt")


def test_app_numbers_match_report():
    """The Thresholds tab at the defaults must reproduce review_band.py's chosen-pair cost."""
    import json

    import app
    from src import config

    _, md, table, load = app.recompute(config.CHOSEN_T_LOW, config.CHOSEN_T_HIGH,
                                       config.BASE_RATE, config.COST_FALSE_CLAIM_DOP)
    summary = json.loads((ROOT / "outputs" / "review_summary.json").read_text())
    chosen = summary["chosen"]
    assert f"{chosen['cost_dop_per_1000']:,.0f} DOP" in md
    assert list(table["truth"]) == ["intact", "damaged"]
    assert int(table.loc[table["truth"] == "intact", "fail (test n)"].iloc[0]) == 0
    assert len(load) == len(config.CONTAINER_SIZES)
    assert set(load["fits_claim_window"]) == {"yes"}


def test_app_clamps_thresholds():
    import app

    assert app.clamp_pair(0.6, 0.4) == (0.6, 0.6)
    assert app.clamp_pair(0.2, 0.9) == (0.2, 0.9)


def test_inspect_runs_on_a_committed_image():
    """The Inspect tab path: score, decision and drawn boxes on a known failure image."""
    import cv2

    import app
    from src import config

    path = sorted((ROOT / "outputs" / "failures").glob("*.jpg"))[0]
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    out, md = app.inspect_image(rgb, 0.25, config.CHOSEN_T_LOW, config.CHOSEN_T_HIGH, config.COST_FALSE_CLAIM_DOP)
    assert out.shape == rgb.shape
    assert "Image score:" in md and "Decision at" in md
    assert any(f"`{d}`" in md for d in ("PASS", "REVIEW", "FAIL"))


def test_static_tabs_build():
    import app

    assert app.cost_curve(0.2, 0.94, 0.01).ndim == 3
    assert len(app.failure_gallery()) == 5
    assert len(app.per_class_table()) >= 4
    il = app.image_level_table(0.2, 0.94)
    assert set(il.columns) >= {"rule", "TPR", "FPR", "FPR Box", "FPR Package"}
