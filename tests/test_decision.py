import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from decision import evaluate, optimize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/decision_analysis'
POLICY = json.loads((DATA / 'frozen_policy.json').read_text())
A = POLICY['assumptions']


def test_exported_test_results_are_reproduced():
    scores = pd.read_csv(DATA / 'test_scores.csv', float_precision='round_trip')
    r = evaluate(scores, POLICY['lower_threshold'], POLICY['upper_threshold'], A)
    np.testing.assert_array_equal(r['counts'], [[248, 10, 3], [2, 30, 355]])
    assert r['cost_per_1000'] == pytest.approx(120.8590, abs=0.0001)


def test_validation_optimization_reproduces_colab_policy():
    scores = pd.read_csv(DATA / 'validation_scores.csv', float_precision='round_trip')
    selected = optimize(scores, A)
    assert selected['lower_threshold'] == pytest.approx(POLICY['lower_threshold'], abs=1e-12)
    assert selected['upper_threshold'] == pytest.approx(POLICY['upper_threshold'], abs=1e-12)
    r = evaluate(scores, selected['lower_threshold'], selected['upper_threshold'], A)
    assert r['cost_per_1000'] == pytest.approx(112.1684, abs=0.0001)
    frozen = evaluate(scores, POLICY['lower_threshold'], POLICY['upper_threshold'], A)
    assert frozen['cost_per_1000'] == pytest.approx(112.1684, abs=0.0001)


def test_boundary_equality_and_cost_accounting():
    frame = pd.DataFrame({'positive': [0, 0, 1, 1], 'anomaly_score': [0, .5, .2, .8]})
    r = evaluate(frame, .2, .8, A)
    np.testing.assert_array_equal(r['counts'], [[1, 1, 0], [0, 1, 1]])
    # Half of each group is reviewed (€100/1,000); all positives create incidents (€100).
    assert r['cost_per_1000'] == pytest.approx(200)
    with pytest.raises(ValueError):
        evaluate(frame, .8, .2, A)


def test_optimum_matches_exhaustive_small_example():
    frame = pd.DataFrame({'positive': [0, 0, 1, 1], 'anomaly_score': [0, .5, .2, .8]})
    grid = np.unique(np.r_[0, frame.anomaly_score, np.nextafter(.8, np.inf)])
    brute = min((evaluate(frame, lo, hi, A)['cost_per_1000'], lo, hi)
                for lo in grid for hi in grid if lo <= hi)
    selected = optimize(frame, A)
    assert (selected['lower_threshold'], selected['upper_threshold']) == brute[1:]
