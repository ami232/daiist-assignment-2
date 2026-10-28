import numpy as np
import pytest
from policy import Scenario, evaluate_policy, outcomes


def test_manual_cost_and_review_load_example():
    result=evaluate_policy([.05,.2,.4,.95],[False,False,True,True],.1,.8)
    # 450 negatives and 50 positives reviewed per 1,000; 50 positives held directly.
    assert result['cost_per_1000']==pytest.approx(450*2.75+50*17.25+50*5)
    assert result['reviews_per_day']==pytest.approx(50)
    assert result['review_minutes_per_day']==pytest.approx(250)
    assert np.sum(result['expected_binary_per_1000'])==pytest.approx(1000)


def test_all_review_agrees_with_explicit_human_baseline():
    result=evaluate_policy([.2,.4],[False,True],0,1)
    assert result['cost_per_1000']==pytest.approx(4200)
    assert result['review_staff_fte']==pytest.approx(500/360)


def test_production_prevalence_is_not_sample_prevalence():
    for truth in [[False,True],[False,False,False,True]]:
        scores=[.1]*len(truth)
        assert evaluate_policy(scores,truth,.5,.5)['cost_per_1000']==pytest.approx(20000)
        assert evaluate_policy(scores,truth,0,0)['cost_per_1000']==pytest.approx(5000)


def test_exact_threshold_boundary_and_equal_thresholds():
    assert outcomes([.1,.2,.3],.2,.3).tolist()==[0,1,2]
    assert outcomes([.1,.2,.3],.2,.2).tolist()==[0,2,2]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        outcomes([.5],.8,.2)
    with pytest.raises(ValueError):
        outcomes([float('nan')],.2,.8)
    with pytest.raises(ValueError):
        Scenario(prevalence=1.2)
    with pytest.raises(ValueError):
        evaluate_policy([.1,.2],[True,True],.2,.8)
