import pytest
from evaluate import box_iou, fixed_threshold_metrics, detection_confusion


def record(detections):
    return {'truth':[[0,.5,.5,.4,.4]],'detections':detections}


def test_iou_identity_and_no_overlap():
    assert box_iou([.1,.1,.3,.3],[.1,.1,.3,.3])==pytest.approx(1)
    assert box_iou([.1,.1,.3,.3],[.6,.6,.8,.8])==0


def test_duplicate_detection_is_not_a_second_true_positive():
    pred={'class':0,'confidence':.9,'xyxy':[.3,.3,.7,.7]}
    result=fixed_threshold_metrics([record([pred,pred])],['Damage','Undamage'],.5)[0]
    assert result['true_positives']==1
    assert result['false_positives']==1
    assert result['recall']==1
    assert result['precision']==.5


def test_wrong_class_appears_off_diagonal_and_is_not_a_damage_hit():
    pred={'class':1,'confidence':.9,'xyxy':[.3,.3,.7,.7]}
    row=fixed_threshold_metrics([record([pred])],['Damage','Undamage'],.5)[0]
    assert row['true_positives']==0
    assert row['false_negatives']==1
    matrix=detection_confusion([record([pred])],['Damage','Undamage'],.5)
    assert matrix[1,0]==1
