"""Cost and review calculations for saved binary image-decision scores.

Inputs must come from real model predictions. Choose thresholds on validation
data, then evaluate the frozen choice on the held-out test split. A score is
the confidence of the business-positive decision, not an estimated euro loss.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Costs:
    false_positive: float
    false_negative: float
    true_positive: float = 0.0
    true_negative: float = 0.0
    review_per_case: float = 0.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in self.__dict__.values()):
            raise ValueError('All costs must be finite nonnegative euro amounts.')


def policy_metrics(y, scores, low, high, costs, base_rate, *,
                   human_sensitivity, human_specificity, daily_volume,
                   review_seconds, productive_seconds_per_person, force_review=None):
    """Expected confusion matrix and costs, weighted to production prevalence.

    score < low: pass; low <= score < high: human review;
    score >= high: fail. low == high is a model-only policy. Values just above
    one allow pass-all and review-all policies to be represented explicitly.
    Human error rates are assumptions, never inferred from the model scores.
    """
    y=np.asarray(y);scores=np.asarray(scores,dtype=float)
    if y.ndim!=1 or scores.shape!=y.shape or len(y)==0:
        raise ValueError('Use equally sized, nonempty one-dimensional arrays.')
    if set(np.unique(y)) != {0,1}:
        raise ValueError('Both positive and negative examples are required.')
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Scores must be finite and between zero and one.')
    if not 0 <= low <= high <= 1.000001:
        raise ValueError('Require 0 <= low <= high <= 1.000001.')
    if any(not np.isfinite(v) or not 0<=v<=1 for v in
           [base_rate,human_sensitivity,human_specificity]):
        raise ValueError('Prevalence and human accuracy must lie in [0,1].')
    if (not all(np.isfinite(v) for v in [daily_volume,review_seconds,productive_seconds_per_person])
        or daily_volume<0 or review_seconds<0 or productive_seconds_per_person<=0):
        raise ValueError('Invalid workload inputs.')
    positive=y==1;negative=~positive
    forced=np.zeros(len(y),dtype=bool) if force_review is None else np.asarray(force_review,dtype=bool)
    if forced.shape!=y.shape:raise ValueError('Forced-review mask must match the score array.')
    review=((scores>=low)&(scores<high))|forced
    fail=(scores>=high)&~forced;passed=(scores<low)&~forced
    pos=lambda mask:np.mean(mask[positive])
    neg=lambda mask:np.mean(mask[negative])
    tp=base_rate*(pos(fail)+pos(review)*human_sensitivity)
    fn=base_rate*(pos(passed)+pos(review)*(1-human_sensitivity))
    tn=(1-base_rate)*(neg(passed)+neg(review)*human_specificity)
    fp=(1-base_rate)*(neg(fail)+neg(review)*(1-human_specificity))
    review_share=base_rate*pos(review)+(1-base_rate)*neg(review)
    total=(tp*costs.true_positive+fn*costs.false_negative+
           tn*costs.true_negative+fp*costs.false_positive+
           review_share*costs.review_per_case)
    review_daily=daily_volume*review_share
    return {
        'low':float(low),'high':float(high),'base_rate':float(base_rate),
        'expected_confusion_per_1000':{'TN':1000*tn,'FP':1000*fp,'FN':1000*fn,'TP':1000*tp},
        'cost_eur_per_1000':float(1000*total),
        'review_share':float(review_share),'review_cases_per_day':float(review_daily),
        'review_hours_per_day':float(review_daily*review_seconds/3600),
        'people_required_fte':float(review_daily*review_seconds/productive_seconds_per_person),
        'sensitivity':float(tp/base_rate) if base_rate else None,
        'specificity':float(tn/(1-base_rate)) if base_rate<1 else None,
    }


def choose_on_validation(y_val,scores_val,costs,base_rate,*,grid=None,max_review_share=1,**workload):
    """Search a documented grid on validation data only, respecting review capacity."""
    if not 0<=max_review_share<=1:raise ValueError('Invalid review capacity.')
    if grid is None:grid=np.r_[np.linspace(0,1,51),1.000001]
    grid=np.unique(np.asarray(grid,dtype=float))
    records=[]
    for low in grid:
        for high in grid[grid>=low]:
            row=policy_metrics(y_val,scores_val,low,high,costs,base_rate,**workload)
            if row['review_share']<=max_review_share+1e-12:records.append(row)
    if not records:raise ValueError('No policy satisfies the supplied constraints.')
    # Prefer fewer reviews, then deterministic thresholds, when cost is tied.
    best=min(records,key=lambda r:(r['cost_eur_per_1000'],r['review_share'],r['low'],r['high']))
    return best,records
