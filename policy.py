"""Image-level operating costs. Detection AP is evaluated separately.

A positive case is a photo with at least one original Damage annotation.
Detector confidence is a score, not a calibrated probability of damage.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Scenario:
    prevalence: float = .10
    missed_cost: float = 200.
    inspection_cost: float = 5.
    review_minutes: float = 5.
    hourly_cost: float = 30.
    cases_per_day: int = 100
    productive_minutes: float = 360.
    closer_inspection_minutes: float = 10.
    human_sensitivity: float = .95
    human_specificity: float = .95

    def __post_init__(self):
        for value in [self.prevalence,self.human_sensitivity,self.human_specificity]:
            if not 0 <= value <= 1:
                raise ValueError('Prevalence and human accuracy assumptions must lie between 0 and 1.')
        if min(self.missed_cost,self.inspection_cost,self.review_minutes,self.hourly_cost,self.cases_per_day,self.closer_inspection_minutes)<0:
            raise ValueError('Costs, time and daily volume must be non-negative.')
        if self.productive_minutes<=0:
            raise ValueError('Productive minutes must be positive.')

    @property
    def review_cost(self):
        return self.review_minutes * self.hourly_cost / 60


def outcomes(scores, lower, upper):
    if not 0 <= lower <= upper <= 1:
        raise ValueError('Use 0 <= pass/review threshold <= review/fail threshold <= 1.')
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Scores must be finite numbers between zero and one.')
    return np.where(scores < lower, 0, np.where(scores < upper, 1, 2))


def evaluate_policy(scores, truth, lower, upper, scenario=Scenario()):
    """Return conditional rates reweighted to the assumed production base rate."""
    truth = np.asarray(truth, dtype=bool)
    action = outcomes(scores, lower, upper)
    if len(action) != len(truth) or not truth.any() or truth.all():
        raise ValueError('Both damaged and non-damaged images are needed to estimate costs.')
    actual = np.array([np.bincount(action[~truth], minlength=3),
                       np.bincount(action[truth], minlength=3)])
    conditional = actual / actual.sum(axis=1, keepdims=True)
    weights = np.array([1-scenario.prevalence, scenario.prevalence])
    production = conditional * weights[:, None]
    # Reviewers may make mistakes; their accuracy is a scenario, not measured here.
    negative_costs = [0, scenario.review_cost + (1-scenario.human_specificity)*scenario.inspection_cost,
                      scenario.inspection_cost]
    positive_costs = [scenario.missed_cost,
                      scenario.review_cost + (1-scenario.human_sensitivity)*scenario.missed_cost
                      + scenario.human_sensitivity*scenario.inspection_cost,
                      scenario.inspection_cost]
    costs = np.array([negative_costs, positive_costs])
    review_share = float(production[:,1].sum())
    minutes = review_share * scenario.cases_per_day * scenario.review_minutes
    # Binary result after assumed human decisions; fractional cells are expectations.
    tn = production[0,0] + production[0,1]*scenario.human_specificity
    fp = production[0,2] + production[0,1]*(1-scenario.human_specificity)
    fn = production[1,0] + production[1,1]*(1-scenario.human_sensitivity)
    tp = production[1,2] + production[1,1]*scenario.human_sensitivity
    closer_minutes=float(fp+tp)*scenario.cases_per_day*scenario.closer_inspection_minutes
    return {'lower':float(lower), 'upper':float(upper),
            'cost_per_1000': float((production*costs).sum()*1000),
            'review_share':review_share, 'reviews_per_day':review_share*scenario.cases_per_day,
            'review_minutes_per_day':minutes, 'review_staff_fte':minutes/scenario.productive_minutes,
            'closer_inspections_per_day':float(fp+tp)*scenario.cases_per_day,
            'closer_inspection_minutes_per_day':closer_minutes,
            'total_inspection_staff_fte':(minutes+closer_minutes)/scenario.productive_minutes,
            'actual_triage_counts':actual.tolist(),
            'expected_triage_per_1000':(production*1000).tolist(),
            'expected_binary_per_1000':(np.array([[tn,fp],[fn,tp]])*1000).tolist()}


def threshold_sweep(scores, truth, scenario=Scenario()):
    return [evaluate_policy(scores,truth,t,t,scenario) for t in np.linspace(0,1,101)]


def choose_policy(scores, truth, scenario=Scenario()):
    """Select on validation only, using an explicit one-percentage-point grid."""
    curve = threshold_sweep(scores,truth,scenario)
    best_single = min(curve, key=lambda p:(p['cost_per_1000'],p['lower']))
    candidates = (evaluate_policy(scores,truth,lo,hi,scenario)
                  for lo in np.linspace(0,1,101) for hi in np.linspace(0,1,101) if lo < hi)
    best_triage = min(candidates,key=lambda p:(p['cost_per_1000'],p['review_share'],p['lower'],p['upper']))
    return {'scenario':asdict(scenario), 'single':best_single, 'triage':best_triage,
            'selection_split':'valid', 'grid_step':.01, 'curve':curve}


def alternatives(scores,truth,choice,scenario=Scenario()):
    return {
        'Do nothing (pass every photo)':1000*scenario.prevalence*scenario.missed_cost,
        'Human review of every photo':1000*(scenario.review_cost
            + scenario.prevalence*(scenario.human_sensitivity*scenario.inspection_cost
                                  +(1-scenario.human_sensitivity)*scenario.missed_cost)
            + (1-scenario.prevalence)*(1-scenario.human_specificity)*scenario.inspection_cost),
        'Closer inspection of every return':1000*scenario.inspection_cost,
        'Model alone':evaluate_policy(scores,truth,choice['single']['lower'],choice['single']['upper'],scenario)['cost_per_1000'],
        'Model with human review':evaluate_policy(scores,truth,choice['triage']['lower'],choice['triage']['upper'],scenario)['cost_per_1000']}
