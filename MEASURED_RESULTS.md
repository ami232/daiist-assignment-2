# Measured parcel inspection results

Automatically summarised from real saved outputs. Report,  No synthetic values are used as model results.

## Dataset
Public source: https://universe.roboflow.com/recog-ayum6/boxes_status/dataset/2
Split image counts: {'train': 389, 'val': 104, 'test': 52}
Exact cross-split duplicate findings: 1.

- train / box: 1170 instances; median relative box area 0.20246276486878356.
- train / box_damaged: 248 instances; median relative box area 0.16676712036132812.
- train / box_opened: 58 instances; median relative box area 0.22590802400437918.
- val / box: 249 instances; median relative box area 0.23264559093757323.
- val / box_damaged: 85 instances; median relative box area 0.1605548858642578.
- val / box_opened: 15 instances; median relative box area 0.091033935546875.
- test / box: 168 instances; median relative box area 0.19279120771314834.
- test / box_damaged: 27 instances; median relative box area 0.13520336151123047.
- test / box_opened: 8 instances; median relative box area 0.45770729997614257.

## Same-test detection comparison

- finetuned: AP50 0.5226; mAP50–95 0.4259.
- coco_zero_shot: AP50 0.0000; mAP50–95 0.0000.
- predict_nothing: AP50 0.0000; mAP50–95 0.0000.

## Fine-tuned per-class measurements

- {"class": "box", "precision_max_f1": 0.7749663056901197, "recall_max_f1": 0.6547619047619048, "AP50": 0.7505371407065059, "mAP50_95": 0.5820531205081656}
- {"class": "box_damaged", "precision_max_f1": 0.3065358274845008, "recall_max_f1": 0.11491503186928617, "AP50": 0.18254467529010074, "mAP50_95": 0.12168721724944416}
- {"class": "box_opened", "precision_max_f1": 0.5859309375879618, "recall_max_f1": 0.625, "AP50": 0.6348114393463231, "mAP50_95": 0.5740188801269513}

## Bounded repair cycle
Selected candidate for the coursework pipeline using validation only. No operational deployment.
Candidate accepted: True. Held-out test inspected during repair: False.

## Validation-selected policy
{
  "review_policy": {
    "low": 0.12,
    "high": 0.28,
    "base_rate": 0.1,
    "expected_confusion_per_1000": {
      "TN": 741.5999999999999,
      "FP": 158.39999999999998,
      "FN": 1.8593750000000004,
      "TP": 98.14062500000001
    },
    "cost_eur_per_1000": 613.5680729166667,
    "review_share": 0.11968750000000002,
    "review_cases_per_day": 239.37500000000003,
    "review_hours_per_day": 1.994791666666667,
    "people_required_fte": 0.33246527777777785,
    "sensitivity": 0.98140625,
    "specificity": 0.8239999999999998
  },
  "model_only_policy": {
    "low": 0.04,
    "high": 0.04,
    "base_rate": 0.1,
    "expected_confusion_per_1000": {
      "TN": 607.5,
      "FP": 292.50000000000006,
      "FN": 0.0,
      "TP": 100.0
    },
    "cost_eur_per_1000": 785.0000000000001,
    "review_share": 0.0,
    "review_cases_per_day": 0.0,
    "review_hours_per_day": 0.0,
    "people_required_fte": 0.0,
    "sensitivity": 1.0,
    "specificity": 0.675
  },
  "selected_on": "validation only",
  "capacity_share": 0.12,
  "capacity_feasible": true,
  "force_review": "no recognised target box at confidence >= 0.25"
}

## Frozen test policy comparisons

- model_with_review: €804.93 per 1,000 images; 21.79% reviewed; 435.7 reviews/day; 0.605 FTE.
- model_alone: €1100.00 per 1,000 images; 0.00% reviewed; 0.0 reviews/day; 0.000 FTE.
- do_nothing_pass_all: €4000.00 per 1,000 images; 0.00% reviewed; 0.0 reviews/day; 0.000 FTE.
- human_check_everything: €474.17 per 1,000 images; 100.00% reviewed; 2000.0 reviews/day; 2.778 FTE.

## Interpretation boundaries

All business costs and review rates depend on the scenario assumptions. Image-level risk scores are not automatically calibrated probabilities. Cost comparisons are expected scenario calculations rather than observed warehouse profit.
The test set is small and public. Exact duplicates, class imbalance, multiple boxes in one image and a different production camera environment can limit generalisation. The dataset health and per-class results must be considered before any shipping decision.
No operational self-improvement or new-damage recognition is claimed without independently verified feedback and a fresh candidate evaluation. The saved feedback cycle records whether this particular repair attempt succeeded.
