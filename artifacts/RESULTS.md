# Results (written by train.py)

## Dataset composition

| class | train | val | test | median box area (% of image, test) |
|---|---|---|---|---|
| Bridge | 894 | 93 | 140 | 4.5% |
| Detached | 1599 | 189 | 274 | 1.8% |
| Double | 375 | 57 | 62 | 2.5% |
| Good | 7017 | 765 | 1108 | 2.0% |
| Insufficient | 657 | 69 | 97 | 1.7% |

- **train**: 3816 images, 2511 contain a defect (66%), 0 unlabelled, 2.8 boxes per image
- **val**: 423 images, 291 contain a defect (69%), 0 unlabelled, 2.8 boxes per image
- **test**: 608 images, 395 contain a defect (65%), 0 unlabelled, 2.8 boxes per image

## Finetuned model vs baselines (same test split)

| model | AP50 | mAP50-95 | image AUC | defect recall @1% false-fail | min cost / 1,000 windows |
|---|---|---|---|---|---|
| yolo11s finetuned | 0.965 | 0.920 | 0.987 | 0.924 | €27.77 |
| yolo11s COCO zero-shot | 0.000 | 0.000 | 0.564 | 0.033 | €142.93 |
| predict nothing (pass everything) | 0.000 | 0.000 | 0.500 | 0.000 | €144.00 |

COCO zero-shot most common labels on solder images: [('bottle', 255), ('person', 85), ('traffic light', 68), ('parking meter', 23), ('sink', 19)]

## Per class (test split)

| class | role | test instances | precision | recall | AP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Bridge | positive (defect) | 140 | 0.964 | 0.958 | 0.986 | 0.938 |
| Detached | positive (defect) | 274 | 0.964 | 0.971 | 0.989 | 0.984 |
| Double | positive (defect) | 62 | 0.921 | 0.935 | 0.972 | 0.872 |
| Good | negative (good) | 1108 | 0.969 | 0.988 | 0.991 | 0.938 |
| Insufficient | positive (defect) | 97 | 0.928 | 0.801 | 0.886 | 0.869 |
| **all** | | | 0.949 | 0.931 | 0.965 | 0.920 |

Box-level confusion matrix at conf ≥ 0.94 (rows = true, columns = predicted):

| true \ pred | Bridge | Detached | Double | Good | Insufficient | background |
|---|---|---|---|---|---|---|
| Bridge | 35 | 0 | 0 | 0 | 0 | 105 |
| Detached | 0 | 218 | 0 | 0 | 0 | 56 |
| Double | 0 | 0 | 10 | 0 | 0 | 52 |
| Good | 0 | 0 | 0 | 384 | 0 | 724 |
| Insufficient | 0 | 2 | 0 | 0 | 4 | 91 |
| background | 0 | 0 | 0 | 0 | 0 | 0 |

Image-level recall by defect type at the band's fail threshold (share of test windows containing that defect that the model fails on its own):

- Bridge: 0.343 of 140 windows
- Detached: 0.824 of 188 windows
- Double: 0.677 of 62 windows
- Insufficient: 0.091 of 88 windows

## Costing

Assumptions: {"fn": 72.0, "fp": 1.5, "tp": 1.5, "tn": 0.0, "review": 0.12, "human_sensitivity": 0.95, "human_false_fail": 0.02, "base_rate": 0.002}

- Best single threshold: **0.87**, €27.77 per 1,000 windows (90% bootstrap interval €12.96–€43.90); recall 0.924, false-fail rate 0.0094
- Best band: pass < **0.40** ≤ review < **0.94** ≤ fail, €12.93 per 1,000, review share 2.44% (90% interval €10.24–€15.96)
- That sends **1952 windows/day** to a person (6.5 h at 12 s each) = **1.00 reviewers** at 6.5 productive h
- Test counts in the band (pos = defective, neg = good): {"pos": {"pass": 8, "review": 193, "fail": 194, "n": 395}, "neg": {"pass": 208, "review": 5, "fail": 0, "n": 213}}

| policy | cost / 1,000 windows | per day |
|---|---|---|
| do nothing (no inspection) | €144.00 | €11,520 |
| person checks every window | €159.99 | €12,799 |
| fail everything | €1500.00 | €120,000 |
| model decides alone | €27.77 | €2,221 |
| model + human review band | €12.93 | €1,035 |

## Sensitivity: what moves the operating point

| assumption | value | best single t | €/1,000 | band lo–hi | €/1,000 | review share |
|---|---|---|---|---|---|---|
| base rate | 0.0005 | 0.87 | 17.50 | 0.59–0.94 | 5.59 | 1.90% |
| base rate | 0.001 | 0.87 | 20.92 | 0.40–0.94 | 8.23 | 2.39% |
| base rate | 0.002 | 0.87 | 27.77 | 0.40–0.94 | 12.93 | 2.44% |
| base rate | 0.005 | 0.59 | 47.13 | 0.40–0.94 | 27.05 | 2.58% |
| base rate | 0.02 | 0.40 | 93.06 | 0.40–0.87 | 78.49 | 1.49% |
| false-negative cost (EUR) | 15 | 0.94 | 16.74 | 0.87–0.94 | 7.14 | 1.02% |
| false-negative cost (EUR) | 36 | 0.87 | 22.30 | 0.40–0.94 | 9.71 | 2.44% |
| false-negative cost (EUR) | 72 | 0.87 | 27.77 | 0.40–0.94 | 12.93 | 2.44% |
| false-negative cost (EUR) | 150 | 0.87 | 39.61 | 0.40–0.94 | 19.90 | 2.44% |
| false-negative cost (EUR) | 300 | 0.40 | 50.23 | 0.40–0.87 | 32.93 | 1.42% |
| false-positive cost (EUR) | 0.5 | 0.59 | 17.01 | 0.40–0.94 | 12.46 | 2.44% |
| false-positive cost (EUR) | 1.5 | 0.87 | 27.77 | 0.40–0.94 | 12.93 | 2.44% |
| false-positive cost (EUR) | 3 | 0.87 | 41.82 | 0.40–0.94 | 13.63 | 2.44% |
| false-positive cost (EUR) | 6 | 0.87 | 69.93 | 0.40–0.94 | 15.04 | 2.44% |

Training settings: {"model": "yolo11s.pt", "epochs": 80, "imgsz": 640, "patience": 20, "seed": 0, "device": "0", "batch": 16, "epochs_run": 80, "best_fitness": 0.89001}