# Numbers produced by train.py

Run settings: {"model": "yolo11n.pt", "epochs": 30, "imgsz": 416, "batch": 8, "seed": 0, "device": "auto", "machine": "Windows-11-10.0.26200-SP0", "roboflow": "artificial-intelligence-82oex/detecting-diseases v5", "trained_at": "2026-10-27 15:24", "minutes": 15.3}

## Class grouping used for the decision

| class | group |
|---|---|
| Beans Angular LeafSpot | fungal (positive) |
| Beans Rust | fungal (positive) |
| Strawberry Angular LeafSpot | non-fungal (negative) |
| Strawberry Anthracnose Fruit Rot | fungal (positive) |
| Strawberry Blossom Blight | fungal (positive) |
| Strawberry Gray Mold | fungal (positive) |
| Strawberry Leaf Spot | fungal (positive) |
| Strawberry Powdery Mildew Fruit | fungal (positive) |
| Strawberry Powdery Mildew Leaf | fungal (positive) |
| Tomato Blight | fungal (positive) |
| Tomato Leaf Mold | fungal (positive) |
| Tomato Spider Mites | non-fungal (negative) |

## Dataset

| split | images | images with fungal | share fungal | images with both groups | images with no box | median box area | boxes > 25% of image |
|---|---|---|---|---|---|---|---|
| train | 2904 | 2372 | 0.820 | 0 | 0 | 11.9% | 28% |
| valid | 1416 | 1151 | 0.810 | 0 | 0 | 11.3% | 25% |
| test | 1173 | 933 | 0.800 | 0 | 0 | 12.0% | 27% |

Instances per class:

| class | train | valid | test | group |
|---|---|---|---|---|
| Beans Angular LeafSpot | 956 | 277 | 147 | fungal (positive) |
| Beans Rust | 1652 | 463 | 243 | fungal (positive) |
| Strawberry Angular LeafSpot | 226 | 193 | 295 | non-fungal (negative) |
| Strawberry Anthracnose Fruit Rot | 166 | 85 | 132 | fungal (positive) |
| Strawberry Blossom Blight | 258 | 161 | 211 | fungal (positive) |
| Strawberry Gray Mold | 207 | 200 | 213 | fungal (positive) |
| Strawberry Leaf Spot | 821 | 779 | 698 | fungal (positive) |
| Strawberry Powdery Mildew Fruit | 278 | 151 | 134 | fungal (positive) |
| Strawberry Powdery Mildew Leaf | 669 | 584 | 607 | fungal (positive) |
| Tomato Blight | 352 | 100 | 41 | fungal (positive) |
| Tomato Leaf Mold | 353 | 101 | 35 | fungal (positive) |
| Tomato Spider Mites | 358 | 99 | 31 | non-fungal (negative) |

## Detection metrics on the test split (finetuned yolo11n)

| class | group | precision | recall | AP50 | mAP50-95 |
|---|---|---|---|---|---|
| Beans Angular LeafSpot | fungal | 0.795 | 0.748 | 0.803 | 0.569 |
| Beans Rust | fungal | 0.848 | 0.765 | 0.882 | 0.606 |
| Strawberry Angular LeafSpot | non-fungal | 0.887 | 0.641 | 0.754 | 0.553 |
| Strawberry Anthracnose Fruit Rot | fungal | 0.636 | 0.500 | 0.441 | 0.306 |
| Strawberry Blossom Blight | fungal | 0.919 | 0.536 | 0.527 | 0.432 |
| Strawberry Gray Mold | fungal | 0.921 | 0.446 | 0.599 | 0.355 |
| Strawberry Leaf Spot | fungal | 0.846 | 0.840 | 0.897 | 0.816 |
| Strawberry Powdery Mildew Fruit | fungal | 0.515 | 0.896 | 0.774 | 0.587 |
| Strawberry Powdery Mildew Leaf | fungal | 0.824 | 0.635 | 0.785 | 0.644 |
| Tomato Blight | fungal | 0.758 | 0.991 | 0.772 | 0.732 |
| Tomato Leaf Mold | fungal | 0.663 | 0.971 | 0.725 | 0.632 |
| Tomato Spider Mites | non-fungal | 0.573 | 0.968 | 0.626 | 0.561 |

Overall: precision 0.765, recall 0.745, mAP50 0.715, mAP50-95 0.566

Confusion matrix (rows = predicted, columns = true; conf 0.25):

| pred \ true | Beans Angular LeafSpot | Beans Rust | Strawberry Angular LeafSpot | Strawberry Anthracnose Fruit Rot | Strawberry Blossom Blight | Strawberry Gray Mold | Strawberry Leaf Spot | Strawberry Powdery Mildew Fruit | Strawberry Powdery Mildew Leaf | Tomato Blight | Tomato Leaf Mold | Tomato Spider Mites | background |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Beans Angular LeafSpot | 32 | 63 | 5 | 0 | 1 | 9 | 78 | 14 | 11 | 5 | 0 | 5 | 3299 |
| Beans Rust | 39 | 97 | 1 | 0 | 9 | 8 | 43 | 4 | 64 | 0 | 0 | 3 | 4014 |
| Strawberry Angular LeafSpot | 15 | 1 | 139 | 2 | 0 | 15 | 155 | 6 | 50 | 1 | 1 | 0 | 6413 |
| Strawberry Anthracnose Fruit Rot | 0 | 0 | 1 | 31 | 4 | 42 | 7 | 29 | 45 | 1 | 0 | 1 | 2768 |
| Strawberry Blossom Blight | 0 | 0 | 0 | 0 | 73 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 204 |
| Strawberry Gray Mold | 0 | 0 | 7 | 9 | 6 | 44 | 4 | 21 | 6 | 0 | 0 | 1 | 1463 |
| Strawberry Leaf Spot | 26 | 25 | 13 | 3 | 10 | 20 | 297 | 0 | 69 | 2 | 4 | 3 | 9676 |
| Strawberry Powdery Mildew Fruit | 7 | 5 | 15 | 12 | 3 | 48 | 0 | 59 | 15 | 0 | 0 | 0 | 4493 |
| Strawberry Powdery Mildew Leaf | 2 | 45 | 3 | 5 | 6 | 10 | 27 | 0 | 249 | 0 | 0 | 0 | 6390 |
| Tomato Blight | 16 | 1 | 8 | 5 | 0 | 1 | 46 | 0 | 15 | 20 | 9 | 7 | 1292 |
| Tomato Leaf Mold | 0 | 1 | 46 | 6 | 0 | 1 | 9 | 0 | 40 | 9 | 13 | 5 | 1131 |
| Tomato Spider Mites | 9 | 4 | 4 | 1 | 2 | 12 | 15 | 0 | 33 | 3 | 7 | 6 | 3071 |
| background | 1 | 1 | 53 | 58 | 97 | 2 | 17 | 0 | 9 | 0 | 1 | 0 | 0 |

## Case-level baselines on the same test split (positive = photo shows a fungal disease)

| model | test precision | test recall | test F1 | TP | FP | FN | TN | note |
|---|---|---|---|---|---|---|---|---|
| Trivial: predict nothing (never fungal) | 0.000 | 0.000 | 0.000 | 0 | 0 | 933 | 240 | every fungal case is missed |
| Trivial: majority class (always fungal) | 0.795 | 1.000 | 0.886 | 933 | 240 | 0 | 0 | precision = test base rate 0.80 |
| COCO yolo11n zero-shot (any 'potted plant' >= 0.25) | 0.895 | 0.018 | 0.036 | 17 | 2 | 916 | 238 | COCO has no disease classes; AP on target classes = 0 by construction |
| Finetuned yolo11n (fungal >= 0.24) | 0.981 | 0.974 | 0.977 | 909 | 18 | 24 | 222 | case-level AUC = 0.956 |

## Cost model

```json
{
  "POSITIVE_NAME": "fungal disease present",
  "NEGATIVE_NAME": "not fungal (pest / bacterial)",
  "C_FN": 60.0,
  "C_FP": 12.0,
  "C_TP": 9.0,
  "C_TN": 0.0,
  "C_REVIEW": 1.5,
  "REVIEW_ACCURACY": 0.95,
  "BASE_RATE": 0.4,
  "CASES_PER_DAY": 300,
  "REVIEW_SECONDS": 120,
  "REVIEWER_HOURS_PER_DAY": 1.0,
  "REVIEWERS_AVAILABLE": 1
}
```

## Threshold sweep (model alone, cost per 1000 production cases)

| threshold | cost_per_1000 | TPR | FPR |
|---|---|---|---|
| 0.010 | 5267.299 | 0.979 | 0.171 |
| 0.080 | 4831.029 | 0.976 | 0.104 |
| 0.150 | 4741.029 | 0.976 | 0.092 |
| 0.220 | 4694.759 | 0.974 | 0.079 |
| 0.290 | 4730.354 | 0.971 | 0.075 |
| 0.360 | 4817.814 | 0.967 | 0.075 |
| 0.430 | 4853.408 | 0.964 | 0.071 |
| 0.500 | 4837.138 | 0.961 | 0.062 |
| 0.570 | 5121.383 | 0.947 | 0.062 |
| 0.640 | 5280.032 | 0.937 | 0.054 |
| 0.710 | 5673.601 | 0.917 | 0.054 |
| 0.780 | 6067.170 | 0.898 | 0.054 |
| 0.850 | 7575.852 | 0.824 | 0.054 |
| 0.920 | 11590.868 | 0.626 | 0.050 |
| 0.990 | 23846.945 | 0.008 | 0.000 |

Minimum: threshold 0.24 -> EUR 4665 per 1000 cases

## Chosen operating point (model + review band)

- t_low = 0.20, t_high = 0.62
- cost per 1000 cases: EUR 4565
- share sent for review: 3.2%  (9 photos/day, 0.32 h/day, 0.32 people at 1.0 h each)
- unconstrained optimum would be [0.20, 0.62) at EUR 4565 with 3.2% reviewed (0.32 people)

Expected confusion matrix per 1000 production cases after review:

TP 389.9 | FN 10.1 | FP 33.4 | TN 566.6

## Priced against the alternatives

| design | cost_per_1000 | review_share | people_needed | cost_per_day |
|---|---|---|---|---|
| Do nothing (never spray on the scout's photo) | 24000.00 | 0.00 | 0.00 | 7200.00 |
| Agronomist checks every photo | 6480.00 | 1.00 | 10.00 | 1944.00 |
| Model alone (t = 0.24) | 4664.76 | 0.00 | 0.00 | 1399.43 |
| Model + review band [0.20, 0.62) | 4565.08 | 0.03 | 0.32 | 1369.52 |

## Sensitivity of the optimum

| C_FN | base rate | best threshold | cost/1000 |
|---|---|---|---|
| 15 | 0.2 | 0.62 | 2391 |
| 15 | 0.4 | 0.62 | 4131 |
| 15 | 0.7 | 0.24 | 6678 |
| 30 | 0.2 | 0.50 | 2562 |
| 30 | 0.4 | 0.24 | 4356 |
| 30 | 0.7 | 0.20 | 6947 |
| 60 | 0.2 | 0.24 | 2782 |
| 60 | 0.4 | 0.24 | 4665 |
| 60 | 0.7 | 0.20 | 7442 |
| 150 | 0.2 | 0.24 | 3245 |
| 150 | 0.4 | 0.20 | 5530 |
| 150 | 0.7 | 0.07 | 8897 |
