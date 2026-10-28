# Assignment 2 Report — Car Damage Inspection

<!-- report-layout: student-english-draft -->

- **Name**: Máximo Garrido Blanco
- **Student ID**: 18994
- **Email**: mgarrido.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

This project considers a rental office handling 100 vehicle returns per day. It uses a photo to decide whether to pass the case, ask an employee to review the photo, or create a provisional damage incident. The current process is physical inspection of every return. The proposed hybrid combines the model with human photo review and incident reporting. It does not include a physical check or automatically charge the customer.

Missing damage is the more expensive mistake. If it is discovered after later rentals, the company may struggle to identify the responsible driver and recover the cost. A false incident mainly wastes staff time. I assume EUR200 average unrecovered loss per missed case and EUR1.67 administration per false incident. These are academic estimates, not measured company costs or rules about customer liability.

One photo represents one return in this prototype. It only shows visible damage; it cannot establish an accident, responsibility, hidden damage or the condition of the whole car.

## Dataset

- **Roboflow workspace / project / version**: `yolov8-z6snb / damage-detect-pzgeu / 1`.
- **Source**: [Damage Detect, version 1](https://universe.roboflow.com/yolov8-z6snb/damage-detect-pzgeu/dataset/1).
- **Licence**: CC BY 4.0.
- **Task and classes**: object detection; `Damage-detected` and `Undamaged`.
- **Export**: original public YOLO export, with images, labels and splits unchanged.

Damage-detected is the positive class. A photo with a damage annotation counts as damaged; one with only Undamaged annotations counts as undamaged. Empty annotations mean unknown condition and are excluded from photo-level cost calculations. All images remain in detection evaluation.

The dataset has no identified multi-camera groups or before/after rental pairs.

## What the dataset's composition implies

| Split | Images | Damage-detected boxes | Undamaged boxes | Photos with known condition |
|---|---:|---:|---:|---:|
| Train | 1,501 | 717 | 895 | 1,501 |
| Validation | 366 | 138 | 272 | 365 |
| Test | 206 | 92 | 119 | 205 |

Validation contains 135 damaged and 230 undamaged photos with known condition; test contains 92 and 113. A photo can contain several boxes, so box and photo counts differ. The median box covers 62.4% of the image for damage and 59.9% for undamaged cars. Some boxes cover much of the vehicle rather than a small damaged area.

The inspection found three related train-validation photo pairs, but no exact file or pixel duplicates across splits. A limited visual-similarity check found no train-test or validation-test candidates. This does not prove that all photos come from different vehicles or sources. One validation and one test photo show vehicles but have empty annotations. Inspected examples include close-up damage photos and commercial-looking undamaged vehicle photos.

Related photos may make validation results look better because the model can recognise patterns similar to ones it has already seen. The size of this effect has not been measured.

The business scenario assumes only 5% of returns have damage. Costs therefore use the error rates for damaged and undamaged photos separately, weighted to this assumed prevalence rather than the dataset's class balance.

## Finetuning and baselines

I compared YOLO11n trained for 10 and 20 epochs with YOLO11s trained for 20 epochs. I selected YOLO11s, using its best validation checkpoint from epoch 19. Each 20-epoch experiment started fresh from the original COCO weights.

The selected run trained on a Tesla T4 in Google Colab, using image size 640, batch size 8, seed 42, two workers, no cache and mosaic disabled. The earlier 10-epoch run was trained locally.

Each model below uses its own minimum-cost validation thresholds, including the cost of staff time:

| Model | Validation mAP50–95 | Hybrid EUR/day | Staff min/day | Damaged photos passed |
|---|---:|---:|---:|---:|
| YOLO11n, 10 epochs | 0.5364 | 33.63 | 100.90 | 0/135 |
| YOLO11n, 20 epochs | 0.5440 | 30.44 | 91.31 | 0/135 |
| YOLO11s, 20 epochs | 0.5322 | 27.59 | 82.78 | 0/135 |

Although YOLO11s has a slightly lower detection score, I accept this trade-off for the business case because its estimated cost and staff workload are lower. These results come from one run per configuration, so the small score differences are not proof of statistical significance. The cost estimates assume a perfect reviewer.

<!-- BEGIN AUTO detection_results -->
Measured secondary evaluation on the same 206 test images:

| Model | AP50, class mean | mAP50–95 |
|---|---:|---:|
| Finetuned YOLO11s, 20 epochs | 0.8151 | 0.5254 |
| Original COCO YOLO11s, zero-shot | 0 | 0 |
| Predict no boxes | 0 | 0 |
<!-- END AUTO detection_results -->

COCO has no classes for damaged or undamaged cars. We tested the original YOLO11s model and checked that its weights matched those used to start training. A prediction labelled `car` was not counted as `Undamaged`. Since it made no predictions for our two target classes, both AP scores were zero and precision could not be calculated. The original predictions and class-matching rules are saved.

Our finetuned model learned to detect damaged and undamaged cars, conditions that COCO could not label with its original categories.

The test set had already been used with an earlier model, so these results are not a new independent test. The current model and hybrid thresholds were fixed before this evaluation. After seeing test results, we changed the assumed missed-damage loss from EUR240 to EUR200 to reflect minor bodywork damage. We repeated the validation sweep, kept the same hybrid thresholds and updated costs using saved test predictions. This cost change did not improve the model's predictions.

## Per-class evaluation and failure modes

<!-- BEGIN AUTO per_class_results -->
Measured secondary test results per box; precision and recall are at the evaluator's F1 point:

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| Damage-detected | 0.8603 | 0.5761 | 0.7170 | 0.3495 |
| Undamaged | 0.8311 | 0.8571 | 0.9132 | 0.7013 |
<!-- END AUTO per_class_results -->

The model is worse at detecting damaged cars. This matters because a damaged car that passes, a false negative, may leave the company to absorb the cost. The assumed loss is EUR200 per missed case.

Box confusion matrix at confidence 0.713699 (rounded) and matching IoU 0.45. Rows are predictions and columns are true labels. The background row shows missed boxes; the background column shows predictions with no matching labelled box:

| Predicted / true | Damage-detected | Undamaged | Background |
|---|---:|---:|---:|
| Damage-detected | 17 | 0 | 2 |
| Undamaged | 6 | 93 | 5 |
| Background / missed | 69 | 26 | 0 |

This cutoff covers automatic incidents. Box metrics differ from final photo decisions because the hybrid also sends intermediate scores to a worker for review. The 205 test photos with known condition followed these initial routes:

| Initial photo route | Damaged | Undamaged |
|---|---:|---:|
| Pass | 2 | 53 |
| Human photo review | 71 | 60 |
| Provisional incident | 19 | 0 |

With the assumed perfect reviewer, the hybrid misses 2/92 damaged photos (2.17%). Both scores are below the lower threshold, so the photos pass without human review. I consider both to be model errors because the damage should be detected.

## Cost matrix and operating point

These inputs are agreed scenario assumptions, not measurements from a rental company:

| Input | Value |
|---|---:|
| Returns/day | 100 |
| Damage prevalence | 5% |
| Staff time value | EUR20/hour |
| Photo review | 1 minute/photo |
| Incident administration | 5 minutes/incident |
| Physical inspection, comparator only | 3 minutes/return |
| Unrecovered loss per missed damage | EUR200 |
| Photo reviewer sensitivity and specificity | 100%, assumed |
| Additional false-incident penalty | EUR0 |

I chose EUR200 because the scenario focuses on minor bodywork damage at normal returns. I assume major accidents follow a different process, potentially involving a tow truck or immediate communication with the company. EUR200 is the average assumed unrecovered loss, applied directly. The original dataset also includes severe damage, so it does not exactly match this narrower scope.

Costs before human photo review:

| True condition | Pass | Provisional incident |
|---|---:|---:|
| Damage | EUR200 missed-damage loss (FN) | EUR1.67 administration (TP) |
| Undamaged | EUR0 (TN) | EUR1.67 unnecessary administration (FP) |

Photo review costs EUR0.33. With a perfect reviewer, a damaged photo then needs EUR1.67 for reporting; an undamaged photo does not. Reporting is administrative work, not another condition check.

A photo's score is its highest Damage-detected confidence, or zero if none is predicted. Scores are not calibrated probabilities. The validation sweep tested 59,340 threshold pairs and selected the lowest expected total cost, including staff time at EUR20/hour. There is no fixed daily time cap.

The chosen thresholds are **0.003750** and **0.713699**, rounded here. Below the lower threshold, the case passes. At or above the lower threshold and below the upper one, an employee reviews the photo. At or above the upper threshold, the model creates a provisional incident. A positive human review also creates an incident.

Since missed damage is expensive, I prioritise human review when there is doubt. This uses more time but allows fewer errors through, saving money. I consider it a good balance because employees still gain time for other duties, such as answering calls and emails. The dashboard shows the cost curves and chosen thresholds.

## What a person still has to check

| Quantity | Validation estimate | Secondary-test estimate |
|---|---:|---:|
| Human photo reviews/day | 57.78 | 54.30 |
| Incidents/day | 5.00 | 4.89 |
| Staff min/day | 82.78 | 78.76 |
| Expected EUR/day | 27.59 | 47.99 |
| Damaged photos passed, observed sample | 0/135 | 2/92 |

In the test sample, 131/205 photos enter review. After weighting to the assumed 5% damage prevalence, this becomes 54.30 reviews per 100 daily returns. Each review takes one minute, plus five minutes to report each true or false incident. Total hybrid work is 78.76 minutes/day, about 1 hour 19 minutes of one employee's time, which could also be shared among staff. This frees approximately 4 hours 6 minutes compared with physical inspection of every return. Work is spread throughout the day; peak arrival times have not been modelled.

Reviewer mistakes can let damage pass and increase costs. I assume 100% sensitivity and specificity as an ideal academic scenario; we have not measured human accuracy.

The main comparison uses the secondary-test rates:

| Operational alternative | Expected EUR/day | Staff min/day | Staff minutes released/day |
|---|---:|---:|---:|
| Physical inspection of every return | 108.33 | 325.00 | 0.00 |
| Model plus doubtful-photo review | 47.99 | 78.76 | 246.24 |
| Model alone | 138.97 | 123.44 | 201.56 |

The model-alone threshold is 0.018087 (rounded), selected by minimum validation cost and then kept fixed on test. Physical inspection assumes all damage is found: 100 × 3 minutes of inspection plus 5 × 5 minutes of reporting gives 325 minutes/day. Staff time has a cost in all three alternatives; time saved is available for other duties.

For the required do-nothing reference, expected loss is EUR1,000/day (100 × 5% × EUR200), with no staff time. This is a technical reference, not one of the three operational options.

## Limitations & next steps

The hybrid's estimated EUR47.99/day includes EUR26.25 of staff time and EUR21.74 of missed-damage loss. The two observed omissions are 2/92 damaged test photos, not two missed returns per day. At 5% prevalence, the expected count is about 0.109 missed returns/day.

Under this project's assumptions, the model could improve cost and staff-time efficiency, freeing employees for other tasks. The value of that time is estimated at EUR20/hour; it does not automatically mean lower payroll or measured cash savings.

For future use, I would propose several cameras taking close-up photos of the car. This has not been implemented or tested, and no rule for combining views has been evaluated. Other limitations are the reused test, related training-validation photos, missing annotations, differences from rental-return photos, ideal human accuracy and assumed costs. The results do not guarantee perfect damage detection.

The professor's checker and dashboard checks passed. A separate one-epoch run on the full dataset verified the full workflow, not a fresh 20-epoch reproduction; the final model and reported results remained unchanged.

## Generative AI use disclosure

I used OpenAI Codex in the following ways:

- **Dataset suitability:** After I selected a dataset, Codex helped me assess whether it was suitable for this assignment, including its classes, size and split quality.
- **Code development:** Codex wrote and adjusted the code following my instructions, decisions and requested changes. This included data preparation, training, evaluation, cost calculations and the Gradio dashboard. At my request, it also proposed scenario estimates and recommended model and threshold options, which I discussed and confirmed. It also helped me understand the metrics and assumptions.
- **Report preparation:** I explained my business problem, finetuning choices, cost assumptions and human-review roles. Codex helped organise and edit those explanations into a clear, well-written and well-structured report.
