# Assignment 2 Report

- **Name**: Celia Sanmartín
- **Student ID**:18798
- **Email**: csanmartin.ieu2022@student.ie.edu
- **Group**: B

## Business problem, and the cost of being wrong

The model supports one decision per electronic board coming off an assembly line: **pass**, **send to a person**, or **fail** (divert for rework). A camera photographs every board, the model detects defects, and the highest detection confidence on a board is its score. The person who acts on it is a quality operator or the system that diverts the board.

A board is positive if it has **at least one defect** of any kind (solder defects or bare-board defects such as shorts or missing holes), because any of them makes the board unfit to ship. I chose this over "solder defects only" because the solder classes in this dataset are thin once the test set is cleaned (see below), so a board-level definition rests on far more evidence.

The two mistakes are very unequal. A **false negative** is a defective board that reaches the customer: replacement, repair in service, return logistics, warranty and reputation. I assumed **150 EUR**, roughly 15-20 times the cost of fixing the same board in the factory. A **false positive** is a good board that gets rejected: reinspection and, sometimes, a scrapped or damaged board. I assumed **12 EUR**. So a miss costs about 12 times a false alarm, which is why the operating point sits well below what a symmetric cost would give. Scenario: industrial control boards, about 5,000 boards a day.

## Dataset

- Roboflow Universe: workspace `pcb-defect-detection-emmts`, project `pcb-defect-detection-datasets-whwcy`, **version 5**.
- https://universe.roboflow.com/pcb-defect-detection-emmts/pcb-defect-detection-datasets-whwcy (public dataset used as is, not forked).
- Object detection, 9 classes: Cold_solder, Excessive_solder, Insufficient_solder (solder) and missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper (bare board). One image is one photographed board.
- It fits the problem because it labels exactly the defect types a final inspection looks for, on board photographs with bounding boxes. I rejected another candidate (`pcb-deffect-detection-solder-sthr7`) because it has no published dataset version and has duplicated class names.

## What the dataset's composition implies

| Class | Train instances | Test (original) | Test (clean) | Mean defect area, % of image (train) |
|---|---|---|---|---|
| Cold_solder | 1,191 | 89 | **48** | 1.91 |
| Excessive_solder | 6,405 | 708 | **65** | 0.16 |
| Insufficient_solder | 16,122 | 1,637 | 238 | 0.80 |
| missing_hole | 6,585 | 754 | 754 | 0.25 |
| mouse_bite | 6,402 | 754 | 754 | 0.21 |
| open_circuit | 6,594 | 662 | 662 | 0.14 |
| short | 6,342 | 660 | 660 | 0.41 |
| spur | 6,783 | 671 | 671 | 0.28 |
| spurious_copper | 6,462 | 781 | 781 | 0.29 |

Splits: train 30,189 / valid 3,363 / test 3,324 images; 5,889 / 609 / 654 of them contain no defect. Defects are tiny (0.14-0.41% of the image for most classes), which is hard for a small detector.

**Data leakage.** Comparing file names (ignoring Roboflow's suffix and `_rotated`) showed that part of the test set also appears in train, and three pairs I opened looked like the same board. I removed those images, leaving a **clean test set of 2,301 images** (2,217 with defects, 84 without). The leak is concentrated in the solder classes: Excessive_solder fell from 708 to 65 test instances and Insufficient_solder from 1,637 to 238. `valid` overlaps with train too, so I did not tune anything on it.

What this means for the results:
- **Cold_solder (48) and Excessive_solder (65) are too thin to trust**; their metrics have wide error bars.
- **Only 84 good boards** are in the clean test set, so any false-positive rate has a large uncertainty.
- The clean test set is **96% defective boards**, nothing like production, where I assume 3%. Accuracy on it is misleading and cost figures must be reweighted by an assumed base rate.
- The leak was found by file name and three visual checks; undetected near-duplicates may remain.

## Finetuning and baselines

Model: **yolo11n**, object detection (the decision is about where defects are, and the board score comes from the detections). Trained on Google Colab (T4 GPU), 8 epochs, image size 640, batch 16, seed 0, all training images, other settings Ultralytics defaults. Weights in `weights/best.pt`.

On the clean test set (2,301 images, 4,633 instances): precision 0.867, recall 0.813, **AP50 0.843**, mAP50-95 0.463.

Board level (positive = board with at least one defect), clean test set, threshold 0.25:

| Model | TP | FP | FN | TN | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Finetuned yolo11n | 2,158 | 2 | 59 | 82 | 0.973 | 0.999 | 0.973 |
| COCO zero-shot yolo11n | 155 | 23 | 2,062 | 61 | 0.094 | 0.871 | 0.070 |
| Trivial: everything defective | 2,217 | 84 | 0 | 0 | 0.963 | 0.963 | 1.000 |
| Trivial: detect nothing | 0 | 0 | 2,217 | 84 | 0.037 | n/a | 0.000 |

The finetuned model is far better than COCO, which finds something on only 7% of defective boards because it has never seen these classes. Against the trivial "everything defective" baseline, accuracy barely moves (0.973 vs 0.963). That is because 96% of the clean test boards are defective, so accuracy is the wrong yardstick here. The real difference is in false positives: the trivial baseline rejects all 84 good boards, the model only 2, at the price of 59 missed defective boards. Which one is better depends entirely on the cost matrix below.

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| Cold_solder (n=48) | 0.811 | 0.714 | 0.767 | 0.606 |
| Excessive_solder (n=65) | 0.640 | 0.338 | 0.408 | 0.261 |
| Insufficient_solder (n=238) | 0.564 | 0.408 | 0.512 | 0.296 |
| missing_hole | 0.984 | 0.989 | 0.991 | 0.547 |
| mouse_bite | 0.976 | 0.983 | 0.992 | 0.494 |
| open_circuit | 0.967 | 0.974 | 0.975 | 0.464 |
| short | 0.973 | 0.971 | 0.979 | 0.525 |
| spur | 0.943 | 0.960 | 0.974 | 0.480 |
| spurious_copper | 0.943 | 0.976 | 0.984 | 0.491 |

Precision and recall per class are at Ultralytics' best-F1 point, not at my operating threshold. The six bare-board classes (AP50 0.97-0.99) carry the headline AP50 of 0.843. The three solder classes drag it down, especially Excessive_solder (recall 0.34) and Insufficient_solder (recall 0.41). Insufficient_solder had AP50 0.94 on `valid`, which overlaps with train, and only 0.51 on the clean test set, so the leak was inflating the solder results. The confusion matrix is in `outputs/val_finetuned/confusion_matrix.png`. The confusion matrix (computed at a confidence of 0.001, so its "background" column is full of very low-confidence boxes and says little about my operating threshold) shows that the solder classes are mostly confused with each other rather than missed: of 65 true Excessive_solder instances, 34 were predicted correctly and 27 as Insufficient_solder (only 4 were missed as background), and 4 of 48 Cold_solder instances were also predicted as Insufficient_solder. Insufficient_solder itself is mostly right (219 of 238). The six bare-board classes sit almost entirely on the diagonal. So part of the low solder recall in the table comes from the model mixing up the solder defect types, not only from failing to see them; for a board-level pass/fail decision that mix-up matters less than a miss, because either label sends the board to rework.

## Cost matrix and operating point

| Item | Value | Where it comes from |
|---|---|---|
| False negative | 150 EUR | Replacement (~40), service diagnosis and repair (~30), return logistics (~20), warranty and reputation (~60). Assumption, not measured. |
| False positive | 12 EUR | Reinspection (~2) plus a scrapped or damaged board in about 1 in 5 cases (~8-10). Assumption. |
| True positive (rework in the factory) | 8 EUR | Fixing a detected defect before shipping. Assumption. |
| True negative | 0 EUR | No cost. |
| One human review | 0.50 EUR | 1 minute at ~30 EUR/hour loaded cost; one minute because the defects are tiny. Reviewers are assumed always right. |
| Base rate of defective boards | 3% | Assumption for production, not a measurement. |

I scored each board by its highest detection confidence and swept one threshold over the clean test set, reweighting good and defective boards by the 3% base rate. The cost-minimising single threshold is **0.45**, at **407 EUR per 1,000 boards**.

Sensitivity (cost with one assumption changed at a time):
- **False-negative cost.** At 30 EUR the review zone adds nothing; at 75 EUR it starts to pay; at 300-600 EUR the zone opens down to 0.02, 31% of boards go to review and about 3.7 people are needed.
- **Base rate.** At 0.5% a review zone does not pay; at 1% barely; at 3% or more it does. At 10%, about 3.5 reviewers.
- **False-positive cost** changes nothing, because none of the 84 good boards scores 0.45 or higher.

The assumption that would change my answer most is the **false-negative cost together with the 3% base rate**. A second, subtler weakness: the optimal threshold 0.45 sits exactly where the first good board would appear, and with only 84 good boards the exact value is an artefact of the sample. What is robust is the conclusion that the model is worth using and that review pays off only when misses are expensive and defects not rare.

## What a person still has to check

Thresholds: **pass below 0.18, human review from 0.18 to 0.46, fail from 0.46 up.** This sends **7.0% of boards** to a person, about **349 boards a day**, at 1 minute each, which needs **0.8 people** (420 useful minutes per person per day). I assume the plant has 2 quality staff, so this is realistic, with slack for the review load to grow.

Cost per 1,000 boards of each design:

| Design | EUR per 1,000 boards |
|---|---|
| Do nothing (everything passes) | 4,500 |
| A person checks everything | 740 (about 12 people, not realistic) |
| Model alone (threshold 0.45) | 407 |
| Model + human review (0.18-0.46) | 363 |

**Chosen design: model plus human review.** It costs about 44 EUR less per 1,000 boards than the model alone (roughly 220 EUR a day at 5,000 boards) for less than one person, and it gives a safety net on uncertain boards. The advantage is modest and shrinks fast if misses are cheaper than I assumed; the cases the zone catches are boards a human would also have to judge.

## Limitations & next steps

- **Threshold chosen and evaluated on the same clean test set**, so it is optimistic. With more time I would build a clean validation set (no near-duplicates) to choose thresholds and keep the test set for the final number.
- **Only 84 good boards.** Zero false positives out of 84 does not show there are none; the true rate could be around 3-4%, which on about 4,850 good boards a day would mean hundreds of false alarms. I would collect a few thousand good boards from the real line before trusting any threshold.
- **Human reviewers are modelled as perfect.** With defects at 0.1-0.4% of the image they will miss some. I would measure reviewer accuracy and add it to the cost model.
- **The 3% base rate and all euro figures are assumptions.** With more time I would get the real defect rate and rework and warranty costs from the plant.
- **Near-duplicate detection used file names and three visual checks.** I would use an image-similarity measure to find duplicates the names hide.
- **Small detector, short training, tiny defects.** I would try yolo11s, a larger image size and more epochs, and look at the solder classes first.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment. My instructor told the group on 26 October 2026 (WhatsApp message to the class) that the report and the rest of the assignment could be done with Claude, with a disclosure at the end.

Claude wrote the code for downloading the dataset, analysing its composition and the train/test overlap, training and evaluating the model, the cost sweep, the sensitivity analysis and the Gradio dashboard. It guided me through the setup (Codespaces, Colab, Roboflow) and drafted this report from my outputs. It also proposed the euro figures, the 3% base rate, the choice of the model-plus-review design and the assumption of two quality staff; I accepted them as proposed.

Decisions that were mine: the problem (defect inspection on electronic boards), the dataset and version, treating a board with any defect as positive, and evaluating on a test set cleaned of images that also appear in train. Anything I could not defend without the code I should not have kept, and I have reviewed the report with that in mind.