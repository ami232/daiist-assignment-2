# Assignment 2 Report

- **Name:** Ignacio Ascanio
- **Student ID:** 18719
- **Email:** iascanio.ieu2022@student.ie.edu
- **Group:** BBADBA 5B

## Business problem, and the cost of being wrong

I developed an AI-assisted road inspection system for a municipality responsible for maintaining its roads. Inspectors normally need to examine large numbers of photographs to identify damage and decide which locations need attention. I fine-tuned YOLO11n to detect six asphalt conditions and support three decisions for each image: **Pass**, **Human Review**, or **Flag**.

The two mistakes have different consequences. Missing actual damage may delay repairs and allow a road defect to worsen; falsely flagging a road can waste inspection time and resources. My objective is therefore not simply to maximize accuracy, but to minimize an estimated operational cost while keeping the human-review workload manageable. I treat potholes and all four crack classes as actionable damage, while **Repair** is excluded because a previously repaired surface does not necessarily need new maintenance.

## Dataset

I used the [Asphalt Distress Detection dataset, version 2](https://universe.roboflow.com/roboflow100vl-full/asphaltdistressdetection-shlk/dataset/2), from Roboflow Universe (workspace `roboflow100vl-full`, project `asphaltdistressdetection-shlk`). It contains **3,144 images** and YOLO-format bounding-box annotations for Alligator crack, Longitudinal crack, Oblique crack, Pothole, Repair, and Transverse crack. Each box identifies the location and class of a road-surface condition.

The dataset is relevant because it lets the model identify specific visible defects rather than merely labeling an entire image as damaged. I retained the dataset's original training, validation, and test splits. During inspection, I noticed that some photographs appeared to show very similar road scenes across splits, which may make evaluation results more optimistic than performance on genuinely new roads.

### Dataset exploration

I counted images and annotations, plotted the number of objects per class, examined bounding-box areas, and displayed six annotated training examples. The training set contains **3,478 objects**, but they are unevenly distributed: Repair has 1,157 annotations, while Pothole has only 63.

Bounding boxes also vary substantially in size. Their mean area is **7.83%** of the image, their median area is **3.95%**, and the smallest box covers only **0.084%**. This suggests that many defects, especially small potholes and thin cracks, may be difficult to localize accurately. The visual examples also showed that one image can contain multiple defect classes.

## What the dataset's composition implies

| Class | Train objects | Validation objects | Test objects |
|---|---:|---:|---:|
| Alligator crack | 297 | 79 | 38 |
| Longitudinal crack | 911 | 288 | 140 |
| Oblique crack | 131 | 30 | 12 |
| Pothole | 63 | 21 | 10 |
| Repair | 1,157 | 311 | 154 |
| Transverse crack | 919 | 258 | 150 |
| **Total objects** | **3,478** | **987** | **504** |

The images are split into **2,200 training**, **628 validation**, and **316 test** images. Oblique crack and Pothole have especially few test annotations (12 and 10), so their per-class results are less stable than those for Repair or the more common crack types. The image-level actionable-damage rate is **438/628 (69.7%)** in validation and **216/316 (68.4%)** in test. This is the rate in this dataset, **not** an estimate of how often a municipality would encounter damaged roads in practice.

The test set resembles the training data in its labels and road-image format, but it may not represent different cities, cameras, weather, or truly unseen road scenes. I therefore cannot claim the measured performance will generalize to real municipal inspections without an independent field test.

## Finetuning and baselines

### YOLO11 fine-tuning

I fine-tuned the **COCO-pretrained YOLO11n** model in **Google Colab using a Tesla T4 GPU**, with **30 epochs**, **640-pixel image size**, **batch size 16**, and **random seed 42**. I chose YOLO11n because it is relatively lightweight and starts with useful pretrained visual features. The selected checkpoint is saved as `weights/best.pt`; the model and its training history are used by the local pipeline and dashboard. The standard `main.py train` path contains the actual training code, while `ASPHALT_USE_SAVED=1` lets me check the downstream analysis using the already trained checkpoint without repeating training.

### Training performance and baselines

Training classification loss decreased from approximately **4.28 to 1.11**, while validation classification loss decreased from **3.51 to 1.13**. Their similar downward trends do not show obvious divergence during these 30 epochs, although they do not rule out leakage or overfitting. The training-history validation mAP50 rose from about **0.14 to 0.69**.

I compared three reference approaches:

| Approach | Validation mAP50 | Interpretation |
|---|---:|---|
| Predict nothing | 0.000 | Misses all 987 validation objects |
| YOLO11n trained from scratch for 30 epochs | ~0.365 | Learns from the asphalt dataset but starts without pretrained weights |
| Fine-tuned pretrained YOLO11n after 30 epochs | ~0.690 | Benefits from transfer learning |

I also tried the original **COCO-pretrained model without asphalt fine-tuning** on six example images. It produced one detection, incorrectly labeling a road repair as a person, while the fine-tuned model produced 16 asphalt-specific detections on those same images. This is a **qualitative illustration using training examples**, not a quantitative COCO-versus-fine-tuned test-set comparison, since the COCO categories do not match the asphalt classes.

A separate evaluation of the saved model on the validation split yielded **74.1% precision, 57.5% recall, 68.01% mAP50, and 43.65% mAP50–95**. The small difference from the training-history mAP reflects different evaluation outputs/settings rather than a new training experiment.

## Per-class evaluation and failure modes

I evaluated the selected model on the **316-image test set**, without changing the model or thresholds based on test results. Overall test performance was **65.22% mAP50** and **39.93% mAP50–95**.

| Test class | Precision | Recall | mAP50 |
|---|---:|---:|---:|
| Alligator crack | 77.18% | 47.37% | 59.29% |
| Longitudinal crack | 72.62% | 64.41% | 71.01% |
| Oblique crack | 58.30% | 41.67% | 38.52% |
| Pothole | 97.81% | 50.00% | 61.96% |
| Repair | 72.25% | 87.01% | 85.69% |
| Transverse crack | 73.68% | 68.67% | 74.82% |

**Repair** has the highest test mAP50 and recall; **Oblique crack** has the lowest mAP50. Pothole precision is high, but recall is only 50%, meaning the model still misses some pothole instances. These values are particularly uncertain because the test set has just ten pothole annotations.

The validation confusion matrix shows concrete failure patterns: the model sometimes confuses **Oblique** with **Transverse cracks**, and some **Potholes** with **Alligator cracks**. It can also produce false alarms or miss parts of larger damaged areas. For a municipality, missing damage is more serious than producing a limited number of extra alerts, which motivates the conservative threshold and the human-review option. The confusion-matrix matching settings are not identical to the image-level business calculations below.

## Cost matrix and operating point

I defined an **illustrative** cost model in euros, rather than using verified municipal accounting data:

| Outcome | Assumed cost | Reasoning |
|---|---:|---|
| False negative: actionable damage passed | **€200** | Represents potential delayed maintenance and follow-up consequences |
| False positive: non-actionable image flagged | **€25** | Represents an unnecessary inspection or administrative response |
| Human review | **€3 per image** | Represents three minutes of staff time at an implied fully loaded rate of €60/hour |

For each image, I take the **highest YOLO confidence among the five actionable classes** as its damage score; Repair is not actionable. The starting inference confidence floor is **0.01**. My cost calculation is `200 × missed-damage images + 25 × unnecessary flags + 3 × human-reviewed images`, assuming that human reviewers make the correct decision.

I first swept a **single automated flag threshold** from 0.05 to 0.95 in steps of 0.05 on the validation set. The best tested point was **0.05**, with **4 missed-damage images**, **35 unnecessary flags**, and an estimated cost of **€1,675**. I then searched pairs of Pass and Flag thresholds under a review-capacity constraint. The best feasible pair was **0.01 and 0.20**, reducing estimated validation cost to **€958**.

The biggest uncertainty is the **€200 cost assigned to missed damage**, together with the assumption that review is reliable. These figures, and the unusually high damaged-image rate in this dataset, strongly influence the selected policy. In sensitivity tests, raising false-alarm costs or lowering human-review costs moved the Flag threshold to 0.35, while the original 0.01/0.20 pair remained selected in four of six scenarios.

## What a person still has to check

Under my selected policy, an image is **passed** when its damage score is below **0.01**, **sent for human review** when it is from **0.01 to below 0.20**, and **flagged** when it is **0.20 or higher**. On the 628 validation images, this results in **117 Pass**, **86 Review**, and **425 Flag** decisions. The review share is **13.7%**.

I assumed **1,000 images per day**, **two available reviewers**, **six productive hours per reviewer per day**, and **three minutes per image**. The team can review **240 images daily** (24% of daily volume). Applying the validation review rate to 1,000 daily images gives approximately **137 reviews**, or **6.8 reviewer-hours**; **two reviewers** are sufficient under these assumptions. The flagged images would still need a separate maintenance or inspection response, which my staffing calculation does not price in full.

I fixed the thresholds using validation data and then applied them unchanged to the **316 test images**. The resulting decisions were **58 Pass**, **47 Review**, and **211 Flag**, with **one missed-damage image** and **eight unnecessary flags**. Estimated test costs were:

| Test strategy | Estimated cost |
|---|---:|
| Pass everything | €43,200 |
| Flag everything | €2,500 |
| Review everything | €948 |
| Fully automated YOLO (validation-selected threshold 0.05) | €900 |
| **Hybrid YOLO + human review** | **€541** |

The hybrid policy costs approximately **40% less** than fully automated YOLO under the assumed cost model. Although reviewing everything has a €948 modeled cost on this test set, reviewing all 1,000 daily images would require about **50 staff-hours per day**, well beyond the capacity of two reviewers. The Gradio app makes it possible to inspect uploaded images, examine per-class results, change operating thresholds and costs, and estimate reviewer staffing without retraining.

## Limitations & next steps

The main limitation is **possible scene-level leakage** in the original Roboflow splits: some images appear to show nearly identical road scenes, even though I did not identify exact duplicate files. Keeping related scenes across different splits can make test performance overly optimistic. Ideally, images from the same road scene or original photograph would be grouped before creating new splits, with augmentation restricted to the training group.

The dataset is also **imbalanced**, with few Oblique crack and Pothole instances, and its high actionable-damage prevalence may not match a municipality's actual image stream. My system makes an **image-level** decision based on the highest detection confidence, so that score is not a calibrated probability that a road needs maintenance. Finally, the cost estimates are hypothetical, human review is assumed to be perfect, and the downstream expense of dealing with correctly flagged defects is not fully represented.

Next, I would test on images from genuinely unseen roads, collect more examples of the rare classes, review false negatives with road-maintenance staff, validate the actual cost assumptions and reviewer accuracy, and recalibrate the decision thresholds using real municipal operating data before deployment.

## Generative AI use disclosure

I used **ChatGPT** to help structure the project, understand YOLO fine-tuning and evaluation, draft and debug parts of the Python notebook and Gradio dashboard, troubleshoot Colab/Codespaces file paths and execution errors. 