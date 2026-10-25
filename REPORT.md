# Assignment 2 Report

- **Name:** Deniz Meten
- **Student ID:** 18758
- **Email:** dmeten.ieu2022@student.ie.edu
- **Group:** BBADBA 5A

## Business problem, and the cost of being wrong

For this project, I wanted to use computer vision for something related to gaming because it is a domain I understand well. The system is designed to support Valorant VOD and replay analysis.

The main decision is whether a gameplay frame contains a visible enemy body and therefore should be considered relevant for review. Instead of a coach or analyst manually going through every sampled frame of a VOD, the model can automatically skip low-confidence frames, flag high-confidence frames, and send uncertain cases to a person.

The positive business class is `enemyBody`.

I treated a false negative as more expensive than a false positive. A false negative means that there actually is an enemy visible but the system misses the frame. This can cause an important combat or positioning situation to be skipped completely and may require the analyst to manually search through the VOD later.

A false positive means that the system flags a frame even though there is no enemy body. This is still a mistake, but the main consequence is that the analyst wastes some time checking an unnecessary frame.

For the cost model, I assumed an analyst cost of €48 per hour.

- False positive: around 15 seconds of unnecessary review = **€0.20**
- False negative: around 120 seconds of additional manual searching = **€1.60**
- Human review of an uncertain frame: around 15 seconds = **€0.20**

This means a false negative is treated as eight times more expensive than a false positive.

These numbers are assumptions rather than values provided by the dataset. The reasoning is that missing a useful combat situation is more costly for this use case than checking one unnecessary frame.

## Dataset

I used a public object-detection dataset from Roboflow Universe.

- **Dataset:** Valorant Object Detection 2
- **Workspace:** `kwan-li-jqief`
- **Project:** `valorant-object-detection2`
- **Version:** `7`
- **Task:** Object detection
- **Format:** YOLOv11
- **Classes:** `enemyBody` and `enemyHead`
- **Dataset link:** https://universe.roboflow.com/kwan-li-jqief/valorant-object-detection2/dataset/7

The dataset was used directly from Roboflow Universe and was not forked or relabelled.

One image represents a Valorant gameplay frame. Bounding boxes identify visible enemy bodies and enemy heads.

Version 7 contains **1,570 images** in total.

The dataset split was:

- Training: **1,416 images**
- Validation: **103 images**
- Test: **51 images**

The number of annotations per class was:

| Split | enemyBody | enemyHead |
|---|---:|---:|
| Train | 1,243 | 1,254 |
| Validation | 87 | 87 |
| Test | 47 | 42 |
| **Total** | **1,377** | **1,383** |

The two classes are therefore very balanced overall.

Object size was another important part of the dataset. The median `enemyBody` bounding box covered around **0.64% of the image**, while the median `enemyHead` bounding box covered only around **0.15%**.

This means heads are much smaller targets than bodies, which makes them harder to detect reliably.

The main weakness of the dataset is the relatively small test split. With only 51 test images, a few additional correct or incorrect predictions could noticeably change the final results. Because of this, the test results should not be treated as a perfect estimate of performance across every possible Valorant match.

## Finetuning and baselines

I used **YOLO11n** for object detection.

I selected the nano version because the dataset is not extremely large and the model is lightweight enough to train and deploy efficiently while still benefiting from pretrained YOLO features.

The final model was trained for **50 epochs** using an image size of **640** and a batch size of **16**. I trained the model locally using an **NVIDIA GeForce RTX 5070 Ti Laptop GPU**.

The model was compared against two baselines on the same held-out test split:

1. Finetuned YOLO11n trained on the Valorant dataset.
2. Pretrained YOLO11n using COCO weights without Valorant finetuning.
3. A trivial baseline that predicts nothing.

For the COCO zero-shot comparison, the generic COCO `person` class was mapped to `enemyBody`, because COCO does not contain the Valorant-specific `enemyBody` and `enemyHead` classes. There is no direct COCO equivalent for `enemyHead`.

The comparison was:

| Model | Main result |
|---|---|
| Finetuned YOLO11n | Precision 0.770, Recall 0.792, mAP50 0.790 |
| COCO YOLO11n zero-shot | 0.00 precision and 0.00 recall for the mapped Valorant classes at threshold 0.50 |
| Predict-nothing baseline | 0.00 recall because it never predicts a positive detection |

For the COCO zero-shot model, the mapped `person` detections produced **37 false positives** for `enemyBody`, but none correctly matched the 47 ground-truth `enemyBody` objects at the required IoU threshold.

For `enemyBody`, the zero-shot result was:

- True positives: **0**
- False positives: **37**
- False negatives: **47**
- Precision: **0.00**
- Recall: **0.00**

For `enemyHead`, the zero-shot baseline also achieved 0 precision and 0 recall because COCO has no direct equivalent for this class.

The trivial predict-nothing baseline also had 0 recall because it never makes a positive prediction.

This shows that generic COCO object detection was not enough for this specific visual problem. Finetuning was necessary for the model to learn the Valorant-specific appearance and annotation style.

The final finetuned model achieved:

| Metric | Result |
|---|---:|
| Precision | 0.770 |
| Recall | 0.792 |
| mAP50 | 0.790 |
| mAP50-95 | 0.297 |

## Per-class evaluation and failure modes

The final per-class test results were:

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| enemyBody | 0.7800 | 0.8298 | 0.8054 | 0.2843 |
| enemyHead | 0.7601 | 0.7546 | 0.7747 | 0.3095 |

`enemyBody` had the strongest AP50 and recall. This is useful because `enemyBody` is also the positive class used for the business decision.

`enemyHead` had lower recall, meaning that the model missed enemy heads more often than bodies.

One likely reason is object size. The median head annotation occupies only around 0.15% of the image compared with around 0.64% for a body. Smaller targets are generally more difficult for the model to detect consistently.

The difference between mAP50 and mAP50-95 is also important. The model performs relatively well when the required bounding-box overlap is moderate, but performance drops when stricter localization is required.

This suggests that the model often identifies the correct enemy but does not always place the bounding box extremely precisely.

## Cost matrix and operating point

The model outputs confidence scores for its detections. Instead of automatically using a threshold of 0.50, I swept confidence thresholds from **0.01 to 0.99**.

For the model-only decision, the cost function was:

`Total cost = FP × €0.20 + FN × €1.60`

The observed positive base rate in the held-out test set was **0.843**, meaning approximately **84.3%** of the test images contained at least one `enemyBody`.

Because the positive class is very common and false negatives are much more expensive than false positives, the cost model strongly prefers a low confidence threshold.

The minimum-cost model-only threshold was:

**0.01**

At this threshold, the image-level result was:

- True positives: **43**
- False positives: **3**
- True negatives: **5**
- False negatives: **0**
- Expected cost per 1,000 cases: **€11.76**

This threshold is much lower than a typical 0.50 threshold, but it follows from the business assumptions.

The model is encouraged to make more positive predictions because incorrectly flagging a frame costs only €0.20, while completely missing an enemy frame is assumed to cost €1.60.

For comparison, at a threshold of **0.50**, the model had:

- True positives: **37**
- False positives: **3**
- True negatives: **5**
- False negatives: **6**

This produces an expected cost of **€200 per 1,000 cases**, which is much higher than the low-threshold operating point because the extra false negatives are expensive.

The dashboard cost curve shows this relationship directly. Once the threshold becomes more conservative, false negatives increase and expected cost rises quickly.

The most important assumption affecting this result is the false-negative cost. If missing an enemy were much less expensive than assumed, the optimal confidence threshold would move higher.

## What a person still has to check

Instead of forcing every frame into only a positive or negative decision, I also implemented a three-outcome system:

- Score below **0.05** → auto-pass / skip
- Score from **0.05 to 0.10** → human review
- Score of **0.10 or above** → auto-flag

The selected assisted review band was:

**0.05 – 0.10**

At this operating point, the test-set results were:

- True positives: **43**
- False positives: **3**
- True negatives: **5**
- False negatives: **0**
- Cases sent to human review: **1 out of 51**
- Human-review share: **2.0%**
- Expected cost per 1,000 cases: **€15.69**

Assuming **2,000 sampled frames per day**, around **39.2 frames per day** would be sent to human review.

At approximately 15 seconds per review, this equals around **0.16 hours of review work per day**, or roughly **10 minutes**.

This is a very small workload compared with checking every frame manually.

The zero false negatives in this specific test result should not be interpreted as proof that the model will never miss an enemy in production. The test split contains only 51 images, so this result is based on a relatively small sample.

### Comparison with the alternatives

Using the same cost assumptions, the alternatives can be priced as follows:

| Approach | Expected cost per 1,000 cases |
|---|---:|
| Do nothing / manually recover missed positive events | €1,349.02 |
| Human review of every case | €200.00 |
| Model-only at threshold 0.01 | €11.76 |
| Model + human review band | €15.69 |

The doing-nothing estimate uses the same 120-second manual-search assumption used for a false negative. With an observed positive rate of 84.3%, relying on manual recovery for positive events produces the highest expected cost.

Having a person review every case avoids relying completely on the model, but at €0.20 per review it would cost **€200 per 1,000 cases** and create substantially more manual workload.

The model-only setup is actually the cheapest option under my current assumptions, at **€11.76 per 1,000 cases**.

The model plus human-review system is slightly more expensive at **€15.69 per 1,000 cases** because it includes the additional review cost.

However, the assisted system provides an extra safety mechanism by sending uncertain cases to a person instead of forcing the model to automatically decide every case.

This means there is a trade-off. If minimizing expected cost is the only objective, the model-only threshold of 0.01 is the preferred option on this test set. If human oversight for uncertain cases is also important, the assisted system provides that oversight for a relatively small additional cost.

## Dashboard

The Gradio dashboard loads the already-trained model from `weights/best.pt` and does not retrain anything when it starts.

It allows the user to:

- Upload a Valorant screenshot and view detected `enemyBody` and `enemyHead` bounding boxes.
- Change the confidence threshold used for predictions.
- Change the lower and upper thresholds used for pass, human review and automatic flagging.
- See the confusion matrix update when the thresholds change.
- See the expected cost per 1,000 cases.
- See the percentage of cases sent to human review.
- See the expected daily human-review workload.
- View the cost-against-threshold curve.
- View per-class precision, recall, AP50 and mAP50-95.

This makes it possible to see the operational effect of changing the model threshold rather than only looking at one overall accuracy number.

## Limitations & next steps

The first limitation is the small test split. With only 51 images, the reported performance and chosen thresholds could change significantly if a larger test set was used.

The second limitation is that the dataset may not represent every possible Valorant situation. Different maps, agents, weapons, abilities, resolutions, graphics settings and recording quality may affect model performance.

Another limitation is that the model only detects visual objects. Detecting an enemy does not automatically mean that a frame is tactically important.

A real coaching tool would also benefit from information such as player position, map location, round state, economy, utility usage and what happens immediately before and after the frame.

The production base rate is also uncertain. I used the test-set base rate of 84.3% because no separate production stream was available.

A real VOD review system could have a very different proportion of frames containing enemies. Since the optimal threshold depends partly on the base rate, a different production distribution could change the preferred operating point.

The business costs are also assumptions based on analyst time rather than measured organizational data.

Finally, the assisted system assumes that a human correctly resolves every case that is sent for review. In practice, human reviewers can also make mistakes.

With more time, I would evaluate the model on a larger and more varied set of Valorant recordings, measure the real production base rate and analyst review times, and then recalculate the cost-optimal thresholds using those real values.

## Generative AI use disclosure

Generative AI was used to assist with boilerplate code structure, Roboflow and YOLO integration, evaluation code, debugging, threshold-sweep implementation, dashboard wiring, and report formatting.

The business assumptions, interpretation of the results, operating decision, and final conclusions were reviewed by me, and I understand the submitted pipeline and the reasoning behind the chosen thresholds.
