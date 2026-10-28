# Assignment 2 Report

- **Name:** Clementine Mathieu
- **Student ID:** 19311
- **Email:** cmathieu.ieu2022@student.ie.edu
- **Group:** BBADBA 5A

## Business problem and the cost of being wrong

### The decision

A remote alarm-monitoring centre monitors indoor CCTV feeds for commercial clients, including warehouses, workshops, nightlife venues and event spaces. The cameras already use built-in analytics to flag events when something changes in the scene, such as movement or a sudden change in brightness.

Each event is sent to the centre with one image. An operator must decide whether it shows fire or smoke and whether the alert should be escalated. Escalation means contacting the site and, if nobody can confirm that it is safe, calling the fire brigade and starting an evacuation.

At the moment, operators make this decision by looking at the image themselves. However, one operator may be responsible for hundreds of cameras, so they cannot review every event immediately. The model can score each event and help decide what happens next: an operator can check the image, or the alert can go directly to the site and fire brigade if the model is allowed to decide on its own.

### What counts as one case?

One case is **one event**, not every frame from every camera stream. If the model analysed every frame, the proportion of positive cases would be close to zero. Also, missing one frame would not necessarily mean missing the fire, because the next frame might capture it. A €50,000 cost per missed frame would therefore not make sense.

Using the single frame attached to each event matches the monitoring centre's current process and makes the cost per case meaningful.

### What counts as a positive case?

An event is positive if its image contains **Fire or Smoke**. Smoke is included because it can appear before flames. Excluding it would remove the early warning that makes camera-based detection useful.

**Lights**—including stage lighting, LEDs, lamps and reflections—are negative examples. They are important because they could be mistaken for fire and cause false alarms. Detecting a light by itself should never trigger an alarm.

### Cost of being wrong

- **False negative: €50,000.** The model passes an event that contains a real fire. The site still has ceiling smoke detectors, so the fire should eventually be noticed. The problem is that it may be detected a few minutes later, after it has grown. The €50,000 estimate represents this additional damage, not the cost of the entire building, which could be worth millions. Using the full building cost would overstate the cost of a miss by one or two orders of magnitude and would make the cost analysis favour escalating every event. At a site without ceiling detectors, such as an unprotected storage yard, the cost of a miss would be much higher and the thresholds would shift strongly towards escalation.
- **False positive: €1,000.** The model escalates an event when there is no fire. This estimate covers a fire-brigade call-out, staff time and disruption at the site, such as evacuating a venue or stopping a warehouse shift. Frequent false alarms may also lead clients to ignore future alerts.
- **True positive and true negative: €0.** A real fire would eventually trigger the ceiling detector, even if the model did not detect it first. The benefit of the earlier warning is already included in the €50,000 cost avoided when a miss is prevented. Counting it again would double-count the benefit.

A missed fire is therefore assumed to cost **50 times more than a false alarm**. This affects all the later decisions. The confidence threshold should be lower than the default of 0.5, and uncertain cases should go to a person at an estimated cost of €1 rather than directly to the fire brigade at €1,000.

The system can therefore have three outcomes: automatically escalate high-confidence cases, send uncertain cases to an operator, and pass low-confidence cases.

## Dataset

- **Workspace:** `pangarap`
- **Project:** `3classes-nmslq` ("3CLASSES")
- **Version:** 3 ("Last")
- **Link:** https://universe.roboflow.com/pangarap/3classes-nmslq
- **Licence:** CC BY 4.0
- **Format/task:** YOLOv11 object detection, with images resized to fit 640 × 640 pixels
- **Size:** 9,977 images (8,456 training / 881 validation / 640 test)
- **Classes:** Fire (0, positive), Lights (1, negative), Smoke (2, positive)

### What the images show

Each image is a single frame from an indoor scene, similar to the image a monitoring-centre operator would see. The dataset includes warehouses and workshops, homes, and dark venues with stage lighting or LEDs. Some images look like wide-angle CCTV footage, while others are photos taken with a phone. The labels contain bounding boxes around visible fire, smoke or artificial light. Some images do not contain any labelled objects.

### Why I chose this dataset

I also considered datasets labelled only with Fire and Smoke, as well as datasets of outdoor wildfires. This was the only dataset I found with a Lights class, which represents the look-alike I expected to cause false alarms. A dataset containing only Fire and Smoke would not let me measure how often a lamp is mistaken for a fire.

The scenes are also indoors and include commercial settings, which are relevant to the clients in my business scenario. The dataset contains negative images with only lights or no labelled objects, which are needed to estimate false alarms.

## What the dataset implies

### Number of labelled objects

The following numbers count bounding boxes from the label files.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Fire | 4,622 | 417 | 305 | 5,344 |
| Lights | 5,235 | 502 | 407 | 6,144 |
| Smoke | 5,291 | 593 | 440 | 6,324 |
| Images | 8,456 | 881 | 640 | 9,977 |

### Types of test images

These counts are used for the frame-level cost analysis.

| Frame type | Test images |
|---|---:|
| Contains Fire or Smoke (positive) | 512 (80%) |
| Lights only | 109 |
| Empty | 19 |

Roboflow created two versions of each training image, including one flipped horizontally. This means the 8,456 training images come from approximately 4,200 original images.

### Object sizes

The box area is measured as a percentage of the image.

| Class | Median box area | Middle 50% | Boxes covering less than 1% of the image |
|---|---:|---:|---:|
| Fire | 2.6% | 1.1–6.5% | 22% (30% in test) |
| Lights | 0.4% | 0.2–1.7% | 70% |
| Smoke | 9.5% | 2.3–21.4% | 11% (20% cover more than a quarter of the image) |

### What this means for the model

- **There are enough labelled objects per class.** Each class has 300–440 test boxes, so its AP50 should be reasonably stable, within a few percentage points.
- **There are too few negative images for a reliable cost estimate.** Only 128 test images contain no Fire or Smoke, and just 19 are completely empty. The false-alarm estimate depends on these 128 images: one additional false alarm changes the estimated cost by about €7,700 per 1,000 cases. Empty scenes are common in real camera feeds, and 19 images are not enough to estimate performance on them reliably.
- **A low Lights AP is expected and is not necessarily a problem.** Around 70% of Lights boxes cover less than 1% of the image. Small LEDs and lamps are difficult for a nano model running at 640 pixels to locate precisely. The more important question is whether Lights are mistaken for Fire, which is examined in the confusion matrix.
- **Smoke may look worse under AP50–95 than it does for the actual alert decision.** Smoke can cover a large area but has unclear edges, making it difficult to draw an exact box. For an alert, the main question is whether smoke is detected somewhere in the frame, not whether the box fits it perfectly.
- **Accuracy is not a useful main metric for this test split.** Since 80% of the test images are positive, a model that always escalates achieves 80% accuracy without looking at the image. Real event streams should contain far more negative events. I therefore focus on frame-level recall and expected cost adjusted to an assumed real-world base rate, rather than the test split's accuracy.
- **The images are not all representative of real CCTV.** Many are phone or internet photos rather than fixed, wide-angle or infrared security-camera footage.
- **There may be overlap between training and test images.** Of the 555 distinct original filenames in the test set, 166 also appear in the training set. Some test images may therefore be near-copies of training images, which could make the test results look better than they really are.

### What I can and cannot conclude

The test results show how well the model distinguishes Fire and Smoke from Lights in the indoor scenes in this dataset. They do not establish the real false-alarm rate on client cameras, especially at night or with infrared, how well the model works on empty scenes, or whether the same results will hold on real CCTV footage.

## Fine-tuning and baselines

### 1. Model and task

I chose **YOLO11n object detection** instead of image classification (`yolo11n-cls`) or the larger `yolo11s` model.

- **Why object detection?** A bounding box shows the operator where to look, which can make each review faster. Detection can also identify both stage lights and a real fire in the same image. A classification model would have to assign one overall label to the image.
- **Why the nano model?** In production, the system may need to score events from hundreds of cameras, so speed matters. YOLO11n also trains in a reasonable time on a free Colab GPU. If smoke recall is the main limitation, the next options to test are `yolo11s` or a larger input size.

### 2. Training setup

I fine-tuned the pretrained `yolo11n.pt` model in Google Colab using a T4 GPU.

- Up to 40 epochs, with early stopping after 10 epochs without improvement.
- Image size: 640; batch size: 16; random seed: 0.
- Default Ultralytics data augmentation.
- Kept the checkpoint with the best validation performance.

All 40 epochs ran, so early stopping did not activate. The best epoch was 37, with validation mAP50 of 0.779 and mAP50–95 of 0.418. Validation mAP50–95 had levelled off from around epoch 35, ranging from 0.409 to 0.418, so more training would probably make little difference.

Test mAP50 was 0.692, lower than the validation score of 0.779. This suggests the test split is harder than the validation split.

### 3. Results on the same 640 test images

I compared the fine-tuned model with the pretrained COCO model used without fine-tuning, as well as two simple baselines: never escalate and always escalate.

The pretrained COCO model does not have Fire, Smoke or Lights classes, so its detection AP is zero by definition. I did not map a COCO class to Fire—for example, "traffic light"—because that would be arbitrary and would make the baseline unfair. Instead, I used the most generous simple rule: escalate whenever the COCO model detects anything, and record what it detects.

| Model / strategy | mAP50 | mAP50–95 | Frame precision | Frame recall | Frame accuracy |
|---|---:|---:|---:|---:|---:|
| Fine-tuned YOLO11n (confidence 0.10, lowest cost) | 0.692 | 0.365 | 0.988 | 0.967 | 0.964 |
| Fine-tuned YOLO11n (confidence 0.25, default) | 0.692 | 0.365 | 0.988 | 0.830 | 0.856 |
| COCO YOLO11n, zero-shot | 0 | 0 | 0.762 | 0.611 | 0.536 |
| Predict nothing | 0 | 0 | — | 0.000 | 0.200 |
| Always escalate | — | — | 0.800 | 1.000 | 0.800 |

A frame is escalated if the model detects Fire or Smoke at or above the stated confidence threshold.

### 4. What changed after fine-tuning?

- **The zero-shot COCO model is not useful for this task.** It performs worse than always escalating on precision, recall and accuracy. Under the cost model, it costs €952,305 per 1,000 cases, almost as much as always escalating. On positive images, it mainly detects people, chairs, couches and TVs. On negative images, it mainly detects people, chairs, TVs and clocks. Its decisions therefore depend on the furniture in the room rather than whether there is a fire.
- **Fine-tuning teaches the model the relevant classes.** mAP50 rises from 0 to 0.692.
- **The fine-tuned model performs much better than the simple baselines.** At its lowest-cost threshold of 0.10, it detects 495 of 512 positive frames and raises 6 false alarms among 128 negative frames. Its estimated cost is about eight times lower than never escalating and 16 times lower than always escalating. The cost analysis is explained in the section below.

## Per-class evaluation and failure modes

The most important things to measure are recall for Fire and Smoke, because missing them is expensive, and how often Lights are mistaken for Fire, because that could trigger a false alarm.

### Per-class results

These results are from the 640-image test split and `artifacts/metrics.json` under `finetuned.per_class`.

| Class | Test instances | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|
| Fire | 305 | 0.805 | 0.744 | 0.815 | 0.480 |
| Lights | 407 | 0.652 | 0.542 | 0.577 | 0.250 |
| Smoke | 440 | 0.841 | 0.639 | 0.685 | 0.366 |
| All | 1,152 | 0.766 | 0.642 | 0.692 | 0.365 |

The precision and recall in this table use the confidence threshold selected by Ultralytics to maximise F1, not the threshold used for my final operating decision. This explains why Smoke recall is 0.639 here, while the confusion matrix at a near-zero confidence threshold shows that 81% of smoke boxes are detected.

### Confusion matrix

The normalised confusion matrix is available in `artifacts/confusion_matrix_normalized.png`.

The matrix is calculated at a very low confidence threshold of 0.001. Its background row contains thousands of very uncertain detections that would not trigger an alarm at a real operating threshold. The errors that matter for the decision are:

| Actual object | Model prediction | Number | Impact |
|---|---|---:|---|
| Smoke | Not detected | 84 of 440 (19%) | Serious: the model may miss an early sign of fire. |
| Fire | Not detected | 14 of 305 (5%) | Serious: the model misses a fire. |
| Fire | Lights | 3 | Serious: the fire may be missed unless Smoke is also detected. |
| Lights | Fire | 1 | False alarm caused by a light. |

Confusing Fire with Smoke (4 boxes) does not add a cost because both classes trigger an alarm. The 26 missed Lights boxes also do not add a cost.

The main finding is that **missing Fire or Smoke is a bigger problem than confusing Lights with Fire**.

### Which class performs best and worst?

**Fire is the strongest class**, with an AP50 of 0.815. Its appearance is relatively clear and consistent. The overall mAP50 of 0.692 is pulled down mainly by Lights.

**Lights has the lowest AP50 (0.577)**, but this matters less for the final decision. Around 70% of Lights boxes cover less than 1% of the image, making them difficult for a small model to locate precisely. Missing a light does not trigger an alarm.

**Smoke is the main weakness for the decision.** Its recall is 0.639, and 84 of 440 smoke boxes are missed completely. At the chosen confidence threshold of 0.10, the model misses 17 of the 512 positive frames:
- 11 contain only Smoke.
- 6 contain both Fire and Smoke.
- None contain only Fire.

The main failure is therefore detecting Smoke when there are no visible flames. This can be the early stage of a fire, when the camera is meant to provide a warning before the ceiling detector does.

### What does the model confuse?

The model rarely mistakes Lights for Fire: this happens for only 1 of 407 light boxes. At a threshold of 0.10:
- 2 of 109 lights-only frames trigger an alarm.
- 4 of 19 empty frames trigger an alarm.

False alarms therefore appear to come more often from ordinary scenes than from lights. This is the opposite of what I expected when I chose the dataset. However, there are only 19 empty frames, so this rate cannot be estimated reliably for real camera feeds.

### Frame-level results

A positive frame contains Fire or Smoke. There are 512 positive and 128 negative test frames.

| Threshold | True positives (caught) | False negatives (missed) | False positives (false alarms) | True negatives | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 (Ultralytics default) | 425 | 87 | 5 | 123 | 0.830 | 0.988 |
| 0.10 (lowest-cost single threshold) | 495 | 17 | 6 | 122 | 0.967 | 0.988 |

Reducing the threshold from 0.25 to 0.10 cuts the number of missed positive frames from 87 to 17, with only one additional false alarm. Since a missed fire is assumed to cost 50 times more than a false alarm, the default threshold of 0.25 is too conservative.

## Cost matrix and operating point

### 1. Cost of each outcome

| Outcome | Cost |
|---|---:|
| False negative (missed fire) | €50,000 |
| False positive (false alarm) | €1,000 |
| True positive / true negative | €0 |
| Human review (about 45 seconds of operator time) | €1 |

The reasons for these estimates are explained in the business-problem section. They are not verified industry averages. The ratio between the costs matters most, so I also test what happens when a missed fire costs €10,000 or €200,000.

### 2. Base rate and cost calculation

I assume that **1% of events involve real Fire or Smoke**. I measure detection and false-alarm rates on the test split, then adjust them to represent 10 positive and 990 negative events per 1,000 cases. I do not use the test split's own 80% positive rate because that would make escalating almost everything appear to be the best strategy, which does not reflect a stream dominated by false alarms.

The 1% base rate is a deliberately high assumption, making the design harder to justify rather than easier. At 300 events per day, it would mean three real fire or smoke events each day, which is probably unrealistic for one centre. The real rate is likely to be lower. The sensitivity analysis below shows that the two-threshold design becomes cheaper as the base rate falls.

The estimated cost per 1,000 events is calculated by adding the cost of missed fires, false alarms and human reviews. The rates are measured on the test split and adjusted to the assumed 1% base rate.

### 3. Threshold sweep and chosen operating point

I tested thresholds from 0.00 to 1.00 in steps of 0.01, using the 640 test images. The results are saved in `artifacts/cost_sweep.csv`, and the cost curve is in `artifacts/cost_sweep.png`.

If the model makes every decision itself, threshold 0.10 has the lowest estimated cost: €63,008 per 1,000 events. This compares with €123,633 at the Ultralytics default threshold of 0.25, €500,000 for never escalating and €990,000 for always escalating.

The curve is uneven because there are only 128 negative images. As a result, the exact minimum should be treated with caution.

### 4. Sensitivity: which assumption matters most?

The table below shows how the cheapest single threshold changes when the base rate or the cost of a missed fire changes. It also shows the cost of the two-threshold design (0.01 / 0.69) when those thresholds are kept fixed. The results are from `metrics.json` under `operating_point.sensitivity`.

| Base rate | Cost of a missed fire | Best single threshold | Single-threshold cost per 1,000 | Two-threshold cost per 1,000 |
|---|---:|---:|---:|---:|
| 1% (baseline) | €50,000 | 0.10 | €63,008 | €1,105 |
| 0.1% | €50,000 | 0.69 | €24,121 | €223 |
| 5% | €50,000 | 0.05 | €88,867 | €5,026 |
| 1% | €10,000 | 0.31 | €34,414 | €324 |
| 1% | €200,000 | 0.05 | €89,063 | €4,035 |

The **base rate has the biggest effect**. If only 0.1% of events involve Fire or Smoke, the best single threshold rises from 0.10 to 0.69. At a 5% base rate, it falls to 0.05.

The cost of a missed fire also matters. If a miss costs €10,000, the best threshold rises to 0.31; if it costs €200,000, the best threshold falls to 0.05.

The single-threshold choice is sensitive to these assumptions, while the two-threshold design is less so. Across the scenarios in the table, the two-threshold design costs between 18 and 108 times less than the best single threshold, and its cost falls as the assumed base rate falls.

## What a person still has to check

### 1. The two thresholds

The model uses the highest confidence score for Fire or Smoke to decide what happens to each event.

| Model score | Outcome |
|---|---|
| Below 0.01 | Pass without an alarm |
| From 0.01 to below 0.69 | Send to an operator |
| 0.69 or above | Escalate automatically to the site and fire brigade |

### How the thresholds were chosen

In `train.py`, I tested every pair of thresholds from 0.00 to 1.00 in steps of 0.01. The chosen pair was the cheapest one that sends no more than 30% of cases to an operator. This is recorded in `costing.sweep_band` / `best_band` and in `metrics.json` under `operating_point.band`.

Using two thresholds is better than relying on a single threshold because the single-threshold minimum is sensitive to the assumptions. Uncertain cases can be checked by a person for €1 rather than sent directly to the fire brigade at an estimated cost of €1,000.

- **Why limit reviews to 30%?** Without a cap, the cheapest option would be to review almost every case: thresholds of 0.00 / 0.69 would cost €995 per 1,000 events but send 99.5% of cases for review. This would create the alarm-fatigue problem the two-threshold design is meant to avoid.
- **Why 30%?** At 300 events per day, 30% would mean 90 reviews, taking about 70 minutes of operator time. This should fit into existing shifts without hiring extra staff. The exact cap does not determine the result: any cap from 13% to 99% produces the same 0.01 / 0.69 pair.
- **Why is the escalation threshold 0.69?** No negative test image scored above this threshold. The model makes an automatic decision only for cases where it did not make a mistake on this test set.
- **Why is the review threshold 0.01?** Only 1 of 512 positive frames scored below it, so very few positive cases are passed without an alarm.

### Expected outcomes per 1,000 events

Using a 1% base rate, the model is expected to handle the following number of events:

| Actual case | Pass | Send to operator | Escalate |
|---|---:|---:|---:|
| Fire / Smoke (10) | 0.02 | 4.80 | 5.18 |
| No fire (990) | 866.25 | 123.75 | 0 |

Around 12.9% of events go to an operator. The model automatically escalates about 52% of real fires without waiting for a person, while almost all of the other 48% go for review.

### 2. Daily workload and staffing

I assume the centre handles 300 events per day. This represents a medium-sized centre with a few hundred cameras, where each camera's analytics raises about one event per day.

- **Reviews:** 12.9% of 300 is about 39 reviews per day, or around 1.6 per hour across a 24-hour period.
- **Time per review:** Each takes about 45 seconds, giving a total of around 30 minutes of operator time per day.
- **Staff needed:** This is about 0.06 of one full-time person's time, so no extra hire should be needed. The centre already has operators on duty 24/7, and one or two extra checks per hour should fit between the alarms they already handle.

The real event volume and the time needed for each review would have to be measured before deployment.

### 3. Comparing the four approaches

The comparison below assumes 300 events per day and a 1% base rate.

| Approach | Cost per 1,000 events | Cost per day | Reviews per day |
|---|---:|---:|---:|
| Do nothing; rely on ceiling detectors | €500,000 | €150,000 | 0 |
| Model decides alone (threshold 0.10) | €63,008 | €18,902 | 0 |
| Person checks every event | €1,000 | €300 | 300 |
| Two thresholds (0.01 / 0.69) | €1,105 | €332 | 39 |

All four approaches assume that operators make the correct decision for every case they review.

### 4. Is the design realistic?

The two-threshold design reduces the workload from 300 reviews per day to around 39 and automatically escalates about half of real fires.

In my cost model, having a person check every event is slightly cheaper: €300 per day compared with €332 for the two-threshold design. I include this result rather than changing the assumptions just to make my design look better.

I would still choose the two-threshold design because the model assumes that operators never miss a fire, which is unlikely in practice. An operator reviewing 300 events a day, 99% of which are false alarms, could become fatigued and miss important cases. The two-threshold design sends about eight times fewer cases to the operator, mostly those that are genuinely uncertain. I would reconsider this choice if the centre's records showed that operators almost never miss a fire, even when handling a full stream of alerts.

The main weakness is the 0.69 automatic-escalation threshold, which is based on only 128 negative test images. It must be tested on real camera footage before the model is allowed to escalate cases without human review.

## Limitations and next steps

1. **The thresholds may be too optimistic.** I selected them using the same 640 images on which I report the results, and some test images may be near-copies of training images.  
   **Next step:** Choose the thresholds using the validation split, report the cost on the untouched test split, and remove test images that share an original image with the training set.

2. **There are too few negative images.** Only 128 test images contain no Fire or Smoke, so the false-alarm estimates are unreliable.  
   **Next step:** Collect thousands of event frames from real client cameras, including night-time and infrared footage.

3. **The images do not fully represent real CCTV, and the model sees only one frame per event.** Many images are photos from phones or the internet.  
   **Next step:** Train and test on real camera footage, and analyse several frames per event to help detect smoke that continues or grows over time.

4. **Smoke without flames is the main failure.** Eleven of the 17 missed positive frames contain only Smoke.  
   **Next step:** Test `yolo11s` or a larger input size. More training epochs are unlikely to help much because validation mAP had already levelled off by around epoch 35.

## Generative AI use disclosure

Generative AI was used to help make the report clearer and more concise, and to assist with the code.
