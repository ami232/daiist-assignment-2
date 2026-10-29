# Assignment 2 Report

- **Name**: Yasmin Tandon
- **Student ID**: 20356
- **Email**: ytandon.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

On a commercial poultry farm, staff walk through each house every day and look at the fresh droppings. This is because the colour and consistency of a dropping are among the earliest visible signs of the three diseases that matter most in a flock: coccidiosis, salmonellosis and Newcastle disease (NCD). At the moment, that call is made by eye, by farm workers whose training varies quite a lot. Essentially, the model takes a phone photo of a single dropping sample and returns one of three outcomes: **pass**, meaning the sample looks healthy and nothing needs to be done; **fail**, meaning disease is suspected, so a confirmatory PCR test is sent and the vet is called; or **human review**, meaning the model is unsure, so a trained vet technician looks at the photo and decides between pass and fail.

Farm staff take the photos during the daily walk-through, a vet technician works through the review queue and the farm vet acts on every fail.

The two possible errors do not cost the same, and the difference between them is extremely large. A false negative (a diseased sample that is passed) lets an infectious disease keep spreading through a house of thousands of birds until someone notices it later. That means mortality, lost weight gain, a bigger treatment bill and, in the case of NCD, a notifiable-disease response. A false positive (a healthy sample that is failed), on the other hand, costs one PCR test and a vet visit that finds nothing. The miss is worse by more than an order of magnitude, and this asymmetry drives everything that follows: the model is tuned for disease recall, the thresholds sit low and the review band is there to catch uncertain cases cheaply, before they turn into either kind of error.

The cost per sample, as assumed in `train.py`, is as follows:

| Outcome | Cost (EUR) | Reasoning |
|---|---|---|
| False negative: diseased sample passed | **2,000** | One extra day of undetected spread in one house before the next sampling round catches it: extra mortality and lost growth across the house, plus a larger treatment course. |
| False positive: healthy sample failed | **80** | A confirmatory PCR panel plus the farm vet's time on a visit that finds nothing. |
| True positive: diseased sample failed | **80** | The same PCR and visit. This is the action we want, but it is not free. |
| True negative: healthy sample passed | 0 | No action. |
| Human review of one sample | **2** | 3 minutes of a vet technician at EUR 40/hour. The reviewer is assumed to decide correctly. |

A miss therefore costs 25 times as much as a false alarm. These figures are assumptions rather than measured ones. However, the sensitivity analysis below shows that the chosen thresholds hold for any miss cost between EUR 500 and EUR 2,000, and only move once a miss is assumed to cost around EUR 5,000 or more.

## Dataset

The dataset comes from Roboflow Universe (workspace `thesis-pr4oh`, project `chicken-droppings-analysis`, version 1: https://universe.roboflow.com/thesis-pr4oh/chicken-droppings-analysis/dataset/1). It is published under a CC BY 4.0 licence and was used as published, not forked, with its native train/valid/test split. The task is object detection with four classes: `Coccidiosis-like`, `Healthy`, `NCD-like` and `Salmonella-like`.

Each image is a phone photo of one dropping on a farm floor (concrete, litter or soil), mostly 288×640 or 640×288 pixels, which is the same kind of photo the deployed system would receive. On export, images were auto-oriented and then resized to fit within 640×640. Roboflow augmented only the training split, with three outputs per source image, using horizontal and vertical flips, 90° rotations, a shear of ±10° and exposure of ±10%.

One of the key reasons for choosing this dataset is that it serves the decision rather than just resembling it. The unit is the same, since one image is one dropping sample, exactly what farm staff would photograph. The classes are the same, being the three diseases the farm needs to act on plus healthy. The capture conditions are also the same: real floors with real clutter rather than studio shots.

There is one caveat. The '-like' suffix suggests the labels are visual diagnoses rather than lab-confirmed ones, and the project page does not say how the labels were made. At best, then, the model can reproduce an expert's visual call, which is exactly why every fail is confirmed by PCR.

## What the dataset's composition implies

| Split | Images | Source images | Coccidiosis | Healthy | NCD | Salmonella |
|---|---|---|---|---|---|---|
| train | 13,644 | 4,548 (each ×3 augmented) | 4,308 | 4,915 | **891** | 4,875 |
| valid | 1,298 | 1,298 | 412 | 473 | **95** | 457 |
| test | 648 | 648 | 212 (197 img) | 232 (205 img) | **51 (38 img)** | 223 (208 img) |

Counts are box instances, with test image counts in brackets. The split is roughly 70/20/10 by source image. Grouping files on the name stem before `.rf.` shows that no source image appears in more than one split, and the valid and test splits are not augmented. This composition has several implications.

1. **NCD results cannot be pinned to a precise number.** NCD has 5–6× fewer instances than the other classes and only 38 test images (51 boxes), so a single image more or less moves NCD image recall by 2.6 points. The 95% Wilson interval on NCD box recall (35/51 = 0.69) is 0.55–0.80. NCD is also the most dangerous disease to miss, which means the class we care about most is the one we can measure least precisely.
2. **It is effectively image classification.** Given that 13,640 of the 13,644 training images contain exactly one class (with 1.1 boxes per image on average), the decision unit is essentially one image, and the cost analysis scores each image by its highest disease-class confidence.
3. **There are no background-only images.** Every image contains a labelled dropping, so the test set says nothing about false alarms on empty floor, feathers or feed. In production, a photo with no dropping in it is quite possible and completely untested.
4. **Box size goes with class.** The median box area as a share of the image is around 6% for Healthy, 9–19% for NCD, 20% for Salmonella and 42% for Coccidiosis. The model can partly use size as a shortcut, so if farm staff photograph droppings from a different distance than the dataset's photographers did, accuracy may drop.
5. **The class mix is nothing like a farm.** 68% of test images (443/648) are diseased, compared with an assumed farm base rate of 5%. Raw test accuracy or precision would therefore overstate how often a fail is real, which is why the cost analysis reweights to the base rate instead of using the test mix.
6. **Photos were collected class by class**, as the filenames start with the class name. Within a class, photos may share a farm, a floor or lighting conditions, which the test split would not separate, so test performance may be optimistic for a new farm.

## Finetuning and baselines

The model is `yolo11s`, finetuned from the COCO-pretrained checkpoint as an object detector. It was chosen over `yolo11n` because healthy droppings are small (around 6% of the image) and the larger model handles small objects better at only a modest extra cost. Detection was preferred to classification both because that is how the dataset is labelled and because the boxes show the reviewer which object drove the call.

All training values are set in `train.py`. Training ran in a Kaggle notebook on a Tesla T4 GPU, where 30 epochs took 1.36 hours. The image size was 640, matching the export, so no further downscaling took place. Training was capped at 30 epochs with an early stopping patience of 5, although early stopping never actually triggered. The optimiser was AdamW with lr0 = 0.001, batch size 16, seed 0 and deterministic mode on. Ultralytics' default augmentations were applied on top of Roboflow's, with mosaic switched off for the last 10 epochs.

The box (CIoU) gain of 7.5 and the DFL gain of 1.5 were left unchanged, but the classification term was replaced by **0.5 × (0.7 × BCE + 0.3 × Focal)**, with focal γ = 1.5 and α = 0.25. The focal component down-weights easy examples (mostly background locations and the well-separated large classes) and so shifts the gradient towards hard and rare cases such as NCD. It can essentially be thought of as telling the model to spend less effort on what it already gets right and more on what it keeps getting wrong. BCE was kept dominant so that the confidence scores remain usable for thresholding. The committed weights are the best checkpoint by validation fitness, saved as `weights/best.pt`.

The results on the same test split (648 images) are below.

| Model | mAP50 | mAP50-95 | Disease recall | Expected cost / 1,000 samples |
|---|---|---|---|---|
| **Finetuned yolo11s** (pass < 0.04, fail ≥ 0.71) | **0.896** | **0.523** | **99.5%** | **EUR 4,535** |
| COCO yolo11s, zero-shot | 0.000 | 0.000 | 0% | EUR 100,000 |
| Trivial: predict nothing (pass everything) | 0.000 | 0.000 | 0% | EUR 100,000 |
| Trivial: majority class (Salmonella-like; 32.1% accuracy) | n/a | n/a | 100% | EUR 80,000 |

COCO zero-shot has no dropping or disease class, so it cannot match a single label. It does still fire on 227 of the 648 test images at confidence 0.25, with irrelevant COCO classes, which shows that an object detector seeing *something* is not the same as it answering the question. For the farm, it is essentially the same as passing everything. The majority-class baseline, on the other hand, fails every sample: it catches every disease but pays for a PCR test and a vet visit on every healthy sample. The finetuned model beats all three by a wide margin on both accuracy and cost, and the gap is larger than the gap between the two trivial baselines themselves.

Validation mAP rose from 0.71 after the first epoch to about 0.91 by epoch 22 and then flattened out (0.905–0.911). However, it was still edging up at epoch 30, so a longer run might add a little more.

## Per-class evaluation and failure modes

For costing, the positive class is *diseased*, meaning any of Coccidiosis, NCD or Salmonella, and each image's score is its highest disease-class confidence.

The detection metrics on test are below (from Ultralytics, with precision and recall taken at its max-F1 confidence).

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| Coccidiosis-like | 0.918 | 0.929 | 0.946 | 0.607 |
| Healthy | 0.882 | 0.906 | 0.920 | 0.501 |
| **NCD-like** | 0.946 | **0.682** | **0.773** | 0.433 |
| Salmonella-like | 0.904 | 0.919 | 0.943 | 0.550 |
| **All** | | | **0.896** | **0.523** |

The image-level confusion matrix below is at the single-threshold operating point (0.13). Each image takes its top detection, and anything below the threshold counts as "none".

| true \ predicted | Coccidiosis | Healthy | NCD | Salmonella | none |
|---|---|---|---|---|---|
| Coccidiosis-like (197) | **197** | 0 | 0 | 0 | 0 |
| Healthy (205) | 0 | **202** | 0 | 2 | 1 |
| NCD-like (38) | 0 | 2 | **34** | 2 | 0 |
| Salmonella-like (208) | 1 | 3 | 0 | **203** | 1 |

Coccidiosis and Salmonella carry the headline number. They are the two largest classes and have an AP50 of about 0.94–0.95 each. NCD, however, drags it down, with a box recall of 0.68 and an AP50 of 0.77, the lowest on every metric. At image level, 34 of the 38 NCD images are classified correctly, with two called Healthy and two called Salmonella. Two NCD images have disease scores of 0.002 and 0.007, meaning the model sees no disease in them at all. No sensible threshold catches them, and they are the only misses at the chosen review band.

Additionally, NCD is the least confident class. Its median disease score is 0.64, compared with 0.74 for Coccidiosis and 0.76 for Salmonella, and only 32% of NCD images score above the fail threshold of 0.71, against 64–69% for the other diseases. NCD therefore leans on human review far more than the other diseases do.

At box level, the largest confusion is true Coccidiosis predicted as Salmonella (40 of 212 boxes). This mostly disappears at image level, since the top detection in those images is still Coccidiosis. In any case, the confusion matters less for the farm, as both are diseases and both lead to a fail and a PCR test, which names the pathogen anyway. Healthy has the lowest precision (0.88), partly because its boxes are the smallest and the hardest to localise.

The training curves, along with Ultralytics' PR, F1, precision and recall curves, are in `outputs/training/` and `outputs/test_eval/`.

## Cost matrix and operating point

Each test image is given a disease score. For every threshold, the pass/review/fail rates are measured separately on the diseased (443) and healthy (205) test images and then combined at the assumed 5% base rate. The cost per sample is 0.05 × (cost on diseased) + 0.95 × (cost on healthy), using the cost table above. Thresholds were swept from 0.00 to 1.00 in steps of 0.01 (`outputs/cost_sweep.json`, plotted in the dashboard).

The 5% figure comes from the fact that droppings are sampled across healthy houses every day. On most days most houses are clean, and a sample is only diseased while an outbreak is under way. Like the costs, 5% is an assumption rather than a measured prevalence.

| Operating point | Thresholds | Cost / 1,000 | Disease recall | Healthy failed | Sent to review |
|---|---|---|---|---|---|
| Model alone (minimum of single-threshold sweep) | 0.13 | EUR 6,196 | 98.9% | 1.5% | n/a |
| **Model + review band (minimum over all pairs)** | **pass < 0.04, fail ≥ 0.71** | **EUR 4,535** | **99.5%** | **0.0%** | **5.1%** |

Given that a miss costs 25 times as much as a false alarm, the single-threshold optimum sits extremely low (0.13). The sweep is therefore willing to accept 3 healthy samples failed out of 205 in order to miss only 5 diseased ones out of 443. These five misses are the two near-zero NCD images and three Salmonella images scoring between 0.05 and 0.12.

**Sensitivity: what moves the answer.**

| Assumption varied | Range tested | Chosen band | Cost / 1,000 |
|---|---|---|---|
| Base rate | 0.5% → 10% | stays **0.04 / 0.71** | EUR 515 → 9,001 |
| Base rate | **20%** | **0.00 / 0.71**: review nearly everything below 0.71 (87% of samples) | EUR 17,747 |
| Cost of a miss | EUR 500 → 2,000 | stays 0.04 / 0.71 | EUR 4,196 → 4,535 |
| Cost of a miss | EUR 200 | 0.09 / 0.71 | EUR 4,123 |
| Cost of a miss | **EUR 10,000** | **0.00 / 0.71**: review 97% | EUR 5,937 |
| Cost of a false alarm | EUR 20 → 500 | stays 0.04 / 0.71 | EUR 1,548 → 25,440 |
| Review time | 1 → 30 min | stays 0.04 / 0.71 | EUR 4,467 → 5,446 |

The assumption that would most change the answer is the base rate, together with the cost of a miss. The band stays stable from 0.5% to 10% prevalence and from EUR 500 to EUR 2,000 per miss. However, if prevalence rises towards 20% (as it might in an outbreak season), or a miss costs EUR 10,000 or more (for example, an NCD case that triggers movement restrictions), the cheapest design is to send almost every sample below 0.71 to a person. At that point, the model is essentially only being used to fast-track confident fails. The false-alarm cost and the review time, by contrast, change the total but not the thresholds.

## What a person still has to check

Any sample with a disease score in the review band (0.04 ≤ score < 0.71) goes to a person. At a 5% base rate, this is **5.1%** of samples. On the test set, the band contains 161 of the 443 diseased images and 7 of the 205 healthy ones. It holds a large share of the diseased samples (36%) because NCD and some Salmonella score in the middle range, but at the farm's 5% base rate that still works out to only about 1 sample in 20.

Assuming 500 samples a day (for example, 10 houses × 50 samples), that means about 25 reviews a day. At 3 minutes each, this comes to around 76 minutes a day, or 0.17 of one person's 7.5-hour working day. This is realistic: one vet technician can clear the queue in a single sitting alongside other duties, and the load stays under half a person even at 10% prevalence, where 6.7% of samples are sent to review.

The table below compares the design against the three alternatives at a 5% base rate.

| Design | Cost / 1,000 samples | People needed | Comment |
|---|---|---|---|
| Do nothing (pass everything) | EUR 100,000 | 0 | every diseased sample is missed |
| Person checks every sample | EUR 6,000 | 3.3 (1,500 min/day) | perfect accuracy is assumed, so this is a lower bound on its real cost |
| Model decides alone (threshold 0.13) | EUR 6,196 | 0 | 5 misses and 3 false alarms per 648 test images |
| **Model + human review band** | **EUR 4,535** | **0.17** | 2 misses, 0 false alarms on test |

Interestingly, the model on its own is slightly *more* expensive than having a person check every sample, since its few misses are costly and review is cheap at EUR 2. The key highlight of this design is that its real value comes from deciding where people look. The review band saves 24% compared with checking everything and cuts staffing from 3.3 people to a fraction of one, and against doing nothing it saves about EUR 95 per sample. The comparison with a person checking everything actually flatters the human, because it assumes a perfect reviewer. A real reviewer would miss some cases, which would make the model + review design look even better.

## Limitations & next steps

1. **NCD is thin and the weakest class.** There are only 38 test images, and box recall is 0.69 (95% CI 0.55–0.80). The next step would be to add NCD images from other sources or farms, evaluate NCD recall on a larger held-out set and compare the current setup against a class-weighted classification loss on NCD.
2. **Two NCD samples score close to zero**, and no threshold catches them. These should be inspected to see whether the label or the image is at fault. If it is the image, this is a real blind spot that needs to be reported to the vet.
3. **There are no background or non-dropping images**, so false alarms on empty floor, feathers or feed remain untested. Background images should be collected or added, and included in the test set.
4. **Box size correlates with class**, so the model may partly be learning a shortcut. It should be tested on photos taken from a different distance, with scale augmentation added if performance drops.
5. **The labels are visual ('-like'), not lab-confirmed.** The model should be validated against PCR-confirmed samples before its score is treated as a disease probability.
6. **There may be farm or session effects.** Photos were collected class by class and the test split is not separated by farm, so the model should be evaluated on photos from a farm that is not in the dataset.
7. **The costs, base rate and review accuracy are all assumptions**, and the reviewer is assumed to be perfect. These should be replaced with the farm's real PCR price, the vet's rate and historical outbreak losses. The dashboard already accepts new cost and base-rate figures live.

