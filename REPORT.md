# Assignment 2 Report

- **Name**: Mariano Delpree
- **Student ID**: 19970
- **Email**: mdelpree.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

On poorly maintained roads, potholes damage tyres, rims and suspension, and in extreme cases cause accidents when the driver does not see them in time. The model runs inside one car and supports one decision, made for every frame of its forward-facing camera (roughly one frame every 50 m of road): **what should the car do about the road ahead?**

There are three possible outcomes:

- **Pass: continue as normal.** The model is confident there is no pothole, so nothing happens.
- **Fail: decelerate.** The model is confident there is a pothole. The car slows down gently by itself. The deceleration is not emergency braking; its purpose is to give the driver time to spot the pothole and steer around it or cross it slowly.
- **Review: warn the driver, without decelerating.** The model is not sure. A chime and a dashboard icon tell the driver to look at the road ahead; the driver is the person who decides whether there is a pothole and what to do.

The system works only for the car it is installed in. Nothing is shared with other cars or stored on a map, so a pothole this car misses is a pothole this car drives over.

The two mistakes are not equally expensive. **All costs in this report are euro-equivalents, not money actually spent.** They put every kind of harm on one scale, including things that have no price: safety, stress, comfort, wasted time and the driver's trust in the system. Only the cost of hitting a pothole is anchored to a real repair bill. The other two costs are judgements about how bad each outcome is compared with that one, expressed in the same units so that they can be added up and compared.

- **Missed pothole (false negative), 40 EUR-equivalent.** The car drives over the pothole at full speed. Not every pothole causes damage, and damage is not always a 500 EUR repair. I assume that out of every 10 potholes hit at speed, one causes damage of about 400 EUR (a tire, a rim or a suspension part) and the other nine cause none. The expected cost of one hit is therefore 400 / 10 = **40 EUR-equivalent**. This is the anchor of the scale. It also has to stand for harms I have not priced separately, such as the jolt and the small risk of losing control, so if anything it is on the low side.
- **Needless deceleration (false positive), 2 EUR-equivalent.** The car slows down when there is nothing there. Almost none of this is money: a few seconds lost, a less comfortable ride, some annoyance for the driver and the traffic behind, and a little less trust in the car. It causes no damage. I judge it to be **20 times less bad** than running over a pothole, so it counts as 40 / 20 = **2 EUR-equivalent**.
- **Warning (review), 0.40 EUR-equivalent each.** A warning does not slow the car, but it is still a distraction: the driver's attention is pulled to the road ahead and away from mirrors and traffic, and every unnecessary warning makes the next one easier to ignore. Being distracted is still less bad than having the car slow down: I judge a warning to cost **5 times less** than a deceleration, so a warning counts as 2 / 5 = **0.40 EUR-equivalent**, whether or not there turns out to be a pothole. Nobody pays this; it measures distraction.

Because a miss costs 20 times a needless slowdown, the car should slow down readily. That does not justify catching every pothole at any price: the operating point is the one that minimises total expected cost per day, which accepts a few misses if catching them would mean braking for nothing all day. The warning sits in between: it is cheaper than a slowdown but not free, so it only pays off for frames where the model is genuinely unsure.

## Dataset

- **Workspace / project / version**: `pothole-ztskn` / `newdataset-pmxsv` / `1`
- **Link**: https://universe.roboflow.com/pothole-ztskn/newdataset-pmxsv/dataset/1
- **Task**: object detection, 4 classes: `Drain Hole`, `Pothole`, `Sewer Cover`, `speed-breaker`
- **Positive class**: `Pothole`

One image is a photo of a road surface with bounding boxes around each pothole and around three other road features. The images are a mix of sources: close-up phone photos taken standing on the road, web-style photos of manhole covers, and frames taken from vehicle-mounted video (the speed-breaker images, which appear to come from Indian city roads). All were resized to 640 × 640 by stretching, and Roboflow applied no augmentation to version 1.

The dataset serves the problem, not just resembles it, for one main reason: the three non-pothole classes are exactly the things most likely to make the car slow down for nothing. Speed bumps, sewer covers and drain holes are dark, round or raised shapes on the road that a pothole detector could easily confuse with a pothole. Because they are labelled as separate classes, the model is explicitly taught to tell them apart, which a pothole-only dataset would not do.

Its main gaps relative to the real task are covered below: it contains almost no plain road, potholes appear in about half the images instead of a small share, and many pothole photos are close-ups rather than views from a car.

## What the dataset's composition implies

Computed from the label files by `train.py` (`outputs/dataset_stats.json`).

| Split | Images | Share | Background-only images | Images containing a pothole |
|---|---:|---:|---:|---:|
| train | 4,679 | 70% | 1 | 2,371 (50.7%) |
| valid | 1,335 | 20% | 1 | 672 (50.3%) |
| test | 670 | 10% | 0 | 332 (49.6%) |
| **total** | **6,684** | | **2** | **3,375** |

Instances per class (images containing the class in brackets):

| Class | Train | Valid | Test | Total |
|---|---:|---:|---:|---:|
| Pothole | 5,044 (2,371) | 1,379 (672) | 668 (332) | 7,091 |
| speed-breaker | 3,778 (2,052) | 1,024 (584) | 502 (294) | 5,304 |
| Sewer Cover | 240 (192) | 53 (48) | 37 (34) | 330 |
| Drain Hole | 88 (88) | 32 (32) | 15 (15) | 135 |

Box area as % of image area (test split; train is similar):

| Class | p10 | median | p90 |
|---|---:|---:|---:|
| Pothole | 0.57% | 4.04% | 23.24% |
| speed-breaker | 0.37% | 5.40% | 18.33% |
| Sewer Cover | 0.79% | 8.08% | 61.34% |
| Drain Hole | 4.80% | 24.97% | 53.76% |

Exact-duplicate images across splits (MD5): 0.

**Which numbers are too thin to trust.** Pothole (668 test instances in 332 images) and speed-breaker (502 in 294) are large enough for stable per-class metrics. Sewer Cover (37 instances in 34 test images) and Drain Hole (15 in 15) are not: for Drain Hole a single image changes recall by almost 7 percentage points, so its per-class AP is only a rough indication. This matters less than it seems, because those classes only need to be good enough not to be mistaken for potholes.

**The test split does not look like deployment.** In the test split, 49.6% of images contain a pothole. On the bad roads this system is meant for, sampled every 50 m, I assume about 3%. Precision measured on the test set will therefore be far too optimistic. With recall *r* and false-positive rate *f* measured per image, precision on the road is

  precision = 0.03·r / (0.03·r + 0.97·f)

For example, a model with 90% recall and a 5% false-positive rate looks about 95% precise on the test split but would be only about 36% precise on the road: almost two in three decelerations would be for nothing. I report precision both ways and use the 3% base rate for all cost calculations.

**Small potholes are where it will struggle.** One in ten potholes covers less than 0.57% of the image, roughly a 39 × 39 pixel box at the 512 px training resolution. These are mostly distant potholes, which are also the ones that leave the car the most time to slow down, so recall on them matters more than the headline figure suggests.

**What I cannot conclude.** No exact duplicates were found across splits, and Roboflow applied no augmentation, so there are no augmented copies of the same photo.

## Finetuning and baselines

- **Where trained**: GitHub Codespace, CPU only (2 cores, no GPU), roughly 20–25 minutes per epoch, resumed from `last.pt` across several sessions
- **Settings**: `yolo11n.pt`, imgsz 512, epochs 15, batch 8, patience 5, seed 0

I used object detection rather than image classification for two reasons. First, labelling the look-alike classes separately teaches the model to distinguish them from potholes, which directly reduces needless decelerations. Second, the box location and size can later be used to judge whether the pothole is in the car's lane and how close it is. For the cost analysis, each image is reduced to one score: the highest `Pothole` confidence among its detections. YOLO11n, the smallest YOLO11 model, was chosen because it has to run inside the car in real time and trains in reasonable time on limited hardware; 512 px input keeps it fast at some cost to small, distant potholes.

Results on the same test split (670 images):

Image-level columns use the chosen operating threshold of 0.08 (see the cost section); at YOLO's default 0.25 the finetuned model has 96.1% recall, 99.1% test precision and 77.0% precision on the road.

| Model | Pothole AP50 | mAP50 (all classes) | Image-level recall | Image-level precision (test, 49.6%) | Image-level precision (road, 3%) |
|---|---:|---:|---:|---:|---:|
| YOLO11n finetuned | 0.847 | 0.853 | 99.7% (331 / 332) | 98.8% | 72.3% |
| YOLO11n COCO, zero-shot | 0 | 0 | 0% | — | — |
| Trivial: always decelerate | — | — | 100% | 49.6% | 3.0% |

The COCO model has no pothole, speed-bump, sewer-cover or drain-hole class, so it cannot detect any of them without finetuning. On the pothole test images it only fires on cars (191), people (168), trucks, motorcycles and other road users. Its score of zero shows that the useful ability comes entirely from finetuning. The trivial "always decelerate" baseline never hits a pothole at speed, but 97% of its decelerations on real roads are for nothing: the car would crawl everywhere.

The gap is larger than I expected for the smallest YOLO model trained on a CPU: mAP50 goes from 0 to 0.85, and on real roads roughly 7 in 10 decelerations would be justified instead of 3 in 100. All 15 epochs ran without early stopping, and validation mAP50 was still rising at the end (0.51 after epoch 1, 0.87 after epoch 13, 0.88 after epoch 15), so the model is under-trained and more epochs would probably help.

## Per-class evaluation and failure modes

| Class | Test instances | Precision | Recall | AP50 |
|---|---:|---:|---:|---:|
| Pothole | 668 | 0.881 | 0.702 | 0.847 |
| speed-breaker | 502 | 0.949 | 0.886 | 0.968 |
| Sewer Cover | 37 | 0.938 | 0.817 | 0.924 |
| Drain Hole | 15 | 0.744 | 0.667 | 0.673 |

**Which class carries the headline and which drags it down.** speed-breaker (AP50 0.97) and Sewer Cover (0.92) pull the 0.85 mAP50 up; Drain Hole (0.67) pulls it down. Pothole, the class that matters, sits right at the average (0.85). Its weakness is recall: at box level it finds only 70% of potholes. The Drain Hole and Sewer Cover figures rest on 15 and 37 boxes, so they are indicative only.

**Confusion matrix** (test split, conf 0.25, rows = predicted, columns = actual):

| Predicted ↓ / Actual → | Drain Hole | Pothole | Sewer Cover | speed-breaker | background |
|---|---:|---:|---:|---:|---:|
| Drain Hole | 11 | 0 | 0 | 0 | 3 |
| Pothole | 0 | 552 | 0 | 1 | 226 |
| Sewer Cover | 1 | 0 | 34 | 0 | 13 |
| speed-breaker | 0 | 0 | 0 | 485 | 116 |
| background (missed) | 3 | 116 | 3 | 16 | — |

**The look-alikes are not the problem I expected.** No sewer cover or drain hole box and only 1 of 502 speed-breakers was labelled as a pothole, so the separate classes did their job. The real errors are on the Pothole class itself: 116 of 668 pothole boxes are missed and 226 pothole boxes have no matching label. These errors mostly sit inside images that do contain a pothole. At conf 0.25 only 3 of 338 pothole-free images get any pothole box, and only 13 of 332 pothole images get none. So the extra boxes are mostly duplicates or fragments of large, irregular potholes, or nearby potholes the annotators did not label. The missed boxes are mostly second and third potholes in a frame where another one was found. For the car, one detected pothole in the frame is enough to slow down, which then also protects it from the others, so this matters much less than the 70% box recall suggests.

**What the image-level errors at the chosen threshold have in common** (all 5 inspected by hand):

- *The one missed pothole* (score 0.04) is a water-filled pothole in a TV news screenshot: the puddle reflects the sky, the photo is an extreme close-up, and a caption banner covers part of it. There is no visible hole edge. With the chosen thresholds the car would continue without even a warning. Water-filled potholes are a real risk after rain.
- *The four false alarms* all come from look-alikes or worn, broken surfaces. They are a drain grate with rows of dark rectangular openings (score 0.799), a cracked and worn crosswalk stripe next to a manhole cover (0.64, the Pothole box sits on the stripe, not the cover), a sewer cover set in broken paving (0.19) and a speed bump with worn zebra paint (0.28). With the chosen thresholds, all four only trigger a warning, and none makes the car slow down; the drain grate is just 0.001 below the deceleration threshold. Worn road paint appears twice, so painted markings are a failure mode to watch on real roads.

## Cost matrix and operating point

All costs are per image, meaning one camera frame (about 50 m of road), for one car and one driver (`costs.json`). They are euro-equivalents of total harm, not only money: the "EUR" figures in this table and in every cost below, including the cost per day, should be read in that sense. Only cost_fn is anchored to a real repair cost; cost_fp and cost_review are set relative to it (20 times and 100 times smaller).

| Field | Value | Range explored | Where it comes from |
|---|---:|---|---|
| cost_fn | 40 EUR | 10 – 300 EUR | Car hits the pothole at speed. One hit in 10 causes about 400 EUR of damage (tyre, rim or suspension), the other nine cause none: 400 / 10 = 40 EUR expected per hit. Only this car is affected. |
| cost_fp | 2 EUR | 0.5 – 5 EUR | Needless deceleration: a few seconds lost and a less comfortable ride, no damage. Judged 20 times less bad than a hit: 40 / 20 = 2 EUR. |
| cost_tp | 0 EUR | 0 – 5 EUR | The car slows down for a real pothole and the driver avoids it. The short slowdown is justified, so it costs nothing extra. |
| cost_tn | 0 EUR | 0 | The car continues as normal. |
| base_rate | 0.03 | 0.005 – 0.15 | Assumed share of frames on badly maintained roads that contain a pothole (one about every 1.7 km). Not taken from the dataset, where it is about 50%. |
| cost_review | 0.40 EUR | 0.05 – 2 EUR | One warning to the driver, without slowing down: a distraction from mirrors and traffic plus a little alert fatigue. Judged to cost 5 times less than a slowdown: 2 / 5 = 0.40 EUR. |
| cases_per_day | 5,000 | 1,000 – 10,000 | One working day of a professional driver (delivery van or taxi): about 250 km in a 7-hour shift, one frame every 50 m. |
| review_seconds | 15 | 5 – 30 | Driver attention one warning takes: hearing the chime, scanning the road ahead, deciding, and settling back. Used for workload, not cost. |
| reviewer_hours_per_day | 7 | 6 – 8 | Driving hours in one shift. |

These figures are my own estimates, not sourced data; the range column shows how far each could plausibly be off.

**Theoretical threshold.** If the model's scores were calibrated probabilities, cost would be minimised by decelerating whenever P(pothole) > cost_fp / (cost_fp + cost_fn) = 2 / 42 ≈ **0.05**. That low bar reflects the 20 : 1 asymmetry. YOLO confidence scores are not calibrated, so the actual threshold comes from the cost sweep below.

**Cost sweep.** If the model decides alone (only continue or decelerate, no warnings), the expected cost per day at threshold *t* is

  cost(t) = 5,000 × [0.03 × (1 − recall(t)) × 40 + 0.97 × FPR(t) × 2]

where recall and FPR are measured per image on the test split.

![Expected cost per day against threshold](outputs/cost_curve.png)

*The blue line is the cost if the model decides alone with a single threshold: the car continues below it and decelerates above it. The shaded band shows how my three-outcome design splits the same score axis: continue below 0.08, warn the driver between 0.08 and 0.80, decelerate at 0.80 or above. The dotted line is that design's cost, 61 EUR per day.*

**Chosen operating point**: threshold **0.08**, giving image-level recall **99.7%** (1 of 332 pothole images missed), false-positive rate **1.2%** (4 of 338 pothole-free images), and an expected cost of **133 EUR per day** (about 30 EUR in potholes hit and 103 EUR in needless decelerations). That is close to the theoretical 0.05, so at the low end the scores behave roughly like probabilities. The curve is steep on both sides: at YOLO's default 0.25 the cost is 321 EUR per day, and by 0.5 it is over 700 EUR per day as more and more potholes are hit at speed.

**Adding the warning.** The cost sweep over both thresholds (`costing.py`, every lower/upper pair in steps of 0.02) finds the cheapest three-outcome design at **continue below 0.08, warn between 0.08 and 0.80, decelerate at 0.80 or above**, for **61 EUR per day**. The lower threshold stays where the model-alone threshold was. The upper threshold rises to 0.80, so frames between 0.08 and 0.80, which the model alone would have slowed down for, now only get a warning.

**Sensitivity (one assumption at a time, others at their base value):**

| Assumption | Values tried | Best single threshold | Best three-outcome band (continue / warn / decelerate) |
|---|---|---|---|
| base_rate | 0.5% / 1% / 3% / 8% / 15% | 0.29 / 0.08 / 0.08 / 0.08 / 0.08 | 0.08–0.80 up to 3%; 0.08–0.64 at 8%; 0.02–0.30 at 15% |
| cost_fn | 10 / 20 / 40 / 100 / 300 EUR | 0.20 / 0.08 / 0.08 / 0.08 / 0.08 | 0.08–0.80 up to 100 EUR; 0.02–0.80 at 300 EUR |
| cost_fp | 0.5 / 1 / 2 / 5 EUR | 0.08 throughout | 0.08–0.20 at 0.5 EUR; 0.08–0.64 at 1 EUR; 0.08–0.80 from 2 EUR |
| cost_review | 0.05 / 0.1 / 0.2 / 0.4 / 0.6 / 0.8 / 1 / 2 EUR | 0.08 throughout | 0.02–0.80 up to 0.1 EUR; 0.08–0.80 at 0.2–0.6 EUR; 0.08–0.64 at 0.8 EUR; 0.08–0.30 at 1 EUR; no warnings at 2 EUR |

I expected the base rate to be the assumption that most changes the answer, and it does have the largest effect on cost and on precision on the road. However, the point where the car stops continuing (0.08) is surprisingly stable: it holds for any base rate between 1% and 15% and any cost_fn from 20 to 300 EUR. It moves only if potholes are rarer than about 1 frame in 100 (to 0.29) or a hit costs as little as 10 EUR on average (to 0.20).

The **band depends mostly on how a warning compares with a slowdown, and my 5 : 1 ratio sits comfortably inside the range where it holds.** The band stays at 0.08–0.80 for any warning cost from 0.20 to 0.60 EUR (a slowdown 3.3 to 10 times worse than a warning), any cost_fp from 2 to 5 EUR, any cost_fn up to 100 EUR and base rates up to 3%. It only changes at the extremes. If a warning were nearly free (0.10 EUR or less), the car should also warn between 0.02 and 0.08. If a warning cost half a slowdown (1 EUR), the car should already decelerate from 0.30. If a warning were as bad as a slowdown, warnings would never pay off and the car should simply decelerate above 0.08.

The assumption I cannot test, and that would change everything, is the **false-positive rate on plain road**. The whole curve rests on 4 false alarms among 338 look-alike images. If ordinary road with shadows and patches produced even a 3% false-alarm rate, the cost of needless decelerations and warnings would almost triple, and fewer than half of all decelerations would be justified.

## What a person still has to check

The "person" in this system is the driver. The car uses two thresholds on the image-level pothole score:

- **Fail, at or above the upper threshold (0.80): decelerate.** The car slows down by itself, and the driver deals with the pothole.
- **Pass, below the lower threshold (0.08): continue as normal.**
- **Review, in between: warn the driver without decelerating.** The driver looks at the road and decides.

A warning costs 0.40 EUR, five times less than a needless slowdown (2 EUR), so frames the model is unsure about get a warning instead of a deceleration. Each warning takes about 15 seconds of the driver's attention.

| Per driver, per 7-hour shift (5,000 frames) | Value |
|---|---:|
| Share of frames that trigger a warning | 2.1% |
| Warnings per day | 106 (about 15 per hour, one every 4 minutes) |
| Driver attention spent on warnings | 27 minutes (6% of the shift) |
| Decelerations per day | 101, all for real potholes on the test split |
| Potholes driven over without a warning | about 0.5 per day |
| Extra people needed | none: the driver handles the warnings |

On the test split, pothole frames lead to a deceleration 67% of the time, a warning 33% and no action 0.3% (the one water-filled pothole). Pothole-free frames lead to no action 98.8% of the time and a warning 1.2%; none led to a needless slowdown. The expected 61 EUR per day breaks down as 42 EUR of warnings and 18 EUR of potholes hit.

**Is that realistic? Yes, with one caveat.** A warning every four minutes and under half an hour of attention per shift is manageable, and every deceleration on the test split was for a real pothole, so the driver has no reason to distrust the car's slowdowns. About half of the warnings are for frames with no pothole (57 false, 49 real per day). That is close to the level at which drivers start ignoring warnings, and an ignored warning on a real pothole costs 40 EUR instead of 0.40 EUR. The design only beats the model alone if the driver reacts to at least 96% of real warnings.

**Design compared with the alternatives** (expected cost per day, one driver, 5,000 frames, 3% base rate):

| Option | How it is priced | Cost per day | Driver attention needed |
|---|---|---:|---:|
| Do nothing (always continue) | 150 potholes/day × 40 EUR, all hit at speed | 6,000 EUR | 0 |
| Always decelerate | 4,850 empty frames/day × 2 EUR | 9,700 EUR | 0 |
| Warn the driver on every frame | 5,000 warnings × 0.40 EUR, assuming the driver never misses one | 2,000 EUR | 21 hours (impossible in a 7-hour shift) |
| Model decides alone (continue or decelerate) | cost(t) at threshold 0.08 | 133 EUR | 0 |
| Three outcomes (my design) | errors outside the band + 0.40 EUR per warning | 61 EUR | 27 minutes |

Two things stand out. First, with no system the driver hits about 150 potholes a day, which at one damaging hit in ten means about 15 repairs a day. That is the 6,000 EUR every option must be judged against, and every model-based option removes about 98–99% of it. Second, leaving everything to the driver does not work: a warning every five seconds costs 2,000 EUR a day and would need 21 hours of attention in a 7-hour shift.

**The three-outcome design costs 61 EUR per day, 72 EUR (54%) less than the model alone.** The warning replaces the needless slowdowns: frames between 0.08 and 0.80, which the model alone would have slowed down for, now only get a cheaper warning, and the car slows down by itself only when it is very confident. These figures rest on few test images (1 pothole missed, 4 pothole-free frames warned, no needless slowdowns), and each image moves a rate by about 0.3 percentage points. Zero needless slowdowns in 338 images is still consistent with a true rate of up to about 0.9% (rule of three), which would add about 86 EUR per day and make the design about 147 EUR per day, slightly more than the model alone. The saving is therefore likely but not certain, and it needs to be confirmed on more pothole-free images.

## Limitations & next steps

**Base rate and false-alarm rate on real roads are unknown.** The dataset has about 50% pothole prevalence and only 2 images of plain road, so the two quantities that drive cost most are assumptions. *Next step:* record a few hours of driving on Madrid roads, label one frame every 50 m, and measure the real base rate and the false-positive rate on ordinary road surfaces.

**No plain-road negatives in training.** The model has never been trained on shadows, patched asphalt, cracks or wet tarmac without another object present. *Next step:* add a few thousand background-only frames to training and test, and re-measure the false-positive rate.

**Frames are not potholes.** As the car approaches, the same pothole appears in several consecutive frames, so a pothole missed in one frame is often caught in the next. Per-frame costs therefore overstate the cost of a single miss. *Next step:* track detections across consecutive frames and evaluate per pothole, with a rule such as "decelerate after two detections in a row".

**Thresholds were chosen on the test split.** `costing.py` sweeps thresholds on the same 670 test images it reports results for, and the optimum rests on a handful of images (1 miss, 4 false alarms). The reported costs are therefore somewhat optimistic. *Next step:* choose the threshold and band on the validation split, then report cost once on the untouched test split.

**Small model, short CPU training.** YOLO11n at 512 px for 15 epochs was what the available CPU could handle, and validation mAP was still rising at the last epoch. Box recall on potholes is only 70%, and I did not measure how much of that comes from small, distant potholes, which give the car the most time to slow down. *Next step:* retrain YOLO11s at 640 px on a GPU until early stopping triggers, and compare recall on the smallest tenth of pothole boxes.

**Cost figures are estimates, and mostly not money.** The one-in-ten damage rate, the 400 EUR repair and the 20 : 1 and 5 : 1 ratios are reasoned, not measured, and the slowdown and warning costs measure inconvenience and distraction rather than spending. The cost per day is therefore a way to compare designs on one scale, not a forecast of what anyone will pay. *Next step:* use insurer claims data on pothole damage to firm up the anchor, and a small driver study (for example, asking drivers to rank a needless slowdown, a warning and a pothole hit) to replace the two ratios.

## Generative AI use disclosure

- AI was used to write the pipeline code (`train.py`, `costing.py`, `app.py`) and to compute and format the dataset tables.
- AI was used to draft and structure this report. All model results, thresholds and conclusions come from my own training and costing runs.
