# Assignment 2 Report

**Deciding when to send a salt truck from dashcam images**

- **Name**: Niccolò Pragliola
- **Student ID**: 19873
- **Email**: npragliola.ieu2022@student.ie.edu
- **Group**: BBADBA 5A
- **Repo**: https://github.com/nicopragliola/daiist-assignment-2 (branch `assignment-2`)
- **Dataset**: Roboflow Universe `udot-classification` / `udot-winter-road-conditions` / version 1

## Summary

I finetuned `yolo11n-cls` on 6,245 dashcam images of Utah roads labeled Clear, Wet, Slush or Snow (90.7% test accuracy against 55.4% for always answering Snow), and turned its probability that a road needs treatment into “pass”, “operator review” or “send a truck”. With €65 for an unnecessary truck pass, €800 for a missed snowy road and a 15% base rate, the cheapest threshold is 0.51, which matches the break-even of 0.54, while an operator review band brings the cost from €17,070 to €10,038 per 1,000 images, close to a perfect model's €9,750, with 1.7 operators instead of 5

---

## 1. Business problem, and the cost of being wrong

On winter nights a road-maintenance control room has to decide, for every stretch of road, whether to send a salt truck, and today this is done by the night-shift supervisor/operator who dispatches trucks segment by segment, partly by looking at road camera images. The problem is volume, as a region of about 6,000 lane-miles split into 1,200 segments of 5 lane-miles and checked every 2 hours over a 12-hour night gives 7,200 images a night, which at 15 seconds each is 30 hours of looking while only two operators are on shift.

The model does the first sort, by scoring each image and giving one of three outcomes:
1. “Pass” below the low threshold, and nothing is done
2. “Operator review” between the two thresholds, where an operator looks at the image for about 15 seconds and makes the call
3. “Send a truck” from the high threshold up

One case is one image of one segment in one 2-hour cycle, and the positive class is “needs treatment” (Snow or Slush) while Clear and Wet are negative.

| | Meaning | Cost |
|---|---|---|
| True positive | Snowy road, truck sent | €65 |
| False positive | Fine road, truck sent | €65 |
| False negative | Snowy road, no truck | €800 |
| True negative | Fine road, nothing done | €0 |

A missed snowy road leaves a segment untreated for two hours while traffic keeps using it, and from crash data (Section 6) that costs about €800 in expected crashes, while an unnecessary truck pass wastes fuel, salt and a driver's time for about €65. A miss is therefore about 12 times worse than a false alarm, and its cost falls on drivers rather than on the road budget. A true positive isn't free either, as the truck still drives, but it buys €800 of avoided crash cost for €65, which means a truck is worth sending once the chance of snow passes about 8% (65 / 800), far below 50%.

I count Wet as negative, as salt is spread to melt snow and slush and a wet road has nothing for salt to remove. This matters as Wet is 20.5% of the images, but the weak point is temperature, as wet pavement can freeze below 0 °C and a camera can't see that, which I treat as a limitation (Section 8) rather than sending a truck to every wet road.

---

## 2. Dataset

| | |
|---|---|
| Roboflow Universe | Workspace `udot-classification`, project `udot-winter-road-conditions`, version 1 (generated January 30th, 2026), used as published and not forked |
| Link | https://universe.roboflow.com/udot-classification/udot-winter-road-conditions/dataset/1 |
| License | CC BY 4.0 |
| Task | Image classification, 4 classes (Clear, Slush, Snow, Wet) |
| Images | 6,245, with no preprocessing or augmentation by the author |

Each image is a frame from a vehicle dashcam on winter roads in Utah, with the car's hood visible at the bottom (5,232 are 3840×2160 video frames and 1,013 are 640×640 with black letterbox bars). Before choosing it I checked that the version-1 export puts every image in exactly one class folder, as the project page lists 8,496 labels for 6,245 images, and that it comes with its own train/valid/test split, so nothing had to be re-split. I rejected the “Roads” dataset (`roads-pjwmf/roads-q0nh0`, version 2), which has an Ice class, but whose images are augmented 320×320 close-ups of asphalt that look nothing like a camera view.

The dataset fits the decision as its four labels are the four surface states that decide whether a truck is needed, and the treatment line falls cleanly between classes, so no relabeling was needed. The one way it differs from my scenario is the camera, as these are dashcam frames from a moving car while a control room would use fixed roadside cameras, mounted higher and at a different angle, so the results are likely optimistic until the model is tested on those cameras.

---

## 3. What the dataset's composition implies

| Class | Train | Valid | Test | Total | Share | Taken at night |
|---|---|---|---|---|---|---|
| Clear | 410 | 117 | 52 | 579 | 9.3% | 24.9% |
| Slush | 762 | 203 | 107 | 1,072 | 17.2% | 9.7% |
| Snow | 2,289 | 678 | 344 | 3,311 | 53.0% | 53.4% |
| Wet | 902 | 263 | 118 | 1,283 | 20.5% | 11.6% |
| **Total** | **4,363** | **1,261** | **621** | **6,245** | | |

The split is 69.9% train, 20.2% valid and 9.9% test, and 69.9% of train and 72.6% of test (451 of 621) needs treatment, while object size doesn't apply as this is classification. I checked two more things, by measuring brightness and by fingerprinting every image:
1. Night images (mean gray level below 45 out of 255) are 53.4% of Snow against 9.7 to 24.9% for the other classes
2. 22.8% of validation and 21.1% of test images (131 of 621) have a near-identical twin in train, using a 256-bit fingerprint where two images that differ in 25 bits or fewer count as the same scene (a cut-off I set by checking pairs by eye)

Clear only has 52 test images, so one image moves its recall by 1.9 points and a rough 95% interval around its 0.788 recall runs from 0.68 to 0.90, which makes its numbers indicative only, but a Clear/Wet mix-up doesn't change the truck decision anyway. Snow being mostly dark means darkness could become a shortcut (“dark means snow”), which would hurt exactly at night when the system runs, and the near-twins mean part of the test measures memory rather than generalization, as the split is by image rather than by video. Section 4 checks both.

Given the test set is 72.6% snowy only because someone filmed winter roads mostly in snow, pricing that mix would count far too many misses, so I measure rates within each true group (snowy and fine) and recombine them at a 15% base rate, my real-world estimate from Section 6. The difference is large, as on the test mix the model's precision is 0.967 while at 15% about one truck in three (75 needless for every 147 needed per 1,000 images) would go to a road that didn't need it.

---

## 4. Finetuning and baselines

I finetuned `yolo11n-cls` (1.5M parameters) from its ImageNet checkpoint, as the decision is one per image and a classifier gives four probabilities that become one score, so the threshold works exactly as in Assignment 1, while detection would mean drawing boxes around a road surface that has no natural box. It's the smallest model the brief allows, it scores an image in about 3 ms (a whole night's 7,200 images in about 22 seconds) and Section 7 shows that with operator review the system is within €288 per 1,000 of a perfect model, so a bigger model wasn't needed.

I trained it on a 4-core Intel Xeon CPU cloud machine (Python 3.12, PyTorch 2.11 CPU build, Ultralytics 8.4.173) for 30 epochs at 224 px, batch 64 and seed 0, with Ultralytics' default augmentation and optimizer (AdamW, learning rate 0.00125), which took 47 minutes. Before training I shrank every image to 640 px on its longest side, as the model only sees 224 px and decoding 4K frames every epoch would waste most of the training time, and I checked that Ultralytics picks the best epoch on the `valid` folder, as without one it would silently use test. The best epoch was 28 (validation accuracy 0.890), as validation accuracy flattened around 0.87 to 0.89 after epoch 10 while training loss kept falling.

I compared it on the same 621 test images against “always Snow” (the majority class, which means a truck everywhere), “predict nothing” (never a truck) and the pretrained model used zero-shot. The pretrained model only knows the 1,000 ImageNet labels, so its “snow score” is the probability it gives to snow-related ones (alp, snowplow, snowmobile, ski, dogsled, bobsled and shovel).

| Approach | 4-class accuracy | “Needs treatment” precision | “Needs treatment” recall |
|---|---|---|---|
| Finetuned `yolo11n-cls` (threshold 0.51) | **0.907** | **0.967** | **0.978** |
| Always Snow (truck everywhere) | 0.554 | 0.726 | 1.000 |
| Predict nothing (never a truck) | n/a | n/a | 0.000 |
| Pretrained zero-shot (threshold 0.01) | n/a | 0.750 | 0.911 |

| Test subset | Images | Accuracy |
|---|---|---|
| All | 621 | 0.907 |
| No near-twin in train | 490 | 0.892 |
| Daytime | 388 | 0.889 |
| Night | 233 | 0.936 |

The gap to the trivial baseline is what I expected, as always answering Snow just scores the Snow share (55.4%) while the model reaches 90.7%. The zero-shot result was more telling, as its cheapest threshold is 0.01, which means it flags almost everything (precision 0.750, barely above the 72.6% snowy share) and costs €64,053 per 1,000, almost the €65,000 of a truck everywhere, while its top labels (alp 74, snowplow 70, snowmobile 63, traffic light 60, car mirror 50) show it sees a snowy hillside or the inside of a car, not the pavement.

On the 490 test images with no near-twin accuracy is 89.2%, 1.5 points below the full split, so the twins flatter the headline slightly but don't explain it. Night images score 93.6% and daytime 88.9%, but I don't read this as the model being better in the dark, as 82% of night test images are Snow, the easiest class, and what it shows is that the model holds up at night, while whether it uses darkness as a shortcut would need a class-by-class day/night breakdown.

---

## 5. Per-class evaluation and failure modes

| Class | Test images | Precision | Recall |
|---|---|---|---|
| Clear | 52 | 0.788 | 0.788 |
| Slush | 107 | 0.812 | 0.850 |
| Snow | 344 | 0.971 | 0.971 |
| Wet | 118 | 0.858 | 0.822 |

| Actual \ Predicted | Clear | Slush | Snow | Wet |
|---|---|---|---|---|
| Clear | **41** | 2 | 2 | 7 |
| Slush | 0 | **91** | 7 | 9 |
| Snow | 1 | 9 | **334** | 0 |
| Wet | 10 | 10 | 1 | **97** |

| At threshold 0.51 | Truck sent | No truck |
|---|---|---|
| Needs treatment (451) | 441 (TP) | 10 (FN) |
| Fine (170) | 15 (FP) | 155 (TN) |

Snow carries the headline, as it's 344 of the 621 test images with precision and recall of 0.971, while on the other three classes together accuracy is 82.7% (229 of 277), with Clear dragging most (0.788), then Wet (recall 0.822) and Slush (precision 0.812). 52 of the 58 errors are between neighboring states on the scale Clear, Wet, Slush, Snow, which fits how the labels cut a continuous change in the road surface into four bins.

Of the 58 errors, 33 don't change the truck decision (16 Snow/Slush mix-ups still get a truck and 17 Clear/Wet mix-ups still get none) while 25 cross the line, which are 10 misses (9 Slush read as Wet and 1 Snow as Clear) and 15 needless trucks (10 Wet read as Slush, 4 Clear as Slush/Snow and 1 Wet as Snow). So the money sits on one boundary, as 19 of the 25 decision errors are between Wet and Slush and 9 of the 10 misses are slush read as a wet road, which makes Slush the hard positive and where better data would pay most.

---

## 6. Cost matrix and operating point

| | Truck sent | No truck | Operator review |
|---|---|---|---|
| Road needs treatment | €65 (TP) | €800 (FN) | €0.125 + €65 |
| Road fine | €65 (FP) | €0 (TN) | €0.125 |

All figures are in `assumptions.json`, with dollars converted at about 0.9 EUR per USD:
1. Truck pass €65 (TP and FP), as UDOT spends about $1M per storm on 24,000 lane-miles, which is about $210 per 5-lane-mile segment per storm, or about $70 per pass assuming 3 passes per storm
2. Missed snowy road €800 (FN), as a crash on snowy pavement costs about €12,000 (174,446 crashes, 41,597 injured and 538 deaths a year in the US, priced at about $4,200 per property-damage crash and $21,000 per injury crash in 2015 values and $1.6M per death), and treating a snowy segment for 2 hours avoids about 0.07 crashes (1,000 vehicles over 2.5 road-miles, a dry crash rate of about 2 per million vehicle-miles that's 15 times higher on snow, cut by 88% by salting)
3. Operator review €0.125, as 15 seconds at €30 an hour including overheads
4. Base rate 15%, as Salt Lake City has about 31 snowy days a year (about 21% of a 150-day winter), and I assume roads are snowy in about two thirds of the night's checks on those days
5. 7,200 images a night with room for 2,880 reviews, as 2 operators spend 6 hours each of a 12-hour shift on reviews at 240 images an hour

The score is P(Snow) + P(Slush), and I measure the share of snowy and fine images that are passed, reviewed or sent a truck (on validation to choose thresholds, on test to report them), and reweight them to 15% snowy, assuming the operator always makes the right call.

```
cost if snowy = passed × 800 + reviewed × (0.125 + 65) + truck × 65
cost if fine  = passed × 0   + reviewed × 0.125        + truck × 65
cost per 1,000 images = 1000 × (0.15 × cost if snowy + 0.85 × cost if fine)
```

I tried every threshold from 0.00 to 1.00 in steps of 0.01 on validation, and for the model alone 0.51 was cheapest, which costs €17,070 per 1,000 images on test (3.3 missed snowy roads and 75 needless trucks per 1,000). Raising the threshold sends fewer trucks, so false positives fall while false negatives rise, and the cost is lowest around 0.51 before rising sharply near 1.0 as €800 misses take over, while near 0 a truck goes almost everywhere and the cost approaches €65,000.

![Cost per 1,000 images against threshold, with the chosen threshold marked](results/cost_curve.png)

A truck is worth sending when €65 is less than the expected €800 of not sending, so once P(snow) passes 65 / 800 = 0.081. Given the model learned on 69.9% snow while reality is 15%, its scores run higher than real probabilities, and converting with odds (0.088 real-world odds times 13.2, the ratio of training to real-world odds) gives a model-score threshold of 0.54, close to the 0.51 the sweep found and the same pattern as Assignment 1 (0.38 found, 0.377 break-even).

| Changed | Value | Best threshold | € per 1,000 (model alone) |
|---|---|---|---|
| Missed snowy road | €200 | 0.90 | 12,823 |
| Missed snowy road | €400 | 0.53 | 15,739 |
| Missed snowy road | **€800 (used)** | **0.51** | **17,070** |
| Missed snowy road | €1,600 | 0.43 | 20,055 |
| Missed snowy road | €3,200 | 0.05 | 23,793 |
| Truck pass | €32.50 | 0.43 | 10,028 |
| Truck pass | €130 | 0.53 | 31,478 |
| Truck pass | €260 | 0.90 | 51,290 |
| Base rate | 5% | 0.53 | 9,513 |
| Base rate | 25% | 0.43 | 24,913 |
| Base rate | 40% | 0.05 | 35,828 |

Changing one figure at a time (full table in `results/sensitivity.csv`), the threshold stays between 0.43 and 0.53 for a miss of €400 to €1,600, a truck pass of €32.50 to €130 and a base rate of 5 to 25%, and the model beats both “do nothing” and “truck everywhere” in every row.

I trust the truck pass most, as it comes from UDOT's own budget and my only addition is 3 passes per storm (2 passes would give about €95 and 4 about €47, both inside the range where the threshold barely moves). I trust the miss cost least, as it multiplies four estimates (a national crash cost, a 15 times risk on snow, an 88% effect from a 1992 study and 1,000 vehicles in two hours) and the traffic figure isn't sourced at all, while the base rate sits in between and the review cost barely matters, as a review is 520 times cheaper than a needless truck.

The assumption that would most change the answer is traffic, as the threshold jumps to 0.90 when a miss costs about three truck passes (€200, which a quarter of the traffic gives) and drops to 0.05 when misses cost €3,200 or snow is 40% common. Given a quiet rural road at 3 a.m. and an interstate can differ by more than four times, I'd set the threshold by road class rather than once for the whole region.

---

## 7. What a person still has to check

Chosen on validation within the operators' capacity, images below 0.03 pass and the rest go to an operator, with only those scoring 1.00 sent straight to a truck.

| On the test images | Pass | Operator review | Send a truck |
|---|---|---|---|
| Fine (170) | 129 | 41 | 0 |
| Needs treatment (451) | 1 | 421 | 29 |

At 15% snowy that sends 34.5% of images to an operator (2,484 a night), which at 15 seconds each is 10.4 hours or 1.7 operators against the 2 on shift, leaving 396 reviews of slack (about 414 per 2-hour cycle against 480).

| Policy | € per 1,000 | € per night | Operators | Missed /1,000 | Needless trucks /1,000 |
|---|---|---|---|---|---|
| Perfect model (reference) | 9,750 | 70,200 | 0 | 0 | 0 |
| Do nothing | 120,000 | 864,000 | 0 | 150 | 0 |
| Truck everywhere | 65,000 | 468,000 | 0 | 0 | 850 |
| Operator checks every image | 9,875 | 71,100 | 5.0 | 0 | 0 |
| Pretrained zero-shot alone | 64,053 | 461,183 | 0 | 13.3 | 685 |
| Finetuned model alone | 17,070 | 122,904 | 0 | 3.3 | 75 |
| **Finetuned model + operator review** | **10,038** | **72,271** | **1.7** | 0.3 | 0 |

On an average night it's realistic but tight, and since reviewing every 15 seconds for hours at night is tiring, the review hours should be split into shorter blocks. A heavy snow night is different, as 93.3% of snowy images go to review against 24.1% of fine ones, so the load reaches capacity at about 23% snowy checks and at 60% it would be about 4,700 reviews (3.3 operators). On those nights I'd lower the high threshold so confident cases go straight to a truck, as few roads are fine and a truck without review is rarely wasted (on an average night, moving it from 1.00 to 0.50 cuts reviews to 13.3% but raises the cost to €14,886 per 1,000).

Doing nothing costs €120,000 per 1,000 images (€864,000 a night) as 150 snowy roads in every 1,000 go untreated, a truck everywhere costs €65,000 and the model alone costs €17,070, while adding the review band brings it to €10,038 (€72,271 a night, about €50,600 less than the model alone). A perfect model would cost €9,750, which is just the trucks that are really needed, so my design is €288 above it (€244 from the one snowy test image it passed, which makes that figure uncertain, and €43 of operator time) against €7,320 for the model alone.

A person checking every image costs €9,875, €163 per 1,000 less than my design, but it needs 7,200 reviews and 30 hours a night, which is 5 operators against the 2 on shift, and the €0.125 per review only prices the seconds spent looking, not three extra people hired for every night shift. The model's value is that it takes 64.5% of images off the operators' screens.

The operator ends up confirming almost every truck, as with the high threshold at 1.00 only 29 of 451 snowy test images go straight to a truck. Sending a truck without a check costs €65 times the chance the road is fine while a check costs €0.125, so review wins whenever that chance is above 0.2%, which is almost every image. Cost alone would review everything, so what limits the band is the operators' capacity, and as about six in ten reviewed images are fine roads, the operator's main job at a 15% base rate is stopping needless trucks.

---

## 8. Limitations & next steps

1. The model has only seen dashcam frames from one state, while 62.5% of test images are daytime and the scenario runs at night. I'd run the system in shadow mode for part of a winter on the control room's own cameras (the model scores every image while operators decide as today, and both are logged), then label those images, finetune and re-run the costing, which also gives a real base rate and tests whether operators are always right
2. Near-twins and small classes make the test optimistic and noisy. I'd re-split so each cluster of near-twins sits in one split, collect more Clear and Slush images, and report per-class results with confidence intervals
3. The camera can't see temperature, so a wet road about to freeze is passed. I'd join each image with pavement temperature from road weather stations, and send any Wet image near or below 0 °C to review whatever its score
4. The miss cost assumes one traffic volume, and each image is judged on its own as if the road had never been salted. I'd price misses per segment from traffic counts, and feed in the time since the segment was last treated so a road salted in the previous cycle isn't sent another truck for the same snow

---

## 9. Generative AI use disclosure

I used Claude to write code before analyzing the results. All decisions about this project were mine, and I chose this topic as I genuinely see an unexplored use case to it. 


