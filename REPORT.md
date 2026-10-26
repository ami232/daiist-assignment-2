# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Carlota Navarro Calle
- **Student ID**: 19992
- **Email**: cnavarro.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

The decision this model supports is whether a banana currently on shelf or in stock should be sold, thrown away, or sent to a staff member for a quick manual check. Grocery and warehouse staff managing shelf stock are the ones acting on it.

The positive class includes the bananas that need to be thrown away because they are overripe or rotten. The negative class includes the bananas that are ripe or unripe and can therefore be sold (stores stock both, since customers buying several at once often want some less ripe for later in the week).

The two mistakes cost different amounts. False positives are the cases in which a sellable banana gets flagged as rotten and thrown away. The cost of this has been estimated to be €0.20, based on the price you would pay for a banana at the supermarket. A false negative occurs when a rotten banana is missed and stays on the store shelf. This is more expensive, for two reasons: a rotten banana in direct contact with others can cause them to spoil too, and visibly rotten fruit on display damages the store’s image. The total cost is assumed to be €3.00, broken down as €1.00 for roughly five neighboring bananas spoiled by contact at €0.20 each (using the same assumption as above), plus €2.00 as the assumed reputational cost of customers choosing not to buy bananas from the store that day.

I have assumed that true positives and true negatives cost €0. Correctly sorting a banana is not an extra cost as no meaningful staff time or resource is spent beyond normal operation.

## Dataset

The dataset comes from Roboflow Universe, workspace roboflow-universe-projects, project banana-ripeness-classification, version 1 (“original-images,” with no augmentation or class modification applied by the dataset author).
Link: https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification/dataset/1

Each image is a single banana photographed close-up against a plain background, labeled with its ripeness stage. The dataset’s raw labels include six classes: freshripe, ripe, freshunripe, unripe, overripe and rotten. Other versions of this same dataset apply augmentation and/or collapse these labels down to only two classes; version 1 is the only one that preserves the full original six, which is why it was chosen.

This dataset serves the business problem because it contains labeled images of bananas at different ripeness stages, allowing the model to distinguish sellable bananas from those that are overripe or rotten and should be removed. The six raw classes were merged into these three for exactly that reason: unripe (freshunripe + unripe), ripe (freshripe + ripe), and rotten (overripe + rotten). This trains the model on the real decision, rather than on a distinction finer than the business needs.

## What the dataset's composition implies

The training set contains 634 unripe, 1174 ripe, and 2123 rotten images (3,931 total), the validation set 167, 339, and 617 (1,123 total), and the test set 110, 154, and 298 (562 total). This is a 70/20/10 split across 5,616 images.

The training set is moderately imbalanced, with rotten outnumbering unripe by about 3.3x. However, the per-class numbers remain trustworthy as no class falls anywhere near the roughly 40-instance threshold below which results become unreliable (the smallest class, unripe, still has 110 test images). 

Object size relative to the image doesn’t apply here, since this is a classification dataset with no bounding box annotations to measure.

The test split does not resemble what a real deployment would see, in two ways. First, its rotten prevalence (53.0%) is far higher than the assumed real-world rate (10%), meaning test set accuracy and precision can overstate how well the model would perform in practice; what does transfer is recall and false-positive rate, since those describe the model’s behavior rather than the test set’s class mix, which is why the costing in Section 4 relies on those two instead of raw test-set counts. Second, these are close-up photos of a single banana against a plain background, not realistic shelf or crate conditions with multiple bananas, occlusion, and variable lighting. Therefore, results from this test set can’t be taken as a reliable estimate of real-world accuracy, only as evidence that the model has learned to distinguish ripeness stages under clean conditions.


## Finetuning and baselines

This is an image classification task rather than object detection, since the decision only requires judging the ripeness of a banana already in frame, there’s no need to locate or count multiple bananas within a larger scene, which is what detection would be for. The model is yolo11n-cls, finetuned from ImageNet-pretrained weights, trained for 30 epochs with patience=10, using Ultralytics’ automatic optimizer selection (AdamW, lr≈0.0014). No further hyperparameter search was performed, since validation accuracy converged smoothly to above 99%.

| Model | Accuracy (test, n=562) |
|---|---|
| Trivial baseline (always predict majority class, rotten) | 53.0% |
| Pretrained zero-shot | not directly comparable (see below) |
| Finetuned model | 99.1% |

For the zero-shot baseline, the assignment specifies that a COCO-pretrained model should be used, which applies to object detection. Since this is classification, the equivalent baseline is pretrained on ImageNet-1k instead. ImageNet-1k has no ripeness classes at all, so the pretrained model cannot output “rotten,” “ripe,” or “unripe”. The label it does have is “banana” so I checked how often it could recognize the images as being a banana. It only did so 65.1% of the time. For instance, green bananas were called cucumber or zucchini, rotten bananas were called dark, textured things like nematode, slug, or eel. This shows the pretrained model picks up on the same color and texture cues that distinguish ripeness, but has no labels to express them as a ripeness judgment.

What changed between the trivial baseline and the finetuned model is a jump from 53.0% to 99.1% accuracy. Given how visually distinct the three ripeness stages are in these close-up, plain-background photos, I expected a strong result but not one this close to perfect. The likely reason is that these images have just one banana per photo, no clutter and no overlap. A real shelf camera would see something messier with multiple bananas, variable lighting and partial occlusion so this result probably overstates how well the model would actually perform in production.


## Per-class evaluation and failure modes

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ripe | 1.000 | 0.994 | 0.997 | 154 |
| rotten | 0.997 | 0.987 | 0.992 | 298 |
| unripe | 0.965 | 1.000 | 0.982 | 110 |

The model performs very well across all three classes, with an overall test accuracy of 99.1%. The model makes only 5 errors across all 562 test images:
| | Predicted: ripe | Predicted: rotten | Predicted: unripe |
|---|---|---|---|
| Actual: ripe | 153 | 1 | 0 |
| Actual: rotten | 0 | 294 | 4 |
| Actual: unripe | 0 | 0 | 110 |

One ripe banana is misclassified as rotten (a false positive costing €0.20), and four rotten bananas are misclassified as unripe (false negatives, costing €3.00 each, €12 total). 

With so few errors, the confusion matrix is close to empty which is a good sign for the model, but it also means the threshold sweep in the next section has limited error to redistribute between false positives and false negatives.

## Cost matrix and operating point

As mentioned above, the two mistakes cost different amounts. False positives are the cases in which a sellable banana gets flagged as rotten and thrown away. The cost of this has been estimated to be €0.20, based on the price you would pay for a banana at the supermarket. A false negative occurs when a rotten banana is missed and stays on the store shelf. This is more expensive, for two reasons: a rotten banana in direct contact with others can cause them to spoil too, and visibly rotten fruit on display damages the store’s image. The total cost is assumed to be €3.00, broken down as €1.00 for roughly five neighboring bananas spoiled by contact at €0.20 each, plus €2.00 as the assumed reputational cost of customers choosing not to buy bananas from the store that day. I have assumed that true positives and true negatives cost €0. Correctly sorting a banana is not an extra cost as no meaningful staff time or resource is spent beyond normal operation.

I have assumed that the real-world base rate is that 10% of bananas checked are actually rotten at any given time for a supermarket. This is different from the test set (see Section 2), which has a way higher proportion of rotten bananas than a real store would.

Sweeping a single confidence threshold against these costs and this base rate, using the model’s recall and false-positive rate (not raw test-set counts, since those are distorted by the test set’s unrealistic rotten rate), gives a cost-minimizing threshold of 0.26, at €3.70 expected cost per 1,000 cases. With a threshold of 0, the costs are €180.00 per 1,000 cases, while with a threshold of 1 the costs are €148 per 1,000 cases. It is crucial to mention that many rotten bananas saturate to a model confidence of exactly 1.0, so even the maximum threshold still catches roughly half of them. 

The assumption that would most change this answer if it were wrong is the €2.00 reputational cost of the false-negative cases. It’s the least measurable of all the cost inputs as there’s no receipt or real transaction behind it, just a judgment call about how customers might react. 

## What a person still has to check

Adding a human-review band between two thresholds (below a low threshold sells automatically, above a high threshold throws automatically, and anything between goes to a staff member) improves on this further. The chosen operating point is a low threshold of 0.04 and a high threshold of 0.76, giving an expected cost of €1.82 per 1,000 cases with 3.2% of cases sent to review. This band is asymmetric because a missed rotten banana costs 15 times more than a wrongly thrown away good one so the system demands near-certainty (below 4% confidence of being rotten) before selling automatically, but accepts comparatively less certainty (above 76%) before throwing automatically.

At an assumed volume of 100 bananas checked per day (a small shop), 3.2% works out to roughly 3.2 bananas per day needing review. At 10 seconds per check, that’s about 32 seconds of staff time daily which means the supermarket staff can easily fit this into their schedule (no need to hire a dedicated role). 

The cost of a single staff check is derived from an assumed wage of €9 per hour: €9 ÷ 3,600 seconds = €0.0025 per second, so a 10-second check costs €0.025 (2.5 cents) per banana. Scaled to checking all 1,000 cases, that’s the €25.00 figure in the “person checks everything” row below.

This design is priced against three alternatives, per 1,000 cases at the assumed 10% base rate:
| Policy | Cost per 1,000 cases |
|---|---|
| Do nothing (sell everything, no system) | €300.00 |
| A person checks every banana manually | €25.00 |
| The model decides alone, no review band | €3.70 |
| Model + human review (chosen design) | €1.82 |

Doing nothing is by far the most expensive, since every rotten banana becomes a false negative. Having a person check everything is far cheaper than doing nothing, but still over 13 times more expensive than the chosen design, since it pays the per-check staff cost on all 1,000 cases rather than just the 3.2% that actually need it. The model deciding alone comes close to the chosen design but isn’t quite as good, since it has to fully commit to sell or throw on every case, including ones it’s unsure about. The review band exists to avoid forcing a costly guess on exactly those ambiguous cases, which is why adding it reduces cost further still.

## Limitations & next steps

One limitation is that the test images don't resemble a real deployment as they're close-up, single-banana photos against plain backgrounds, not shelf conditions with multiple bananas, occlusion and variable lighting. With more time, I would collect and label a small set of real shelf camera images and re-evaluate on those specifically. I would likely also need to pair this classifier with an object-detection step first, to locate and crop individual bananas out of a realistic photo before classifying each one.

A second limitation is that the 10% base rate is assumed rather than measured, and the cost-minimizing threshold depends on it. This figure would vary depending on the provider of the supermarket so if I were deploying this model I would pull actual shrinkage or spoilage records to replace this assumption with a measured rate, and re-run the threshold sweep against it.

Finally, the €2.00 reputational cost is also based on assumptions. With more time, I would try to find a proxy for this figure by, for example, checking whether stores see a measurable dip in banana sales on days following a visibly rotten display, or interviewing a store manager about how they currently think about this tradeoff.

## Generative AI use disclosure

I used AI for boilerplate code (e.g: dataset download and folder merging), debugging environment errors, looking up real-world figures (Spanish banana prices and minimum wage) to ground my cost assumptions, and explaining ML concepts I didn’t understand as they came up. The business framing, class merging, cost assignments, base rate assumption, and final thresholds were my own decisions.
