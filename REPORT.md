# REPORT.md: Fresh vs rotten fruit boxes, cost-based thresholds

## 1. Business framing

**Decision.** At the end of a packing line, each **box of fruit** (apples and bananas) is photographed. The model supports one decision per box: **pass** (ship), **fail** (send to the manual re-sort station) or **human review** (a person opens and checks the box). One *case* is one box, so the unit of decision is the box, not the individual fruit.

**Who acts on it.** The line supervisor and two packing-line reviewers. Boxes that pass are shipped. Failed boxes go to a re-sort station where a person removes the bad fruit and repacks.

**Positive class.** A box is *positive* if it contains at least one rotten fruit (`rotten-apple` or `rotten-banana`). This is the thing we must catch.

**Why one error is worse.**

| Outcome | Meaning | Cost | How it is built |
|---|---|---|---|
| FN | rotten box ships to the customer | **€34** | refund of the box €20 + return shipping €6 + complaint handling €8 (15 min at €32/h) |
| FP | good box rejected | **€3** | wasted manual re-sort: 10 min at €18/h |
| TP | rotten box rejected | **€3** | same re-sort, but it is necessary work |
| TN | good box passes | **€0** | nothing happens |

A missed rotten box costs about **11 times** more than a false alarm (34 vs 3), because the customer sees the problem and the company pays a refund, shipping and support time, while a false alarm costs a few minutes of a person's time. This asymmetry drives the later choices: it pushes the thresholds toward rejecting early rather than passing late.

**Base rate.** I assume **5%** of production boxes contain at least one rotten fruit. This is *not* the share in the test data (62.5% of the test images are positive), so the production rate is applied on top of the rates measured on the images.

**Volume.** 20,000 boxes per day, 2 reviewers available, 7 productive hours each, 15 seconds to check one box.

**Not included in the costs:** the loss of a customer who stops buying after a bad experience. This makes the FN cost an **underestimate**; including it would push the thresholds lower (reject more).

## 2. Dataset

**Source (public, not forked).** Roboflow Universe, workspace **`pbl-ez66k`**, project **`rotten-and-fresh-fruits`**, version **1**. Licence CC BY 4.0. Object detection, 4 classes: `fresh-apple`, `fresh-banana`, `rotten-apple`, `rotten-banana`.

**Splits.**

| Split | Images | Notes |
|---|---|---|
| Train | 504 | Roboflow augmentation ×3 (noise, rotation ±15°), so about **168 original photos** |
| Valid | 48 | not augmented, 62 instances |
| Test | 24 | not augmented, 34 instances |

**Instances per class (not augmented splits).**

| Class | Valid instances | Test instances | Test images |
|---|---|---|---|
| fresh-apple | 11 | 8 | 6 |
| fresh-banana | 16 | 3 | 3 |
| rotten-apple | 21 | 13 | 9 |
| rotten-banana | 14 | 10 | 6 |

Train instances per class (augmented x3): fresh-apple 207, fresh-banana 159, rotten-apple 147, rotten-banana 180.

**What this implies.**

- The **test split has only 24 images**, and `fresh-banana` has only **3** test instances. Any number measured on it is very noisy. A perfect score on 24 images only means "no error in 24 tries".
- Test is **62.5% positive** (15 of 24 images); production should be mostly good boxes (about 5% bad). Test accuracy therefore says little about production, which is why costs use the assumed base rate.
- Augmentation only touches train, so valid and test are untouched photos. I cannot rule out that near-duplicate photos exist across splits, which would make results look better than they are.
- The photos are mostly **a single fruit or small group on a plain background**, not whole packed boxes. A case in my framing is a box, so there is a gap between the data and the deployment.

## 3. Training and evaluation

**Setup.** `yolo11s`, 50 epochs, image size 640, batch 16, seed 0, trained on **Google Colab with a Tesla T4** in about 8 minutes. Weights are committed at `weights/best.pt`. Running `main.py train` twice gave **identical test metrics**, so the pipeline is reproducible.

**Detection results on the test split.**

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.972 | 1.000 | 0.995 | 0.765 |
| fresh-apple | 0.954 | 1.000 | 0.995 | 0.780 |
| fresh-banana | 0.950 | 1.000 | 0.995 | 0.731 |
| rotten-apple | 0.994 | 1.000 | 0.995 | 0.897 |
| rotten-banana | 0.990 | 1.000 | 0.995 | 0.651 |

Precision and recall are at the confidence that maximises F1. The positive classes are `rotten-apple` and `rotten-banana`, and the headline number is **recall on the positive classes (1.000)**.

**Where it fails.** On the test split nothing fails, which is a warning rather than good news. The weakest class is **`rotten-banana`**: its mAP50-95 is the lowest (0.651), and on the *validation* split its recall was only **0.786** (AP50 0.797). So the model is least sure exactly on the positive class that matters most. The overall number hides this.

**Anecdotal failure.** I tried two photos in the dashboard that are not from the dataset. A bunch of fresh bananas on a white background was correctly passed (`fresh-banana`, rotten score 0.000). A clearly overripe, spotted banana on a white background was also **passed with rotten score 0.000** (it was predicted `fresh-banana` at 0.44). This is **one image**, not evidence about the whole error rate, but it shows the model can miss a rotten fruit with no warning signal, and it shows that background and labelling criteria differ from the training data.

**Baselines (same test split, image level, threshold 0.5).** An image is called positive if any `rotten-*` box scores above the threshold.

| Model | TP | FP | FN | TN | Accuracy | Recall |
|---|---|---|---|---|---|---|
| Fine-tuned yolo11s | 15 | 0 | 0 | 9 | 1.000 | 1.000 |
| COCO yolo11n, zero-shot | 0 | 0 | 15 | 9 | 0.375 | 0.000 |
| Trivial (always "rotten") | 15 | 9 | 0 | 0 | 0.625 | 1.000 |

The fine-tuned model beats both. The COCO model has no "rotten" class, so it can never flag a rotten box and behaves like "do nothing". In the test data rotten boxes are the majority, so the trivial baseline predicts "rotten" for everything. In production, where rotten is a minority, the trivial baseline is "predict nothing", and that is the *do nothing* strategy priced below.

## 4. Costing and threshold

**Score.** For each image, `score` is the highest confidence among `rotten-apple` and `rotten-banana` boxes (0 if none). The ground-truth label is positive if the image contains any rotten box.

**Data used for the cost curve.** The test split alone (24 images) is too small, so the curve is computed on **Valid + Test pooled: 72 images (41 diseased, 31 healthy)**. Caveat: the validation split was also used to select the best weights during training, so its scores are slightly optimistic.

**Sweep.** I swept a single threshold from 0 to 1 and priced each confusion matrix at the assumed 5% base rate. On these data the cost is **flat at about €150 per 1,000 boxes from threshold 0.02 up to about 0.55**, because the model separates the images almost perfectly. `argmin` would pick a meaningless 0.01, so I take the **middle of the flat region: 0.28**. A bootstrap over 200 resamples gives a median best threshold of 0.28 (90% interval 0.28 to 0.42) and a minimum cost of €150 in every resample.

**Why €150 is the floor.** Of 1,000 boxes, 50 are rotten, and each must go to the re-sort station (TP €3), which is €150. A perfect model cannot go below it.

**Strategies compared (per 1,000 boxes, base rate 5%).**

| Strategy | Cost | Review share | People needed |
|---|---|---|---|
| Do nothing (ship everything) | €1,700 | 0% | 0 |
| A person checks every box | €313 | 100% | 11.9 |
| Model alone, threshold 0.28 | €150 | 0% | 0 |
| Model + human review band | €150 | 0% in these data | 0 |

At 20,000 boxes per day, the model saves about **€1,550 per 1,000 boxes, roughly €31,000 per day**, compared with doing nothing. A person checking every box would need 11.9 people and I only have 2, so that option is not available.

**Sensitivity (the answer changes if...).**

- *Model miss rate.* If the model lets through a fraction *m* of rotten boxes, cost per 1,000 ≈ 150 + 1,550·*m*. This equals the €313 of checking everything when **m ≈ 10.5%**. With 0 misses in 41 rotten images, the 95% upper bound on *m* is about 3/41 ≈ **7.3%** (rule of three), which gives about €263 per 1,000, still below €313. This only holds if real boxes look like the dataset.
- *Base rate.* The break-even miss rate falls as rotten boxes become more common: about 45% if only 1% of boxes are bad, 10.5% at 5%, and about **4% if 20% are bad**.
- *FN cost.* A box is worth rejecting when P(rotten) × 34 > 3, i.e. when P(rotten) > 3/34 ≈ **9%**. If a rotten box only cost €16 (refund of one fruit €2 + €6 + €8), the cutoff would rise to 3/16 ≈ **19%**. If lost customers were counted, FN would go up and the cutoff would fall.

## 5. Human review

**Three outcomes.** `score < 0.05` pass, `0.05 ≤ score < 0.30` human review, `score ≥ 0.30` fail.

- **Fail from 0.30.** Rejecting early is cheap (a €3 re-sort), missing is expensive (€34), and 0.30 sits next to the centre of the flat minimum (0.28) and the bootstrap interval. Failed boxes also pass through a person at the re-sort station, so "fail" is not final.
- **Pass below 0.05.** Here the model sees essentially no sign of rotten fruit.
- **Review in between.** This is where the model is unsure.

**Load.** On the 72 scored images, no image falls inside the band, so **0% of boxes go to review in the data**. That is a real result but probably too optimistic for production, where new boxes will not all be this easy. Planning numbers at 20,000 boxes per day and 15 seconds per check:

- each 1% of boxes in review is 200 boxes per day, about 0.83 hours, about **0.12 people**;
- 10% in review needs about 1.2 people;
- the 2 reviewers can check at most **3,360 boxes per day, 16.8%** of volume. This is my capacity cap, and if the review band attracts more than 16.8% of boxes in production, the thresholds must be narrowed.

**Realistic?** Yes for any review share up to about 16.8%. Checking every box is not realistic (11.9 people).

**Limit of the score.** The overripe banana scored 0.000, so it would be **passed** even with a review band, because the band is defined on the rotten score. A band on that score cannot catch a case where the model is confidently wrong. A possible improvement is also sending to review the boxes where the model is unsure what it sees (for example, a `fresh` score below about 0.5), which I did not implement.

## 6. Dashboard

Run with `uv run python main.py app`. It loads `weights/best.pt` and the files in `outputs/`, and never trains. It has three tabs and three controls:

- **Try an image:** upload a photo and see boxes plus the decision (pass, human review or fail) at the current thresholds.
- **Cost & review:** the confusion matrix per 1,000 boxes (rows are the true class, columns are pass, review, fail), expected cost per 1,000, share sent for review, people needed against the 2 available, the cost-versus-threshold curve with the chosen threshold and the current review band marked, a table of the four strategies, and a bootstrap button for uncertainty. A selector chooses which scored images feed the curve (test only, valid only, or both pooled).
- **Per-class results:** per-class precision, recall, AP50 and mAP50-95, the YOLO confusion matrix, and the image-level comparison against the two baselines.
- Sliders: low threshold, high threshold, and assumed base rate.

`check.py` passed on a clean clone (key, weights, app).

**Simplification.** The image-level score is the highest `rotten-*` box confidence. A box with several fruits is scored by its worst fruit.

## 7. Limitations (honest summary)

1. The test set is tiny (24 images, 3 `fresh-banana` instances), so a perfect result is weak evidence.
2. The cost curve rests on 72 images in which the model makes no errors, so the choice of threshold is driven by reasoning about costs, not by an observed minimum.
3. Validation images were used to select weights, so pooled scores are optimistic.
4. A real overripe banana on a white background was passed with score 0.000. Training photos have different backgrounds, and I do not know how strict the "rotten" labels are.
5. Cases are boxes in my framing, but the dataset contains photos of one fruit or a small group.
6. Customer loss is not in the FN cost, and every cost figure is a stated assumption, not a measured value.
