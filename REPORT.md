# Assignment 2 Report

- **Name**: Paula Candiles Puente
- **Student ID**: 18670
- **Email**: pcandiles.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

The decision I support is the end-of-line cap inspection on a bottling line: for every
filled and capped bottle, should it **pass** to packing, **fail** and be pulled from the
line, or be **sent to a person** for a visual check? Today that decision is made by an
operator looking at the bottle. The model looks at a photo of the cap and gives each
bottle a defect score; the line controller acts on the score, and a quality inspector
handles the bottles sent for review.

The two mistakes do not cost the same:

- **False negative (a defective bottle passed): €50.** A broken, loose or missing cap
  reaches the customer. The product can leak, spoil or be contaminated, which means a
  complaint, a refund, possible retailer penalties and damage to the brand. I built the
  figure from its parts so that it is not just a guess:
  - refund or replacement of the bottle: about €5
  - handling the complaint (staff time, shipping a replacement): about €10
  - penalty or chargeback from the retailer: about €15
  - brand damage, and the risk of losing the customer or triggering a wider check: about €20

  This is an assumption. On a real line I would replace it with the company's own
  complaint costs from finance.
- **False positive (a good bottle rejected): €5.** The bottle is pulled and its value is
  lost. I itemised it as:
  - product: €2.00
  - bottle and cap: €0.50
  - filling and labelling already done: €1.00
  - handling or disposal: €0.50
  - lost margin: €1.00
- **True positive and true negative: €0.** Catching a defect or passing a good bottle is
  the line working as intended.
- **Human review: €0.08 per check.** I derived this from the inspector's wage rather
  than choosing it: 15 seconds per check at about €20 per hour is 15 ÷ 3,600 × €20 ≈ €0.08.
  I assume the inspector makes the right call.

A missed defect is therefore **10 times as expensive** as a wrongly rejected bottle. This
asymmetry drives everything below. I treat **defective** as the positive class, I care
most about recall on it, and I accept some false positives to keep false negatives near
zero.

I assume a production **base rate of 5% defective bottles**. A real bottling line is
usually much better than 5%, so this is a cautious, round number: if the design works at
5%, it should work even better on a cleaner line. It is also far below the dataset's 62%
to 80% defective, which makes clear that the dataset is not a realistic picture of
production. Because of that gap, every cost I report is re-weighted from the measured
rates to this 5%. The exact figure is not critical: I test base rates from 1% to 20%
below, and the ranking of the four strategies does not change. This is also an
assumption. On a real line I would replace it with the defect rate measured by running
the model in shadow mode.

## Dataset

- **Workspace / project / version**: `product-defect-detection` / `bottle-defects-o4gnx` / **version 28**
- **Link**: https://universe.roboflow.com/product-defect-detection/bottle-defects-o4gnx/dataset/28
- **Task**: object detection, used as a public dataset as it is (not forked).

One image is a grayscale close-up of one bottle's cap and neck ring against a dark
background. Most test images (100 of 112) carry one box, and 12 carry two. There
are five classes: **Good Cap**, **Broken Cap**, **Broken Ring**, **Loose Cap** and **No Cap**.
The four defect classes are exactly the failure modes the inspection is meant to catch,
and Good Cap gives the model an explicit picture of an acceptable bottle. That is why
the dataset serves the decision rather than just resembling it: its classes map one to
one onto pass (Good Cap) and fail (everything else).

The decision unit is the **bottle**, not the box. A bottle is defective if any of its
labels is a defect class. Its defect score is the **highest confidence among its
non-Good Cap detections**, or 0 if there are none.

## What the dataset's composition implies

Images per split: **398 train, 113 validation, 112 test**.

Instances per class (train / validation / test):

- **Good Cap**: 152 / 43 / 23
- **No Cap**: 112 / 32 / 17
- **Loose Cap**: 89 / 23 / 46
- **Broken Cap**: 35 / 13 / 15
- **Broken Ring**: 35 / 8 / 23

Median box area as a share of the image: Broken Cap about 27–32%, Good Cap and No Cap
about 25%, Loose Cap about 14%, and **Broken Ring only 3.3–6.0%**.

What this implies:

- **Thin classes.** Broken Cap has only **15 test instances** and Broken Ring **8
  validation instances**. One more or one fewer detection moves their recall by 7 and
  12 points, so I cannot trust their per-class numbers to the decimal.
- **Small objects.** Broken Ring is a small object, and I trained at 320 px. I expected
  its localisation (mAP50-95) to be weak, and it is (see below).
- **Defect share far above production.** **61.8% of training bottles and 79.5% of test
  bottles are defective**, against my assumed 5% in production. Raw accuracy on the test
  split is therefore meaningless as a business number. This is why I re-weight every
  rate to the 5% base rate, and why my review band converts scores to production odds.
- **Few good bottles to judge false positives on.** The test split has only **23 good
  bottles**. A single false positive is a **4.3% false-positive rate**, so the cost of
  letting the model decide alone rests on one or two bottles.
- **The splits do not match each other.** Test has 46 Loose Cap and 23 Broken Ring
  instances against 23 and 8 in validation. Thresholds tuned on validation are tested on
  a different mix. That is a fair test, but it also means the test numbers will not hold
  exactly on a production line with yet another mix.

## Finetuning and baselines

I finetuned **YOLO11n** for detection, starting from the COCO-pretrained checkpoint, for
**50 epochs at 320 px, seed 0**, on my MacBook Pro with the CPU build of PyTorch pinned
by the project's environment. I chose detection over classification because the defect
is a property of one small region (the cap or the ring), and detection lets me show the
inspector where the problem is. I chose the nano model because the dataset is small and
the training had to run on a laptop CPU.

Results on the same 112-image test split, at bottle level (threshold 0.08, chosen on
validation):

| Model | TP | FP | FN | TN | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| **Finetuned YOLO11n** | 89 | 1 | 0 | 22 | **0.991** | **0.989** | **1.000** |
| COCO YOLO11n, zero-shot | 0 | 0 | 89 | 23 | 0.205 | 0.000 | 0.000 |
| Trivial: predict nothing | 0 | 0 | 89 | 23 | 0.205 | 0.000 | 0.000 |
| Trivial: majority class ("defective") | 89 | 23 | 0 | 0 | 0.795 | 0.795 | 1.000 |

At box level, the finetuned model reaches **mAP50 = 0.933** and **mAP50-95 = 0.717**.
The COCO model scores **mAP50 = 0**: none of its 80 classes is a cap state, so no box can
match a label. It is not blind. It found something in **25 of 112** test images, mostly
"cake" (15), "book" (5), "bottle" (2) and "vase" (2), but it cannot judge a cap, so as an
inspector it behaves exactly like predict-nothing.

The finetuned model **beats both baselines**. The majority-class baseline has perfect
recall only because it rejects every bottle, including all 23 good ones. The gap is
larger than I expected, but the 0.991 accuracy should not be read as production
performance, for the base-rate reason above.

## Per-class evaluation and failure modes

Per-class detection results on the test split:

| Class | Precision | Recall | AP50 | mAP50-95 | Test instances |
|---|---|---|---|---|---|
| No Cap | 0.972 | **1.000** | **0.995** | 0.935 | 17 |
| Good Cap | 0.902 | 0.957 | 0.981 | **0.940** | 23 |
| Loose Cap | 0.918 | 0.973 | 0.976 | 0.797 | 46 |
| Broken Ring | 0.907 | 0.913 | 0.914 | **0.260** | 23 |
| Broken Cap | **0.757** | **0.667** | **0.801** | 0.651 | 15 |

The headline mAP50 is carried by **No Cap, Good Cap and Loose Cap**, which are large and
visually obvious. Two classes drag it down:

- **Broken Cap** is the weakest class on every measure, with **recall 0.667**. It also has
  the fewest test instances (15), so this number is itself uncertain.
- **Broken Ring** is found (AP50 0.914) but badly localised (**mAP50-95 0.260**): the model
  knows roughly where the ring is but not its exact outline. This matches its tiny box
  size at 320 px.

The confusion matrix, counted at confidence **0.25**, shows what the model confuses with
what:

- **Broken Cap**: 10 of 15 found, **2 predicted as Good Cap**, 3 missed. A broken cap read
  as a good one is the most dangerous error in this problem.
- **Broken Ring**: 22 of 23 found, 1 missed.
- **Good Cap**: 22 of 23 correct, 1 predicted as Loose Cap.
- **Loose Cap**: 45 of 46 found, 1 missed.
- **No Cap**: 17 of 17 found.
- **False boxes on background**: 6 Broken Cap, 6 Broken Ring, 4 Loose Cap and 3 Good Cap.

At bottle level these box errors mostly disappear: **no defective test bottle was
passed** at the chosen threshold. A bottle's score uses any defect detection down to
very low confidence, and a defective bottle usually carries at least one. The one
bottle-level error is a **good bottle scored 0.923**, which the model is confident is
defective. That single bottle is the false positive in the table above, and it becomes
important in the costing.

## Cost matrix and operating point

The cost matrix is set out in the first section: **€50** per false negative, **€5** per
false positive, **€0** for correct decisions and **€0.08** per human check, at a **5%**
base rate. Rates measured on each split are re-weighted to that base rate, and every
cost is expressed per 1,000 bottles.

I swept the confidence threshold from 0.01 to 0.99 in steps of 0.01, choosing on the
**validation** split and reporting on the **test** split:

- **0.01**: €220.9 on validation, €1,239.1 on test (too many good bottles rejected)
- **0.05**: €110.5 on validation
- **0.08 to 0.27**: **€0 on validation**. Every good validation bottle scores at most 0.080
  and every defective one at least 0.276, so the validation split is perfectly separated.
- **0.28**: €35.7 on validation (defective bottles start being passed)
- **0.90**: €571.4 on validation, €964.9 on test (recall falls to 0.70)

The sweep takes the first minimum, **0.08**. On test that threshold costs **€206.5 per
1,000 bottles**, all of it from the one good bottle scored 0.923: 1/23 × 95% × €5 × 1,000.
The flat zero-cost region from 0.08 to 0.27 shows that the exact threshold is not
pinned down by the data. It is pinned down only by where the few bottles happen to fall.

**The assumption that would most change my answer is the cost of a human check**,
because it decides both how wide the review band is and how many people it needs.
With the band rule below, a 30-second check (€0.17) raises the cost to €15.5 per 1,000
and the staffing to 2.2 people, and a 60-second check (€0.33) raises it to €30.1 and 4.4
people. Above about €0.21 per check (about 37 seconds at €20 per hour), letting the
model decide alone becomes cheaper than having a person check everything. The base rate matters less than I expected: between
**1% and 20%**, the review share moves from 5.3% to 23.5% and the cost from €4.2 to €18.8
per 1,000, but the ranking of the four strategies does not change.

## What a person still has to check

My first version chose the review band the same way as the threshold, by minimising
validation cost over every (pass-below, fail-at-or-above) pair. It returned **0.08 / 0.09**,
an empty band that sent **0%** of bottles to a person. The reason is the same perfect
separation: on validation there is no bottle in the uncertain zone, so a band can only add
cost. Changing the costs did not help: with €5 and €0.08 the band was still empty.

I therefore set the band from the cost matrix instead of from the data. With *p* the
chance that a bottle is defective, a person is worth paying when a check costs less than
the expected cost of either automatic decision:

- review beats **pass** when **p > €0.08 / €50 = 0.16%**
- review beats **fail** when **p < 1 − €0.08 / €5 = 98.4%**

I read the model's score as *p* after shifting it from the training split's defect rate
(61.8%) to the 5% production rate, and rounded to the threshold grid. This gives:

- **Pass below 0.05**
- **Send to a person between 0.05 and 0.99**
- **Fail at or above 0.99**

On the test split this sends **9.1% of bottles to a person**: every defective bottle,
plus the one good bottle scored 0.923. No bottle is failed automatically, because no test
bottle scores 0.99 or more. The fail outcome is rare by design: with a check at €0.08 and
a wrong rejection at €5, it is almost always cheaper to ask a person first. In practice the
model clears the obviously good bottles and the inspector confirms the rest.

Workload, assuming **20,000 bottles a day**, **15 seconds per check** and **420 inspection
minutes per person per day**:

- **about 1,826 checks per day**
- **about 7.6 hours of checking**
- **about 1.1 full-time inspectors**

That is realistic: one inspector per shift, with some slack from a second person. If the
real defect rate is 20% instead of 5%, it rises to about 2.8 inspectors, which is still
manageable.

Priced against the three alternatives on the test split, per 1,000 bottles:

- **Doing nothing** (pass everything): **€2,500**
- **A person checks everything**: **€80**, but **11.9 full-time inspectors**
- **The model decides alone** (threshold 0.08): **€206.5**
- **Model plus human review** (my band): **€7.3**, with **1.1 inspectors**

The model alone loses to a person checking everything, entirely because of one
confident mistake. The combination is about **11 times cheaper than full human
inspection** and needs **one-eleventh of the staff**, and it is the only option in which
no defect is missed and no good bottle is thrown away on the test split.

## Limitations & next steps

- **The band relies on the score reading as a probability.** YOLO confidences are not
  calibrated, and I could not calibrate them because validation has no bottles in the
  uncertain range. With more data, I would fit a calibration curve (for example isotonic
  regression) on a larger validation set and recompute the band from it.
- **The false-positive estimate rests on 23 good test bottles.** One bottle drives the
  whole €206.5 of the model-alone option. I would collect several hundred images of good
  bottles from the real line before trusting any false-positive rate.
- **Broken Cap is weak and thin** (recall 0.667, 15 test instances), and 2 broken caps
  were read as good. I would add Broken Cap examples, ideally from the real line, before
  relying on the model for that defect.
- **Broken Ring is poorly localised** (mAP50-95 0.260) because the boxes are small at
  320 px. I would retrain at 640 px, or crop to the neck before detection, and check the
  gain on this class specifically.
- **The dataset does not look like production.** It has 62–80% defective bottles, and
  the images are uniform webcam close-ups against a dark background (the filenames
  suggest a single capture session in July 2022), unlike a moving line with varying
  light. Before deployment, I would run the model in shadow mode on the real
  line, with every bottle still checked by a person, to measure the true base rate and
  false-positive rate.
- **The human is assumed to be always right.** If inspectors miss some defects, the review
  band needs a cost for that, and the fail threshold should move down.

## Generative AI use disclosure

I used Claude (Anthropic), through Claude Code in VS Code, for the following:

- **Code**: changing the dashboard's colour palette; reviewing the repository against the
  assignment; computing the confusion matrix at confidence 0.25 instead of 0.001;
  recording what the COCO model detects; clearing notebook outputs and fixing
  `.gitignore`; diagnosing a local SSL certificate error with Roboflow; and implementing
  the cost-based review band in `train.ipynb`, which Claude proposed after the
  validation-based search returned an empty band.
- **Report**: Claude drafted the report text from my pipeline's results. The cost
  figures, their justifications and the decisions listed below are mine.

The decisions are mine: the problem and the dataset; the €5 false-positive breakdown; the
€0.08 review cost derived from a €20 hourly wage; the requirement that the review band
should not be empty; and the 5% base rate and €50 false-negative cost.
