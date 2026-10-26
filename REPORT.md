# Assignment 2 Report

- **Name**: Diana de Lorenzo
- **Student ID**: 20119
- **Email**: ddelorenzo.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

**The decision.** A convenience-store chain (the Carrefour Express or
petrol-station shop type) has a fixed camera on each drinks fridge that takes
a photo about once an hour. For every photo the model decides whether the
fridge has an empty gap (out-of-stock) that needs refilling. A store associate
acts on it: an alert means "go to this fridge and refill it". The unit of
decision is **one fridge photo**, and the **positive class is "the photo shows
at least one empty gap"**.

**Why a gap costs money.** These stores have few staff, usually busy at the
till, so an empty gap can stay unnoticed for around **2 hours** before someone
sees it and refills it. Drinks are very substitutable: if Coca-Cola Zero is
missing, many customers take a Pepsi or a regular Coca-Cola, and that sale is
not lost. But a customer who wanted exactly that drink can leave without
buying, or buy it at the bar or shop next door.

**False negative (a gap the model misses): about €1.70.** It is the margin,
not the price, that is lost. A drink sells for ~€2, and at a ~35% margin the
store loses ~€0.70 per lost sale. I assume 40% of the customers who find the
gap leave instead of taking another drink, and that about 6 customers want
that drink during the ~2 hours the gap stays unnoticed:
€0.70 × 40% × 6 ≈ **€1.70**.

**False positive (an alert for a full fridge): about €0.47.** An associate
walks to the fridge, sees it is full and goes back, ~2 minutes in a small
store. A retail associate in Spain costs the store ~€14/hour including social
security: 2/60 × €14 ≈ **€0.47**. Refilling a real gap also takes time, but
that work has to be done anyway, so it is not a cost of the model
(true positive = €0).

**Human review: about €0.04 per photo.** A person looks at the photo on a
screen for ~10 seconds and decides: 10/3600 × €14 ≈ **€0.04**. I assume the
person is right.

**The asymmetry.** A missed gap costs about **3.6 times** as much as a false
alarm (€1.70 vs €0.47). So I would rather send an associate to a full fridge
than leave a gap for two hours, and recall on the `empty` class matters more
than precision, but false alarms are not free: the threshold cannot simply be
pushed to zero. The money per photo is small; the case for the model only
works because a chain has many fridges and takes many photos a day.

## Dataset

- **Workspace / project / version:** `tossoengineering` /
  `retail-shelf-detection-peimi` / **v1**
- **Link:** https://universe.roboflow.com/tossoengineering/retail-shelf-detection-peimi/dataset/1
- **License:** CC BY 4.0. Used as published, not forked.
- **Task:** object detection, 2 classes: `empty` (an empty gap on a fridge
  shelf) and `product` (a drink on the shelf).
- **What one image is:** a photo of a drinks fridge in a US convenience store
  (Coca-Cola, Sprite, Mountain Dew, juices), resized by Roboflow to 640×640
  (stretched). Every gap and every product has its own box. Some samples are
  in `artifacts/dataset_samples.jpg` (gaps in red, products in green).

**Why it serves the problem.** It shows exactly the decision the store needs:
a drinks fridge photographed from the front, with the empty gaps labelled. The
`empty` boxes are what triggers an alert; the `product` boxes teach the model
what a full slot looks like, so it does not fire on every dark area. I turn the
boxes into the business decision at photo level: a photo's score is the
highest `empty` confidence on it, and that score is what the thresholds act
on.

## What the dataset's composition implies

Computed from the label files by `train.py` (saved in `artifacts/metrics.json`):

| | Train | Validation | Test |
|---|---:|---:|---:|
| Images | 390 | 24 | 11 |
| Images with at least one gap | 175 (45%) | 13 (54%) | 5 (45%) |
| `empty` boxes | 448 | 37 | **20** |
| `product` boxes | 4,504 | 238 | 205 |

| Box area as % of the image | 10th pct | Median | 90th pct |
|---|---:|---:|---:|
| `empty` | 0.23% | **0.8%** | 10.7% |
| `product` | 0.43% | 2.3% | 10.9% |

**Class imbalance, and the hard class is the one that matters.** There are
about 10 products for every gap. It is also easier to recognise whether a can
is there than whether there is a gap: a can has a shape and a colour, a gap
has neither, it is just an empty shelf or a dark background, and it is small
(median 0.8% of the image). So I expect the model to be good at `product` and
worse at `empty`, and the headline average will hide that. The important class
is `empty`, because finding gaps is what makes money, so I report it
separately and treat it as the positive class.

**The test split is too small to trust on its own.** 11 photos and 20 gaps:
one gap more or less moves recall on `empty` by 5 points, and with only 5
photos with a gap and 6 without, photo-level rates move in steps of 17–20
points. To trust the result I would need many more test photos, from several
stores and several days. Before paying for cameras in 200 stores I would run a
**pilot in 2 stores** to see how it really works. The pipeline also reports a
bootstrap interval for the cost, to show how much of the number is noise.

**The dataset has far more gaps than a real store.** Whoever built it wanted to
teach a model what a gap looks like, so 45% of the photos have one; in a real
store the fridge is almost always nearly full, and I assume **10%** of photos
have a gap (see costs). If I used the dataset's 45% in the cost, the
calculation would believe gaps happen four times as often as they do, and the
model would look much more valuable than it is. That is why the cost
re-weights the per-class rates to a 10% base rate. It also means the model has
seen few full fridges, which is what production mostly looks like, so to
avoid false alarms I would train it with many more photos of full fridges.

**US fridges, not Spanish ones.** The photos are from US convenience stores.
Many drinks look similar across countries (red Coca-Cola, blue Aquarius,
orange Fanta), and the model looks for gaps rather than brands, so what can
change most is the fridge model, the shelves and the lighting. Some training
photos are also rotated or have black squares pasted in (augmentation by the
author), while my cameras would be fixed at one angle. Before launch I would
take and label photos from Spanish fridges, with Spanish drinks, and finetune
on them. **What I cannot conclude:** that the test numbers will hold in a
Spanish store, or that they are precise to more than roughly ±10–20 points.

## Finetuning and baselines

**Model and task.** I chose YOLO11n with object detection because it is a
good balance between efficiency and usefulness for the business. The model
was trained on a laptop without a GPU, so a smaller model is more practical
and takes less time to train (yolo11s is roughly three times slower). As the
cameras only take one picture per hour, the model does not need to be
extremely fast. Detection is also more useful than classification because it
identifies the exact location of empty spaces, helping staff know where
products need to be restocked. Finally, the dataset already includes bounding
boxes, making object detection a suitable choice.

**Where and how it was trained.** On a laptop CPU (Windows, no GPU), via
`uv run python main.py train`: yolo11n from the COCO checkpoint, 30 epochs,
640×640, batch 16, early-stopping patience 15, seed 0. Around 4–5 minutes per
epoch. Weights in `weights/best.pt`.

All three models are scored on the same 11 test photos with the same
evaluator in `train.py` (COCO-style AP; precision/recall at confidence 0.25,
IoU 0.5). For COCO zero-shot I map COCO classes that can sit on a shelf
(bottle, cup, bowl, vase, fruit…) to `product`; COCO has no class that means an empty space.

| Model (test split) | AP50 `empty` | AP50 `product` | **AP50 (mean)** | mAP50-95 (mean) |
|---|---:|---:|---:|---:|
| **Finetuned yolo11n** | **0.92** | **0.98** | **0.95** | **0.57** |
| COCO yolo11n zero-shot | 0.00 | 0.54 | 0.27 | 0.13 |
| Predict nothing | 0.00 | 0.00 | 0.00 | 0.00 |

At photo level (does this fridge have a gap?), at the chosen threshold:

| Model (test, 11 photos) | Gaps caught | False alarms | Accuracy | € per 1,000 photos |
|---|---:|---:|---:|---:|
| **Finetuned yolo11n** (t = 0.39) | 5 / 5 | 0 / 6 | 100% | **0** |
| COCO zero-shot | 0 / 5 | 0 / 6 | 55% | 168 |
| Predict nothing | 0 / 5 | 0 / 6 | 55% | 168 |

The difference is large, but it is important to interpret the results
carefully. COCO cannot detect empty spaces because it was trained on general
object categories and does not include empty fridge spaces as a specific
class. Although it can detect some bottles, identifying only 35 out of 205 is
not reliable enough for this business, and at photo level it is no better than
doing nothing. My model performs much better, with an AP50 of 0.92 for
detecting empty spaces. However, the result may be too optimistic because the
training and test images come from the same videos and may contain very
similar frames (for example `video-IMG_5316_mp4-18` is in test and
`video-IMG_5316_mp4-20` in validation). Therefore, the model should be tested
on completely different videos and fridges before concluding that it
generalises well.

## Per-class evaluation and failure modes

Positive class: **`empty`**. Finetuned yolo11n, test split:

| Class | Boxes | Precision@0.25 | Recall@0.25 | AP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `empty` | 20 | 0.75 (18 / 24) | 0.90 (18 / 20) | 0.92 | **0.42** |
| `product` | 205 | 0.91 (195 / 214) | 0.95 (195 / 205) | 0.98 | 0.72 |

Object-level confusion matrix (Ultralytics, conf 0.25, IoU 0.45; rows =
predicted, columns = true; also `artifacts/confusion_matrix.png`):

| predicted \ true | `empty` | `product` | background |
|---|---:|---:|---:|
| `empty` | **17** | 1 | **8** |
| `product` | 0 | 198 | 15 |
| background (missed) | **3** | 6 | – |

The model mainly fails by detecting empty spaces where no space has been
labelled (8 boxes) and by missing some real empty spaces (3 of 20). Some of
the false detections could correspond to dark areas or spaces that were not
labelled correctly in the dataset. The model rarely confuses empty spaces with
products (once), which is positive for the business. Although the mAP50-95
score for `empty` is relatively low at 0.42, against 0.72 for `product`, the
exact position of the bounding box may not be critical if it still helps the
employee identify the correct fridge. The model can also classify a picture
correctly even if some boxes are inaccurate, because detecting just one empty
space may be enough to alert the employee. That is why the headline number
(AP50 0.95) is carried by `product`, while the business depends on `empty`.

## Cost matrix and operating point

| Outcome | € per photo | Where it comes from |
|---|---:|---|
| False negative (missed gap) | 1.68 | €0.70 margin × 40% leave × 6 customers in ~2 h |
| False positive (false alarm) | 0.47 | 2 min × €14/h |
| True positive / true negative | 0 | refill work is needed anyway / nothing happens |
| Human review | 0.04 | 10 s × €14/h, reviewer assumed correct |

**Base rate: 10%** of fridge photos have a gap in production (the dataset has
45%; see the composition section). Per-class rates measured on the photos are
re-weighted to 10% before pricing, so

cost per photo = 10% × (FN rate × €1.68 + review rate × €0.04)
               + 90% × (FP rate × €0.47 + review rate × €0.04).

**The sweep.** `train.py` sweeps the threshold from 0 to 1 in steps of 0.01
(`artifacts/cost_sweep.csv`, plotted in the dashboard), choosing on the
**validation** split and reporting on **test**, so the threshold is not fitted
to the photos it is judged on. On validation every photo with a gap scores
≥ 0.71 and every full fridge ≤ 0.07, so every threshold from 0.07 to 0.71
costs €0. To break that tie I take the **middle of the cheapest range,
t = 0.39**, because it leaves the largest margin on both sides: the lowest
tied value (0.07) sits right on top of the highest-scoring full fridge in
validation (0.068), and on test it would have produced a false alarm on a full
fridge scoring 0.37 (€70 per 1,000 photos instead of €0).

**Break-even.** A false negative only costs more than a false alarm while
€0.70 × 40% × customers > €0.47, i.e. while **more than ~1.7 customers** want
that drink during the 2 hours. Below that, missing a gap is cheaper than a
false alarm and the threshold should move up.

**The assumption that would most change the answer.** I think the number of
customers looking for the product is the most important assumption because it
directly affects the cost of a missed empty space. If only two customers look
for the drink instead of six, the cost of a false negative would fall to
€0.56, which is close to the €0.47 cost of a false alarm. This would reduce
the economic advantage of using the model. This assumption is also difficult
to estimate accurately without observing real customer behaviour. In
contrast, the number of pictures taken and the time required to review them
can be measured more easily. Therefore, I would test different
customer-demand scenarios to determine the point at which using the model
becomes economically worthwhile.

## What a person still has to check

**Operating assumptions.** 200 stores × 6 fridge cameras × 16 photos a day
(one an hour, 16 opening hours) = **19,200 photos a day**; 10 seconds to
review one; a reviewer spends 6 hours of the day reviewing. I think these
numbers are reasonable as a starting point for a convenience store chain,
although they are estimates that should be validated in real stores.

**Who reviews.** I would choose a central team to review doubtful pictures.
Store employees are already busy serving customers and working at the
checkout, so adding this task could interfere with their main
responsibilities. A central team would make the process more consistent and
easier to manage across all stores, although its cost must be considered.

**What is realistic.** I would aim to keep the workload below two full-time
employees. At 10 seconds per picture and 19,200 pictures per day, reviewing
every picture would require approximately 53 hours of work per day, equivalent
to almost nine people working six hours each. Therefore, the model must filter
out most pictures and only send doubtful cases for manual review. Otherwise,
the system could become too expensive to justify.

**The band.** Below **0.08 → pass**, from **0.70 up → fail** (send an
associate), in between → **review**. Chosen on validation as the **widest band
among the cheapest ones**: it covers exactly the score range where validation
has no photos at all, the scores the model has no evidence about.

| Test photos (11) | pass | review | fail |
|---|---:|---:|---:|
| has a gap (5) | 0 | 0 | 5 |
| no gap (6) | 5 | **1** | 0 |

The one full fridge that scored 0.37 goes to review. Re-weighted to the 10%
base rate that is **15% of photos sent to review → 2,880 reviews a day →
8 hours of reviewing → 1.33 people** at 6 review hours each.

| Policy (per 1,000 photos, test) | € | People reviewing |
|---|---:|---:|
| Do nothing (no alerts) | 168.00 | 0 |
| A person checks every photo | 38.90 | ~8.9 |
| Model decides alone (t = 0.39) | **0.00** | 0 |
| Model + review band (0.08–0.70) | **5.83** | 1.33 |

Bootstrap over the 11 test photos (90% interval) for the band: €0–17.50 per
1,000 photos.

**Why I keep human review.** I would keep human review as a safety measure.
Although the model alone costs €0 on the test set compared with €5.83 per
1,000 pictures with the review band, there are only 11 test pictures, so this
result is not enough to prove that the model will always work correctly. A
full fridge scored 0.37, very close to the 0.39 threshold; without human
review, a small change in the score would have turned it into a false alarm,
and a gap scoring a little lower than usual would be missed. The band sends
both kinds of doubtful case to a person. Across 200 stores (19,200 photos a
day) the band costs about **€112 a day**, against about **€3,226 a day** for
doing nothing (€168 per 1,000 photos), so it is a cheap insurance. However,
these costs should be validated with real data before implementation.

**Is 1.33 people realistic?** I think 1.33 full-time employees is a reasonable
starting estimate because it stays below my limit of two employees. Compared
with reviewing every picture manually, which would require approximately nine
employees, the model could substantially reduce the workload. However, this
estimate depends on the percentage of pictures that require human review. If
the percentage increased from 15% to 30%, the workload would approximately
double, requiring around 2.66 employees. This would exceed my original limit,
so the system's feasibility depends on keeping the number of doubtful
pictures low in real stores.

## Limitations & next steps

| Limitation I found | What I would do |
|---|---|
| **Train, validation and test come from the same videos.** Neighbouring frames of one video sit in different splits, so the test score is probably optimistic. | Re-split by video (all frames of a video in one split), and test on fridges and stores the model has never seen. |
| **The test split is tiny** (11 photos, 20 gaps), so every rate moves in steps of 5–20 points and the €0 cost proves little. | Collect a few hundred labelled photos, and run a **pilot in 2 stores** before paying for cameras in 200. |
| **US fridges, not Spanish ones**, and augmented training photos while my cameras would be fixed. | Take and label photos of Spanish fridges and drinks from the real camera position, and finetune on them. |
| **Few full fridges** in the data (45% of photos have a gap, I assume 10% in reality), so false alarms in production may be higher than measured. | Add many photos of full fridges, which is what production mostly looks like, and re-measure the false-alarm rate. |
| **The cost of a missed gap depends on a guess**: ~6 customers in 2 hours. Below ~1.7 customers a missed gap costs less than a false alarm. | Measure it in the pilot: sales per drink per hour and how long gaps really stay empty, then re-run the sweep (`costs.py`). |
| **The review load depends on how many photos are doubtful.** If 30% instead of 15% fall in the band, it needs ~2.7 people, above my limit of 2. | Track the review share in the pilot; if it grows, retrain with the reviewed photos (reviewers' decisions become new labels) to shrink the band. |

## Generative AI use disclosure

Claude Code (Anthropic) wrote the code (`costs.py`, `train.py`, `app.py`),
found candidate datasets on Roboflow Universe, and produced the factual tables
in this report from the pipeline's outputs. For the analysis, Claude
explained each concept and proposed starting estimates (margins, minutes,
base rate); I chose the problem, decided to switch from perfumery to drinks
fridges so the dataset would fit, changed or accepted each figure (for
example, that a gap stays unnoticed for about 2 hours), and wrote my own
answers, which Claude translated and formatted into English. I reviewed the
code and the final text and can explain every decision.
