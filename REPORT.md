# Assignment 2 Report

- **Name**: Bruno Sanchis
- **Student ID**: 20262
- **Email**: bsanchis.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

**Decision.** A vending operator has 200 snack and drink machines and wants to know,
from one photo of each machine's front taken each day, which slots still hold
product, how many are empty, and what kind of product each one holds. Today a
restocker finds this out by visiting. With the photo, the van is loaded and the
route is planned before anyone leaves.

**Positive class = "there is a product in this slot"** (one detected box). Empty
slots are estimated as `machine capacity − detected products`, because in a spiral
machine every stocked slot shows its front item behind the glass.

**The two errors cost different amounts:**

- **False positive (phantom product): €6.00.** The model sees a product in a slot
  that is actually empty. The slot counts as full and is not refilled until the next
  visit. €6 = ~€0.80 margin per unit × ~1.5 sales per slot per day × ~5 days to the
  next visit.
- **False negative (missed product): €0.50.** The model misses a product that is
  there. The slot counts as empty and the restocker checks it or carries stock for
  nothing: ~2 minutes at ~€15/h.
- **TP and TN cost €0.** That is the normal operation.

A false positive is therefore **about 12× worse** than a false negative. So we
prefer a **high confidence threshold**: we would rather miss a product (cheap) than
invent one (expensive). Everything below follows from this ratio.

**Base rate:** I assume **80% of slots are occupied** when the photo is taken, since a
machine halfway through its refill cycle has about 1 slot in 5 empty. It is used to
turn product counts into total slots: slots = products / 0.8. It only scales the
cost and does not move the optimal threshold. The assumption that does move the
threshold is the FP cost (see the sensitivity analysis).

## Dataset

- **Roboflow Universe:** workspace `amits-pwfut`, project `vending-niqv5`, **version 8**.
- **Link:** https://universe.roboflow.com/amits-pwfut/vending-niqv5/dataset/8
- **License:** CC BY 4.0. Used as published (not forked).
- **What one image is:** a photo of a vending machine's front, taken from outside
  through the glass, with every visible product boxed.
- **Classes (6):** Chocolate, Energy Drink, Juice, Snack, Soda, Water.

**Why it serves the problem.** It is the angle a restocker or a cheap camera would
actually capture (front view, through glass, with reflections), and it uses
coarse product categories, which is the granularity a van-loading decision needs.

**Datasets I rejected:**
- `sc-pewleydown-com/ivsnew` has a real "empty" class, but its camera sits *inside*
  a drinks machine, looking down at bottle tops. Our photos would come from outside,
  so it does not match production.
- Retail shelf and cooler datasets (`retail-coolers`, `retail-inventory-ai`) are not
  vending machines, and most of them do not say *which* product is in each slot.
- No public dataset of vending machines photographed from outside has an "empty
  slot" label. That is why empty slots are derived from capacity instead.

## What the dataset's composition implies

| Split | Images | Instances | Instances / image | Median box area (% of image) |
|---|---|---|---|---|
| train | 312 | 11,096 | 35.6 | 0.79% |
| valid | 73 | 2,521 | 34.5 | 0.71% |
| test | 73 | 2,516 | 34.5 | 0.75% |

| Class | train | valid | test |
|---|---|---|---|
| Chocolate | 2,148 | 559 | 551 |
| Energy Drink | 998 | 237 | 242 |
| Juice | 2,364 | 515 | 515 |
| Snack | 2,614 | 611 | 578 |
| Soda | 1,812 | 358 | 382 |
| Water | 1,160 | 241 | 248 |

What this means:

- **Every class has at least 240 test instances**, so the per-class numbers are not
  thin. Energy Drink and Water are about half the size of the others, so their
  per-class numbers are the least stable.
- **The objects are tiny:** a median box is ~0.75% of the image (about 55×55 px at
  640 px input). This is why resolution mattered (Run 2).
- **Train is augmented ×2 by Roboflow** (version 8 uses flip, small rotation, shear,
  hue and saturation), so there are only ~156 distinct training photos. Valid and
  test are not augmented.
- **The test split probably flatters the model.** The photos look like a small
  number of machines photographed repeatedly. Valid and test are likely the same
  machines in the same light as train, so a new machine (other products, other
  lighting) will score lower than our test numbers. The dataset has no machine IDs,
  so this cannot be checked.
- **The data has no empty-slot labels.** The "empty slots" figure is an estimate
  (capacity − detections). It is not validated against labelled empty slots, and the
  number of empty slots in the test images is itself assumed through the base rate.

## Finetuning and baselines

- **Model and task:** object detection with `yolo11n`. Detection, not classification,
  because a photo contains about 35 products and the decision is per slot.
- **Where trained:** MacBook M3 (Apple MPS GPU) via `uv run python main.py train`.
- **Shared settings:** batch 16, seed 0, optimizer "auto" (AdamW, lr 0.001).

| Run | Setting | Train time | Test mAP50 | Test mAP50-95 |
|---|---|---|---|---|
| Run 1 | yolo11n from COCO, 50 epochs at 640 px | 32 min | 0.964 | 0.676 |
| **Run 2 (final)** | Run 1 weights + 20 more epochs at 800 px | +16 min | **0.970** | **0.682** |
| Run 3 | none: I kept Run 2 (see below) | – | – | – |

The baselines are compared on the same test split, class-agnostic (product yes/no).
COCO does not know our six classes, so for it only "is there a product" is comparable.

| Model | AP50 (product yes/no) | Threshold | TP | FP | FN | € / 1,000 slots |
|---|---|---|---|---|---|---|
| **Fine-tuned yolo11n (Run 2)** | **0.972** | 0.80 | 2,253 | 62 | 263 | **160** |
| COCO yolo11n zero-shot (bottle/cup/food classes count as product) | 0.602 | 0.45 | 175 | 8 | 2,341 | 387 |
| Trivial: predict nothing | 0.000 | – | 0 | 0 | 2,516 | 400 |

- The fine-tuned model **beats both baselines by a wide margin**.
- COCO finds some bottles, but at any useful confidence it misses ~93% of products:
  snack bags and chocolate bars are not COCO objects.
- The gap is larger than I expected for COCO's AP50. It looks decent only because
  it ranks the few bottles it finds well. At its best threshold it still costs
  almost as much as predicting nothing.

**Why Run 2 and not something else.** Run 1 failed by *missing* products at high
confidence (488 FN at its threshold), and the products are tiny, so I raised the
resolution.
- A full retrain at 960 px was measured at ~3.6 min per epoch (~3 hours), so I
  stopped it.
- At 800 px an epoch takes ~48 s. Continuing from Run 1 for 20 epochs cut the test
  FN at the chosen threshold from 488 to 263.

**Why no Run 3.** The Run 1 → Run 2 gain was small (+0.6 mAP50). The validation
curve was flat at the end of Run 2, and the remaining options were either slow
(960 px, a bigger model) or expected to give less than one point. I chose to spend
the time on the cost analysis instead.

## Per-class evaluation and failure modes

Run 2 on test (Ultralytics metrics). **Positive class = product present.**

| Class | Precision | Recall | AP50 | Test instances |
|---|---|---|---|---|
| Chocolate | 0.954 | 0.945 | 0.967 | 551 |
| Energy Drink | 0.979 | 0.973 | 0.977 | 242 |
| Juice | 0.945 | 0.950 | 0.973 | 515 |
| Snack | 0.957 | 0.956 | 0.976 | 578 |
| Soda | 0.967 | 0.961 | 0.969 | 382 |
| **Water** | **0.929** | 0.948 | **0.956** | 248 |

Confusion matrix at the chosen threshold 0.80 (rows = true, columns = predicted,
`bg` = background/no box). It is saved in `outputs/confusion_matrix.png`.

| true \ pred | Choc | Energy | Juice | Snack | Soda | Water | bg (missed) |
|---|---|---|---|---|---|---|---|
| Chocolate | 494 | 0 | 0 | 6 | 0 | 0 | 51 |
| Energy Drink | 0 | 208 | 0 | 0 | 0 | 0 | 34 |
| Juice | 0 | 0 | 457 | 0 | 2 | 4 | 52 |
| Snack | 5 | 0 | 0 | 506 | 0 | 0 | 67 |
| Soda | 0 | 3 | 3 | 0 | 344 | 0 | 32 |
| Water | 0 | 0 | 0 | 0 | 0 | 221 | 27 |
| bg (phantom) | 17 | 1 | 11 | 17 | 10 | 6 | – |

- **Headline:** Energy Drink and Snack carry the highest AP50. **Water** is the
  weakest class: lowest precision and AP50, and the second-smallest class.
- **The model rarely confuses categories.** The only real mix-up is Snack ↔
  Chocolate (11 boxes), and both are wrapped bars or bags.
- **The real failure is missed products.** 263 of 2,516 (10%) fall below 0.80. The
  most missed are Snack (67), Juice (52) and Chocolate (51): small, partly hidden
  items at the back or bottom of the machine.
- **Phantom products (62)** appear mostly as Chocolate and Snack. These are the
  expensive errors.

## Cost matrix and operating point

| Outcome | Cost | Where the figure comes from |
|---|---|---|
| FP (phantom product, empty slot not refilled) | €6.00 | €0.80 margin × 1.5 sales/day × 5 days until next visit |
| FN (product missed, wasted check) | €0.50 | ~2 min of restocker time at €15/h |
| TP / TN | €0 | normal operation |
| Human review of one flagged box | €0.0125 | 3 s at €15/h |

- **Base rate:** 80% of slots occupied.
- **Cost per 1,000 slots** = 1000 × (FP·6 + FN·0.5) / (products / 0.8).

**Sweep.** I swept the confidence threshold from 0.05 to 0.95 in steps of 0.05. I
**chose the threshold on the validation split** and report the cost on test, so
the test set never tunes anything.
- The curve (in `outputs/sweep.csv` and dashboard tab 3) falls steeply from ~1,500 €
  at 0.05, where there are many phantoms, to its minimum, then rises again above
  0.90, where too many products are missed.
- **Chosen operating point: 0.80 → €160 per 1,000 slots on test.**

**Sensitivity: what moves the threshold.** The FP cost is the most uncertain number,
because it depends on how often machines are visited. Repeating the validation
sweep with other FP costs gives:

| FP cost (€) | 1 | 2 | 3 | 6 | 10 |
|---|---|---|---|---|---|
| Optimal threshold | 0.65 | 0.75 | 0.75 | **0.80** | 0.85 |

If machines are visited daily, so a phantom costs only ~€1, the threshold should
drop to ~0.65. Changing the base rate does not move the threshold at all, because it
only rescales the cost.

## What a person still has to check

**Three outcomes, per detected box:**
- confidence ≥ **0.90** → accept as product (pass)
- confidence < **0.10** → treat as empty slot (fail)
- in between → a person looks at the crop (review)

I picked these two thresholds as the cheapest pair on validation whose review load
fits the staff available. The reviewer is assumed always to be right: they keep
real products and delete phantoms.

**Load (test results scaled to the operation):**
- 200 machines × 40 slots = 8,000 slots/day.
- **57%** of slots are sent to review: ~4,560 boxes/day × 3 s ≈ **3.8 h/day**.
- That fits **one person part-time (4 h)**. It is realistic but tight: with 20%
  more machines, it no longer fits.

| Option | € / 1,000 slots | € / day (8,000 slots) | Fits 4 h of staff? |
|---|---|---|---|
| Do nothing (no system; empty slots go unnoticed until the visit) | 1,200 | 9,600 | – |
| Model alone, threshold 0.80 | 160 | 1,281 | yes, no staff needed |
| **Model + human review, band 0.10–0.90 (chosen)** | **57** | **456** | **yes, 3.8 h** |
| Person checks every slot | 12.5 | 100 | **no: needs 6.7 h/day** |

**Conclusion.** With the staff available, **model + review is the cheapest option
that is feasible**: 2.8× cheaper per slot than the model alone, and 21× cheaper
than doing nothing.
- On paper, a person checking every slot is cheaper. But it needs 6.7 h/day, which
  means ~2.7 more staff hours a day than are available (about a third of a
  full-time employee more).
- It also relies on 3 seconds per slot being enough to check a whole photo slot by
  slot, which is optimistic.
- **Break-even:** if a full manual check takes more than ~14 s per slot
  (57 € / 1,000 slots ÷ €15/h), model + review wins even with unlimited staff.

## Limitations & next steps

- **Few machines, test probably near-duplicate of train.** Expect worse results on
  new machines. *Next:* collect photos from 10–20 of our own machines, split by
  machine, and re-measure.
- **Empty slots are estimated, not detected.** The figure depends on knowing each
  machine's capacity and assumes one visible front item per stocked slot. *Next:*
  label a few hundred photos with an explicit `empty` class, or store each machine's
  slot layout.
- **The FP cost drives the threshold** and depends on visit frequency. *Next:* use
  the operator's real sales per slot and visit schedule, and pick the threshold
  per route.
- **The review load is close to the limit (3.8 of 4 h).** *Next:* narrow the band, or
  sort review boxes by expected cost so the person sees the riskiest ones first.
- **Water is the weakest class and Snack/Chocolate are sometimes mixed.** This only
  matters for van loading. *Next:* more Water examples.
- **Higher resolution was too slow on a laptop** (960 px ≈ 3 h). *Next:* retrain at
  960 px on a GPU (e.g. Colab T4) to see whether missed products drop further.

## Conclusion
The model detects the products well and almost never confuses categories. Its most common error is not detecting small or partly hidden products, e.g. snacks, juices and chocolate bars at the back or bottom of the machine. For the most important decision: as inventing a product where there isn't one is 12x more expensive than not seeing it, we operate at a 0.80 confidence threshold and send the doubtful detections (between 0.10 and 0.90) for a human to supervise. This lowers the cost from €160 to €57 per 1,000 slots and needs 3.8 of the 4 review hours available per day. Our vending machine dataset is not very large, so before deploying this project we should create a dataset of our own by taking pictures of vending machines, e.g. the vending machines on the IE campus (there are a lot).

## Generative AI use disclosure

I used Claude (Anthropic) through Claude Code as a coding assistant:
- It searched Roboflow Universe for candidate datasets and wrote `train.py` and
  `app.py`.
- It ran the training on my laptop and drafted this report from the results.
- For every decision it gave me options with pros and cons, and **I chose**: the
  dataset, the positive class and framing, the FP/FN/review costs, the base rate,
  the staffing assumptions, the model and resolution of each run, stopping at
  Run 2, and the final recommendation.
- I reviewed the numbers and can explain each one.
