# Assignment 2 Report

- **Name**: Juan Diaz-Bernardo Mier
- **Student ID**:19785
- **Email**:jdiazbernard.ieu2022@student.ie.edu
- **Group**: BBADBA A

## Business problem, and the cost of being wrong

Decision: given a photo of a leaf sample, should that field section be
sprayed/treated or left alone? Acted on by a farmer or field technician
doing routine leaf checks from phone photos.

Positive class for costing: "Disease present" (Powdery mildew + Rust
merged); per-class results still reported separately for all 3 classes.

False negative (Healthy predicted, actually diseased): disease spreads
undetected until the next check, causing more crop damage and often needing
a harsher, costlier treatment later.

False positive (diseased predicted, actually healthy): one unnecessary
spray — chemical, labor, minor crop stress, small long-run resistance-
buildup cost. Real but much smaller than a missed outbreak.

Not symmetric: a false positive costs one wasted treatment on one section;
a false negative risks spreading to neighboring sections before anyone
notices, compounding the longer it goes undetected.

## Dataset

Workspace: `simpleteam`, project: `plant-health-dtdb8`, version: `1`.
https://universe.roboflow.com/simpleteam/plant-health-dtdb8

One image = one leaf, photographed to show its condition. Classes:
Healthy, Powdery mildew, Rust. (A 4th "Unlabeled" folder in the raw export
is not a real class — un-annotated leftovers, excluded from everything
below.) This serves the problem directly: the model judges a single leaf's
disease status from a photo, exactly what a technician would want judged.

## What the dataset's composition implies

| Class          | Train | Valid | Test | Total |
|----------------|-------|-------|------|-------|
| Healthy        | 36    | 7     | 2    | 45    |
| Powdery mildew | 162   | 15    | 8    | 185   |
| Rust           | 171   | 17    | 8    | 196   |
| **Total**      | 369   | 39    | 18   | 426   |

- **Class imbalance**: Healthy has ~1/4 the images of either disease class.
  A model could score well overall by defaulting toward "diseased" — read
  Healthy's precision/recall with that in mind.
- **Thin Healthy test split**: only 2 images — one misclassification swings
  recall by 50 points. Flagged, not hidden.
- **Small overall train set** (369 images, 3 classes): expect improvement
  over baselines, not state-of-the-art numbers, and watch for overfitting.
- **Test set's class mix (~89% diseased) doesn't match a real field**
  (mostly healthy, most of the time). This is why the costing section below
  uses a different, lower assumed base rate rather than the test set's own
  balance.

## Finetuning and baselines

**Model/task**: `yolo11n-cls` (classification, not detection) — each image
is one leaf shown whole, nothing to localize, so classification fits the
data better than forcing bounding boxes onto single-subject photos.

**Training**: 25 epochs requested; early stopping (patience=8) at epoch 12;
best weights from epoch 4. CPU, imgsz 224, batch 16, AdamW (auto-selected).
~3.5 minutes total, trained locally.

**Baselines**: the brief names "pretrained COCO zero-shot," but COCO's
object classes don't apply to leaf-disease classification. Substituted
**CLIP** (`openai/clip-vit-base-patch32`), prompted zero-shot with text like
"a photo of a plant leaf that is rust" — the equivalent "big pretrained
model, no finetuning" baseline for this task type. Second baseline: always
predict the majority training class (Rust).

| Model | Accuracy |
|---|---|
| Finetuned yolo11n-cls | **0.833** |
| CLIP zero-shot | 0.611 |
| Majority-class baseline | 0.444 |

The finetuned model beats both by a wide margin — the gap over majority-
class shows it's learning real visual distinctions, not just exploiting
imbalance; the gap over CLIP shows finetuning on domain images matters more
than generic pretrained vision-language knowledge here.

## Per-class evaluation and failure modes

Positive class for the headline number: "Disease" (Powdery + Rust).

**Finetuned model** confusion matrix (rows = true, cols = predicted):

|                | Healthy | Powdery | Rust |
|----------------|---------|---------|------|
| True Healthy   | 2       | 0       | 0    |
| True Powdery   | 0       | 7       | 1    |
| True Rust      | 1       | 1       | 6    |

| Class | Precision | Recall | Support |
|---|---|---|---|
| Healthy | 0.67 | 1.00 | 2 |
| Powdery mildew | 0.88 | 0.88 | 8 |
| Rust | 0.86 | 0.75 | 8 |

Overall accuracy: 0.833.

**Where it fails**: Rust has the lowest recall (0.75) — one rust leaf
misclassified as healthy, one as powdery mildew. The rust→healthy miss is
the most consequential failure in the whole evaluation: the model was
**100% confident** this leaf was healthy (P(disease) = 0.000). A
confidently-wrong prediction, not a borderline one — see the cost section
for why this matters.

**CLIP** struggled specifically with Rust (recall 0.25 on the same test
images) — it misses most rust cases using only generic text-image matching,
never having seen a labeled rust example. **Majority-class** always
predicts Rust, so gets Rust's recall to 1.0 while being useless for the
other two classes — a clear illustration of why accuracy alone is a bad
summary statistic.

## Cost matrix and operating point

| Outcome | Cost |
|---|---|
| False negative (missed disease) | €50 |
| False positive (unnecessary spray) | €5 |
| True positive (correctly caught) | €5 |
| True negative (correctly left alone) | €0 |

False negative ≈10x a false positive: a missed disease spreads before the
next check and often needs costlier treatment later, versus one wasted
spray. TP and FP share €5 since the *action* (spray this section) is
identical — only its justification differs.

**Assumed base rate: 10%** of field sections actually diseased at a given
check — deliberately *not* the test set's own ~89% (built to contain enough
disease examples to evaluate, not representative of a real field). **This
is the assumption most likely to change the answer if wrong**: at a much
higher real base rate (e.g. an active outbreak), false negatives matter
proportionally more, pushing the optimal threshold lower.

Swept confidence threshold on P(disease) = 1 − P(healthy) from 0.00–1.00
(step 0.01), computing expected cost per 1,000 cases at each point.

**Chosen threshold: 0.38** → TPR 0.9375, FPR 0.0, **€781.25 per 1,000
cases** — the sweep's minimum. Cost stays flat and near-minimal from ~0.38
to ~0.91, partly because only 18 test images means few distinct confidence
values to produce a smooth curve.

## What a person still has to check

Two thresholds: pass edge 0.30, fail edge 0.55, on P(disease). Below 0.30:
no action. At/above 0.55: treat automatically. Between: sent to a person.

On the 18-image test set: Pass 2, Review 1, Fail 15 — **5.6% review share**.

**Staffing**: assume a regional cooperative processing 1,000 photos/day
(consistent with the 10% base rate above). 5.6% review share → **56
cases/day**. At 3 min/review → 2.8 hours/day → **under half of one
full-time person (0.47 FTE)** — folds into an existing technician's duties
rather than justifying a dedicated hire.

**Priced against the three alternatives** (1,000 cases/day, €15/hr wage):

| Alternative | Error cost/day | Labor cost/day | Staffing | Total/day |
|---|---|---|---|---|
| A. Do nothing | €5,000 | €0 | 0 | **€5,000** |
| B. Human checks everything | ~€0 | €750 | 8.3 FTE | **€750** |
| C. Model alone (single threshold) | €781.25 | €0 | 0 | **€781.25** |
| D. Our 3-way system | €781.25 | €42 | 0.47 FTE | **€823.25** |

**Honest reading**: B is cheapest in raw euros but needs 8+ full-time
reviewers — unrealistic for a small cooperative, and ignores that a human
checking 1,000 images/day would likely fatigue and err more than this
table assumes. D costs slightly *more* than C here, not because a review
band is inherently worse, but because 18 test images is too few to show its
benefit: the one genuinely dangerous error (the 100%-confident rust→healthy
miss) didn't land in the review band, because the model's confidence sat at
the extreme (0.000), not the ambiguous middle the band targets. No
threshold placement could have caught it — a model limitation, not a
tuning problem. The review band is still worth keeping as low-cost
insurance against this kind of failure, even though this sample doesn't
happen to show the payoff.

## Limitations & next steps

- **Test set is tiny (18 images)** — every metric here is a rough signal,
  not a precise estimate. *Next*: source a larger, better-balanced dataset
  (more Healthy images) before trusting these numbers for deployment.
- **Class imbalance** risks the model defaulting toward "diseased". *Next*:
  collect more Healthy images, or use class-weighted loss.
- **One confidently-wrong prediction** shows real miscalibration; no
  threshold can fix a maximally-confident wrong answer. *Next*: investigate
  what made that image different (lighting, angle, disease stage), and try
  calibration techniques (e.g. temperature scaling).
- **Base rate and costs are assumptions, not measured figures.** *Next*:
  get real numbers from an agricultural extension service and rerun the
  sweep.
- **Images are mostly studio/stock photos** (filenames like "istockphoto-
  ..."), not messy real field photos. *Next*: validate on real field-
  captured images before trusting this in production.

## Generative AI use disclosure

Used Claude (Anthropic) as a collaborator under my direction