# Assignment 2 Report

- **Name**: Paula Evangelista
- **Student ID**: 20150
- **Email**: pevangelista.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

I picked a hygiene gate at the entrance of the filling hall in a luxury cosmetics
plant. A camera takes one photo of each operator before they go in, and the model
decides one of three things: **pass** (mask on, the door opens), **fail** (no mask,
the door stays shut until they put one on) or **review** (the model is not sure, so
a supervisor looks at the photo). The model only checks the mask. It does not
identify anyone.

The class I care about is **NO-Mask**, because that is the one I must not miss.

The two mistakes do not cost the same:

- **Missing a no-mask (false negative): about €300.** If an unmasked person gets
  in, there is a small chance the product gets contaminated and a batch is put on
  hold. I assumed a 2% chance of that and a €15,000 cost per hold, so the expected
  cost is 0.02 × 15,000 = €300.
- **False alarm (false positive): about €1.** A masked person is stopped by
  mistake and a supervisor needs around 30 seconds to check, at roughly €25 an
  hour, plus a little waiting time for the operator.

So a miss is about 300 times worse than a false alarm. That is why I will set the
threshold low: I would rather stop a few people by mistake than let one through.
It is also why I use a review band. If I only used a low threshold, I would stop
too many masked people, so the uncertain cases go to a person instead.

I assume 2% of entries are without a mask. This is an assumption, not data. Most
people comply, but a few are rushed or have just taken it off. My headline metric
is recall on NO-Mask, together with the share of cases sent to review. I do not
use overall mAP as the main number because it treats both classes the same and
hides this asymmetry.

The assumption that would change my answer the most is the cost of a miss. If a
miss cost much less (for example, a 0.2% chance of a hold), the best threshold
would be higher and fewer cases would need review.

## Dataset

- **Workspace / project / version:** `roboflow-universe-projects` /
  `mask-wearing-iskms` / **version 6** ("roboflow train v3")
- **Link:** https://universe.roboflow.com/roboflow-universe-projects/mask-wearing-iskms/dataset/6
- **Task:** object detection with 2 classes, `Mask` and `NO-Mask`. Each box is
  drawn around a face.
- **What one image represents:** one photo of one or more people, taken at
  close range, with a box around each face and a label saying whether that
  face has a mask on. Many of the images look like indoor webcam-style photos.

**Why it serves my problem and does not just resemble it.** At the gate, the
decision is whether a face has a mask on or not, and this dataset is labelled
for exactly that, with the two classes I need. The boxes also let me turn the
model output into a per-image decision: an image fails if it contains a
NO-Mask detection above the threshold, and passes if it has a Mask detection
and no NO-Mask. The photos are close and mostly frontal, which is close to what
a gate camera would see.

**Where it only partly fits.** I am being honest about three gaps:
- The masks are everyday masks. A cleanroom would also need hairnets, hoods
  and goggles, which this dataset does not cover, so I only claim to check
  the mask.
- The lighting and backgrounds are not those of a plant.
- Some images contain several people, while the gate would only see one.

I chose version 6 because it's valid and test splits are original images, and
only the training split is augmented. That means the test results are not
inflated by copies of training images.

## What the dataset's composition implies

| Split | Images | Mask instances | NO-Mask instances |
|---|---|---|---|
| Train | 7,413 | 5,932 | 4,693 |
| Valid | 706 | 554 | 505 |
| Test | 353 | 329 | 218 |

**Object size.** I measured this on the test split (547 boxes). The median face
box is about 12% of the image width and 19.5% of the image height, which is
about 2% of the image area, or roughly 77 x 125 pixels after the 640x640 resize.
58% of the boxes cover less than 5% of the image area. So the faces are small,
and I expect this to hurt recall, especially on NO-Mask, because a small face
with a hidden or blurry mouth area is the hardest case to judge and it is also
the mistake that costs me the most.

**What the numbers tell me.**
- **The training counts are inflated.** Train is augmented three times, so the
  7,413 images are about 2,470 real ones. Valid and test are original images,
  so the test results are not inflated by copies.
- **Both classes have enough data to train on.** Neither is close to 100
  instances, which is where I would stop trusting a number.
- **NO-Mask is the thin class in test, with 218 instances.** Its recall can
  move by a few points just from which images ended up in the split. For a
  recall near 90%, the 95% interval is about plus or minus 4 points
  (1.96 x sqrt(0.9 x 0.1 / 218)). So I will not treat small differences
  between models, or small changes in the threshold, as real.

**Does the test split look like what I would really see?** No, in two ways.
- **Class mix.** About 40% of test instances are NO-Mask (218 of 547), but at my
  gate I assume only 2% of entries are without a mask. Test precision would
  therefore be far too optimistic, because in real life almost every alarm would
  be a masked person. Before I put any euro value on the confusion matrix, I
  reweight it to the 2% base rate.
- **Face size.** Many faces here are small (median about 2% of the image area),
  so the photos are often wider scenes than what a gate camera would capture. A
  gate camera at close range gives larger faces, so on this point the test is
  probably harder than my real use case. This is the one place where my results
  are likely conservative, not optimistic.

**What I can't conclude from my results.**
- I can't say how many false alarms the gate would really produce. I can only
  estimate it after reweighting to the assumed 2% base rate.
- I can't say it works under plant lighting or with cleanroom clothing
  (hairnets, hoods, goggles), because the dataset has neither.
- I can't rule out that the same person appears in both train and test. I saw
  the same person several times in the preview images, so if this happens, the
  test score is a bit optimistic.
- I can't conclude small differences in NO-Mask recall, because of
  the 218 test instances.

## Finetuning and baselines

**Model and task.** I used object detection with yolo11s, finetuned from the pretrained COCO checkpoint (`yolo11s.pt`). I chose detection because the dataset has a box around every face and one photo can hold several people with different labels. I turn the detections into a gate decision afterwards: an image's NO-Mask score is the highest NO-Mask confidence found in it. I chose size s over n because the faces are small (median about 2% of the image area) and a GPU made the bigger model cheap to train. I did not run a full yolo11n comparison, so I cannot say how much that choice helped.

**Training setup.** Google Colab, Tesla T4 GPU, Ultralytics 8.4.190, 40 epochs (1.2 hours), image size 640, batch 16, seed 0. Ultralytics chose the optimizer automatically (AdamW, learning rate 0.0017) and applied its default augmentations on top of the dataset's own. `weights/best.pt` is the epoch with the best validation score. It was one run with one seed.

**Baselines on the same test split** (353 images, 158 of them with at least one no-mask face; an image counts as positive if it has a NO-Mask face; confidence 0.25).

| Model | Caught | Missed | False alarms | Precision | Recall |
|---|---|---|---|---|---|
| Finetuned yolo11s | 157 | 1 | 2 | 98.7% | 99.4% |
| COCO zero-shot, rule: any person = NO-Mask | 157 | 1 | 115 | 57.7% | 99.4% |
| Predict nothing (everyone passes) | 0 | 158 | 0 | n/a | 0% |

COCO has no mask classes, so as it comes, it can never output NO-Mask. That is why I gave it the "person = NO-Mask" rule, which is the most it can do. Box-level results for the finetuned model on test: mAP50 0.935 and mAP50-95 0.646 (on validation: 0.951 and 0.714).

**What changed.** Against predicting nothing, the model catches 157 of 158 no-mask images instead of none. Against the COCO rule, recall is the same but false alarms drop from 115 to 2. The COCO rule only reaches 99.4% recall because it flags 272 of 353 images. At my assumed 2% base rate, its precision would fall to about 3%, roughly one correct alarm in 30.

**Was the gap as large as I expected?** On precision, yes. I expected COCO to lose on recall too, and it did not, which shows that recall alone would have hidden how useless that baseline is. The test mAP is a little lower than validation. My guess is that part of it comes from choosing `best.pt` on the validation set, but I did not test that.

The final evaluation outputs came from rerunning `SKIP_TRAIN=1 python main.py train` on the saved weights: the first evaluation ran out of GPU memory in the same process as training. I then fixed that in `train.py` (smaller prediction batches, freeing memory first) and checked the whole pipeline end to end with a quick run (`QUICK=1`, 2 epochs on a subset). I did not rerun the full 40-epoch run in one process after the fix.

## Per-class evaluation and failure modes

The positive class is **NO-Mask**. Box-level results on the test split (precision and recall are at the confidence that maximises F1):

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| NO-Mask | 0.951 | 0.922 | 0.953 | 0.686 |
| Mask | 0.932 | 0.874 | 0.918 | 0.606 |
| all classes | | | 0.935 | 0.646 |

Confusion matrix at confidence 0.25 (boxes; rows predicted, columns true; "background" means no face):

| | true Mask | true NO-Mask | true background |
|---|---|---|---|
| predicted Mask | 295 | 3 | 35 |
| predicted NO-Mask | 4 | 208 | 12 |
| predicted background | 30 | 7 | 0 |

**Which class carries the headline.** NO-Mask carries it (AP50 0.953, and 99.4% image-level recall). Mask is the weaker class (recall 0.874, AP50 0.918). That matters less for the gate, because the decision only uses the NO-Mask score.

**Where it breaks.**
- **Masks and no-masks are rarely confused with each other.** That happens 7 times (4 + 3). The errors are faces the model misses entirely (30 Mask, 7 NO-Mask) and boxes where there is no face (35 Mask, 12 NO-Mask).
- **NO-Mask boxes at 0.25:** 208 of 218 found (95.4% recall) and 208 of 224 predictions correct (92.9% precision).
- **The one image I miss at 0.25** is a group photo with 1 unmasked and 3 masked faces; its NO-Mask score is 0.056. The two false alarms at 0.25 are one photo with no labelled face (score 0.47) and one with two masked faces (0.69).
- **AP50-95 is far below AP50** (0.646 against 0.935). My reading is that the model finds the faces but the boxes are not tight on small faces. I did not check this visually. The gate only needs to know that a NO-Mask box exists, so it matters little for the decision.
- **What I did not check:** results by face size, skin tone, glasses, head coverings or lighting. I cannot claim the model works equally well for everyone.


## Cost matrix and operating point

All of these figures are assumptions. I have no plant data.

| Item | Value | Where it comes from |
|---|---|---|
| Missed no-mask (false negative) | €300 | 2% chance an unmasked entry leads to a batch hold, times €15,000 per hold |
| False alarm (false positive) | €1 | 30 seconds of a supervisor at about €25/hour (€0.21) plus the operator waiting at a locked door; I rounded up |
| Correct block | €0.10 | about 15 seconds' delay while the mask goes on |
| Correct pass | €0 | nothing happens |
| One human review | €0.21 | 30 seconds of a supervisor at €25/hour |
| Reviewer misses a no-mask | 2% | assumption: one look at a photo |
| Base rate of no-mask entries | 2% | assumption: most operators comply |

A review costs less than a false alarm because it happens on a screen before the door decision, so nobody walks to the door. A false alarm locks the door, and someone has to come over.

**Reweighting.** The test set is 40% no-mask images, so I reweight the rates to my 2% base rate before costing: cost per entry = 2% × (cost of positive cases) + 98% × (cost of negative cases).

**The sweep.** I swept the threshold on the **validation** set (706 images, 327 positive) and reported the result on the untouched **test** set. The single-threshold cost is lowest at 0.10 (€41 per 1,000 on validation) and flat between about 0.10 and 0.30 (€41 to €46), so the exact minimum matters little. The plot is in `outputs/cost_curve.png` and in tab 3 of the dashboard. For the two-threshold design (pass below 0.10, fail from 0.72, review in between), I searched both edges on validation, with the review load capped at 12% of cases (60 reviewer minutes a day). The cap did not bind.

| Design | Cost per 1,000 entries, test | Same on validation |
|---|---|---|
| Do nothing | €6,000 | €6,000 |
| A person checks everyone | €330 | €330 |
| Model alone, threshold 0.10 | €70 | €41 |
| Model + human review, 0.10 and 0.72 | **€54** | €18 |

**Why test is higher than validation.** One test image (the group photo, score 0.056) is a miss at 0.10. One miss in 158 positives is worth about €38 per 1,000 entries (0.0063 × 2% × €300 × 1,000). On validation, 0 of 327 positives were missed. Over both splits, it is 1 miss in 485. Resampling the test images 5,000 times, the hybrid costs €52 at the median, with a 95% interval of **€11 to €137**. It was cheaper than a person checking everything in every resample (worst case €291 against €330). This treats images as independent.

**What would change my answer.**
- **What is robust.** The gate beats doing nothing unless a miss costs less than about €0.43. It beats a person checking everyone even if reviewing were free, because a person who misses 2% of cases still costs €122 per 1,000. Only a person who never misses and checks each photo in about 5.6 seconds would tie. Across base rates from 0.1% to 10% and miss costs from €1 to €3,000, a person checking everyone was never the cheapest design.
- **What is fragile: the review band.** It only pays while a false alarm costs more than about €0.47 (I assume €1), while a miss costs less than about €930 (I assume €300; above that, failing uncertain cases automatically beats trusting a reviewer), and while reviewers miss less than about 6%. Its benefit is small: about €16 per 1,000 entries over the model alone.
- **The assumption I trust least** is the cost of a miss (€300), because it sets both the threshold and whether review pays. The model's own miss rate is also uncertain (1 in 485). If the true rate were 2% instead of 0.6%, the hybrid would cost about €136 per 1,000, still under half of a person checking everyone.


## What a person still has to check

My thresholds are 0.10 and 0.72. On the test set they send **3.1% of entries** to a person (3.7% on validation). At 1,000 entries a day, that is about **31 reviews a day** (37 on validation), at 30 seconds each, which is **16 minutes a day** (19 on validation), or **0.04 of a person** (420 productive minutes a day). In the test set the reviewed images are 16 of 353: 10 with a no-mask face and 6 with only masked faces.

**Is that realistic?** In headcount, yes: one supervisor spends under 20 minutes a day. The real constraint is response time. The operator waits while the supervisor looks, so someone must be reachable (for example on a tablet) and answer within about a minute. At shift change several operators arrive together, and 31 a day is only an average. I did not model those bursts. The reviewer should see the photo on screen only, and the photo should not be stored.

**Priced against the alternatives** (per 1,000 entries, which is one day at my assumed volume):

| Design | Cost | Person time | Comment |
|---|---|---|---|
| Do nothing | €6,000 | none | every no-mask entry gets in |
| A person checks everyone | €330 | 500 min = 1.19 people | more than one full-time person, and people miss 2% |
| Model alone (0.10) | €70 | none | no human in the loop |
| Model + review (0.10 / 0.72) | **€54** | 0.04 people | my design |

Over 250 working days, the hybrid saves about €69,000 a year against a person checking everyone, and about €4,000 a year against the model alone. Against doing nothing it saves about €5,900 a day, but that figure rests entirely on my miss-cost assumption, so it is the least certain number in this report.

## Limitations & next steps

- **The threshold rests on very little evidence.** The test set has 158 positive images and one miss moves the cost by €38 per 1,000 (interval €11 to €137). Next: evaluate on at least 1,000 no-mask images and report the interval with every cost.
- **The dataset is not the plant.** It has everyday masks, no hairnets, hoods or goggles, ordinary lighting, and many photos with several people, while the gate sees one. Faces here are small (median about 2% of the image), so a close-range gate camera would probably be easier. Next: collect a few hundred labelled photos from the real gate camera (with consent) and re-evaluate before deploying.
- **Possible leakage between train and test.** I noticed the same person in several preview images, so the splits may share identities and the test score could be optimistic. I did not check. Next: split by person or source video, or at least look for near duplicates across splits.
- **The costs are assumptions:** €15,000 per hold, 2% hold probability, 2% base rate, 2% reviewer miss rate, 1,000 entries a day, 30 seconds per review. Next: replace them with plant data (entry logs, batch-hold records, a time study of a review). The dashboard already has sliders for base rate and miss cost.
- **No fairness check.** I did not measure miss and false-alarm rates by skin tone, glasses, head coverings, beards or lighting. A gate that blocks some groups more often is a real harm. Next: build a labelled audit set and compare the rates by group before any use.
- **One run, one seed, one model size.** I cannot separate real differences from run-to-run noise. Next: three to five seeds, and compare yolo11n, yolo11s and a larger image size (for example 960) on small faces.
- **One score per image.** Taking the highest NO-Mask score is wrong for photos of several people or a face in the background. Next: crop to the person at the door before scoring.
- **Privacy.** Photos of operators are personal data, and mask status could be treated as health-related. I only used a public dataset. A real deployment would need a legal basis, no image retention, and a data protection impact assessment.

## Generative AI use disclosure

I used Claude (Anthropic) as an assistant throughout this assignment. It helped me to think through the problem choice, to draft the report sections, to write and debug `train.py` and `app.py` (most of the code was written by Claude from my instructions), to read error messages, and to check my numbers. I chose the scenario and the dataset, ran the training on Colab and every evaluation myself, checked the reported numbers against the saved outputs, and decided which assumptions to keep.
