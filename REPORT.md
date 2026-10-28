# Assignment 2 Report

- **Name**: Aitana Damotte Huedo
- **Student ID**: 18564
- **Email**: adamotte.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

**How to run it.**

- `uv run python main.py train` runs the whole pipeline: download, describe, train, test, cost.
- `uv run python main.py app` opens the dashboard.
- Every assumption is in [config.yaml](config.yaml).
- All numbers below come from [results/](results/). All figures are in
  [results/plots/](results/plots/).

## Business problem, and the cost of being wrong

**The decision.** I work in a clothing factory. At the end of the sewing line, a camera takes a photo
of every garment before it is packed. For each photo, the model chooses one of three actions:

- **Pass**: pack and ship the garment.
- **Review**: a quality-control (QC) inspector looks at the spot the model marked, and decides.
- **Reject**: the garment goes to the rework station to be fixed.

**Who acts on it.**

- QC inspectors do the reviews.
- The rework station handles rejected garments.
- The QC manager sets the thresholds and decides how many people to staff.

**What the model looks for.** I treat "garment has a defect" as the **positive class**.

- *Positive class: the thing I am trying to find. Here, a defective garment.*
- The model draws **boxes** around defects. *A box is a rectangle around a defect, with a label and
  a confidence score.*
- *Confidence: a number from 0 to 1 that says how sure the model is about a box.*
- I decide per garment. A garment's score is the confidence of its most confident box.
- If that score is at or above the **threshold**, the garment is flagged as defective. *Threshold:
  the cut-off score I choose; above it the model says "defect".*
- I care whether the garment ships, not which defect it has. The defect type only tells the
  reviewer what to look for.

**The two kinds of mistake.**

- **False negative**: the model misses a real defect, and a bad garment ships.
- **False positive**: the model flags a good garment, a false alarm.
- *True positive: a real defect that the model catches.*
- *True negative: a good garment that the model lets through.*

**What each outcome costs** (per garment, in euros; all in `config.yaml`):

| Outcome | Cost | Where the number comes from |
|---|---|---|
| False negative: defect shipped | **€30** | Return shipping and handling (~€8), refund and customer service (~€4), garment written off at factory cost (~€12), bad review and buyer chargeback (~€6). *A chargeback is a penalty a retail buyer charges for faulty goods.* |
| False positive: good garment rejected | **€3** | A second check at the rework station (~3 min at €16/h, ~€0.80), handling and waiting (~€0.70), and some rework that was not needed (~20% × €7). |
| True positive: defect caught | **€5** | Average repair (re-sew, spot clean) or selling it cheaper as a "second". |
| True negative: good garment shipped | €0 | Nothing extra happens. |
| Running the model | €0.01 | Camera station and computer, ~€25k spread over 3 years of production. |

**Why a missed defect is worse.**

- A missed defect reaches the customer. I pay for the return, the refund, the lost garment and the
  damage to the brand.
- A false alarm stays inside the factory. Someone looks at the garment again and sends it on.
- Catching a defect saves me €30 − €5 = €25. A false alarm costs me €3.
- So a miss is **about 8 to 10 times worse** than a false alarm.

**How this shapes everything after it.**

- **The number I care about most is recall.** *Recall: the share of real defects the model catches.*
  It matters more than **precision**. *Precision: the share of the model's alarms that are real
  defects.*
- **In theory, the cut-off is low.** I should flag a garment once the chance it is defective is above
  3 / (3 + 25) ≈ 11%.
- **The base rate pulls the other way.** *Base rate: how often a defect really happens.* Only ~5% of
  garments are defective. So even rare false alarms on the other 95% add up fast.
- **That is why I add a review lane.** A person cheaply double-checks the unsure cases.

## Dataset

**What I used.** Roboflow Universe, workspace **`sqc`**, project **`fabric-defect-detection-ehhyx`**,
version **2**.

- Link: <https://universe.roboflow.com/sqc/fabric-defect-detection-ehhyx/dataset/2>
- License: CC BY 4.0. I used it as it is and did not fork (copy) it.

**What one image is.** One 640×640 photo of fabric or a garment surface, with at least one labelled
defect. There are five **classes**. *A class is a type of defect the model learns.*

- **gray stitch**: a wrong or visible stitching line;
- **hole**;
- **selvet**: a fault in the selvage, the woven edge band of the fabric;
- **stain**;
- **tear**.

**Why it fits my problem.** These are the defects an inspector looks for at the end of the line. The
photos are close-ups of real fabric with folds, prints and uneven light, close to what a camera at
an inspection table would see.

**The data is cut into three splits.** *A split is a separate part of the data.*

- **Train**: the photos the model learns from.
- **Valid** (validation): photos used to tune choices, like the threshold.
- **Test**: photos kept aside to measure the final result honestly.

**Two datasets I tried first and rejected:**

1. **`hammy/fabric-defect-lcird-1ifel` v1** (my first choice) is broken.
   - 2,490 of its 2,491 label files are empty. The whole dataset has **one** box (a "stain").
   - It has no hole, line or knot examples.
   - It has no valid or test split.
2. **`yolov7-vdg8u/fabric-defect-detection-jdyz3` v3** looks better on paper: 157 test images and
   all four classes in test. But its test split has **leakage**. *Leakage: test photos that also
   appear in train, so the model has already seen the answers.*
   - I compared each test photo with every train photo. I shrank both to 64×64 grey pixels and
     measured the average difference.
   - **All 157 test photos and all 500 valid photos have an identical copy in train.** The
     difference is 0.2 out of 255, which only comes from re-saving the file. Unrelated photos differ
     by about 34.
   - A test score there would measure memory, not how the model does on new garments.

The same check on sqc v2 finds only 2 of 61 test photos and 2 of 123 valid photos with a copy in
train. So sqc v2 is the only one of the three I can trust.

## What the dataset's composition implies

I counted this from the label files ([results/dataset_stats.json](results/dataset_stats.json),
[results/plots/dataset.png](results/plots/dataset.png)).

| Split | Images | Unique source names | Boxes | Defect-free images | gray stitch | hole | selvet | stain | tear |
|---|---|---|---|---|---|---|---|---|---|
| train | 1,266 | 353 | 2,131 | **0** | 357 | 243 | 536 | 581 | 414 |
| valid | 123 | 115 | 177 | **0** | 19 | 44 | 30 | 49 | 35 |
| test | 61 | 57 | 103 | **0** | 27 | **3** | 27 | 28 | 18 |

**How big the defects are**, as the middle (median) box size in % of the photo, train / test:

| Class | Train | Test | Notes |
|---|---|---|---|
| gray stitch | 15.8% | 13.1% | |
| hole | 1.5% | 1.6% | the smallest 10% of holes are only 0.16% of the photo |
| selvet | 8.9% | 2.5% | |
| stain | 3.1% | 3.4% | |
| tear | 2.3% | 2.8% | |

Holes are tiny. Gray stitch and selvet boxes are long bands that cover a big part of the photo.

**What this means for my results:**

1. **There are only 3 holes in the test set, in 3 photos.** Any hole number on test is luck, not
   evidence. So I judge holes on the valid split instead, which has 44.
2. **The test set is small: 61 photos, with 18 to 28 boxes per class.** One photo more or less moves
   a class's recall by 4 to 6 points.
   - At my chosen threshold, the model catches 49% of defective photos.
   - The **95% interval** is **37% to 61%**. *95% interval: the range where the true value most
     likely lies, given how little data I have.*
3. **No split has a single defect-free photo.** In a real factory ~95% of garments are fine.
   - So the test set alone cannot measure false alarms on good garments.
   - A model that rejects everything would score 100% accuracy on it. *Accuracy: the share of photos
     judged correctly.*
   - **My fix:** I cut **defect-free crops** (256×256 pixel squares that touch no labelled defect)
     out of the valid and test photos. I got 301 valid crops and 131 test crops.
   - A crop is 1/6.25 of a full photo. So I estimate the chance a good garment's photo raises an
     alarm as 1 − (1 − crop alarm rate)^6.25. *I call 6.25 the "area factor".*
   - This is my weakest assumption. I test it in the sensitivity analysis below.
4. **The train set is three copies of each photo.** Its 1,266 images come from about 422 real
   photos. Roboflow saved each one three times with a small rotation and some speckle noise. *This
   is called **augmentation**: making slightly changed copies to have more training examples.*
   - So the model has seen much less variety than "1,266" suggests.
   - One **epoch** is really three passes over each real photo. *Epoch: one full pass of training
     over the train set.*
5. **The test photos are not exactly like the factory.** They are hand-held photos of loose fabric,
   with folds, shadows and many prints.
   - A factory camera would have fixed light and a flat garment, which could be easier.
   - But it would show whole garments, at a different scale.
6. **Roboflow stretched every photo to 640×640**, so shapes are a little squashed. I fed photos to
   the model the same way.

**What I cannot conclude:**

- I cannot give a reliable number for holes.
- My false-alarm rate is an estimate from crops, not a measurement on real good garments.
- The euro costs are not exact to the euro. The ranges I trust are the bootstrap intervals below.

## Finetuning and baselines

**The model.** I **finetuned** **yolo11n** for **object detection**.

- *Finetune: start from a model that already learned general images, and keep training it on my
  photos.*
- *Object detection: the model finds defects and draws a box around each one.*
- yolo11n starts from **COCO**. *COCO is a big public dataset of 80 everyday objects, like people,
  cars and bananas.*

**Why I made these choices.**

- **Detection, not classification.** *Classification would only say "defect" or "no defect" for the
  whole photo.* A box tells the reviewer where to look, so a review takes seconds, not a full
  inspection.
- **The nano model ("n").** It trains on a normal computer processor (CPU) in under 3 hours.
- **Not the larger yolo11s.** It would take about 3× longer (~9 hours). With only 61 test photos, a
  small gain would be lost in the noise.

**Where and how I trained:**

| Setting | Value |
|---|---|
| Machine | Claude Code cloud session (Linux computer, 4-core Intel Xeon CPU, no GPU), torch 2.11 CPU, Ultralytics 8.4.173 |
| Command | `uv run python main.py train` |
| Epochs | **30** (it could stop early after 10 epochs without improvement; it did not) |
| Image size / batch | **640** pixels / **16** photos at a time |
| Optimizer | Ultralytics "auto" → AdamW, learning rate 0.00111 |
| Seed | 0, so the run can be repeated exactly |
| Augmentations | Ultralytics defaults (mosaic, flips, colour changes) |
| Training time | **171 min** |
| Best epoch | 25 (valid mAP50 0.600, mAP50-95 0.314) |
| Weights (the saved model) | `weights/best.pt` |

I did not need Google Colab.

**One change after training.**

- I made the defect-free crops smaller, from 320 to 256 pixels. That gave me more of them: 131
  instead of 63 on test.
- Then I re-ran the testing and costing with `uv run python train.py --skip-training`. This uses the
  same saved model.
- Running `main.py train` with today's `config.yaml` reproduces everything.

**How I measure detection quality:**

- **IoU** (intersection over union): how much a predicted box overlaps the real one, from 0 to 1.
- **AP50**: one score from 0 to 1 for how well the model finds and ranks defects of one class. A
  box counts as correct if its IoU is at least 0.5.
- **mAP50**: the average of AP50 over all classes. This is my headline number.
- **mAP50-95**: the same, but averaged over stricter overlap rules (IoU from 0.5 to 0.95). It
  rewards boxes that fit the defect tightly.

**The baselines.** *A baseline is a simple comparison model, to check the real model adds value.*

- **COCO zero-shot**: the original COCO model, used without any training on my photos.
- **Predict nothing**: a trivial model that always says "no defect", so every garment passes.
- **Always reject**: a trivial model that always says "defect". This is the **majority class** on
  this test set, because every test photo is defective. *Majority class: the most common answer.*

**Results on the same test split (61 photos):**

| Model | mAP50 | mAP50-95 | Class-agnostic AP50 | Threshold (picked on valid) | Defects caught (image) | Good garments rejected | € per 1,000 if it decides alone |
|---|---|---|---|---|---|---|---|
| **Finetuned yolo11n** | **0.633** | **0.339** | **0.595** | 0.67 | 49.2% | 4.7% | **€1,029** |
| COCO yolo11n, zero-shot | 0 | 0 | 0.034 | 0.80 | 1.6% | 4.7% | €1,623 |
| Trivial: predict nothing (always pass) | 0 | 0 | 0 | – | 0% | 0% | €1,500 |
| Trivial: majority class on this split (always reject) | – | – | – | – | 100% | 100% | €3,100 |

*Class-agnostic AP50: AP50 when I ignore the defect type and only ask "did it box a defect?".*

**What the table shows:**

- **My finetuned model beats both baselines** on every detection score and on cost.
- **The COCO model scores near zero, as I expected.**
  - COCO's 80 classes include no fabric defects. So it can never name a defect, and its mAP is 0 by
    design.
  - To be fair, I also gave it credit for boxing *anything* on a defect (class-agnostic AP50). It
    still scores only 0.034.
  - It drew 412 boxes on the 61 test photos, with labels like "bed", "airplane" and "banana". To it,
    folds and prints look like objects.
  - **It costs more than doing nothing** (€1,623 vs €1,500). It rejects good garments but catches
    almost no defects.
- **Finding 1: the "always reject" baseline beats my model on test accuracy.**
  - Every test photo is defective, so "always reject" is 100% correct on it. My model is 49%
    correct.
  - That is why accuracy on this test set means nothing for the business.
  - With a realistic 5% base rate, "always reject" is the most expensive option of all (€3,100).
- **Finding 2: my model alone costs more than a person checking everything** (€1,029 vs €957).
  - At its cheapest threshold it still lets half the defects through.
  - It only wins when people review its unsure cases (see below).

The gap to the COCO model is as big as I expected. My model's own level (mAP50 0.63) is modest. I
expected that from ~422 real photos and a small model trained on a CPU.

## Per-class evaluation and failure modes

These numbers come from Ultralytics
([results/plots/per_class.png](results/plots/per_class.png)). Ultralytics reports precision (P) and
recall (R) at the threshold where the **F1 score** is best. *F1: one number that balances precision
and recall.*

| Class | Test boxes | Test P | Test R | Test AP50 | Test AP50-95 | Valid AP50 | Valid AP50-95 |
|---|---|---|---|---|---|---|---|
| gray stitch | 27 | 0.67 | 0.48 | 0.602 | 0.375 | **0.754** | **0.497** |
| hole | **3** | 0.89 | 0.67 | 0.665 | **0.482** | 0.568 | 0.306 |
| selvet | 27 | 0.81 | 0.47 | 0.562 | **0.196** | 0.493 | 0.205 |
| stain | 28 | 0.80 | 0.64 | **0.722** | 0.399 | **0.757** | 0.371 |
| tear | 18 | 0.66 | 0.54 | 0.616 | 0.242 | **0.423** | **0.192** |
| **All** | 103 | 0.77 | 0.56 | **0.633** | **0.339** | 0.599 | 0.314 |

At my own threshold (box confidence ≥ 0.67, IoU ≥ 0.5, test set):

| Class | Precision | Recall |
|---|---|---|
| gray stitch | 0.88 | 0.26 (7/27) |
| hole | 1.00 | 0.33 (1/3) |
| selvet | 0.83 | 0.19 (5/27) |
| stain | 0.87 | 0.46 (13/28) |
| tear | 1.00 | 0.28 (5/18) |

**Confusion matrix** (test set, confidence ≥ 0.67;
[results/plots/confusion_test.png](results/plots/confusion_test.png)).

- *A confusion matrix is a table of what was really there (rows) against what the model said
  (columns).*
- *The diagonal is correct answers.* "Background" means no defect: a missed defect in the last
  column, or a false box in the last row.

| true \ predicted | gray stitch | hole | selvet | stain | tear | background (missed) |
|---|---|---|---|---|---|---|
| gray stitch | **7** | 0 | 0 | 0 | 0 | 20 |
| hole | 0 | **1** | 0 | 0 | 0 | 2 |
| selvet | 0 | 0 | **5** | 0 | 0 | 22 |
| stain | 0 | 0 | 0 | **13** | 0 | 15 |
| tear | 0 | 0 | 0 | 0 | **5** | 13 |
| background (false box) | 1 | 0 | 1 | 2 | 0 | – |

**What carries the headline number: stain.**

- It has the best AP50 on both splits: 0.72 on test, 0.76 on valid.
- It also has the most training boxes (581).
- Gray stitch is strong on valid (0.75) but weaker on test.

**Where the model fails:**

1. **It never mixes up one defect type with another.** All the cells between defect types are 0.
   - Its errors are *misses*: 72 of 103 test boxes at 0.67. It also draws 4 false boxes on clean
     fabric.
   - For the business this is good news: when it draws a box, it is almost always a real defect.
   - The problem is that it stays silent too often.
2. **Selvet boxes are badly placed.** AP50-95 is 0.196 on test and 0.205 on valid.
   - Selvet faults are long bands. The model finds the fault but not its exact length.
   - At 0.67 it finds only 1 of 17 large selvet boxes.
3. **Tear is the weakest class on valid.** AP50 is 0.42 and precision 0.41. Tears on printed fabric
   look like part of the print.
4. **It misses small stains.** On test at 0.67:
   - it finds 12 of 18 large stains (more than 2% of the photo);
   - it finds 1 of 8 medium ones and 0 of 2 small ones;
   - on valid: 17 of 23 large, 6 of 16 medium, 0 of 10 small.
5. **Holes.** On valid (44 holes) it finds 2 of 10 small holes and 6 of 15 large ones. Test has only
   3 holes, so I do not judge holes on test.
6. **The six hardest test photos show the pattern**
   ([results/plots/failures.png](results/plots/failures.png)). Their best scores are only 0.06 to
   0.23:
   - a tear on a yellow flower print;
   - a faint tear on white fabric;
   - a faint stain on white fabric;
   - a selvet fault on a patterned sheet;
   - a gray stitch that looks like decorative stitching;
   - a 1 mm dark spot.

   **It breaks on low contrast and busy prints.**

## Cost matrix and operating point

*A cost matrix gives a euro cost to each outcome (caught, missed, false alarm, correct pass).*
*The operating point is the threshold I actually use.*

**Base rate.** I assume **5%** of garments reach the end of the line with a visible defect.

- Clothing factories usually measure 2–8% at final inspection. *Factories call this DHU, "defects
  per hundred units".*
- So 5% is a middle value.

**How I pick the threshold.**

1. **I try every threshold from 0 to 1, in steps of 0.01.** For each one I measure two rates:
   - recall: the share of defective photos flagged;
   - false-alarm rate: the share of good garments flagged (from the crops).
2. **I turn each pair into an expected cost per garment**, with p = base rate:
   p·[recall·€5 + (1−recall)·€30] + (1−p)·[false-alarm·€3] + €0.01.
3. **I pick the cheapest threshold on the valid split**, then report its result on test. This way I
   never tune on the numbers I report.

![cost vs threshold](results/plots/cost_vs_threshold.png)

**The model alone.** The cheapest threshold on valid is **0.67** (€1,058 per 1,000). On test at
0.67:

- it catches **49%** of defects (95% interval 37–61%);
- it flags 1 of 131 crops (0.8%; interval 0.1–4.2%), which means **4.7%** of good garments are
  rejected;
- it costs **€1,029 per 1,000** garments.

**The test curve's own cheapest point is 0.57** (€803).

- The valid curve is flat from 0.5 to 0.67. The test curve dips before 0.67.
- So 0.67 is a bit careful. I keep it, because picking on test would be cheating.

**Why the threshold is so high.**

- Each extra 1 point of false alarms costs 950 × €3 / 100 = €28.5 per 1,000.
- Each extra 1 point of recall saves only 50 × €25 / 100 = €12.5 per 1,000.
- Good garments are 19 times more common than bad ones, so false alarms weigh more.

**My assumptions, and when they would change my answer.**

This is a **sensitivity analysis**. *Sensitivity analysis: change one assumption at a time and see if
the answer changes.* For each value, I re-chose the thresholds on valid and re-priced all four
options on test ([results/sensitivity.json](results/sensitivity.json),
[results/plots/sensitivity.png](results/plots/sensitivity.png)).

- **A missed defect costs €30.**
  - I assumed €30 because a return, a refund and a written-off garment cost about that for a ~€40
    shop item.
  - The answer changes if it drops below **~€6**: then doing nothing is cheapest.
  - Above **€40**, the model-alone threshold drops to 0.53 and the review band gets wider.
  - Model + review beats a person checking everything at every value from €5 to €100.
- **A false alarm costs €3.**
  - I assumed €3 because a good garment that is rejected only needs a short second look.
  - If it drops below **~€2.55**, the model alone becomes cheaper than a person checking
    everything.
  - At €2, the model-alone threshold falls to 0.53.
  - Above €10, the model alone is not worth running (threshold 0.93).
  - Model + review stays cheapest from €0.5 to €20.
- **Fixing a caught defect costs €5.**
  - I assumed €5 for an average repair.
  - If a repair cost **~€29 or more**, about as much as shipping the defect, catching defects would
    stop paying.
- **The base rate is 5%.**
  - I assumed 5% from typical factory defect rates.
  - The model alone beats a person checking everything only if the base rate drops below **~4.2%**.
  - The model-alone threshold is 0.93 at ≤1%, 0.67 at 2–5%, 0.53 at 7–8%, and 0.30 at ≥15%.
  - Model + review is the cheapest option across the whole range from 0.5% to 30%.
- **A reviewer costs €16 per hour.**
  - I assumed €16/h, the full cost of a QC inspector in Spain (salary plus social security).
  - Above **~€18.7/h**, the model alone beats a person checking everything.
  - Model + review stays cheapest from €5 to €40/h.
- **The area factor is 6.25.** This is an assumption about my crops, not about the business.
  - If it drops below **~5**, the model alone beats a person checking everything (at 4 it costs
    €758).
  - Model + review stays cheapest from 1 to 12.

**The assumption that matters most is the false-alarm rate on good garments**, because I could only
estimate it from crops.

- It decides whether the model alone beats a person. With an area factor below ~5 it does; above
  ~5 it does not.
- It does **not** change my main recommendation, which is model + human review.

## What a person still has to check

**My two thresholds.** I chose them on valid, as the cheapest pair that needs at most 2 reviewers:

| Highest box confidence | Action |
|---|---|
| below **0.38** | pass |
| **0.38 to 0.93** | human review |
| **0.93 or above** | reject straight to rework |

- Without the staff limit, the cheapest low threshold on valid would be 0.11.
- That would need 3.7 reviewers, which I do not have.
- So the number of people, not my taste, set the low threshold at 0.38.

**What happens on the test set at a 5% base rate** (expected garments per 1,000; real test counts in
brackets):

| | Pass | Review | Reject |
|---|---|---|---|
| Actually defective | 8.2 (10 of 61 photos) | 41.8 (51) | 0 (0) |
| Actually good | 782.6 (127 of 131 crops) | 167.4 (4) | 0 (0) |

**How much work that is for people:**

- **20.9% of garments go to review.** At 4,000 garments a day, that is **837 garments a day**.
- **One check takes 30 seconds.** The reviewer sees the photo with the box, then looks at that spot
  on the garment.
- That makes **418 minutes a day**, or **1.05 reviewers**. One person has ~400 working minutes in an
  8-hour shift.

**What the result is:**

- **75% of defects are caught.** 84% of defects reach a reviewer, who confirms 90% of them.
- **Only 0.4% of good garments are wrongly sent to rework.**
- **Nothing scores 0.93 or more on test.** So in practice the reject lane is empty, and a person
  confirms every reject. I am fine with that: the model is not sure enough to reject garments on its
  own.

**Is it realistic? Yes, with two people.**

- One reviewer at 1.05 **FTE** is tight. *FTE: full-time equivalent, one person working full time.*
  A check every 30 seconds for a whole shift is tiring.
- On the valid split, the same thresholds need 1.92 reviewers. The load changes with the mix of
  fabrics.
- So I would staff **two inspectors** at the review station, each spending part of the day on other
  QC work. That fits the 2 people the QC team can give.
- If the factory grows to 10,000 garments a day, the best low threshold rises to 0.59 to stay within
  2 people. The cost then rises to €822 per 1,000.

**Compared with the other options** (test split, 5% base rate, per 1,000 garments;
[results/plots/strategies.png](results/plots/strategies.png)):

| Strategy | € per 1,000 | People needed | Defects caught |
|---|---|---|---|
| Do nothing (ship everything) | €1,500 | 0 | 0% |
| A person checks every garment (90 s each, 80% recall) | €957 | **15** | 80% |
| Model alone (threshold 0.67) | €1,029 | 0 | 49% |
| **Model + human review (0.38 / 0.93)** | **€607** | **~1–2** | 75% |

**Model + human review is the cheapest option.**

- It saves **€350 per 1,000** against a person checking everything. That is about **€1,400 a day**.
- It saves **€893 per 1,000** against doing nothing.

**The saving holds up when I test it.** I used a **bootstrap**. *Bootstrap: redo the calculation 2,000
times on random re-draws of the test photos, to see how much the result moves.*

- Model + review costs €508–€725 per 1,000 (95% interval).
- It beats a person checking everything in **100%** of the re-draws.
- The model alone ranges from €772 to €1,341, so it may or may not beat a person.

**Where the €607 goes:**

- €246 for defects the model let pass;
- €314 for reviewed defects (repairs, plus the 10% the reviewer misses);
- €28 of review time;
- €10 for good garments a reviewer wrongly sends to rework (2% of those reviewed);
- €10 for running the model.

The biggest cost left is missed defects. So the next euro saved comes from better recall, not from
cheaper reviews.

**When review would stop paying.** Model + review stays cheaper than a person checking everything
unless:

- one review takes more than **~5 minutes**; or
- a reviewer catches fewer than **~27%** of the defects shown to them; or
- a full manual check takes less than **~11 seconds**.

**One more person.** With a third reviewer, the low threshold drops to 0.23 and the cost to €549 per
1,000. That is worth discussing with the QC manager.

## Limitations & next steps

1. **No defect-free photos.**
   - My false-alarm rate comes from 131 crops of defective photos and an area factor of 6.25.
   - **Next:** photograph 500 good garments on the real line and score them. That gives a real
     false-alarm rate and removes the area-factor guess.
     
2. **A tiny test set** (61 photos, 3 holes). Recall at 0.67 has an interval 24 points wide.
   - **Next:** label 300 more line photos, with at least 30 boxes per class, before trusting
     per-class numbers.
   - Until then, judge holes on valid only.
     
3. **Copies in train.** There are ~422 real photos, not 1,266. The model has seen few fabric types.
   - **Next:** add more real photos rather than more copies. Focus on printed and white fabrics,
     where it fails.
    
4. **The photos are fabric close-ups, not whole garments under a fixed camera.**
   - **Next:** run a two-week shadow trial. The model scores every garment, but people still decide.
   - Then compare its calls with the inspectors' and re-pick the thresholds on that data.
     
5. **Low recall on faint and small defects** (small stains, tears on prints, thin selvet bands).
   - **Next:** try yolo11s, or a bigger image size (960), on a GPU.
   - **Next:** cut photos into tiles, so small holes look bigger to the model.
   - **Next:** add more examples of low-contrast defects.
     
6. **The garment decision can be right for the wrong reason.** A defective photo counts as caught
   even if the box is on the wrong spot.
   - **Next:** in the shadow trial, record whether the reviewer found the defect *at* the box.
     
7. **The best threshold moves between splits** (valid best 0.67, test best 0.57).
   - **Next:** pick thresholds on a larger, combined set (valid plus shadow-trial data).
   - Re-check them every month as the fabric mix changes.
     
8. **The costs are my estimates, not the factory's real accounts.** The sensitivity analysis shows
   my recommendation holds over wide ranges. The exact euro totals do not.
   - **Next:** get the real return rate, refund cost and rework time from finance and QC, and put
     them in `config.yaml`.

## Generative AI use disclosure

I used Claude Code (Anthropic) throughout this assignment.

**What Claude Code did:**

- wrote all the code (`train.py`, `app.py`, `costing.py`, `config.yaml`);
- ran the training and testing in a cloud session;
- checked the candidate datasets (label counts and the leakage test that ruled out two of them);
- picked sqc v2 using the fallback rule I gave it;
- proposed the euro figures, the 5% base rate, the staffing numbers and the review time;
- drafted the text of this report and of DECISIONS.md.

**What I did:**

- I chose the business problem: garment QC with pass, review and reject.
- I set the positive class and the rule that a garment is defective if any box is above the
  threshold.
- I described what each mistake costs: returns, refunds and bad reviews for a miss; re-checks and
  rework for a false alarm.
- I chose the candidate datasets and the fallback rule.
- I reviewed and approved every decision in this report:
  - the cost figures and base rate;
  - picking thresholds on valid;
  - the defect-free-crop method;
  - the staffing limit;
  - Rewrote this report in simpler language.
  - the final recommendation.
