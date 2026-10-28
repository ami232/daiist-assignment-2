# Decisions: study guide for the written check

Plain-language summary of every decision in Assignment 2. For each one: what I chose, why, what I
rejected, and what would change my mind. The numbers are the ones in REPORT.md.

---

## The one-paragraph story

A factory photographs every garment before shipping. A YOLO model looks for five kinds of fabric
defect. If its most confident box is below **0.38**, the garment ships. Between **0.38 and 0.93**, a
person checks the boxed spot. At **0.93 or above**, it goes to rework.

- A missed defect costs **€30**; a false alarm costs **€3**.
- With a **5%** defect rate, this costs **€607 per 1,000 garments**:
  - versus **€957** for a person checking everything (15 people);
  - versus **€1,500** for doing nothing;
  - versus **€1,029** for the model alone.
- It needs about **1 to 2 reviewers** for 4,000 garments a day.

---

## Key terms in one line each

- **Positive class**: the thing I am trying to find. Here, "garment has a defect".
- **Box**: a rectangle the model draws around a defect, with a label and a confidence.
- **Confidence**: a number from 0 to 1 that says how sure the model is about a box.
- **Threshold**: the cut-off confidence I choose. At or above it, the model says "defect".
- **False negative**: a real defect the model misses, so a bad garment ships.
- **False positive**: a good garment the model flags, a false alarm.
- **True positive / true negative**: a defect caught / a good garment correctly passed.
- **Base rate**: how often a defect really happens (I assume 5%).
- **Precision**: of the model's alarms, the share that are real defects.
- **Recall**: of the real defects, the share the model catches.
- **IoU**: how much a predicted box overlaps the real one, from 0 to 1.
- **AP50**: one score from 0 to 1 for how well the model finds one class (a box counts if IoU ≥ 0.5).
- **mAP50**: AP50 averaged over all classes. My headline number.
- **mAP50-95**: the same, but with stricter overlap rules, so it rewards tight boxes.
- **Confusion matrix**: a table of what was really there (rows) against what the model said
  (columns).
- **Train / valid / test split**: photos to learn from / to tune choices / to measure the final
  result.
- **Leakage**: test photos that also appear in train, so the test is not honest.
- **Augmentation**: slightly changed copies of photos (rotation, noise) to have more training
  examples.
- **Epoch**: one full pass of training over the train set.
- **Finetune**: start from a model that already learned general images and keep training it on mine.
- **COCO zero-shot**: the original model trained on 80 everyday objects, used without any training
  on my photos.
- **Baseline**: a simple comparison model, to check that the real model adds value.
- **Bootstrap**: redo the calculation many times on random re-draws of the test data, to see how
  much the result moves.
- **95% interval**: the range where the true value most likely lies.
- **Sensitivity analysis**: change one assumption at a time and see if the answer changes.
- **FTE**: full-time equivalent, one person working full time.

## 1. The business problem

**What I chose.** End-of-line quality control in a clothing factory: pass / review / reject for each
garment photo.

- **Positive class:** "garment has a defect".
- **The rule:** a garment counts as defective if *any* box scores at or above the threshold. Its
  score is its highest box confidence.

**Why.**

- The two errors cost very different amounts. A missed defect goes to the customer: return,
  refund, written-off garment, bad review. A false alarm stays in the factory: someone looks again.
- I decide per garment, not per box, because the business question is "ship it or not?". It does
  not matter which defect it is, or how many.

**Alternatives I rejected.**

- *Per-box decisions:* the factory ships garments, not boxes.
- *Classifying the defect type as the main output:* the type matters only to tell the reviewer what
  to look for.

**What would change it.** If one defect type were much worse than the others (say holes are always
returned, stains rarely), I would give each class its own cost and threshold.

## 2. The costs

| Cost | Value | Why |
|---|---|---|
| False negative (defect shipped) | €30 | return shipping and handling ~€8 + refund and service ~€4 + garment written off ~€12 + reputation and buyer chargeback ~€6 |
| False positive (good garment rejected) | €3 | 3-minute re-inspection at €16/h ~€0.80 + handling ~€0.70 + some needless rework (~20% × €7) |
| True positive (defect caught) | €5 | average repair or downgrade to "seconds" |
| True negative | €0 | nothing happens |
| Running the model | €0.01 per garment | ~€25k camera and computer over 3 years |
| Reviewer | €16/h, 30 s per check, catches 90% | loaded wage of a Spanish QC inspector; the box tells them where to look |
| Full manual inspection | 90 s per garment, catches 80% | inspecting a whole garment unaided; tiredness lowers recall |
| People also make false alarms | 2% of good garments | humans are not perfect either |

**The key ratio.**

- Catching a defect saves €30 − €5 = **€25**. A false alarm costs **€3**. So a miss is about 8× worse.
- In theory, flag a garment if P(defect) > 3 / (3 + 25) ≈ **11%**.

**What would change it.**

- Missed-defect cost below **~€6** → doing nothing is cheapest.
- False-alarm cost below **~€2.55** → the model alone beats a person checking everything.
- Repair cost of a caught defect at **~€29 or more** → catching stops paying.

## 3. The base rate

**What I chose.** **5%** of garments are defective. Factories typically see 2–8% at final inspection.

**Why it matters.** 95 of every 100 garments are good. So even a small false-alarm rate creates many
false rejects. This is why my model-alone threshold is high (0.67).

**What would change it.**

- Below **~4.2%**, the model alone beats full manual inspection.
- The threshold falls as defects get more common: 0.93 at ≤1%, 0.67 at 2–5%, 0.53 at 7–8%, 0.30 at
  ≥15%.
- Model + review is cheapest across the whole 0.5–30% range.

## 4. The dataset

**What I chose.** `sqc/fabric-defect-detection-ehhyx` **version 2**, the latest version, unforked.

- 5 classes: gray stitch, hole, selvet, stain, tear.
- 1,266 train, 123 valid and 61 test images, all 640×640.

**What I rejected, and why.**

- `hammy/fabric-defect-lcird-1ifel` v1: **broken**. One box in 2,491 images, and no valid or test
  split.
- `yolov7-vdg8u/fabric-defect-detection-jdyz3` v3: **leaked**. All 157 test and 500 valid images are
  pixel-identical copies of train images. A test score there measures memory.
- I tested for leakage by shrinking every image to 64×64 grey and measuring the mean difference to
  the closest train image. A difference of 0.2 out of 255 means the same photo.

**sqc's own weaknesses:**

- **No defect-free images.**
- **Only 3 holes in test.**
- **The test split is small** (61 photos).
- **Train is 3× augmented** (~422 real photos).

## 5. What the dataset composition implies

| What | Consequence |
|---|---|
| Hole: 3 test boxes | The hole test number is an anecdote. Judge holes on valid (44 boxes). |
| 61 test photos | Recall 49% has a 95% interval of **37–61%**. Per-class numbers move 4–6 points per image. |
| 0 defect-free images | I cannot measure false alarms directly. "Always reject" scores 100% accuracy on test, which means accuracy is useless here. |
| Train = 3 copies of ~422 photos | Less variety than it looks. One epoch is really 3 passes. |
| Hand-held photos of loose fabric | Not exactly a factory camera, so production results may differ. |
| Gray stitch and selvet are big bands; holes are tiny (median 1.5% of image) | Holes and small stains are the hardest to find. |

**My fix for the missing good garments.**

- I cut **defect-free crops** (256×256 px, touching no labelled box) out of the valid and test
  photos: 131 on test, 301 on valid.
- A crop is 1/6.25 of a photo. So P(good photo flagged) = 1 − (1 − P(crop flagged))^6.25.
- **Why not upscale the crops?** That would change the pixel scale the model was trained on. I
  predict crops at 256 px so the fabric looks the same size.
- **The weak point.** This assumes each part of a photo raises false alarms independently. If the
  factor were below ~5, the model alone would beat full manual inspection.

## 6. The model and training

**What I chose.** **yolo11n** detection, finetuned from COCO.

| Setting | Value |
|---|---|
| Epochs | 30 (patience 10) |
| Image size | 640 |
| Batch | 16 |
| Optimizer | AdamW (auto) |
| Seed | 0 |
| Hardware | 4-core CPU in a cloud session |
| Time | 171 min |
| Best epoch | 25 |

**Why these choices.**

- **Detection:** a box tells the reviewer where to look. That keeps a review at 30 seconds instead
  of a full inspection.
- **Nano model:** it fits on a CPU in under 3 hours.
- **Image size 640:** holes are tiny, so I did not shrink the images.
- **30 epochs:** train is already 3× augmented, so 30 epochs ≈ 90 passes per real photo.

**Rejected.**

- **yolo11s:** about 3× slower (~9 h). With 61 test photos, a small gain would be inside the noise.
- **Classification:** no location, so slower reviews.
- **Colab:** not needed, because the dataset was small enough for the CPU.

## 7. Baselines

| Model (test split) | mAP50 | mAP50-95 | € per 1,000 alone |
|---|---|---|---|
| Finetuned yolo11n | **0.633** | **0.339** | €1,029 |
| COCO yolo11n zero-shot | 0 | 0 | €1,623 |
| Predict nothing (always pass) | 0 | 0 | €1,500 |
| Always reject (majority class on test) | – | – | €3,100, but **100% test accuracy** |

- **Why COCO scores near zero.** COCO has 80 everyday classes (person, car, banana…) and none for
  fabric defects.
  - It cannot name a defect, so class-aware mAP is 0.
  - Even counting *any* box on a defect, it gets only AP50 0.034.
  - It drew 412 boxes like "bed" and "airplane" on folds and prints. It is worse than doing nothing.
- **Finding 1:** "always reject" beats my model on test accuracy, because the test set has no good
  garments. That is why I judge by cost at a realistic base rate.
- **Finding 2:** my model alone (€1,029) loses to a person checking everything (€957). It only wins
  with a review lane.

## 8. Per-class results and failures

- **The headline is carried by stain:** AP50 0.72 on test, 0.76 on valid, and the most training
  boxes.
- **Weakest:**
  - **selvet box placement:** AP50-95 0.196 on test (0.205 on valid); long bands are hard to box exactly;
  - **tear on valid:** AP50 0.42, precision 0.41; tears on prints look like the print.
- **It never confuses one defect type with another.** All errors are misses (72 of 103 test boxes
  at 0.67) plus 4 false boxes. When it draws a box, it is usually right. It is just too quiet.
- **Size matters.** It finds 12 of 18 large stains but 0 of 2 small ones on test, and 0 of 10 small
  stains on valid.
- **The hardest photos:** a tear on a floral print, a faint tear or stain on white fabric, a selvet
  on a patterned sheet, a decorative-looking gray stitch, and a 1 mm spot. **Low contrast and busy
  prints.**

## 9. Choosing the threshold

**What I chose.** I swept thresholds from 0 to 1 in steps of 0.01 and computed the expected cost at
each. I took the **minimum on the valid split**, then reported on test.

- **Model alone:** 0.67. On test: recall 49%, 4.7% of good garments rejected, €1,029 per 1,000.

**Why valid, not test.** If I picked the threshold on test, the test number would be optimistic. The
test curve's own minimum is 0.57 (€803), which shows how much the best point moves between small
splits.

**Why so high.** One point of false alarms costs 950 × €3 / 100 = €28.5 per 1,000. One point of
recall gains only 50 × €25 / 100 = €12.5. Good garments are 19× more common.

## 10. The two thresholds and human review

**What I chose.** Pass below **0.38**, review from **0.38 to 0.93**, reject at **0.93** or above.

- I picked this pair by minimum cost on valid, **with at most 2 reviewers**.
- Without the limit the low edge would be 0.11, but that needs 3.7 reviewers.

**Load on test:**

- **20.9%** go to review = **837 garments a day**.
- At **30 s** each, that is **418 min a day**, or **1.05 reviewers** (400 productive min per person).
- The same thresholds on valid need 1.92 reviewers, so I staff **2 people**.

**Outcome.**

- 75% of defects caught; 0.4% of good garments wrongly sent to rework.
- Nothing reaches 0.93 on test, so every reject is confirmed by a person.

**What would change it.**

- With a 3rd reviewer: low edge 0.23, €549 per 1,000.
- With only 1 reviewer: low edge 0.52, €684.
- At 10,000 garments a day: low edge 0.59, €822.
- If one review took more than ~5 minutes, or reviewers caught fewer than ~27% of shown defects,
  full manual inspection would win.

## 11. The four options

| Option | € per 1,000 | People | Defects caught |
|---|---|---|---|
| Do nothing | 1,500 | 0 | 0% |
| Person checks everything | 957 | 15 | 80% |
| Model alone (0.67) | 1,029 | 0 | 49% |
| **Model + review (0.38 / 0.93)** | **607** | **1–2** | **75%** |

- **Bootstrap** (2,000 resamples): model + review costs €508–725 per 1,000 and beats manual
  inspection in 100% of the resamples.
- **The saving:** about **€350 per 1,000**, or about €1,400 a day, against manual inspection.
- **The biggest remaining cost is missed defects** (€246 per 1,000 from defects the model passes),
  so recall is what to improve next.

## 12. The config file and the dashboard

- **`config.yaml`** holds every assumption: costs, base rate, garments per day, review minutes,
  wage, reviewer recall, reviewers available, thresholds and training settings.
  - `train.py --skip-training` re-runs the costing in minutes.
  - The dashboard reads the file at startup.
- **The dashboard** (`main.py app`):
  - loads `weights/best.pt` and the saved scores, and never trains;
  - lets you upload a photo and see the decision;
  - has two threshold sliders, a base-rate slider and a split choice;
  - shows a live confusion matrix, cost per 1,000, review share and reviewers;
  - shows the cost curve with 0.67 marked;
  - shows per-class tables with their own confidence slider and box confusion matrix;
  - has baseline, dataset and sensitivity tabs.

---

## 15 likely exam questions, with short answers

1. **Why is a false negative worse than a false positive here?**
   A missed defect reaches the customer: return, refund, written-off garment, reputation, about €30.
   A false reject is re-checked inside the factory for about €3. Net of the €5 repair, a catch saves
   €25, so a miss is about 8× worse.

2. **What is your positive class, and how do you go from boxes to a garment decision?**
   "Garment has a defect." The garment's score is its highest box confidence. If that is at or
   above the threshold, the garment is flagged.

3. **Why did you not use the first dataset you picked?**
   It had one labelled box in 2,491 images, and no valid or test split. You cannot train or test on
   that.

4. **Why not the yolov7-vdg8u dataset, which looked bigger?**
   Every valid and test image was an identical copy of a train image. Its test score would measure
   memorisation, not performance on new garments.

5. **The test set has no defect-free images. How did you measure false positives?**
   I cut 256-pixel windows with no labelled box out of the photos. Then I scaled the crop false-alarm
   rate to a full photo with 1 − (1 − q)^6.25. That is my weakest assumption.

6. **Why does the COCO model score near zero?**
   COCO's 80 classes contain no fabric defects. It cannot name one, so mAP is 0. Even counting any
   box on a defect, it gets AP50 0.034. It boxes folds as "bed" or "airplane".

7. **Did your model lose to a trivial baseline?**
   Yes, on test accuracy. "Always reject" scores 100% because every test photo is defective. At a
   5% base rate it is the most expensive option (€3,100 per 1,000). So accuracy is the wrong measure
   here.

8. **Which class carries your headline mAP50 of 0.633, and where does the model fail?**
   Stain (AP50 0.72). It places selvet boxes badly (AP50-95 0.196). It misses tears on printed
   fabric, and small or faint stains and holes. It never mixes up defect types; it misses them.

9. **Why did you choose the threshold on valid, not test?**
   Choosing on test would make the test result optimistic. The test optimum (0.57) differs from the
   valid one (0.67), which shows that risk.

10. **Why is your model-alone threshold so high (0.67)?**
    95% of garments are good. Each point of false alarms costs €28.5 per 1,000, while each point of
    recall gains only €12.5.

11. **How did you choose the two review thresholds?**
    By minimum expected cost on valid over every pair, with at most 2 reviewers. That gave 0.38 and
    0.93. Without the limit, 0.11 would be cheaper but needs 3.7 people.

12. **How many garments go to review, and how many people does that need?**
    20.9% = 837 a day. At 30 s each, that is 418 minutes = 1.05 reviewers on test (1.92 on valid). I
    staff 2.

13. **Compare the four options per 1,000 garments.**
    Do nothing €1,500. Person checks all €957 (15 people). Model alone €1,029. Model + review €607
    (1–2 people). Model + review wins in 100% of bootstrap resamples.

14. **Which assumption matters most, and how would the answer change?**
    The false-alarm rate on good garments, because it is estimated from crops. With an area factor
    below ~5, the model alone beats manual inspection. Model + review stays best across all values.
    For whether to inspect at all, the missed-defect cost matters: below ~€6, doing nothing is
    cheapest.

15. **What would you do next with more time or data?**
    - Photograph 500 good garments on the line to measure false alarms directly.
    - Add more source photos of white and printed fabric.
    - Run a two-week shadow trial to re-pick the thresholds.
    - Try yolo11s or a larger image size on a GPU for small defects.
    - Replace my cost estimates with real figures from finance.
