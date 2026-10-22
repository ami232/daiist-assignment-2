# Assignment 2 Report

**Name:** Ruben Segura
**Student ID:** 19061
**Email:** rsegura.ieu2022@student.ie.edu
**Group:** BBADBA 5B

---

## Business problem, and the cost of being wrong

A packhouse sorts fruit from growers and packs it into crates for a supermarket chain. The model supports the decision an operator makes today: for one piece of fruit, does it show a visible defect (rot, mould, heavy bruising) or not? A reject is pushed off the belt and sent to juice or waste; a pass goes into a crate, onto a pallet and into a lorry.

**A false negative** is a defective piece that gets packed. It keeps rotting in transit, spoils the fruit next to it in the crate, and surfaces as a refused crate, a quality claim and penalty points on the retailer's scorecard. I price it at **10 EUR** per escaped piece: an average over the many cases that cost almost nothing and the few that trigger a claim.

**A false positive** is a good piece rejected, costing the wholesale value of the piece, about **0.25 EUR** (1 EUR/kg, 180-200 g). It is usually downgraded to juice rather than destroyed, so the figure is conservative.

A false negative therefore costs about **40 times** a false positive. That ratio drives everything below: reject when unsure, and judge the model on recall over the defect classes rather than on overall mAP.

**Base rate:** I assume 5% of pieces reaching the line are genuinely defective, since the fruit has already been roughly graded in the field. That matters as much as the cost ratio: with 19 good pieces for every bad one, even a small false positive rate wastes a lot of good fruit. Correct decisions cost 0 EUR on both sides, and **human review** costs 0.05 EUR per piece (operator at 15-20 EUR/hour fully loaded, a few seconds per piece).

## Dataset

| Field | Value |
| --- | --- |
| Workspace | intelligent-fruit-quality-grading-system |
| Project | fruit-quality-inspection-system |
| Version | 6 (generated 11 May 2024) |
| Link | https://universe.roboflow.com/intelligent-fruit-quality-grading-system/fruit-quality-inspection-system/dataset/6 |
| Task / License | Object detection / MIT |
| Images | 3,304 at 640x640 (auto-orient, stretch resize, no augmentation) |

The dataset is used as published and was not forked, so anyone can download the same version I trained on. One image is a close-up of one or a few pieces of fruit against a plain background, across six classes: good and bad apple, banana and orange.

The labels encode the quality judgement itself, not just which fruit is in the picture, so the good/bad split maps directly onto the pass/reject decision. Because the task is detection, the background is a free negative class and each piece gets its own box and confidence. A dataset labelling only fruit type would look relevant but would not support the decision.

### What the composition implies

Instances per class, counted from the label files:

| Class | Train | Val | Test |
| --- | --- | --- | --- |
| bad apple | 689 | 136 | 64 |
| bad banana | 726 | 182 | 105 |
| bad orange | 1,075 | 228 | 108 |
| good apple | 712 | 101 | 14 |
| good banana | 1,237 | 337 | 184 |
| good orange | 1,244 | 270 | 105 |
| **Images** | **2,431** | **594** | **279** |

The median box covers 13% of the image area in train and 17% in test, with the 90th percentile near 40%. These are large objects.

Three things follow, and they limit what I may conclude later. **good apple is too thin to trust** at 14 test instances, where one extra miss moves recall by seven points. **The test split is not composed like the training split**: apples are 49% bad in train but 82% bad in test, while bananas and oranges keep their proportions, so the apple numbers are measured on a population resembling neither train nor a real line. **The images are easier than production**: fruit filling 13-17% of a frame on a plain background is studio photography, not a camera above a moving belt with thirty small pieces, motion blur and changing light, and the model published on Universe for this dataset reports 99.3% mAP50, high enough to suggest near-duplicates across splits.

So I cannot conclude that this model would hold its accuracy on a real conveyor, nor that its performance on apples is reliable in either direction.

## Finetuning and baselines

**Model:** yolo11n, object detection. Detection rather than classification because a defect is local, a brown patch on one side of an orange: a classifier gives one label per image, a detector gives one box and one confidence per piece, which is what a per-piece reject decision needs. I chose the n size over s because what this assignment grades is the operating point, not the last point of mAP, and a smaller model is the realistic choice at line speed.

**Training:** Google Colab on a T4, 40 epochs, image size 640, batch 16, seed 0, from the pretrained COCO checkpoint. The weights are committed to weights/best.pt and are what the dashboard loads. The pipeline runs with: uv run python main.py train

Results on the test split (279 images):

| Model | mAP50 | mAP50-95 | Recall on defect classes |
| --- | --- | --- | --- |
| Finetuned yolo11n | 0.994 | 0.922 | 1.00 |
| Pretrained COCO yolo11n, zero-shot | 0.025 | 0.011 | 0.00 |
| Trivial: predict nothing | 0.00 | 0.00 | 0.00 |
| Trivial: flag every piece as defective | n/a | n/a | 1.00 |

The COCO model is not useless here: apple, banana and orange are all COCO classes, so it finds the fruit. What it cannot do is say anything about quality, because "rotten" is not a concept it was trained on, so its recall on the bad classes is zero by construction rather than by accident. The value of finetuning is entirely in learning the quality axis, not in locating fruit. The trivial baselines bracket the problem from the other side: predicting nothing is the "do nothing" option, flagging everything is the "human checks everything" option, and both are priced properly below.

The gap was bigger than I expected, and it is not really a compliment to the model. On the collapsed defect decision the finetuned model separates the test split perfectly, with no false positives and no false negatives anywhere between thresholds of 0.13 and 0.39. A result that clean on 279 images says more about how easy the split is than about how good the model is, which is why every number below is treated as an upper bound.

## Per-class evaluation and failure modes

The positive class is "defective", meaning the three bad classes together. A piece is flagged if the model detects any bad class on it, scored by the highest confidence given to any bad detection in the image.

| Class | Precision | Recall | AP50 | AP50-95 |
| --- | --- | --- | --- | --- |
| bad apple | 0.997 | 1.000 | 0.995 | 0.928 |
| bad banana | 0.980 | 0.917 | 0.987 | 0.881 |
| bad orange | 0.998 | 1.000 | 0.995 | 0.919 |
| good apple | 0.982 | 1.000 | 0.995 | 0.987 |
| good banana | 0.993 | 0.989 | 0.995 | 0.893 |
| good orange | 0.998 | 1.000 | 0.995 | 0.922 |

Confusion matrix: see results/confusion_matrix.csv and the dashboard.

good banana carries the headline number, since its 184 instances make it the largest test class and the six-class average follows it. The class dragging the average down is **bad banana**, the only one under 0.99 recall at 0.917, and with 105 test instances I trust that figure. good apple looks perfect at 1.000 recall, but on 14 instances that is not a measurement, so I draw no conclusion about apples.

The quality errors split 14 to 11 towards over-rejection: 11 good bananas, 2 good oranges and 1 good apple called bad, against 5 bad oranges and 3 bad bananas called good, 2 bad oranges called good apples and 1 bad banana missed entirely. Fruit-type confusions that keep the quality label cost nothing and I ignore them. So the model errs towards throwing good fruit away, which at 40:1 is the direction I would have chosen. Two caveats: these counts come from the validation pass, which scores every box down to 0.001 confidence (the 923 phantom bad banana boxes are the same artefact and disappear above 0.25), and at my operating point the image-level decision makes no errors at all on this split.

The six-class average hides the thing I care about, which is whether a defective piece is caught at all. That is why the operating point is chosen on the collapsed defect-versus-good decision rather than on mAP.

 ## Cost matrix and operating point

 | Outcome | Cost (EUR per piece) | Where the figure comes from |
 | --- | --- | --- |
 | False negative (defective piece packed) | 10.00 | Average of cases where little happens and cases where a crate is refused or a claim is raised, including spoilage of neighbouring fruit |
 | False positive (good piece rejected) | 0.25 | Wholesale value of one piece, about 1 EUR/kg at 180-200 g, often partly recovered as juice |
 | True positive / true negative | 0.00 | The piece had no value to lose; normal operation |
 | Human review of one piece | 0.05 | Operator at 15-20 EUR/hour fully loaded, a few seconds per piece |

 Assumed base rate: 5% of pieces on the line are genuinely defective.

 **How the sweep works.** The test split does not have the prevalence of a real line, so its raw counts would give the wrong answer. The pipeline instead computes rates conditional on the true class (what fraction of truly defective pieces are caught at this threshold, what fraction of truly good pieces are wrongly flagged) and reweights them with the 5% base rate. The curve is saved to results/cost_curve.png and redrawn live in the dashboard, where the base rate and all three costs can be changed.

 **Chosen operating point: 0.25, at an expected cost of 0.00 EUR per 1,000 pieces.**

 The minimum is a plateau, not a point. Every clean test image scores at most 0.12 and every defective one at least 0.39, so any threshold between 0.13 and 0.39 gives the same zero errors. The sweep reports 0.13 only because it is the first grid point inside the gap, and shipping the edge of a plateau sits one unlucky image away from the cost jumping, so I deploy the middle. The width of that gap is also the clearest evidence that this split is easier than a real line.

 **The two numbers that frame everything.** At a 5% base rate, doing nothing costs 500 EUR per 1,000 pieces and having a person check every piece costs 50 EUR. Manual inspection is ten times cheaper than shipping the defects, so the benchmark to beat is 50, not 500, and the question is whether the model gets below it while sending far fewer pieces to a person.

 **The assumption that would most change the answer** is the 10 EUR false negative, the figure I am least sure of because it averages over outcomes I have not measured. At 2 EUR the ratio drops from 40:1 to 8:1, the optimal threshold moves up, and the system should accept more escapes; below roughly 1 EUR the review band stops paying for itself, since 0.05 EUR of review is no longer cheap relative to the error it prevents. The base rate pushes the same way: at 1%, doing nothing costs only 100 EUR per 1,000 and the case for an aggressive threshold weakens a lot.

 ## What a person still has to check

 Three outcomes: a piece is packed automatically below the lower threshold, rejected automatically above the upper one, and sent to a person in between.

 | Decision | Value |
 | --- | --- |
 | Pass | below 0.25 |
 | Send to a person | between 0.25 and 0.50 |
 | Reject | above 0.50 |
 | Share sent to a person | 0.03% (15 checks a day at 50,000 pieces) |

 **Does the review load fit the people available?** Assuming a line of 50,000 pieces a day, 10 seconds per manual check and 7.5 productive hours per shift, the 0.25 / 0.50 band sends 0.03% of pieces to a person: 15 checks a day, about 2.5 minutes of work, 0.005 FTE. That is a rounding error rather than a staffing decision, because almost nothing in this split lands near the boundary. The honest way to size it is to ask what happens on a messier line: at the widest band I would still consider sensible, 0.10 to 0.90, the model sends 1.6% of pieces to a person, which is 802 checks a day, 134 minutes and 0.30 FTE. Even that worst case is a third of one person, well inside the three FTE a packhouse this size treats as a normal quality team, so the design is realistic across the whole range.

 | Option | Cost per 1,000 pieces | Comment |
 | --- | --- | --- |
 | Do nothing, ship everything | 500 EUR | Every defective piece escapes |
 | A person checks every piece | 50 EUR | Needs about 18 people at 50,000 pieces/day |
 | Model decides alone | 0.00 EUR (at 0.25) | No review cost, but every model error is paid at full price |
 | Model plus review band (my design) | 0.02 EUR | The band is pure overhead on this split |

 **Is the band worth it?** Not on the numbers I can measure: the model alone costs 0.00 EUR per 1,000 on this test split and the band adds 0.02 EUR. I keep it because a measured cost of exactly zero is not a cost I believe, since it means the split contains no case near the boundary, not that the model never errs. Treated as insurance, the band pays for itself as soon as the true escape rate exceeds roughly one piece in 700,000. Full manual inspection, meanwhile, looks cheap per thousand only because each check is small: at 50,000 pieces a day it is about 2,500 EUR and 18 people working flat out, which is exactly what the band is there to avoid.

 ## Limitations & next steps

 **The test split is not a fair sample of production:** studio close-ups on plain backgrounds, large objects, and apples that are 82% defective in test against 49% in train. With more time I would ignore the published split, pool all 3,304 images, group them by source so near-duplicates cannot land on both sides, and re-split. Then I would collect a few hundred frames from a real belt camera and use only those as the test set, because that is the distribution the decision is made in.

 **good apple has 14 test instances,** so no conclusion about apples is safe. The cheap fix is moving images from train to test for that class; the better fix is labelling a few hundred more apple images, about a day of work.

 **Human review is assumed to be perfect,** counted as always correct at 0.05 EUR, while real operators disagree and get tired. I would have two operators independently grade the same 500 pieces and feed the agreement rate into the cost model, which would raise the cost of the band and push the thresholds apart.

 **The decision is scored per image, not per piece.** Taking the highest defect confidence in an image works here because most images contain one or two pieces, but on a belt with thirty in frame one bad piece would condemn the whole frame. In production I would score each detected box separately, which the detector already supports.

 **All costs are reasoned, not measured.** The 10 EUR false negative is the weak point. With access to the business, the first thing I would do is pull twelve months of quality claims and rejected deliveries and replace it with the real average.

 ## Generative AI use disclosure

 Per the syllabus AI Policy, I used generative AI on this assignment and disclose it here.

 **What AI was used for:** searching Roboflow Universe and comparing candidate datasets; writing the boilerplate in train.py and app.py (the Roboflow download, the Ultralytics training call, the threshold sweep, the matplotlib plots, the Gradio layout); computing the per-class instance counts and object-size statistics from the label files; and drafting and formatting the text of this report.

 **What is mine:** the choice of business problem, the choice of this dataset over the alternatives, every figure in the cost matrix and the base rate, the decision to collapse six classes into a binary defect decision, and the placement of the two thresholds.

 **Verification:** I have read and checked everything submitted, and changed what I did not agree with.
 
