# Assignment 2 Report

- **Name**: Vittorio Fialdini
- **Student ID**: 18843
- **Email**: vpereirafial.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

The client is a small steel fabrication shop in the euro area that sells welded parts for about €600 each, over three shifts. Today three inspectors, one per shift, check every weld by eye: 5 minutes each, 270 welds a day.

The model looks at a photo and gives each weld a score. Each weld then gets one of three decisions: pass (it ships), send to review (a person checks it), or fail (it goes to repair). The positive class is `Bad-weld`.

- Good weld sent to repair (false positive): €46.94. 55 minutes of work (grind out 15, re-weld 30, re-inspect 10) at €40.3 an hour, plus €10 of material.
- Bad weld that gets through (false negative): €900. The piece comes back, a new one is sent with fast transport, and the customer trusts the shop less. Priced at 1.5 times the €600 average sale.

Missing a bad weld costs about 19 times more than a needless repair. So the model should lean towards flagging. But only about 2% of welds are bad, so false alarms on the other 98% add up fast. A 5-minute review (€3.36) is much cheaper than either mistake, which is why the middle "review" option matters.

## Dataset

- Workspace `defect-detection-fxtd5`, project `weld-defect-lze1x`, version 1, used as published (not forked): https://universe.roboflow.com/defect-detection-fxtd5/weld-defect-lze1x/dataset/1
- 838 photos, all resized to 640 x 640. One image shows one or more weld seams, often several steel strips on a table.

Three classes:

- Good-weld: a box around a whole weld seam that is fine.
- Bad-weld: a box around a whole weld seam that needs repair.
- Defect: a smaller box around one flaw (a crack, a hole, spatter). Defects sit inside bad welds: 622 of the 734 bad welds contain a defect box, but only 1 of the 1,109 good welds does.

Why it fits the problem: the shop's decision is made per weld seam, and this dataset labels each seam as good or bad, which is exactly that decision. It also marks where the flaws are, so the model learns what makes a weld bad, not just what a weld looks like. Many photos are frames from videos of welded strips on a bench, which is close to what a camera at an inspection station would see.

## What the dataset's composition implies

| Split | Images | Bad-weld | Good-weld | Defect |
|---|---|---|---|---|
| train | 586 | 536 | 764 | 962 |
| valid | 168 | 134 | 235 | 225 |
| test | 84 | 64 | 110 | 107 |

Thin numbers. The test split has only 64 bad welds. One weld more or less moves recall by about 1.6 points, so per-class results on test are rough, not precise.

Object size: The median bad weld covers 4.2% of the image, a good weld 2.2%, a defect only 1.2%. About 4% of defects are smaller than 32 x 32 pixels. Defects are the hardest class to find.

Does test look like real life? Not fully:

- 40% of the seams in the dataset are bad, against about 2% in a real shop. I reweight all costs to the 2% base rate.
- About 60% of the images are frames from six videos. 8 test images are near-copies of a train image, so those are easier than new photos.
- The other images come from the web (rusty car panels, stock photos, screenshots). They look nothing like the shop.

So I cannot conclude how the model will do on the client's own welds, lighting and camera. The exact euro figures are estimates, not promises.

## Finetuning and baselines

I used object detection, not classification, because one photo often shows several welds and each weld needs its own decision.

Both `yolo11n` and `yolo11s` were finetuned from COCO weights on a Kaggle GPU with the same settings: a time limit of 0.18 hours (about 11 minutes) per model, early stopping after 20 epochs with no gain, image size 640, batch 32, seed 42. The model with the higher mAP50-95 on the valid split was kept:

| Model | Epochs run | val AP50 | val mAP50-95 |
|---|---|---|---|
| yolo11n (kept) | 125 | 0.757 | 0.484 |
| yolo11s | 73 | 0.719 | 0.457 |

The small model won because it fit more epochs into the same time.

Results on the same 84 test images:

| Model | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| Finetuned yolo11n | 0.73 | 0.67 | 0.70 | 0.44 |
| COCO yolo11n, zero-shot | n/a | 0 | 0 | 0 |
| Predicts nothing | n/a | 0 | 0 | 0 |

The COCO model has no weld class, so it scores zero. It saw benches, books and cats instead. The gap is as large as I expected. A fairer test is cost: the COCO model costs about as much as doing nothing (€18,000 per 1,000 welds), while the finetuned model alone gets to €10,719.

## Per-class evaluation and failure modes

The positive class is **Bad-weld**.

| Class | Precision | Recall | AP50 |
|---|---|---|---|
| Bad-weld | 0.70 | 0.75 | 0.74 |
| Good-weld | 0.79 | 0.74 | 0.79 |
| Defect | 0.69 | 0.53 | 0.55 |

Good-weld carries the headline number. Defect drags it down.

Confusion matrix on the test boxes (confidence 0.25 or more):

- Of 64 bad welds: 38 found, 10 called good, 6 called defect, 10 missed.
- Of 110 good welds: 79 found, 14 called bad, 15 missed.
- Of 107 defects: 56 found, **48 missed**. Small flaws are the main weakness.
- The model also draws boxes where nothing is labelled: 48 defects, 31 good welds, 23 bad welds.

Where it breaks (`failures.png`):

- Bad welds passed with a score near 0: strips that look smooth in a clean lab photo, and web photos such as a rusty car panel that look nothing like the training images. A third of bad welds in non-video images are passed, against 12% in video frames.
- Good welds scored highest (0.80 to 0.85): busy photos with many small, rough-looking pieces.

So the model works best on images that look like its training videos.

## Cost matrix and operating point

Cost of one weld, in euros:

| Truth | Pass | Review | Fail |
|---|---|---|---|
| Bad | 900 | 178.26 | 46.94 |
| Good | 0 | 3.36 | 46.94 |

Sources: labour €40.3 an hour (Eurostat 2025, euro-area industry). A visual check takes 5 minutes, middle of the 2 to 8 minute range (Therness). Inspectors miss 15% of bad welds (Harris & Chaney, 1969), so a reviewed bad weld costs the check plus 85% of a repair plus 15% of an escape. Base rate 2% of welds bad (TWI repair-rate survey, mild steel).

Weld score: the highest confidence of any `Bad-weld` or `Defect` box at least half inside the weld.

Sweep: I tried every threshold from 0 to 1 (steps of 0.01) on the valid split and took the cheapest: **0.62**. On test it costs €10,719 per 1,000 welds, catches 67% of bad welds and fails 9% of good ones. The best on test is 0.63, so it holds.

The assumption that matters most is the cost of an escaped weld. Model with review beats checking every weld only while an escape costs under about €1,125 (a sale under €750). It also loses at a 3% base rate with a €900 escape. The saving is small, so a small error flips the answer.

## What a person still has to check

Thresholds: pass below 0.20, review from 0.20 to 0.82, fail at 0.82 or above (chosen on valid).

This sends 22% of welds to a person: about 58 welds a day, at 5 minutes each 4.9 hours, or 0.65 of one shift. One reviewer on the day shift can do this, instead of today's three inspectors. Night-shift welds wait until morning.

Is it realistic? Yes, but with little slack. If a review takes 8 minutes instead of 5, it needs 7.8 hours, more than one shift.

On test, per 1,000 welds:

| Option | Cost | Bad caught | Good failed | Reviewers |
|---|---|---|---|---|
| Do nothing | €18,000 | 0% | 0% | 0 |
| Person checks every weld | €6,856 | 85% | 0% | 3 |
| Model alone (0.62) | €10,719 | 67% | 9% | 0 |
| **Model + review (0.20 to 0.82)** | **€6,373** | 75% | 1% | 1 |

The model with review is the cheapest, but only by about 7% over checking everything. If the reviewer is paid for a full shift, the cost is €6,768, almost the same as checking everything. With shift pay a wider band (0.05 to 0.86) costs €5,377, because the reviewer's spare time is free.

## Limitations & next steps

The test set is small (64 bad welds), so every number is rough. Next: add bootstrap confidence intervals, and use cross-validation over train and valid to pick the thresholds.
Video frames leak between splits. Near-copies of train images make test look easier. Next: split by video so frames of one video stay in one split.
Defects are often missed (48 of 107), mostly because they are small. Next: train at a larger image size (for example 1024) and for longer than 11 minutes. `yolo11s` stopped at 73 epochs and may be undertrained.
The cost figures are estimates, and the margin over checking everything is only 7%. Next: get the shop's real warranty cost and time a few reviews. Then run the model in shadow next to the inspectors for a month before trusting it.

## Generative AI use disclosure

I used AI (Claude) to write code for this assignment: downloading the dataset, the training calls, the cost and threshold functions, the plots and the Gradio dashboard. I also used it to help research plausible numbers for the cost model, such as the labour cost per hour, the share of bad welds, the time for one visual check and how often inspectors miss a bad weld. 
