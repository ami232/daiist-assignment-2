# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Pablo Torres**
- **19910**
- **ptorres.ieu2022@student.ie.edu**
- **Group**: [BBADBA 5B]

## Business problem, and the cost of being wrong

*I look at hard hats on a construction site. A fixed camera takes one photo every 5 minutes and the model says if a worker is in the photo without a helmet. If it says yes, the site supervisor goes to check. One photo is one case. I assume 20 cameras and 10 hours a day, so 2,400 cases a day.
The two mistakes do not cost the same:

- **False alarm** (the model says no helmet, but everybody has one): the supervisor walks over for nothing, about 5 minutes. 5 min x 30 EUR/h = **2.50 EUR** (the 30 EUR/h is my assumption).
- **Miss** (a worker has no helmet and the model says nothing): real risk of a head injury and of a labour inspection fine. I set it at **50 EUR**, 20 times a false alarm.

The 50 EUR is my estimation. What I found: an average work accident with sick leave in Spain lasts 46.4 days (2025 data) and a lost working day costs about 83 EUR (an older CCOO estimate), so about 3,900 EUR only in lost days. Serious occupational safety infractions can be fined from 2,451 to 49,180 EUR (LISOS). But one missed photo is only one of 12 per hour, so most of the time nothing happens. 50 EUR is like saying that 1 in 100 missed photos ends in an incident of about 5,000 EUR.*

## Dataset

- **Roboflow Universe**: workspace `joseph-nelson`, project `hard-hat-workers`, **version 10** (`raw_AllClasses`). https://universe.roboflow.com/joseph-nelson/hard-hat-workers
- Author: Northeastern University, China. License: public domain. I did not fork it, I downloaded it as it is with the API.
- One image is one photo of workers at a work site. Three classes: `helmet` (a head with a hard hat), `head` (a head without hard hat) and `person`.
- I used the raw version, with no augmentation, so the test photos cannot be copies of train photos made by augmentation.
- It serves my problem because `head` is exactly "worker without helmet", and the photos with only helmets are the negatives
- Splits as given: train 4,916, valid 1,413, test 706 images (70/20/10).*
- 
## What the dataset's composition implies

| Class | Train: instances (images) | Valid | Test | Median box area |
|---|---|---|---|---|
| head | 4,612 (911) | 1,339 (256) | 726 (140) | 0.6% of the image |
| helmet | 13,919 (4,503) | 3,913 (1,297) | 1,915 (636) | 0.8% |
| person | 450 (154) | 101 (32) | 64 (23) | 8% |

- **`head` has enough data to trust.** 140 test photos have a head. A detection rate of 91% has a margin of about +-5 points. There are 566 test photos without a head, so a false alarm rate of 1.8% (10 photos) is between 1% and 3%.
- **`person` is too thin.** 64 test boxes in 23 photos and only 450 in train. I do not use it for any decision.
- **The test split is not production.** 20% of the test photos have a head and I assume 2% in real life. That is why I reweight.
- **`helmet` says very little.** It is in 90% of the test photos, so a model that always says "helmet" already gets 0.90 at image level. Only `head` is informative.
- **Heads are small.** A median head covers 0.6% of the photo, about 8% of its side. I expect misses with workers far from the camera.

## Finetuning and baselines
***Task: object detection** with `yolo11s`, starting from the COCO pretrained checkpoint. I need to find the head, not to classify the whole photo, because one photo can have helmets and heads at the same time. For the decision, the score of a photo is the highest confidence of a `head` box in it.

**Where and how I trained:** Google Colab, Tesla T4 GPU, 32 minutes. 640 px, batch 16, 25 epochs (it used all 25, early stopping with patience 8 never triggered).

Results on the same test split:

| | Finetuned yolo11s | COCO yolo11s, zero-shot | Predict nothing |
|---|---|---|---|
| Image-level AP for `head` (is there a head in the photo?) | **0.959** | 0.198 | 0.198 |
| Cost per 1,000 cases, 2% base rate (EUR) | **175** | 1,000 | 1,000 |
| mAP50 on `head` boxes | **0.969** | not possible, COCO has no head class | - |

COCO gets exactly the chance level (0.198 is just the share of photos with a head, 140 of 706) because it does not know the class `head`, so it never alerts and costs the same as doing nothing. Only `person` has the same name in both, and there both models are at chance. So COCO is not really a rival, it only shows the task needs finetuning. The fair comparison is predict nothing: 175 EUR against 1,000 per 1,000 cases. I expected a big gap, and it is. I think it is partly because head against helmet is an easy visual difference and the train and test photos come from the same pool.

## Per-class evaluation and failure modes

*Positive class: `head`. Test split, precision and recall at the best-F1 confidence:

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| head | 0.946 | 0.919 | **0.969** | 0.680 |
| helmet | 0.970 | 0.936 | 0.985 | 0.687 |
| person | 1.0 | 0.0 | 0.030 | 0.018 |
| all | - | - | 0.661 | 0.462 |

The overall mAP50 of 0.661 is an average of 0.97, 0.98 and 0.03, so it hides what matters. **`head` carries the headline number** (AP50 0.969). **`person` drags the average**: the model never predicts it (precision 1.0 with recall 0.0 means no prediction, it is not a good result).

Confusion matrix at confidence 0.25 (test, 726 heads, 1,915 helmets, 64 persons):

- Of 726 heads: 691 found, **15 called helmet**, 20 not detected.
- Of 1,915 helmets: 1,856 found, 5 called head, 54 not detected.
- Of 64 persons: 63 not detected, 1 called helmet.
- The model also invents boxes on background: 66 as head and 131 as helmet.

Head and helmet are almost not confused with each other. The dangerous mistakes for my problem are the 15 heads called helmet and the 20 missed heads (35 of 726, 4.8%)*

## Cost matrix and operating point

| | Model alerts | Model does not alert |
|---|---|---|
| Worker without helmet | TP: 2.50 EUR (supervisor goes and fixes it) | FN: **50 EUR** |
| Nobody without helmet | FP: **2.50 EUR** | TN: 0 |

Sources are in the first section: 2.50 EUR is 5 minutes of a supervisor at 30 EUR/h (my assumption) and 50 EUR is my guess based on accident and fine data. Looking at a photo costs 0.08 EUR (15 seconds at 20 EUR/h, also my assumption). Base rate: **2%** (assumed, test has 20%).

I swept the confidence threshold from 0.01 to 0.99 and took the one with the lowest expected cost. I chose it on the validation split and measured it on test, so it is not tuned on the same photos. The cost curve is in the dashboard and the numbers in `outputs/cost_sweep_*.csv`.

**Operating point (model alone): threshold 0.42.** Cost per 1,000 cases: 175 EUR on valid and 175 EUR on test, against 1,000 for doing nothing and 2,500 for alerting on every photo. On test it finds 128 of 140 positive photos (91.4%) and gives 10 false alarms in 566 photos (1.8%). In production (2,400 photos/day, 48 real cases): about 44 caught, 4 missed and 42 false alarms, so **85 alerts a day**, about 7 hours of the supervisor walking. Cost 419 EUR/day against 2,400 doing nothing.

When I used 250 EUR for a miss in an earlier run, the best threshold was 0.19 instead of 0.42, so the threshold moves a lot with this number.

**The assumption that would change my answer the most is the miss cost (50 EUR).** By hand, with the threshold fixed at 0.42, the model costs about 1.7 x (miss cost) + 89 EUR per 1,000 cases, and doing nothing costs 20 x (miss cost). They are equal at about **4.9 EUR**, roughly two supervisor trips. If a missed photo costs less than that, doing nothing is cheaper. The same way, with 50 EUR for a miss, the model beats doing nothing as long as more than about **1 in 1,000 photos** has a worker without helmet. I did this by hand, I did not run a full sensitivity sweep, and a re-chosen threshold would be a bit better than these numbers.

## What a person still has to check

I use two thresholds on the `head` score, chosen on valid and measured on test:

- below **0.08**: pass, nobody looks
- between 0.08 and 0.84: **a person looks at the photo** (15 s) and goes only if they see a real case
- above **0.84**: alert, the supervisor goes

On test, with 2,400 photos/day: 92.6% pass, **6.2% go to review (149 photos/day)** and 1.2% alert (**29 alerts/day**). The review takes 149 x 15 s = **37 minutes a day, 0.10 people** (6 productive hours per person). About 3.8 real cases a day slip through.

| Strategy | EUR/day | People looking at photos |
|---|---|---|
| Do nothing | 2,400 | 0 |
| Model alone (threshold 0.42) | 419 | 0 |
| A person checks every photo | 426 | 1.67 (10 h/day) |
| **Model + human review** | **321** | **0.10** |

(Alerting on every photo would cost 6,000 EUR/day.)

The design is 23% cheaper than the model alone, and it cuts the alerts from 85 to 29 a day (7 hours of walking down to 2.4). It is realistic: 37 minutes of looking at photos fits in the day of one safety technician, no new hire. If each look took 1 minute it would be 2.5 hours a day, 0.4 people.

All of this uses my assumptions: 15 s per photo, a person misses 5% of real cases and 6 productive hours a day. A person checking everything only beats the model alone if they miss less than 4.7% of cases, and would only beat my design if they miss less than 0.4%, which I do not believe after 10 hours of looking at photos.

## Limitations & next steps
- **`person` is not learned** (450 train boxes). I do not use it. With more labelled data it could be used, if it is needed.
- **The test split may have near-copies of train photos** (frames of the same video) and the photos come from sites in China, not a fixed camera. I could not check it. Next step: split by site or video, and test with a few hundred photos from the real cameras.
- **My costs and rates are guesses**: miss cost 50 EUR, base rate 2%, a person misses 5%, 15 s per photo. I only did the break-even by hand and not a sensitivity sweep. Next step: ask the safety department for real accident and fine data, and sweep these values.
- **The model may still improve**: it used all 25 epochs without early stopping. Next step: train longer.
- **Mistakes in a row are not independent.** I thought of a rule of two photos in a row before alerting, but it is not in the dashboard and I cannot test it because my photos have no order in time. It would remove false alarms from a worker who takes the helmet off for a moment, but mistakes that repeat (the same yellow bucket looking like a head) would not be fixed. Next step: record some video of a real site.
- **Reproducibility.** `uv.lock` pins a CPU version of PyTorch, so in Colab I had to reinstall the CUDA version by hand, and Linux needs `libgl1` and `libglib2.0-0`. Training on a CPU is too slow. Next step: pin the CUDA build and add the system packages to the setup notes.

## Generative AI use disclosure

*In completing this assignment, I used generative AI tools (Claude, by Anthropic) as a support.*
