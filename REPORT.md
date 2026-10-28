# Assignment 2 Report

- **Name**: Isabella Lentini
- **Student ID**: 20211
- **Email**: ilentini.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

The decision is whether a worker can walk onto a construction site or has to be
stopped because they are not wearing a hard hat. I am thinking of a medium site
of around 300 workers, where everyone goes through one entrance gate, usually
twice a day (in the morning and after lunch). That is about 600 gate passes a
day. Right now a gate supervisor checks this by eye, which means someone has to
stand there and watch every person that walks in.

A camera at the gate takes a picture of each worker, and the model looks for a
head without a hard hat. One gate pass is one case. I treat "worker without a
hard hat" as the positive class, because that is the thing the site wants to
catch. There are three possible outcomes: the worker is let in, the worker is
stopped, or, when the model is not sure, the snapshot is sent to the gate
supervisor's tablet. Checking a snapshot takes about 10 to 15 seconds.

The two mistakes do not cost the same. A false positive means stopping a worker
who is actually wearing a hard hat. They lose maybe a minute, and the supervisor
loses some time too, so I think this costs less than 1 euro. Of course, if it
happens too often, workers will start ignoring the system, so false alarms are
not free. A false negative means letting someone in without a hard hat. Most of
the time nothing happens, but if something falls on them it can be a serious
head injury, with sick leave, insurance and legal costs, and the company can
also be fined by the labour inspectorate. Even taking into account that an
accident on any single day is unlikely, I believe a miss costs around 35 times
more than a false alarm, about 25 euros per case. I set the exact
figures, and where they come from, in the cost section.

At a site in the EU with an entrance check, I expect about 3 out of 100 workers
to show up without a hard hat, mostly because they forget it after a break. So
positives are rare, which means that if the threshold is too low, most alerts
will be false alarms.

This asymmetry drives the rest of the project. Since a miss is much worse than a
false alarm, I care more about recall on the "no hard hat" class than about
precision, the threshold should be on the low side, and uncertain cases should
go to the supervisor instead of being let through.

## Dataset

I use the Hard Hat Workers dataset from Roboflow Universe:

- **Workspace**: joseph-nelson
- **Project**: hard-hat-workers
- **Version**: 5
- **Link**: https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/5
- **License**: public domain

It is an object detection dataset. Each image is a photo of one or more
workers, mostly on construction sites, and every head has a box around it.
There are two classes: `helmet`, a head wearing a hard hat, and `head`, a head
without one. The original project also has a `person` class, but version 5
drops it. That works for me, since it only has 615 instances and does not help
the decision. I picked version 5 because it has no augmentation baked in, so
the test images are real photos. I stayed away from version 14, which relabels
every helmet as a head and would make my problem impossible to learn.

This dataset serves the problem because what it labels is exactly what the
gate has to decide: is there a head without a hard hat. `head` is my positive
class. To turn boxes into one decision per case, I count an image as positive
if the model finds at least one `head` above the threshold, and negative
otherwise.

The main difference from my problem is the camera angle. I mean, most photos
show part of a site from a distance with several people in them, while a camera
at the gate would see one person close up. I come back to this in the next
section.

## What the dataset's composition implies

| Split | Images | `head` (no hard hat) | `helmet` | Images with at least one `head` |
|---|---|---|---|---|
| Train | 4,916 | 4,612 | 13,919 | 911 (18.5%) |
| Valid | 1,413 | 1,339 | 3,913 | 256 (18.1%) |
| Test | 706 | 726 | 1,915 | 140 (19.8%) |

I computed these from the label files. The split is about 70/20/10. There are
about three helmets for every bare head, so `head` is the minority class, but
with 726 instances in test I think it is still big enough for its per-class
precision and recall to mean something. The number that worries me more is at
the image level, since that is what my decision runs on: only 140 test images
have a bare head in them. So one missed image moves recall by 0.7 points, and
the margin of error is about 2.5 points either way. Differences between
thresholds smaller than that are just noise.

The objects are very small. The images are around 500 by 333 pixels, and the
median `head` box covers 0.6% of the image, so about 32 by 32 pixels, with 72%
of them under 1%. On top of that, the scenes are crowded: a median of 3 heads
per image and up to 35. A positive image has about 5 bare heads on average.

This has three consequences. First, the test set has a bare head in 19.8% of
the images, while I expect 3% at the gate. So I cannot take precision from the
test set as it is, since with fewer positives there will be more false alarms
for every real catch. Instead, I use the true positive and false positive rates
from the test set and reweight them with my 3% base rate in the cost section.
Second, in the test set a positive image usually has several bare heads, so the
model only needs to find one of them to get the image right. At the gate there
is one person and one head, so I believe the image-level recall I measure is
optimistic. Third, the model learns from small heads seen from far away, while
at the gate a head fills a big part of the frame. All three splits come from
the same source, so the test set looks like the training data, not like my
camera. In the end, my numbers show how well the model does on this kind of
photo, not how it will do at a real gate. To know that, I would need to test it
on photos taken at an actual entrance.

When I looked at the images, I also found that many of the photos with a bare
head are not construction sites at all, but offices, meetings and officials.
This makes the `head` class easier than it would be at a real gate, which I
come back to in the failure modes.

## Finetuning and baselines

I treat this as object detection, since the model has to find each head in the
image and say whether it has a hard hat, and the dataset is labelled with
boxes. I finetuned `yolo11n` from the pretrained COCO checkpoint on my own
laptop (Apple M2, CPU only), with image size 416, batch 16, seed 0 and
Ultralytics' default augmentation. I first planned `yolo11s` at size 640 for 50
epochs, but on my laptop one epoch took about 12 minutes even with the smaller
model, so I went with 10 epochs, about 2 hours in total. Size 416 is close to
the images' real width of about 500 pixels, so not much detail is lost. The
validation mAP50 was still going up at the end (0.880 after the first epoch,
0.947 after the ninth), so I believe more epochs would still help a bit.

All three models are compared on the same 706 test images. The image-level
columns use my decision: an image fails if its highest `head` score is at or
above the threshold, and the cost is per 1,000 gate passes at my 3% base rate.

| Model | mAP50 | mAP50-95 | Recall (no hard hat) | False alarm rate | Accuracy | EUR per 1,000 |
|---|---|---|---|---|---|---|
| Finetuned yolo11n, threshold 0.14 | 0.958 | 0.649 | 0.907 | 0.025 | 0.962 | 154 |
| COCO zero-shot, "fail if a person is detected" | 0 | 0 | 1.000 | 0.961 | 0.229 | 728 |
| Trivial, everyone walks in | 0 | 0 | 0.000 | 0.000 | 0.802 | 750 |

The COCO model has no class for heads or hard hats, so it cannot do the task at
all, and its mAP is 0. The closest rule it can follow is to stop anyone it
sees, which catches everybody but also stops 96% of the workers who are wearing
one. The trivial baseline gets 80% accuracy while catching nobody, which I mean
is exactly why accuracy is the wrong number for this problem. The finetuned
model cuts the cost by about 80% compared with both. I expected a big gap,
since neither baseline can actually tell a hard hat apart, but I did not expect
this much after only 10 epochs on a laptop.

## Per-class evaluation and failure modes

My positive class is `head`, a head without a hard hat.

| Class | Instances | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|---|
| `head` (positive) | 726 | 0.942 | 0.891 | 0.948 | 0.643 |
| `helmet` | 1,915 | 0.945 | 0.908 | 0.967 | 0.656 |
| All | | | | 0.958 | 0.649 |

`helmet` carries the headline number: it has more than twice as many examples
and the higher AP. `head`, the class I actually care about, is a bit behind on
recall. The confusion matrix at 0.25 confidence shows where. Out of 726 bare
heads, 668 are found as `head`, 30 are not found at all, and 28 are called
`helmet`. That last one is the worst mistake for me, since that worker would
walk in. The opposite almost never happens: only 1 of the 1,915 hard hats is
called a bare head.

The average also hides the most important thing. Half of the 140 test images
with a bare head show only people without hard hats, and many of them are not
construction sites at all, but offices, meetings and officials. The model
catches all 70 of them. The other 70 show a mixed group, with some people
wearing a hard hat and someone who is not, which is exactly the situation at my
gate. There it catches only 57, so 81%. All 13 missed images are of this kind:
the bare head is small and at the back, or half hidden behind two people with
hard hats. So I believe the real recall at the gate is closer to 81% than to
the 91% in the table.

Then the false alarms. When I looked at the three images with the highest score
and no bare head in the labels, all three actually have people without a hard
hat that were never labelled: a crowd of residents, a villager next to a
technician, and a man on a scooter. So some of the "false alarms" are the model
being right and the labels being wrong. This means the false alarm rate of 2.5%
is probably too high, and my cost estimate is slightly pessimistic.

## Cost matrix and operating point

| Outcome | EUR per case | Where it comes from |
|---|---|---|
| False positive (worker with a hard hat stopped) | 0.70 | 1 minute of the worker and 30 seconds of the supervisor, at 26.51 EUR per hour worked (INE, average labour cost in Spain, Q4 2025, all sectors, since I could not find one for construction only) |
| True positive (worker without a hard hat stopped) | 2.50 | The worker walks back for a hard hat, about 5 minutes, plus 30 seconds of the supervisor |
| False negative (worker without a hard hat let in) | 25.00 | See below |
| True negative | 0 | |

The false negative is the hard one, so I split it:

- **Injury, 0.20 EUR.** Construction has 6,014 accidents with sick leave per
  100,000 workers a year (INSST, 2024). Over 220 working days, that is about
  1.4 in 10,000 per half shift. I assumed 5% of them are head injuries a hard
  hat would prevent (I could not find a source for this), and that one costs
  the company about 30,000 EUR. I mean, an accident with sick leave costs
  around 6,000 EUR (Ibermutua), and on top of that the employer pays a 30 to
  50% surcharge on the worker's benefits when safety equipment was missing.
- **Fine, 5.60 EUR.** Not making workers wear protection is a serious offence
  (LISOS art. 12.16), fined from 2,451 to 49,180 EUR (art. 40.2). I assumed
  about 2 inspections a year, a 50% chance the inspector notices, and the
  minimum fine.
- **Site stoppage, about 7 EUR.** The inspector can stop the works. One day
  stopped for 300 workers costs about 63,600 EUR (300 × 8 hours × 26.51),
  times a small chance per miss.
- **Safety culture, about 12 EUR.** This one is my own judgement, not data:
  every worker who gets in without a hard hat shows the others that the check
  is not really enforced.

So about half of the 25 EUR comes from data and half from my judgement. The
base rate is 3%, as explained in the business problem section.

Since the test set has 20% positives and not 3%, I do not use its counts
directly. For each threshold, I take the share of bare-head images caught
(recall) and the share of hard-hat images stopped (false alarm rate), and
weight them with 3% and 97%:

cost per gate pass = 0.03 × (recall × 2.50 + (1 − recall) × 25) + 0.97 × (false alarm rate × 0.70)

I swept the threshold from 0 to 1 in steps of 0.01. The dashboard's "Cost
curve" tab shows the full curve:

| Threshold | 0.02 | 0.06 | 0.10 | **0.14** | 0.20 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|---|
| Recall | 0.921 | 0.914 | 0.914 | **0.907** | 0.900 | 0.879 | 0.864 | 0.857 |
| False alarm rate | 0.085 | 0.044 | 0.034 | **0.025** | 0.021 | 0.018 | 0.012 | 0.012 |
| EUR per 1,000 | 186 | 163 | 156 | **154** | 157 | 169 | 175 | 180 |

The minimum is at 0.14, with 154 EUR per 1,000 gate passes, 90.7% recall and
2.5% false alarms, so that is the threshold I use. The curve is quite flat
between 0.10 and 0.22, all within 5 EUR, and one extra missed image already
changes the cost by about 5 EUR. So, honestly, with only 140 positive test
images anything in that range would be just as good. The threshold is lower
than the usual 0.25 because a miss costs so much more than a false alarm.

| If... | Cheapest threshold | EUR per 1,000 |
|---|---|---|
| My assumptions (miss 25, base rate 3%, false alarm 0.70) | 0.14 | 154 |
| A miss costs 15 EUR | 0.14 | 127 |
| A miss costs 10 EUR | 0.20 | 112 |
| A miss costs 6 EUR (only what data supports) | 0.55 | 97 |
| A miss costs 75 EUR | 0.09 | 284 |
| Base rate 1% | 0.34 | 62 |
| Base rate 10% | 0.09 | 464 |
| A false alarm costs 3 EUR | 0.37 | 198 |

The assumption that would most change my answer is the cost of a miss,
especially the safety culture part, since it is not backed by data. The
threshold stays between 0.10 and 0.22 as long as a miss costs more than about
10 EUR and the base rate is around 3% or more. If a miss is only worth the 6
EUR I can back with data, the cheapest threshold jumps to 0.55, so the model
would stop fewer workers and let more of them through.

## What a person still has to check

I use two thresholds. If the highest `head` score is below 0.09, the worker
walks in. If it is 0.55 or above, the worker is stopped. Anything in between
goes to the supervisor's tablet. I did not pick these by hand: I tried every
pair of thresholds on a 0.01 grid and took the one with the lowest cost, with
the same costs as before plus the cost of a review. A review takes about 15
seconds of the supervisor and 15 seconds of the worker waiting at the gate, so
about 0.24 EUR. I assume the supervisor gets it right 95% of the time, since
the snapshots are small and sometimes blurry.

On the test set this sends 2.6% of gate passes to review. With 600 passes a
day, that is about 16 reviews and 4 minutes of the supervisor a day. Even at
the busiest moment, when about 300 workers arrive in half an hour in the
morning, it is around 8 reviews, so one supervisor who is already on site can
do it next to their normal job, and no extra person is needed. I think this is
realistic even if real gate photos are harder than the test set. If ten times
as many cases went to review, it would still be about 40 minutes a day.

Out of the roughly 18 workers a day who show up without a hard hat, the model
stops about 15, sends about 1 to the supervisor, and lets about 1.5 walk in.

| Policy | EUR per 1,000 passes | EUR per day | Checked by a person | Person time per day |
|---|---|---|---|---|
| Do nothing, everyone walks in | 750 | 450 | 0% | 0 |
| A person checks everyone (gate guard) | 496 | 298 | 100% | full 8-hour shift |
| Model alone, threshold 0.14 | 154 | 93 | 0% | 0 |
| **Model with review band (0.09 / 0.55)** | **148** | **89** | **2.6%** | **about 4 minutes** |

For the guard, I counted the full shift (8 hours at 26.51 EUR, so 0.35 EUR per
pass), since someone checking everyone has to stand at the gate all day, also
when nobody is coming in. The guard also makes the odd mistake, at the same
95%.

The review band saves only about 6 EUR per 1,000 passes compared with the model
alone, which is about 800 EUR a year. Of course, that is not the main reason to
have it. The real difference is in how many workers with a hard hat are stopped
for nothing: 14 a day with the model alone, and about 5 with the band. I did
not put a price on annoyed workers, but I believe that if the system stops
people for nothing every day, they will stop taking it seriously. Compared with
a guard, the model with the band saves about 46,000 EUR a year (220 working
days).

What review cannot fix is the misses. The 13 bare-head images the model misses
all have a score between 0.00 and 0.10, so the model is confident and wrong,
and no review band catches them without sending almost everyone to the
supervisor. So about 1 in 12 workers without a hard hat still gets in, and from
the failure analysis, at a real gate with mixed groups it could be closer to 1
in 5. The guard misses fewer of them, but costs more than three times as much.
I also did not include the cost of the camera and tablet, but I believe a few
thousand euros would be small next to these savings.

## Limitations & next steps

- **The test images do not look like my gate.** Most are crowded scenes with
  small heads seen from far away, and many bare heads come from offices and
  meetings. I would put a camera at a real site entrance for a couple of weeks,
  label a few hundred of those photos, and test on them, and finetune on them
  too if the results drop.
- **The model misses bare heads in mixed groups.** All 13 misses are a small or
  half-hidden head next to people with hard hats, and the model is confident
  about them, so review does not help. At the gate I would place the camera so
  it sees one person at a time from close, for example at a turnstile. I would
  also train longer, with `yolo11s` and a bigger image size on a GPU, since
  validation was still improving after 10 epochs.
- **Some labels are wrong.** The top false alarms had people without hard hats
  that nobody labelled. I would relabel the test images where the model and the
  labels disagree the most, and then recompute the false alarm rate and the
  costs.
- **Half of the miss cost is my judgement.** The 12 EUR for safety culture has
  no source. I would ask the company for its own incident, fine and insurance
  history, and if it cannot be backed up, use 6 EUR and the 0.55 threshold that
  goes with it.
- **The base rate is a guess.** I assumed 3%. A guard could count for one week
  how many workers arrive without a hard hat, and I would rerun the sweep with
  that number.
- **Few positive test images.** Only 140 images have a bare head, so recall
  moves by 0.7 points per image, and thresholds between 0.10 and 0.22 cannot
  really be told apart. A bigger test set from real gate photos would fix this
  too.
- **Things I did not price.** The camera and tablet, and privacy: filming workers at the gate falls under GDPR, so workers have to be informed and the photos cannot be kept longer than needed.

## Generative AI use disclosure

I used Claude Code (Anthropic) throughout this assignment: to structure the
project, find the dataset and the sources for the cost figures, and write the code. I made the decisions myself, choosing from the options it gave me (the problem, dataset, training settings, costs, thresholds and review design), and I reviewed every section.
