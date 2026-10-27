# Assignment 2 Report

- Name: Clara Zsigmond
- Student ID:
- Email: crubinstein.ieu2022@student.ie.edu
- Group: PPLEDBA 5A

## Business problem, and the cost of being wrong

The decision I decided to pursue is whether or not a worker is wearing the appropriate safety equipment, in particular a hard hat. Someone or, in this case, a model, needs to look at an image to decide whether or not the guideline of wearing a hard hat is being respected.

If the model finds that a worker is not wearing it, the supervisor would go and check to make sure. In the case the model is right, the supervisor will enforce the rule so all workers wear hard hats, thus avoiding a potential fine or safety liability in case of an injury. If the model alerts that a worker is not wearing it but the worker is, all that happens is the supervisor will walk over and verify, maybe wasting a few minutes (or not even "wasting", as it's part of his job as a supervisor so these are just normal work hours not wasted ones). 

If the model does not alert, however, and there is a worker not wearing a hard hat, consequences are much more serious. There are several levels to the severity of this misidentification:
- real risk of injury -> can cause workers' compensation claims, legal costs and loss of workers.
- fines -> in the United States, depending on how the Occupational Safety and Health Administration (OSHA) classifies the violation, companies can be fined up to $165,514 (this applies if management intentionally ignored head protection rules, or if the plant has been cited for the same safety violation recently). To be realistic, we will consider the more typical fine values. A serious violation where management knew or should have known about the lack of hard hats can be up to $16,550, and can be multiplied by the number of workers in violation.

I want to add that these costs can add up, but they are only realized IF the violation is found out and fined or if an injury takes place and companies must pay compensation to workers. This is taken into consideration when defining cost in the model.
Needless to say, a false negative is significantly more costly and dangerous than a false positive. Where a FP causes at the very most a little extra work from a supervisor, a FN can go up to a hundred thousand dollar fine (averaging lower at about $10,000 to $20,000, since serious violations are around $16,000).

The costs are converted to euro in the calculations but are taken considering OSHA regulations and USA precedent cases.

## Dataset

- Workspace: joseph-nelson
- Project: hard-hat-workers
- Version: 10
- Link: https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/10

Each image is a photo of a work site with one or more workers in it. Every head in the image is labelled with one of three classes:
- `head` -> a bare head, so a worker without a hard hat
- `helmet` -> a head with a hard hat on
- `person` -> a whole person (only used on a small number of images)

This dataset works for my problem because the `head` class is exactly the violation I care about, so the model finds the bare head directly instead of guessing compliance from the whole person. I treat `head` as the positive class, and an image counts as positive if it has at least one bare head in it. The images are also close to what a site camera would see: several workers in one frame, often far away, from different angles, at outdoor and indoor sites. Some of them are stock photos (a few have watermarks), so they are cleaner and more posed than real CCTV footage. I come back to this below.

## What the dataset's composition implies

| Split | Images | Images with a bare head | `head` | `helmet` | `person` |
|---|---|---|---|---|---|
| Train | 4,916 | 911 (18.5%) | 4,612 | 13,919 | 450 |
| Valid | 1,413 | 256 (18.1%) | 1,339 | 3,913 | 101 |
| Test | 706 | 140 (19.8%) | 726 | 1,915 | 64 |

| Class (training split) | Median box size (% of image) | Share of boxes under 1% of the image |
|---|---|---|
| `head` | 0.6% | 72% |
| `helmet` | 0.8% | 56% |
| `person` | 8.2% | 7% |

What this means:
- `helmet` and `head` have plenty of data. With 726 bare heads and 1,915 helmets in test, I can trust the per-class numbers for the two classes that matter.
- `person` is too thin to trust. There are only 64 instances in 23 test images, and only 154 of the 4,916 training images have it labelled at all, even though almost every image has people in it. The label is used inconsistently, so the model learns to mostly ignore it. Any number on `person` is noise, but the decision does not depend on it.
- The objects are tiny. The typical bare head takes up 0.6% of the image, and almost three quarters are under 1%. That is why I kept the full image size of 640 for training. It also means a camera mounted further away than in these photos would make things harder.
- Positives are far more common here than on a real site. About 1 in 5 images has a bare head, and I assume 1 in 20 in production (see the cost section). That is why the cost is calculated on positive and negative images separately and then re-weighted to my base rate, so the test split's mix does not leak into the cost.
- The image level test numbers have a margin. There are only 140 positive test images, so a recall of 95% at image level is really somewhere between roughly 91% and 99%.
- The test split looks more like the training data than like my site. All three splits come from the same export, so similar photos (and sometimes frames from the same scene) end up in both train and test, which makes the test score optimistic. I also cannot conclude anything about night shifts, rain, dust, fixed CCTV angles or hard hat styles that are not in the data.

## Finetuning and baselines

I finetuned YOLO11n for object detection, starting from the pretrained COCO checkpoint. I picked detection over classification because one image usually has several workers in it, and the safety officer needs to know which one is missing a hard hat. A whole-image classifier would also struggle, since the bare head is often less than 1% of the image. I chose the nano model because I trained locally on my MacBook (Apple MPS), and it is the size that would realistically run on a camera feed.

Training settings: 20 epochs, image size 640, batch 16, early stopping patience 8, seed 0, full training split.

All three models are scored on the same test split with the same code:

| Model | mAP50 | mAP50-95 | `head` AP50 | `helmet` AP50 | `person` AP50 |
|---|---|---|---|---|---|
| Finetuned YOLO11n | 0.660 | 0.459 | 0.960 | 0.982 | 0.039 |
| COCO YOLO11n, zero-shot | 0.013 | 0.009 | 0.000 | 0.000 | 0.039 |
| Predict nothing | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The COCO model has no `head` or `helmet` class, so the only thing I can map across is its `person` class. It scores zero on the two classes my decision depends on, so almost all of the gain comes from finetuning. The gap is about as large as I expected, since the pretrained model was never trained to tell a bare head from a hard hat. I did not expect the finetuned model to be no better than COCO on `person` (0.039 for both), which means finetuning taught it nothing about that class. The overall mAP50 of 0.66 is pulled down by `person`. On the two classes that matter it is around 0.97.

## Per-class evaluation and failure modes

| Class | Test instances | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| `head` (positive) | 726 | 0.938 | 0.942 | 0.960 | 0.672 |
| `helmet` | 1,915 | 0.949 | 0.942 | 0.982 | 0.682 |
| `person` | 64 | 1.000 | 0.000 | 0.039 | 0.023 |

Confusion matrix on the test split (confidence 0.25, IoU 0.45, rows = true, columns = predicted):

| True \ Predicted | `head` | `helmet` | `person` | missed |
|---|---|---|---|---|
| `head` | 695 | 14 | 0 | 17 |
| `helmet` | 6 | 1,851 | 0 | 58 |
| `person` | 0 | 0 | 0 | 64 |
| background (false detection) | 88 | 181 | 0 | |

The overall number is carried by `helmet` and `head`, and `person` drags it down. The precision of 1.0 on `person` means nothing here, because the model never predicts the class at all (recall 0, all 64 missed).

What the model gets wrong:
- It rarely mixes up a bare head and a hard hat. Only 14 bare heads were called helmets and 6 helmets were called bare heads. Those 14 are the most dangerous mistakes, since the model sees the worker and says they are fine.
- Most errors are heads it misses or heads it invents. 17 bare heads were missed entirely, and there are 88 false `head` boxes on things that are not labelled as heads. From the examples I looked at, these tend to be tiny, far away workers or people seen from behind.
- At image level, two positive test images in `artifacts/examples/` get a score of 0.0, so the model sees no bare head at all. In one, an older grey-haired man in a shirt stands next to two workers wearing hard hats. He looks more like a site manager or visitor than a worker, and the model misses him completely. In the other, about fifteen workers stand in a crowd, and my guess is the bare head is partly hidden behind the others in the back row. On the negative side, a fisheye photo of electricians on a power pole gets sent to review (score 0.13). The workers in the field are tiny, and the warped lens makes their helmets hard to read.

Overall the model is very good on clear, close up workers. It struggles with people who do not look like typical workers, crowded scenes where heads overlap, and very small or distorted heads. A real site camera will produce a lot of these.

## Cost matrix and operating point

One case is one camera snapshot of a work area, and it is positive if at least one worker in it has a bare head. The model gives each snapshot a score, which is the highest `head` confidence anywhere in the image.

| Outcome | Cost | Where it comes from |
|---|---|---|
| False negative (bare head missed) | €1,784.50 | Expected fine plus expected injury cost: ($16,550 x 10% + $42,000 x 1%) x 0.86 €/$ |
| False positive (false alarm) | €10 | The supervisor walks over for nothing. It is part of the job, but it still takes around 15 minutes away from other work, at roughly €40/hour |
| True positive (correct alarm) | €0 | The supervisor walks over and finds a real violation, which is their job, so it costs nothing extra |
| True negative | €0 | Nothing happens |
| Human review | €0.50 | Someone looks at one flagged snapshot for about 30 seconds |

For the false negative I used the expected cost of the two things that can go wrong, meaning the size of each one times the probability that it happens:
- The fine. One OSHA serious violation is $16,550. Realistically not every missed bare head ends up with an OSHA citation, so I went with a 10% chance. It is a low probability, but when it does happen it is really expensive. That comes to €1,423.
- The injury. The National Safety Council puts the average cost of a workplace injury that needs medical attention at around $42,000 in medical bills and lost wages. A head injury from one missed bare head is much less likely than a fine, so I used a 1% chance. That comes to €361.

Together that gives €1,784.50 per miss. I did not add a separate lawsuit cost, because in the US an injured worker is normally covered by workers' compensation and cannot sue the employer directly. I also left out the legal fees for an OSHA investigation, so this is a conservative number.

Base rate: 5%. I assume 1 in 20 snapshots on a real site has a bare head in it. The test split has 20%, but those photos were collected to show violations, so they have far more of them than a normal working day would. The cost is calculated on positives and negatives separately and then weighted to 5%.

I swept the threshold from 0 to 1 in steps of 0.01 and calculated the expected cost per 1,000 snapshots at each one. The full cost against threshold curve, with the chosen threshold marked, is in the dashboard under "Cost and review". I picked the threshold on the validation split and then reported it on the test split, so the operating point is not tuned on the numbers it is judged by.

| Threshold | Bare heads caught (test) | False alarms (test) | Cost per 1,000 (test) |
|---|---|---|---|
| 0.01 | 96.4% | 14.5% | €4,563 |
| 0.06 (chosen) | 95.7% | 7.1% | €4,495 |
| 0.25 | 95.0% | 3.4% | €4,780 |
| 0.50 | 89.3% | 1.4% | €9,694 |
| 0.80 | 72.1% | 0.7% | €24,923 |
| 1.00 (never alarm) | 0% | 0% | €89,225 |

The cheapest threshold on validation is 0.06, which is very low. That makes sense with my costs. A missed bare head costs almost 180 times more than a false alarm, so the model should alarm on almost anything that looks like a bare head. Above 0.30 the cost goes up fast, since every bare head that slips through costs €1,784.

The assumption that would most change my answer is the probability of the fine. I tested it:
- Between 2% and 25%, the threshold stays at 0.06.
- At 1% or lower (a miss worth about €500 or less), the best threshold moves up to 0.25, because false alarms start to matter more.
- Even with a 0% chance of a fine, the injury cost alone (€361) keeps the model far cheaper than doing nothing.
- The review band stays at 0.06 / 0.90 the whole way from 0% to 25%.

So the decision holds unless the real chance of a fine is much lower than I think. The base rate matters too. If more than about 15% of snapshots had a bare head, it would be cheaper to send the supervisor to every snapshot.

## What a person still has to check

I use two thresholds, again picked on validation and reported on test:
- score below 0.06 -> pass, no alarm
- 0.06 to 0.90 -> review, a person looks at the snapshot
- score of 0.90 or more -> fail, the alarm goes straight to the supervisor

| Snapshots with... | Pass | Review | Fail |
|---|---|---|---|
| a bare head | 4.3% | 88.6% | 7.1% |
| no bare head | 92.9% | 7.1% | 0% |

With the high threshold at 0.90, very few snapshots skip the reviewer. Looking at a snapshot costs €0.50 and a false alarm walk costs €10, so it is much cheaper to have someone check first and only send the supervisor when the bare head is confirmed. The low threshold of 0.06 makes sure almost no bare heads pass without anyone looking.

The review load:
- Weighted to my 5% base rate, 11.1% of snapshots go to review.
- I assume 10 cameras, one snapshot every 3 minutes, over a 10 hour shift, so 2,000 snapshots a day. That means about 223 reviews a day.
- At 30 seconds per review, that is just under 2 hours of work.
- With 420 productive minutes in one person's day, it needs 0.27 of a reviewer.

That is realistic. One person (for example someone in the site office who also does other safety admin) can clear it with plenty of time left, so I am well within the one reviewer I have available.

Compared with the alternatives (expected cost per 1,000 snapshots, test split, 5% base rate):

| Policy | Cost per 1,000 | Cost per day (2,000 snapshots) | People needed |
|---|---|---|---|
| Do nothing | €89,225 | €178,450 | 0 |
| COCO model zero-shot (best threshold) | €8,845 | €17,691 | 0 |
| Model decides alone (threshold 0.06) | €4,495 | €8,991 | 0 |
| Model + human review (0.06 / 0.90) | €3,880 | €7,759 | 0.27 |
| A person checks everything | €500 | €1,000 | 2.38 |

Doing nothing is by far the worst option, which is why the model is worth building in the first place. The model with review beats the model alone by about €600 per 1,000 snapshots, because the reviewer gets rid of most of the €10 false alarm walks.

On paper, having a person check every snapshot is the cheapest option. That only holds because I assume the reviewer never makes a mistake, and it needs 2.4 full-time people when I have one. Someone looking at 2,000 snapshots a day will get tired and start missing heads. At €1,784 per miss, the reviewer only has to miss about 4% of bare heads for that option to cost more than model + review. So with one reviewer available, model + review is the cheapest design I can actually staff. If I could hire two or three more people and keep their accuracy high, checking everything would be worth looking at.

## Limitations & next steps

- Each snapshot is priced as its own fine. If the same worker has no hard hat for an hour, that is 20 snapshots and 20 possible fines in my model, but OSHA would cite it once. This makes the "do nothing" cost too high. Next step: group snapshots by worker and time window so one violation counts once.
- The reviewer is assumed to always be right, which is why "check everything" looks so good. Next step: measure how accurate a real reviewer is on a sample of these snapshots, and add the cost of reviewer misses.
- The test images are stock and collected photos, not CCTV. They are cleaner, closer and better lit than a fixed site camera. Next step: collect a few hundred frames from the actual cameras on site, label them, and test on those before trusting the numbers.
- The model misses people who do not look like typical workers (like the visitor in a shirt) and heads in crowds. Next step: add more images of visitors, managers and crowded scenes to training, and try YOLO11s or a larger image size to help with the small heads.
- `person` is barely used. The label is too sparse to learn. I would drop it from training, or relabel it consistently, so it stops pulling down the overall mAP.
- The 10% fine and 1% injury probabilities are my own estimates. The answer holds for any fine probability between 2% and 25%, but I would want real numbers on how often OSHA inspects sites like ours, and on head injury rates, to pin them down.

## Generative AI use disclosure

AI was used to build the Gradio dashboard and help me write the code given my own design decisions. For the REPORT.md, it was used at the end to better format my own responses (such as formatting the tables.) and confirm all my values matched, as I reevaluated costs a few different times and reran leaving some old numbers without realizing.
