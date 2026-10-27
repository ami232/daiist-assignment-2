# Assignment 2 Report

- **Name**: Eli Homsi Valencia
- **Student ID**: 18938
- **Email**: ehomsi.ieu2022@student.ie.edu
- **Group**: BBADBA 

## Business problem, and the cost of being wrong

The model is intended to be used by car rental companies, which would take a picture to each car when a customer returns it. The model would cheack each photo for new body damage, and then, if it detects a damage above certain confidence threshold, the car is flagged to have certain damage. Moreover, the image would be revised by a supervisor when the model is unsure, and if the damage is real and new, the customer would be charged and sends the car for repair.

A false alarm for this model would represent a waste of time for the company's staff, that would have to revise a car that had no any new damage. On the other side, a missed damage would mean that a damaged car would be marked as clean, which could cost the company money since they can't bill any customer; there would not be any clear path or clue about which customer was the one who damaged the car, which will make this issue untracable and therefore, the damage would need to be covered by the company.

I assume a false alarm could costs about €3: roughly 5 minutes of a supervisor’s time at about €20 per hour (about €1.70), plus a small delay before the car can be rented out again. On the other side, a missed damage could costs, lets say, about €400, the average repair bill the company has to cover itself because it can no longer bill the customer who caused it. A damaged car that gets flagged also costs about €3, because the supervisor reviews it before it is billed.

In this sense, missed damage is therefore the worse error (€400 against €3), because a review costs minutes of staff time while an unbilled repair costs hundreds of euros. Because of this, the confidence threshold should lean low, so the model flags more borderline cars and misses fewer real damages. A person reviews the cars where the model is unsure, and the model acts alone only when it is very sure.

## Dataset

* Dataset : yolov8-z6snb, project damage-detect-pzgeu, V1. Link: https://universe.roboflow.com/yolov8-z6snb/damage-detect-pzgeu/dataset/1.

* Each image is a photo of one car, with a box around it labelled damaged or undamaged. In my pipeline, one image is one returned car, and the question is whether it shows damage. There are two classes: Damage-detected and Undamaged. My positive class is Damage-detected, since missing a damage is the costly mistake in my business problem. An image is flagged when the model finds at least one Damage-detected box above the confidence threshold.

* This dataset fits the problem because each photo shows one car with a label, so one photo is one decision, like one returned car, and it has an Undamaged class, so I can count false alarms as well as misses. Most car damage datasets only have damaged cars, so they cannot measure false alarms. However, the match is not perfect, since the photos look like auction or stock images, not phone photos taken at a rental return. Also, many of the test cars are damaged, which is far more than a real rental business would see. For this reason, I treated the results as a best case and adjust them to realistic conditions in the cost analysis.


## What the dataset's composition implies

The dataset has 717 damaged and 895 undamaged cars in train, 138 and 272 in valid, and 92 and 119 in test. The three groups have 1,501, 366 and 206 photos. Both classes have enough examples, so no class is too small to trust. Each car box covers about 60% of the photo, so the cars are big and easy to see.

The test set is small. With 92 damaged cars, missing one car lowers the share of damaged cars caught by about 1%. So small differences between results may just be luck.

The test set also does not match real life, since around half of its cars are damaged, but in a real rental business far fewer are. The photos are clean pictures of single cars, so I cannot say how the model works with phone photos, bad light or dirty cars.

## Finetuning and baselines

I finetuned YOLO11s on my car damage dataset. I use detection because the dataset is labelled with a box around each car, and for each box the model returns a class (Damage-detected or Undamaged) alongside a confidence level between 0 and 1. That confidence is what lets me set thresholds later: the score of a photo is the highest confidence among its Damage-detected boxes, or 0 if it has none. 

I started from YOLO11s weights pretrained on COCO, a dataset of everyday objects, swapped its final layer for one with my two classes, and trained it on my training photos. I trained on Google Colab with a T4 GPU for 30 epochs (30 full passes over the training photos), with images resized to 640 pixels and 16 photos at a time. After each epoch the model was checked on the validation photos, and I kept the best version. I chose the small size because it trains fast on a free GPU.

I compared the three options on the same test photos. A photo counts as flagged if its score is 0.5 or higher, a provisional threshold that I replaced later.

| Option | Damaged cars caught (of 92) | Clean cars flagged by mistake (of 114) | Share of photos right |
|---|---|---|---|
| Finetuned YOLO11s | 66 | 3 | 85.9% |
| YOLO11s pretrained on COCO, no training on my data (zero-shot) | 0 | 0 | 55.3% |
| Never flag anything | 0 | 0 | 55.3% |

The reuslts show that my model beats both the other two otpions. The COCO model is the same YOLO11s without my training; it knows everyday objects but has no damage class, so it cannot flag any damaged car. On the other side, the "Never flag anything" is the simplest possible baseline, since it says every car is clean. It is still right more than half of the time, because more than half of the test cars really are clean. So I compare models by how many damaged cars they catch, not by the share of photos they get right.

The gap is as large as I expected, since the the COCO modl and the "never flag anything" option catches none of the damaged cars, while my model catches most of them, because only my model was trained to know what damage looks like. It still misses more than a quarter of the damaged cars at 0.5, which is the costly mistake, so the final design uses two thresholds instead of one.

## Per-class evaluation and failure modes

In my rental case, the positive class is Damage-detected, because a damaged car that is returned to the company and marked as clean is a costly mistake. After making a per-class evaluation, these are the results on the test photos:

| Class | Precision | Recall | AP50 |
|---|---|---|---|
| Damage-detected (positive) | 0.868 | 0.815 | 0.870 |
| Undamaged | 0.898 | 0.888 | 0.959 |

The overall AP50 of 0.915 looks strong, but Undamaged (0.959) carries it. Damage-detected, the class that costs money, is the weaker one (0.870). Its recall of 0.815 means the model does not find roughly 1 in 5 damaged cars, and each of those is a repair of about €400 that nobody pays.

This is what the model mixes up on the test photos:

| True class | Called damaged | Called undamaged | Not found |
|---|---|---|---|
| Damaged (92 cars) | 83 | 8 | 1 |
| Undamaged (119 cars) | 8 | 111 | 0 |

The model makes the same number of mistakes in both directions: 8 damaged cars called undamaged and 8 undamaged cars called damaged; however, for the business scenario they are not equal. A damaged car called undamaged is a repair of about €400 that is not billed, while an undamaged car called damaged usually only costs a supervisor about €3 to check. One more damaged car was not found at all. The model's own matrix also has a very large "background" column, which counts boxes drawn at almost zero confidence, so I do not read it as real false alarms, and I measure false alarms per photo in the cost analysis.

Where it breaks is on damaged cars that get a low score. The number of damaged cars the model finds depends on the confidence level used to measure it: the recall and the matrix above use levels set by the evaluation, while here I count a photo as flagged at a threshold that I choose. At a threshold of 0.5 the model flags 66 of the 92 damaged cars, so 26 would go back out as clean. At a very low threshold of 0.02 it flags 88, and only 4 are missed; this is why my final design passes a car only when its score is very low (below 0.12) and sends the unsure cars to a person. 

## Cost matrix and operating point

I proposed a cost for each outcome for one returned car, based on how a car rental branch would work.

| What happens | Proposed cost | Reasoning |
|---|---|---|
| A person reviews a car, damaged or not | €3 | about 5 minutes of a supervisor's time at about €20 per hour, plus a small delay before the car can be rented again |
| A damaged car is missed (marked clean) | €400 | the average repair, which the company pays itself because it can no longer bill the customer who caused it |
| A clean car is failed automatically, with no person (wrong charge) | €250 | a refund, a complaint and the risk of losing the customer; used in the next section |
| A clean car passes, or a damaged car is failed correctly | €0 | nothing extra happens |

A missed damage is therefore much worse than a false alarm (€400 against €3), so the threshold should lean low. I express every cost per 1,000 returns of cars from a rental branch, assuming that 5% of them (50 cars) have new damage, so the results are easy to compare across designs. My test photos have about 45% damaged cars, so I apply the model's results on those photos to a more realistic mix of 50 damaged and 950 clean cars.
I tested every threshold from 0 to 1 and took the cheapest, with resulted to be 0.02, at €1,388 per 1,000 returns, against €5,835 at the provisional 0.5. However, at a threshold of 0.02 the model catches almost all damaged cars (88 of 92), but it sends about 17% of all returns to a person. I considered that too much work, so my final operating point is a two-threshold design in the next section, while I keep 0.02 as the cheapest single threshold.

The figure that would change the answer most is the cost of a missed damage. The best single threshold stays at 0.02 as long as a miss costs at least about €100, and it only rises if a miss costs much less. The share of damaged cars matters too: with far fewer damaged cars, a higher threshold would be best. The two thresholds of the final design also depend on the 10% cap on reviews (a 5% cap would give 0.34 and 0.57) and on the €250 cost of a wrong automatic charge, whose effect I did not test. I also picked the threshold on the same test photos that I report on, and only 92 of them are damaged, so the result is probably a bit optimistic.

## What a person still has to check

I use two thresholds on the model's score. A car with a score below 0.12 passes without a person, a car with 0.70 or more fails without a person, and a person reviews the cars in between. I chose them as the cheapest pair that sends at most 10% of returns to a person, using the €250 cost of a wrong automatic charge. I tried caps of 5%, 10% and no cap; with no cap about 15% of returns go to a person (close to the 17% of the single threshold), which I considered too much, so I chose 10% (the 5% cap is discussed below).

For every 1,000 returns, this sends about 84 cars to a person (8.4%). For a large branch with 500 returns a day, that is about 42 reviews a day. At 5 minutes each it is about 3.5 hours a day, and at 6 productive hours per person that is about 0.6 of a person. This looks realistic for a large branch, since one part-time reviewer is enough, but it depends on my assumption of 5 minutes per review and on the reviewer never making a mistake, which I did not test.

I priced four designs for 1,000 returns, with 5% damaged cars:

| Design | Cost per 1,000 returns | People needed at 500 returns a day |
|---|---|---|
| Do nothing | €20,000 | 0 |
| A person checks every car | €3,000 | about 7 |
| The model decides alone (best single threshold, 0.51) | €10,036 | 0 |
| Model plus review (my design) | €2,426 | about 0.6 |

Doing nothing costs 50 damaged cars times €400. The model deciding alone is expensive, because it misses damaged cars and wrongly charges clean ones, whike my design is the cheapest. Compared with checking every car, it saves about €575 per 1,000 returns (about €290 a day) and needs about 6 fewer people (about 0.6 instead of about 7).

Reviewing less saves time, not money. With a cap of 5%, the best design costs €5,052, more than having a person check every car, because a review costs only €3. So my 10% cap is a decision about workload, and it costs about €1,100 more per 1,000 returns than a design with no cap.

## Limitations & next steps

**Limitations**

- **The photos do not look like real returns.** The dataset photos look like auction or stock images of single cars, not phone photos taken at a rental return. 

- **The test mix is not real life.** About 45% of my test cars are damaged (92 of 206), while I assume far fewer in a real branch. This does not change how well the model detects damage, but it changes the euro figures, which is why I adjust them to a realistic mix.

- **Every cost is a proposal.** The costs proposed are just an assumption, based on my own reasoning and kwoledge of how a rental car company would work. With more time, maybe I could have done an extensive researhc and try to look for real data of these costs, such as repair invoices, the cost of a wrongly charged customer and the time a reviewer takes.

- **The boxes cover most of the car (about 60%).** The model mostly answers "damaged or not" and does not show where the damage is. A reviewer would have to look for it by hand. 

**Next steps**

I consider that one of the main next steps I could do to imrpove the model is to make it more realistic by showing where the damage is and what type it is. For example, the model would detect if it is a dent, a scratch or a broken light; this would help in three ways: the reviewer would see the damage straight away, the review would be faster, and the charge would be more specific and different depend on the type of damage; a small scratch would cost much less to repair than a broken light or a dented door, so the charge to the customer would be fairer and the cost of a missed damage would no longer be one average figure. 

## Generative AI use disclosure

*I used Claude to guide me through the assignment steps, brainstorm the problem, dataset and limitations, support me with the coding, debug setup and code errors, and check my work against the requirements. All the work is based on my own judgment and decisions.*