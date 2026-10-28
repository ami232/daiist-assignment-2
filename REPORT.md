# Assignment 2 Report

- **Name:** Daniel Teixidor
- **Student ID:** 19971
- **Email:** dteixidor.ieu2022@student.ie.edu
- **Group:** BBADBA 5A

## Business problem, and the cost of being wrong

The model helps a construction site detect workers who are not wearing helmets. Twenty cameras take a picture every five minutes during a ten-hour day, which gives 2,400 pictures per day. Each picture is either passed, sent to a safety officer for review, or triggers an alert.

I assume that 2% of pictures show a violation. Missing a violation costs €60, because the worker stays unprotected and there may be injury or fine risks. Any alert costs €8 because a supervisor must go and check the site; this applies to both false alarms and correct alerts. A true negative costs nothing. Human review costs €0.50 per picture. A missed violation is therefore about 7.5 times more expensive than a false alarm, so the model should favour recall on unprotected heads.

## Dataset

I used the public Roboflow Universe dataset `joseph-nelson/hard-hat-workers`, version 5 (workspace `joseph-nelson`, project `hard-hat-workers`, version 5). It has two classes: `head` (a worker without a helmet) and `helmet`. One image is a photo of one or more workers.

The dataset fits the problem because the model must find heads and helmets in pictures of workers. However, the images are mostly normal, ground-level photos, not footage from high-mounted security cameras.

## What the dataset's composition implies

| Split | Images | Images with a `head` | `head` objects | `helmet` objects |
|---|---:|---:|---:|---:|
| Train | 4,916 | 911 (18.5%) | 4,612 | 13,919 |
| Validation | 1,413 | 256 (18.1%) | 1,339 | 3,913 |
| Test | 706 | 140 (19.8%) | 726 | 1,915 |

There are about three times fewer `head` objects than `helmet` objects, even though detecting an unprotected head is the main goal. Heads are also very small: the median head box covers 0.60% of the image. This makes them harder to detect.

The test set has 19.8% positive images, but I assume the real rate is 2%, so the cost results are reweighted to that rate. The dataset has no empty scenes and does not look like real CCTV footage. I cannot conclude how well the model will work on site cameras until it is tested on real footage.

## Finetuning and baselines

I fine-tuned YOLO11n, a small object-detection model, starting from COCO-pretrained weights. I chose detection rather than classification because one picture can contain several workers, and one person without a helmet is enough to trigger an alert. A picture's score is the highest `head` confidence in it.

I trained at 640-pixel resolution for 40 epochs, with batch size 16 and the AdamW optimiser, on Google Colab. Training took 51 minutes.

| Model | Image-level AUC | Threshold | Recall | False-alarm rate | Cost per 1,000 pictures |
|---|---:|---:|---:|---:|---:|
| Fine-tuned YOLO11n | 0.978 | 0.54* | 88.6% | 1.2% | €376 |
| COCO zero-shot (person score as a proxy) | 0.677 | 0.95 | 1.4% | 0.2% | €1,199 |
| Predict nothing | 0.500 | None | 0% | 0% | €1,200 |

\* Cost-minimising threshold when the model decides alone (no human review), from the threshold sweep. Costs are reweighted to a 2% violation rate.

The fine-tuned model is much better than both baselines and cuts the estimated cost by about two thirds compared with doing nothing. The COCO model can detect people, but it has no notion of helmets, so it performs almost the same as predicting nothing.

## Per-class evaluation and failure modes

| Class | Precision | Recall | AP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| `head` | 0.942 | 0.930 | 0.962 | 0.676 |
| `helmet` | 0.953 | 0.939 | 0.983 | 0.684 |

I treat `head` as the positive class, since it is the violation. The model performs slightly better on helmets than on unprotected heads.

| Confusion matrix | True head | True helmet |
|---|---:|---:|
| Predicted head | 622 | 174 |
| Predicted helmet | 102 | 1,730 |
| Missed / background | 2 | 11 |

The two tables are computed differently by YOLO's validator. Precision and recall are taken at the confidence that maximises F1. The confusion matrix keeps boxes down to a very low confidence (0.001) and matches each true box to one predicted box, the most confident overlapping one, whatever its class. So a confident `helmet` box can claim a true head even when a `head` box was also predicted. That is why head recall reads 85.7% (622/726) in the matrix but 0.930 in the table. The low cut-off also explains the thousands of low-confidence boxes on background, which are not shown above. The cost analysis uses a swept threshold instead.

The most important error is a bare head labelled as a helmet: 102 of the 726 true heads were classified this way. This can cause a violation to be missed. Only two heads were missed completely. In the other direction, 174 helmets were wrongly labelled as heads, which causes false alarms. Caps, hoods, light hair and very small, distant faces are the likely causes of these confusions.

## Cost matrix and operating point

| | Violation present | No violation |
|---|---:|---:|
| Alert | €8 | €8 |
| Pass | €60 | €0 |
| Human review | €0.50 + €8 alert | €0.50 |

These figures are my estimates, not measured costs. I swept the confidence threshold and computed the expected cost per 1,000 pictures at a 2% violation rate. When the model decides alone, I chose a threshold of 0.54: €376 per 1,000 pictures, with 88.6% recall and a 1.2% false-alarm rate. The sweep also has a second, almost equal minimum at 0.65 (€370, 86.4% recall, 0.9% false alarms). The difference between them is a single positive test image. With only 140 positive images, one image moves the cost by about €7 per 1,000, so the two thresholds cannot really be told apart. I keep 0.54 because it catches more violations, and a missed violation is the expensive error.

An alert is worthwhile when the chance of a violation is above €8/€60, or about 13.3%. The assumption that would change the answer most is the €60 cost of a missed violation: if it were lower, the optimal threshold would rise and fewer alerts would be sent. The answer also changes if the real violation rate differs from 2%.

The threshold was chosen and evaluated on the test set, which may make the result look slightly better than it really is.

## What a person still has to check

My rule is:

- Score below 0.10: pass the picture.
- Score from 0.10 to below 0.86: send it for human review.
- Score of 0.86 or higher: trigger an alert.

The alert threshold is higher than the 0.54 used for the model alone, because uncertain pictures now go to a person instead of triggering an alert. Only pictures where the model is very confident skip review.

About 6.5% of pictures go to review, which is around 157 pictures per day. At 30 seconds each, this takes about 1.3 hours per day. I assume one safety officer has six productive hours a day and can review around 720 pictures, so one person can handle the load with plenty of spare time.

Of all violations, about 66% go to review, 29% trigger an alert directly, and 5% are missed. This assumes the reviewer is always correct and checks each picture straight away.

| Option | Cost per 1,000 pictures | Cost per day | Main issue |
|---|---:|---:|---|
| Do nothing | €1,200 | €2,880 | All violations are missed |
| Person checks every picture | €660 | €1,584 | Needs about 3.3 people |
| Model alone | €376 | €902 | Misses about 11% of violations |
| Model plus human review | €245 | €587 | Lowest cost; fits one reviewer |

The model plus review saves about €2,300 per day compared with doing nothing, and about €315 per day compared with the model alone.
## Limitations & next steps

1. **Different camera conditions:** Test the model on real CCTV footage from the site.
2. **Small or unclear heads:** Add more examples of distant workers, caps and hoods. Try a larger model or a higher image resolution.
3. **No empty scenes:** Add pictures with no workers to measure false alarms on empty frames.
4. **Cost assumptions:** Check the real violation rate and agree realistic costs with site safety staff.
5. **Threshold selection:** Save full-precision scores, choose the thresholds on the validation set, and evaluate them once on the test set.
6. **Human review:** Measure how accurate and fast the safety officer is in practice.
7. **Pictures over time:** Track workers across frames to measure how long a violation lasts.

## Generative AI use disclosure

I used AI to help with basic coding tasks, such as downloading the dataset, setting up model training and creating plots. I reviewed and adapted the code as needed. I also used it for formatting the report and editing its wording.
