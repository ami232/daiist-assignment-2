# Assignment 2 Report: Fire and smoke detection for cameras

- **Name**: Jimena Sánchez
- **Student ID**: 20104
- **Email**: jsanchezcurt.ieu2022@student.ie.edu
- **Group**: PPLEDBA 5A

## Business problem, and the cost of being wrong

For this assignment, I assumed I was an fire and smoke alarm receiving center monitoring 200 indoor cameras in common areas of residential care homes, houses and so on. Each camera sends one frame per minute, with our data set the center receives 288,000 frames per day. 

For each frame, the model may do three things (multi class classification). It may pass review or alarm. A pass means no action is taken. A review means that a human operator will receive the frame alongside its previous one and we'll decide whether to pass or alarm. Finally if the model decides to alarm, it means that the center runs. It's fire protocol straight away, calling 112 and alerting night staff. 

Although we hold this multi class classification, in terms of decision level, it gets collapsed into a binary. A positive frame is one that contains smoke or flame well, the negative is one that contains neither. Once a fire has been alarmed it becomes handled by a person and leader frames no longer go through the decision. This decision is based on the fact that we are monitoring places like care homes where residents often cannot evacuate themselves and staff is usually not enough to do the evacuation securely, which means that older group people are the most likely to die in building fires. We know that in reality ceiling smoke detectors are mandatory but in large rooms they often react late. Seeing smoke earlier gives a person a headstart. Thus, the business frame is not to replace detectors, but rather to decide which of the 288,000 frames a day deserve a human check.

In terms of the cost of mistakes, there is a clear asymmetry. On the one hand, false negatives (missed fire frames). These are assigned a cost of 50,000. One frame equals one minute, so a positive frame starts the response one minute later. Domestic fires furnished rooms can go from a first flame to flashover in about five minutes. On the other hand, we assigned the cost of a false positive (false alarm) a cost of 500. This essentially represents the cost of running the entire protocol for nothing. Although there is a clear opportunity cost associated with this (ranging from nothing to the cost of a life. If had that dispatch not been called they could've saved someone who was really in danger), lives are always a priority, so it is important that the ratio of cost between false negatives and false positives represent a sort of catastrophic versus 'only money, so in some way cheaper scenario. I also associated a cost to the true positive, which is the correct alarm. This is associated to the one frame per minute structure, even if we catch the fire quickly that minute can still can be a cost. However, a cost of zero was assoc assigned to a correct pass. Furthermore a review costs to the model's eyes 0.14 dollars. This term from a simple operation. We assume an average of 20 seconds wasted by an operator that is charging 25 per hour.

The status that has a particularity, which is the growth aspect of what we are measuring. Fire does not exist, isolated in one frame, but appears in one and would most likely if not intervened results in smokier and more fiery frames following it. Thus we may catch it at different points, and I associated a cost to this. A model that catches the first smoke costs, the rooms, contents, and possibly cleaning. I assumed it cost of 10,000 for this. If a model only catches it after a flash over, it may cost part of the building, the evacuation and security risks that come from the homes closure and so on and so for this we assigned a $250,000 cost. This last cost is associated to the cost of a false negative as the 240,000 difference over five minutes (which is the average time we assume it takes a room to go from a first flame to a flashover) is about $50,000 per minute of delay.

This means that according to our cost structure, a misfire alarm costs 100 times more than a false alarm. That ratio is why the pass threshold sits so low, why there is a review band instead of just a single threshold why operators are told to if in doubt, escalate and why I focused on recall of positive class before precision when evaluating this model. 

It is worth noting the possible issue of double counting, and how it was handled. A fire stays in front of a camera for many minutes. If misses were independent, a model that caught only 60% of positive frames would still catch 94 of frames within three minutes, yeah, I still price every missed frame at a full 50,000. This decision is based on two justifications. First that positive classes are defined as unhandled minutes so each mist frame really means one more minute of fire growth frames after an alarm are not counted so we're avoiding a double billing problem in this aspect. The second comes precisely from the fact that misses are not independent. The model misses for reasons that last within frames. For example, real instead of synthetic imagery, small or distant smoke darkness of the night and so on. On the test set, recall at the chosen threshold is 95% on the composited indoor smoke, 10% on real fire photos and 0 of 7 on real video frames. A fire that messes a minute one is likely to be missed again at two. Per frame recall is therefore a pessimistic yet fair measure of how fast a fire is caught.

Overall, the evidence of the value we offer regarding the The business problem is obvious. Without it, the alternative is to have a person watching every frame (meaning 288,000 a day about 246 full-time operators which include a high cost) or to have nobody watching and depend only on ceiling detectors. With our solution operators look at about 0.7% of frames and the center still reacts to most fire within minutes.

## Dataset

- **Workspace / project / version:** `dangerzone-6iqvs` / `smoke_fire-idefg` / version `5`. Link: https://universe.roboflow.com/dangerzone-6iqvs/smoke_fire-idefg/dataset/5. 

Regarding the classes, we have a binary structure where a 0 represents a fire and 1 represents smoke. Because this is an object detection task the background is a free negative class. It is at the decision level where I merge these two classes and I make a frame positive if it contains at least one fire or smoke box because the business action following is similar.

In this project, one still is treated as one camera frame of a minute interval. In practice, the data contains four sources stitched together with file names telling them apart, which made me have to solve for possible data leakage. The first file name structure type composes about 77% of the data set and shows indoor rooms such as bedrooms, dining rooms and lounges with smoke composited at high, medium or low densities. This type of image contains only smoke, never fire. The second type represents about 16% and shows everyday web photos of people, food, objects and so on, which are almost all unlabeled and negative. The third type represents about 6% and contains real news style photos of fires, which are almost all labeled with a fire box. Finally, there is a fourth type with frames from fire videos, but it represents about 1% of the data set.

I chose this dataset because it resembles the scenes my cameras see, which are furnished indoor rooms, especially with cases of faint smoke, which is what the early stage decision is all about. Moreover, the dataset is public and large enough to fine-tune on.

There are two issues, however, with the data set that must be noted. First, the smoke is synthetic. It has a uniform structure and soft edges that real smoke may not resemble. Secondly, the negatives are not what my camera sees when nothing is wrong. The camera would see an empty lounge at night, steam from a food trolley or dust in a sunbeam, while the negatives of the data set are mostly holiday snapshots, so nothing in them is too hard to reject.

To expand on the most important flaw of the data set, the file naming structure coincides almost perfectly with the labels. The model never reads the file name, but the name marks which source an image comes from, and each source has its own look (composited rooms, holiday snapshots, news photos), so the model could learn to recognise the source instead of the smoke. This was tested, and a rule that reads only the file name reached 99.6% accuracy on the original split. Moreover, Roboflow made about three augmented copies of each training photo, and 299 source photos ended up in both train and test. Thus, I handled both problems in `train.py` before training. First, the pipeline groups every copy by its source photo and re-splits the 7,139 source photos into a 70/15/15 split, stratifying by family and label and keeping one copy per source in val and in test, so no source photo appears in more than one split. Secondly, for each positive frame it crops the largest region that stays 5% clear of every box and adds it as a background frame in the same split. This means that every family now has both labels, and the file name rule drops to 73% accuracy by alarming on 85% of the clear test frames. This is done in code on the downloaded version, so the public dataset is used as it is and anyone re-running the pipeline gets the same split. However, we must anticipate that a few crops may be mislabeled, given that they are partial views of a scene and that faint smoke can spill past its box.

Regarding the resulting split, the following was obtained:

| Split | Images | Background (no labels) | Fire only | Smoke only | Both | Fire-or-smoke prevalence |
|---|---|---|---|---|---|---|
| train | 14,456 | 5,385 (37.3%) | 351 | 8,512 | 208 | 62.7% |
| val | 1,483 | 475 (32.0%) | 4 | 999 | 5 | 68.0% |
| test | 1,464 | 458 (31.3%) | 12 | 989 | 5 | 68.7% |

The train split contains 916 fire boxes and 9,179 smoke boxes, the val split 15 and 1,019, and the test split 36 and 1,002. In terms of box sizes, smoke boxes are large (a median of 37% of the image, with only 0.6% under 1% of it), while fire boxes are small (a median of 3.7%, with 12% under 1% of the image).

![Dataset overview](weights/dataset_overview.png)

In our split, fire is the thin class. In the test split there are 1,002 smoke boxes and only 36 fire boxes, which come from 17 frames, so every frame-level number for fire rests on very few images and must be read as rough. Moreover, there is a disparity in frequency that must be noted. Test prevalence is 68.7% while the production base rate is 0.01%. In other words, positives are about 7,000 times more common in the test set than on real cameras. Therefore there are limitations when translating accuracy and precision measured on the test set onto production. Thus the pipeline only uses the test set to measure recall and false positive rate, and we reweight them to the base rate before costing anything.

## Finetuning and baselines

For this task I used yolo11n (nano) for object detection, fine-tuned from COCO weights. Object detection was better suited than classification given that the operator needs to see where the smoke is in order to judge it in 20 seconds, and the dashboard draws the boxes. I chose the nano version because in production one frame per minute from 200 cameras is a light load, so a small model can run on a cheap machine at the alarm center, although the price of this choice is a weaker detection of small objects, which matters for early smoke.

Regarding the training, the model was trained on a Tesla T4 GPU in Google Colab, with 640 px images, batch 16, AdamW, seed 0 and patience 10, using the default Ultralytics augmentation. Given the time available, I trained for 5 epochs, which took 15.6 minutes (3.1 minutes per epoch) on 14,456 training images. The best epoch was the last one, with a validation precision of 0.686, a recall of 0.613, an mAP50 of 0.682 and an mAP50-95 of 0.451. This means that validation was still improving, so the model is undertrained and more epochs should improve it, especially for fire. The full settings are saved in `weights/run_meta.json`.

To evaluate it, I compared the fine-tuned model against four baselines on the same test split, at frame level and with costs computed at the production base rate:

| Model or policy | mAP50 | Frame recall | Frame FPR | Cost per 1,000 frames | TP/FN/FP/TN (test) |
|---|---|---|---|---|---|
| Fine-tuned model alone (threshold 0.657) | 0.677 | 92.2% | 0.0% | €433.80 | 928/78/0/458 |
| COCO zero-shot | 0.000 | 0.0% | 0.0% | €5,000.00 | 0/1006/0/458 |
| Predict nothing | 0.000 | 0.0% | 0.0% | €5,000.00 | 0/1006/0/458 |
| Always alarm | n/a | 100.0% | 100.0% | €500,000.00 | 1006/0/458/0 |
| Source-only rule (file name, no pixels) | n/a | 99.3% | 84.7% | €423,622.87 | 999/7/388/70 |

On the one hand, the COCO zero-shot model has no fire or smoke class, so it never raises an alarm and behaves exactly like predicting nothing, which costs €5,000 per 1,000 frames in missed fires. What it does detect on positive frames are cars, people, chairs, potted plants and beds, which confirms that the scenes are mostly furnished rooms. On the other hand, fine-tuning takes mAP50 from 0 to 0.677. At the cost-optimal threshold the model catches 92.2% of positive frames with zero false alarms on the 458 clear frames, for €434 per 1,000 frames, which is less than a tenth of doing nothing. Always alarming costs €500,000 per 1,000 frames because every frame triggers a €500 response.

Regarding whether the gap is as large as I expected, it is, and this time the source-only rule confirms that it is not explained by the file names. With no pixels at all, the rule still catches almost every positive, but it alarms on 85% of the clear frames, because the same-source crops break the link between the source and the label, so it costs €423,623 per 1,000 frames. The fine-tuned model, on the other hand, separates smoke from clear crops of the very same rooms, which shows that it learned to see the smoke rather than the source.

## Per-class evaluation and failure modes

Regarding the per-class results, the positive class is a frame that contains fire or smoke (any ground truth box of either class). Overall, the model reaches an mAP50 of 0.677 and an mAP50-95 of 0.433 on the test split, and smoke carries this headline number while fire drags it down. Smoke has a precision of 0.955, a recall of 0.971, an AP50 of 0.986 and an AP50-95 of 0.752 over 1,002 test instances, while fire has a precision of 0.324, a recall of 0.417, an AP50 of 0.369 and an AP50-95 of 0.114 over only 36 instances (precision and recall are measured at the confidence that maximises the mean F1, while AP does not depend on a threshold).

At frame level, at the chosen model-alone threshold (0.657), smoke frames are caught 93.3% of the time, while fire frames are caught only once out of 17. Fire boxes are smaller and get lower confidence, so most fire frames score in the middle of the range. In other words, they do not trigger an alarm on their own, but they do land in the review band, where 12 of the 17 fire frames (70.6%) reach a person. Thus, for fire, the design depends on the human reviewer.

![Detection confusion matrix](weights/confusion_detection.png)

In terms of what the model confuses with what, the confusion matrix at a confidence of 0.25 shows that fire and smoke are never confused with each other, and confusing them would not matter anyway because both lead to the same action. The real failure is missed objects, since 27 of the 36 fire boxes and 28 of the 1,002 smoke boxes (2.8%) are not detected at all. The background row (15 fire and 32 smoke boxes predicted where no box exists) is mostly extra or misplaced boxes inside images that do contain smoke, given that none of the 458 clear frames raises an alarm and only 3 reach the review band.

Moreover, to check whether the model sees smoke rather than the source, I measured the ROC AUC within each family. On the composited indoor rooms, where the clear frames are crops of the same rooms, the AUC is 0.9999 and recall at the threshold is 95.1%, so the model separates smoke from the same scenes without it almost perfectly. On real imagery the model is weaker, with an AUC of 0.89 on real fire photos and 0.77 on video frames, and a recall at the threshold of 2 of 20 fire photos and 0 of 7 video frames, although most of these reach the review band. The model is therefore very good at the smoke it was mostly trained on and less reliable on real imagery, which is what production will be.

It is also worth noting the smoke false positives that I expect in production but cannot measure here, such as steam from food trolleys or showers, dust in sunlight, fog on windows, the glare of a lamp at night, or the cigarette or vape of a resident.

Overall, the model beats both baselines where it matters, on cost and on catching positives, at €434 per 1,000 frames against €5,000 for both COCO and doing nothing, and it does so on a split where neither leakage nor the file name can explain the result.

## Cost matrix and operating point

In terms of costs, as derived in the business section, a false negative costs €50,000, a false positive €500, a true positive €500, a true negative nothing and a review €0.14 per frame (20 seconds of an operator charging €25 per hour). This puts the ratio between false negatives and false positives at 100 to 1.

Regarding the base rate, I assumed 0.0001, meaning that 1 frame in 10,000 shows an unhandled fire. That is 28.8 positive frames a day across 200 cameras, or roughly six incidents a day lasting about five unhandled minutes each, which is one incident per camera per month. This figure is deliberately high, since it counts every smoke or flame event that the protocol requires acting on and not only the fires that would become disasters (for example a smouldering bin, an overheating appliance or a pan fire in a kitchenette). I could not find a measured rate per camera minute for the common areas of care homes, so this is the input I trust least. The rest of the inputs are 200 cameras sending 1,440 frames a day each (288,000 in total), 6.5 productive hours out of 8 paid hours per operator per day, a floor of 4.2 full-time operators to keep one seat staffed 24/7, and 5 operators available. For human review, I assumed that operators escalate 95% of real fires and 2% of frames with no fire.

![Cost vs threshold](weights/cost_curve.png)

Regarding the sweep, the model-alone cost curve is lowest at a threshold of 0.657, at €434 per 1,000 frames (with a 95% confidence interval from €355 to €517, from 1,000 stratified bootstrap resamples of the test set). At that threshold frame recall is 92.2% (from 90.6% to 93.8%) and the false positive rate is 0%. The threshold is that high because of the base rate. Negatives outnumber positives 10,000 to 1, so at the margin one more point of false positive rate costs 0.01 × 500 = €5 per frame, while one less point of recall costs only 0.0001 × 0.01 × 50,000 = €0.05 per frame. In other words, even though a missed fire is 100 times worse than a false alarm, false positives are about 100 times more expensive in total, because there are many more chances to make them.

The operating point I chose is the three-way band, which passes frames that score below 0.26, sends frames between 0.26 and 0.66 to review, and alarms at 0.66 and above. It costs €227 per 1,000 frames (with a 95% confidence interval from €152 to €323), almost half of the model alone, and its interval barely overlaps the model-alone one. The review band exists because of the 100 to 1 asymmetry. Frames the model is unsure about (which include most fire frames) are too risky to pass and too uncertain to alarm, and a 20 second human look at €0.14 is far cheaper than either mistake.

To test how stable this choice is, I scaled the cost of a false negative and the base rate from a tenth to ten times their assumed values, holding everything else fixed. The three-way design is recommended from about a fifth of the assumed values upwards, and as the inputs grow the review band widens downwards (its lower threshold falls from 0.50 to 0.20). The assumption that would most change my answer is the base rate. Below 1.3 × 10⁻⁵ (about 4 positive frames a day across all cameras), the model alone becomes cheaper than the three-way design, because the 24/7 operator seat then costs more than the fires the review catches. The same happens if the cost of a missed frame falls below €6,917, if a false alarm costs more than €4,306, or if a review takes longer than 186 seconds.

However, I still pick the three-way design when the base rate is uncertain, because I priced being wrong in each direction while keeping the band fixed at 0.26 and 0.66:

| True base rate | Fire frames per day | Three-way per 1,000 | Model alone per 1,000 | Overpay if I chose three-way | Overpay if I chose model alone |
|---|---|---|---|---|---|
| 10⁻⁶ | 0.3 | €70.01 | €4.34 | €65.67 | none |
| 10⁻⁵ | 2.9 | €84.29 | €43.38 | €40.91 | none |
| 10⁻⁴ (assumed) | 28.8 | €227.15 | €433.80 | none | €206.65 |
| 3 × 10⁻⁴ | 86.4 | €544.63 | €1,301.39 | none | €756.76 |
| 10⁻³ | 288 | €1,655.78 | €4,337.97 | none | €2,682.19 |

If I choose the three-way design and the true rate is low, I overpay at most about €66 per 1,000 frames. If I choose the model alone and the rate is high, I overpay up to €2,682, and that extra cost is missed fires, which in a care home can mean deaths. Thus, the worst case is far smaller for the three-way design, so I choose it, and I would then measure the real rate during a pilot and move the thresholds in the dashboard if needed.

## What a person still has to check

Regarding the thresholds, frames that score below 0.26 pass, frames between 0.26 and 0.66 go to review and frames at 0.66 and above raise an alarm. This sends 0.66% of frames to a person, which means 1,888 frames a day. At 20 seconds each, that is 10.5 hours of looking per day, which needs 1.6 full-time operators of actual work at 6.5 productive hours each. However, any review service needs one seat staffed 24/7, which takes 4.2 full-time operators, so staffing is set by coverage and not by workload. The center has 5, so the design fits with room to spare, since 5 operators could review up to 2.03% of frames (5,850 a day). With this band, 22 of every 1,000 true fire or smoke frames are missed.

Regarding whether this is realistic, it is. The seat is busy about 40% of the time (10.5 of about 27 productive hours), which leaves margin for the bursts that a real incident produces, since it flags several frames in a row from one camera, and the dashboard groups frames by camera so that a person sees one incident rather than several frames. The weak spot is the false escalations of the operators. At a 2% false escalation rate, 1,888 reviewed frames a day produce about 38 false call-outs a day, so the realistic protocol is that the operator first phones the night staff on site and only calls 112 if they confirm, which cuts the cost of a false escalation and favours the three-way design even more.

In terms of the alternatives, I priced the design against doing nothing, having a person check every frame and letting the model decide on its own, all at the same base rate and costs:

| Policy | Cost per 1,000 frames | Cost per day | Share reviewed | Operators needed | Missed per 1,000 true fire frames | Feasible |
|---|---|---|---|---|---|---|
| Do nothing | €5,000.00 | €1,440,000 | 0% | 0 | 1,000 | yes |
| A person checks every frame | €10,467.44 | €3,014,623 | 100% | 246.2 | 50 | no |
| Model alone at 0.657 | €433.80 | €124,934 | 0% | 0 | 78 | yes |
| Three-way at 0.26 and 0.66 | €227.15 | €65,421 | 0.66% | 4.2 | 22 | yes |

Doing nothing misses every fire, and having a person check every frame would need 246 operators against the 5 available. Thus, the three-way design saves about €59,500 a day against the model alone and misses more than three times fewer fire frames (22 against 78 per 1,000), mainly because humans catch the fire frames the model is unsure about. The absolute euro figures are large because the base rate is a high planning figure, but the ranking holds above a base rate of 1.3 × 10⁻⁵.

## Limitations & next steps

The first limitation is that the model is undertrained, since it was trained for only 5 epochs and validation was still improving. With more time I would train for 60 to 100 epochs with early stopping on a GPU, which should mostly help fire.

Secondly, fire is the thin class, with only 36 fire boxes in 17 test frames once duplicates are removed, and fire recall is weak on its own (1 of 17 at the threshold, 12 of 17 with review). This could be improved with more and more varied fire data, a larger model (yolo11s), a higher input resolution for small objects and a higher class weight for fire.

Moreover, the same-source negatives are crops and not real empty rooms, and the false positive rate is measured on easy negatives. To address both, I would collect a week of real frames from the pilot cameras with no fire, including steam, glare and night lighting, and measure the false positive rate at 0.26 and 0.66 on those frames before re-running the cost sweep.

Another limitation is that the smoke is synthetic and there is no small, early, real indoor smoke in the data. I would add real indoor fire test footage (such as burn videos from fire safety laboratories) and track recall separately for boxes under 1% of the image.

It is worth noting that frames are scored one at a time, while a real fire persists. Alarming only when the score stays above a threshold for 2 consecutive minutes would cost one minute of delay but should cut false alarms sharply, because steam and glare rarely persist.

Furthermore, the base rate is assumed and not measured, and it is what decides between the three-way design and the model alone. To measure it, I would run the model in shadow mode for three months while operators log every real smoke incident, and switch to the model alone if it comes out below 1.3 × 10⁻⁵.

Finally, there is the issue of privacy. Cameras in care homes process images of residents and are subject to the GDPR and the Spanish LOPDGDD, so I would restrict cameras to common areas, show operators only the flagged stills, keep passed frames for a short time only, and complete a data protection impact assessment before deployment.


