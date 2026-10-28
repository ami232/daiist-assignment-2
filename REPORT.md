# Assignment 2 Report


- **Name**: Emilia Granja
- **Student ID**: 18875
- **Email**: egranja.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Business problem, and the cost of being wrong

The company is a fruit distributor that handles around 100,000 apples, bananas and oranges every day. Before the orders get packed for supermarkets, a camera positioned above the conveyor belt captures each fruit, and a machine learning detector identifies it, labeling it as fresh or rotten and assigning a confidence score to that prediction. The business decision has three possible outcomes, the fruit can be shipped if it is considered fresh, discarded if it is considered rotten, or sent to a human inspector if the prediction is uncertain. The decision uses the model’s highest rotten confidence score for each fruit, together with two thresholds, below the lower threshold the fruit is shipped, above the upper threshold it is discarded, and in between it goes to a human inspector. The objective is to reduce waste and inspection costs while also preventing rotten fruit from reaching customers.

Two errors can happen here. A false positive is when a good fruit gets incorrectly identified as rotten and discarded, and this costs around €0.20 per fruit. This estimate comes from wholesale prices at Mercamadrid, weighted by the company’s fruit mix. A false negative is when a rotten fruit gets incorrectly identified as fresh and shipped, and its estimated cost is €2.00, which is the €0.20 value of the fruit plus €1.80 in expected follow on costs. The additional €1.80 is an estimate, for example, if one in ten shipped rotten fruits leads to a customer claim that costs €18, the average claim cost per shipped rotten fruit works out to €1.80. These numbers are estimates and not verified company accounting figures, and the €2.00 false negative cost is especially uncertain since actual complaint rates and claim costs can vary quite a bit.

The two errors are not symmetric, a false negative costs about ten times as much as a false positive (€2.00 ÷ €0.20 = 10). Assuming a 3% base rate of rotten fruit (this is justified in the cost section), 100,000 fruits contain around 3,000 rotten fruits per day, and if all of these were shipped, the estimated cost would be €6,000 per day (3,000 × €2.00). This is why the business should prioritize avoiding false negatives rather than just maximizing overall accuracy. The two thresholds should reflect the relative costs of the errors, favouring the detection of rotten fruit even if this means some good fruit gets discarded along the way. Human review also has a cost, so it should only be used for uncertain cases where the inspection is expected to reduce losses enough to justify the extra expense.

## Dataset

This dataset comes from Roboflow Universe, workspace project-irggm, project -w4yds, version 1: https://universe.roboflow.com/project-irggm/-w4yds/dataset/1. It has 23,904 images and a Public Domain license. The version was auto-oriented and stretched to 640×640, with no augmentations applied. Most images show one fruit per photo, usually close-up; the median is 1 fruit per image, although a few images have a lot more, up to 174 in some cases. The dataset has 6 classes: fresh_apples, fresh_banana, fresh_oranges, rotten_apples, rotten_banana, rotten_oranges. For my decision, the positive classes are the rotten ones, since those are the cases that would trigger action, like discarding the fruit or sending it to review.

This dataset fits my problem because one image can represent one decision on the conveyor line, which matches the same unit I used in the cost analysis. It also covers the three fruits I care about and separates fresh from rotten, which is exactly what the detector needs to do. However, it is still a bit different from a real conveyor: the images are mostly stock-style fruit photos with plain backgrounds, and some even have rotated objects with black corners, so they don’t look exactly like what a factory camera would capture.

## What the dataset's composition implies

The dataset is split into train, valid and test sets with 16,609 / 4,852 / 2,443 images, which is a 70/20/10 split, and about 58% of the images contain rotten fruit. In the class counts, rotten_apples is the biggest class with 15,055 instances, while the smallest test classes are the oranges, with 336 each. Almost all objects are large in the image; apples take up a median of about 8-9% of the image, while bananas and oranges are usually around 20-60%. A few boxes are almost zero-sized, which may be labelling errors.

These numbers are useful, but they still have limits. 336 test instances give a reasonable overall picture, but it is not enough to make very fine per-class comparisons, especially if some test images are rotated copies of training images, and that means the number of truly new examples could be lower than it looks. The test set also does not really match a real conveyor line, because 58% rotten in test is very different from the assumed 3% rotten on a real line, and at that base rate, the same false positive rate creates many more false alarms, so precision would change a lot in practice. This is why the cost analysis uses the test error rates reweighted to the assumed 3% base rate, instead of using test precision directly.

So even if the model gets very high test scores, that does not prove it will work just as well on a real conveyor; it also does not tell us the precision we should expect when rotten fruit is only 3% of the cases in real use.

## Finetuning and baselines

I used object detection rather than classification because some images contain more than one fruit, and I needed the model to know where the fruit is, not just what class it belongs to. I used YOLO11n because it is the nano version, so it is faster and lighter, which matters for a conveyor setup where decisions need to be made quickly. I finetuned YOLO11n from its COCO-pretrained weights for 20 epochs at 640×640, with batch size 16 and seed 0. Training used AdamW with Ultralytics' default augmentation on a Google Colab T4 GPU, and it took about 1.4 hours.

Results on the same test split (2,443 images):

- **Fine-tuned YOLO11n** (6 classes, fruit type × fresh/rotten): mAP50 = 0.964, mAP50-95 = 0.828
- **COCO YOLO11n, zero-shot** (fruit type only: apple/banana/orange): mAP50 = 0.442, mAP50-95 = 0.321
- **Trivial baseline, predict nothing** (6 classes): mAP50 = 0.000, mAP50-95 = 0.000

Fine-tuning changed two things: the last layer went from COCO's 80 classes to my 6 fruit classes, and the model learned fresh vs. rotten from my 16,609 training images. The gap with COCO was about what I expected on the real task, because COCO cannot predict rotten at all, and that means its rotten score is always 0, so every fruit falls below the lower threshold and gets shipped. For my business decision, COCO zero-shot behaves the same as "do nothing, ship everything," which costs €60 per 1,000 fruits in my cost analysis. I was a bit surprised that COCO only reached 0.44 even on fruit type, but that makes sense since these are close-up images where the fruit fills most of the frame, the backgrounds are plain and unlike COCO's normal scenes, and rotten fruit can look unusual on top of that. The fine-tuned score is very high too, and that is a little suspicious because of the possible near-duplicates between train and test, so the test result looks strong, but it does not prove the model will work just as well on a real conveyor line.

## Per-class evaluation and failure modes

Per-class results on the test split:
- **fresh_apples**: precision 0.814, recall 0.900, AP50 0.929
- **fresh_banana**: precision 0.974, recall 0.955, AP50 0.977
- **fresh_oranges**: precision 0.964, recall 0.961, AP50 0.978
- **rotten_apples**: precision 0.900, recall 0.854, AP50 0.943
- **rotten_banana**: precision 0.968, recall 0.937, AP50 0.961
- **rotten_oranges**: precision 0.991, recall 0.973, AP50 0.994

The best class is rotten_oranges, with the highest AP50 (0.994), very strong recall (0.973) and precision of 0.991. Bananas and oranges are all above 0.96 AP50, so they seem easier for the model, probably because they are usually large in the image and rotten bananas change colour very clearly. The class that drags the result down is the apples: fresh_apples has the lowest precision (0.814), and rotten_apples has the lowest recall among the rotten classes (0.854). That matters most because missing rotten apples is the expensive mistake. What is surprising is that rotten_apples is also the biggest class in the dataset (15,055 instances), so the problem is not lack of data. It is more likely because apples are smaller in the image, there are often many apples together, and apple rot can be just a small spot.

In the detection confusion matrix, 271 fresh apples were predicted as rotten apples, which are false positives at €0.20 each. The more expensive error is the 53 rotten apples predicted as fresh apples, which are false negatives at €2.00 each. For apples, 271 × €0.20 = €54.20 in false positives, while 53 × €2.00 = €106 in false negatives. There are many more false positives, but the false negatives cost about twice as much in total, so recall on rotten fruit matters more. For oranges, 74 of 336 fresh oranges were predicted as rotten, split between rotten oranges and rotten apples. Also, 37 rotten oranges were predicted as rotten apples, but that still counts as rotten, so the business decision does not change. Only 29 objects were missed completely. The large background column is mostly low-confidence boxes that my thresholds filter out. At the decision level, the model works better, because doubtful cases go to a person instead of being shipped or discarded, and the decision uses the highest rotten score in each image, so many box-level confusions disappear. For rotten images (1,407), 1,325 were discarded, 78 went to review, and only 4 were shipped. For fresh images (1,036), 0 were discarded, 13 went to review, and 1,023 were shipped.

The 4 shipped rotten images are the real failure cases. In all four, the rotten score was 0, so the model predicted fresh with high confidence (0.74-0.96) and the correct fruit type. One clear failure was an apple with brown rot only on the lower side. Another borderline case was a slightly wrinkled orange with no visible mould. A third case was a bitten apple that was labelled rotten, but the model said fresh with 0.96, and it may actually be right, which suggests the label is questionable. The review band catches uncertainty, but not confident mistakes. This is why cleaner labels matter: not every “error” is a model mistake, and some of them may just be bad annotations.

## Cost matrix and operating point

The cost matrix uses three outcomes. A false positive means good fruit is discarded, and I set that at €0.20, which comes from the fruit value itself, based on Mercamadrid wholesale prices, average piece weight, and the fruit mix. A false negative means rotten fruit is shipped, and I set that at €2.00, about €0.20 for the fruit plus €1.80 in expected follow-on cost. For example, if 1 in 10 shipped rotten fruits leads to an €18 claim, that is €1.80 per rotten fruit on average. Human review costs €0.0174 per fruit, based on about €12.50/hour and 5 seconds per check, and the €12.50/h comes from the 2026 Spanish minimum wage plus employer social security. Correct decisions are assumed to cost €0, and the reviewer is assumed to always be right.

I used an assumed base rate of 3% rotten, which is lower than the test set, where about 58% of the images are rotten, so the costs use the test error rates reweighted to the assumed 3%, not test precision directly. The 3% is reasonable because the FAO reports around 5.5% post-harvest loss and 7.5% distribution loss for fresh produce in Europe, and this fruit has already been sorted once at origin.

I swept thresholds from 0 to 1 in steps of 0.02, trying every lower and upper threshold combination. The model-alone curve shows that at threshold 0, everything is discarded, which is very expensive at about €194 per 1,000 fruits. At threshold 1, everything is shipped, which costs €60 per 1,000 fruits. The cheapest single-threshold point is around 0.50, at €1.69 per 1,000 fruits. My chosen operating point is 0.22 / 0.80, which gives €0.41 per 1,000 fruits.

I expected the €2.00 false negative cost to be the assumption that would change the answer most, so I tested it. With €1, the best band is 0.28-0.80, and the cost is €0.31 per 1,000 fruits; with €2, it is 0.22-0.80 and €0.41; with €5, it stays 0.22-0.80 and rises to €0.67. So the band barely moves, and even at €5 the cost is still far below having a person check everything (€17.36), which makes the decision robust to this assumption. The bigger risk is that the test error rates are optimistic; if the model makes more mistakes on a real conveyor, more rotten fruit would be shipped, the costs would go up, and the review band might need to be wider.

## What a person still has to check

With my chosen thresholds, 0.22 / 0.80, fruit below 0.22 is shipped, above 0.80 is discarded, and anything in between goes to a person. At a 3% rotten base rate, about 1.4% of all fruits are sent to review, which is around 1,383 checks per day if the line processes 100,000 fruits a day. At 5 seconds per check, that is about 1.9 hours a day, so it is only a part-time task for one person. If I use 2 seconds instead, it drops to 0.77 hours a day, and the thresholds do not change.

That is realistic enough for a real line, because less than two hours of review a day is manageable. But it still depends on a few assumptions: the reviewer is assumed to be perfect, and fatigue or busy periods could reduce accuracy. Also, 5 seconds per fruit is only an estimate, so the workload could change in practice.

Compared with the alternatives, the review band is much better. Doing nothing means shipping everything, which costs about €60 per 1,000 fruits or €6,000 per day. A person checking everything costs about €17.36 per 1,000 fruits, or roughly €1,736 per day, and at this volume that is around 139 hours a day, or about 17 people. The model alone at threshold 0.50 costs €1.69 per 1,000 fruits or about €169 per day, while model + review band costs €0.41 per 1,000 fruits or about €41 per day. So the review band cuts the model-only cost by about 4 times; it adds a safety net for uncertain cases, but checking everything is far too expensive.

## Limitations & next steps

- Test scores are probably optimistic: The dataset includes rotated copies like rotated_by_75_Screen-Shot..., so near-duplicates may appear in both train and test, which means the number of truly new examples could be lower than it looks. Fix: remove near-duplicates before splitting, for example with perceptual hashing, and split by original photo.
- The images do not really look like a conveyor: Many are stock-style photos, ImageNet images, screenshots, or plain-background fruit photos, not a real line camera view. Fix: collect real conveyor images in a pilot and finetune on those.
- Some labels look questionable: There is a bitten apple labelled rotten, and some boxes cover almost the whole image or are nearly zero-sized. Fix: audit and relabel, at least the test set, and remove images that are out of scope.
- Confident mistakes are not caught by the review band: The apple that was rotten only on its lower side was predicted as fresh with 0.74 confidence, so the review band would not stop it. Fix: use rollers or multiple cameras so every side is visible, add more examples of subtle rot, and do random checks on shipped fruit.
- The cost figures are not fully verified: The €2.00 false negative cost, the 3% base rate, and the perfect reviewer are all assumptions. Fix: measure the real claim rate, rotten share, reviewer speed, and reviewer accuracy in a pilot with the distributor.

## Generative AI use disclosure

I used Claude (Anthropic) during this assignment. It suggested several business problems and compared datasets on Roboflow Universe, and I chose the problem and the dataset I liked. Claude wrote the code for dataset_stats.py, train.py, costs.py, and app.py, set YOLO11n as the model in the training code, and explained why the nano version fits a conveyor line. It also helped me fix setup errors caused by my Intel Mac, Google Colab, and missing libraries, and explained concepts I was unsure about, such as fine-tuning. It helped me find public sources for my cost figures (Mercamadrid prices, the 2026 Spanish minimum wage, and FAO loss estimates), but I chose the final values. For the report, I wrote each section myself; Claude gave me comments on my drafts with the results I was missing and corrections to punctuation and structure. The business story, the cost assumptions, the choice of the 5-second base scenario, and the interpretation of the results are my own. I also used DeepL to translate parts of my drafts from Spanish to English.