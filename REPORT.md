# Assignment 2 Report

- **Name**: Rakan Hourani
- **Student ID**: 19907
- **Email**: rhourani.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Business problem, and the cost of being wrong

The model was created for a fruit packing line. Every apple, banana and orange that goes past ends up with one of three outcomes: it gets packed, it gets thrown away, or it gets sent to a worker at a review station who checks it by hand. The sorting line acts on the model’s score, and the worker handles the cases where the model is not sure.

The two mistakes are not equally bad. If a rotten fruit is packed, the rot spreads to the fruit around it in the box (‘nesting’), and the buyer might complain or send the box back. Based on assumptions, I put this at €2.00 per rotten fruit packed. If a good fruit is thrown away, the packer only loses that one fruit, which is about 0.2 kg at around €0.95/kg, so €0.20. This means a missed rotten fruit is about 10 times worse than a false alarm.

This affects every decision after it. I defined the positive class as rotten (any of the three rotten classes). The model should lean towards flagging fruit, so recall on rotten fruit matters more than precision. I think that the threshold has to be chosen by cost and not accuracy, because accuracy treats both mistakes as equally bad, when they are not.

## Dataset

The main dataset is from Roboflow (workspace fruit-onxba, project rotten-fresh-fruitclassification, version 2), and I used the 1 519 original images without Roboflow’s augmented copies (https://universe.roboflow.com/fruit-onxba/rotten-fresh-fruitclassification). I also used a second dataset only for checking (workspace objectdetectionlearning-4empo, project food-freshness-classification-zu3j5, version 1), which has 1 188 photos with the same six classes but taken by other people. The model never trains on it.

One image is one fruit, labelled as one of six classes: fresh or rotten apple, banana and orange. I picked this dataset because it is the same decision the line has to make, one fruit at a time, rotten or not, and it is on fruit that Spanish packers actually handle.

I trained it on Codespaces CPU, using yolo11n-cls from the pretrained checkpoint, 20 epochs, image size 224, batch 32 and seed 42 (12 minutes).

## What the dataset's composition implies

| Class | Train | Valid | Test | External |
|---|---|---|---|---|
| fresh apples | 145 | 33 | 16 | 199 |
| fresh bananas | 137 | 50 | 17 | 198 |
| fresh oranges | 132 | 28 | 18 | 195 |
| rotten apples | 169 | 48 | 24 | 196 |
| rotten bananas | 363 | 82 | 62 | 200 |
| rotten oranges | 139 | 37 | 18 | 200 |
| **total** | **1,085** | **278** | **155** | **1,188** |

The test split is small, with 155 images and only 16 to 24 per class apart from rotten bananas, so one mistake moves a class’s recall by 4 to 6 points and small differences between classes can’t really be trusted. It is also imbalanced, because rotten bananas are a third of the training data, which means the overall score leans a lot on bananas.

The dataset is about 65% rotten, while a real line is only about 3% rotten, so I re-weighted all the costs to a 3% base rate so that the numbers describe the line and not the dataset. It also doesn’t look like production, since every image is a close-up where the fruit fills almost the whole frame (640x640, filling about 94% of frame), whereas a packing line has many fruit on a belt, under different light and seen from one side. This is why I also tested the model on the second dataset.

Only a handful of images are almost identical to a training image, so leakage does not explain the scores. I also removed one training image labelled ‘Unlabeled’, because that class doesn’t exist in validation or test and a class that can’t be tested should not be learned.

## Finetuning and baselines

| Model | Images | Accuracy | Rotten precision | Rotten recall |
|---|---|---|---|---|
| Fine-tuned yolo11n-cls (test) | 155 | 97.4% | 99.0% | 100% |
| Baseline: majority class (test) | 155 | 40.0% | 67.1% | 100% |
| Baseline: COCO zero-shot (test) | 155 | 14.2% | – | 0% |
| Fine-tuned, external photos | 1,188 | 56.2% | 83.1% | 57.9% |

I chose image classification instead of detection, because the decision is made per fruit and each image is only one fruit, so boxes don’t add anything.

On the test images, the fine-tuned model gets an accuracy of 97.4% and catches every rotten fruit (100% recall). I compared it to two baselines. The first always predicts the majority class (‘rotten bananas’), which gets 40% accuracy. The second is the COCO yolo11n model used zero-shot, which only gets 14% accuracy and doesn’t catch any rotten fruit. On the external photos, however, the fine-tuned model only gets 56% accuracy.

The fine-tuned model beats both baselines by a lot. The majority baseline catches every rotten fruit, but only because it calls everything rotten, which would mean throwing away all the good fruit as well. COCO knows what an apple, banana or orange is but has no idea what ‘rotten’ is, and it only found a fruit in 34% of these close-up photos.

The gap on the test split was bigger than I expected, but it does not hold. On the photos from the other dataset, accuracy drops and the model misses 42% of the rotten fruit. This shows that the test split looks too much like the training data to tell us how the model would really perform.


## Per-class evaluation and failure modes

| Class | Test precision | Test recall | External precision | External recall |
|---|---|---|---|---|
| fresh apples | 1.00 | 0.81 | 0.60 | 0.56 |
| fresh bananas | 0.90 | 1.00 | 0.46 | 0.89 |
| fresh oranges | 0.94 | 0.94 | 0.51 | 0.54 |
| rotten apples | 0.96 | 1.00 | 0.76 | 0.37 |
| rotten bananas | 1.00 | 1.00 | 0.66 | 0.96 |
| rotten oranges | 1.00 | 1.00 | 0.41 | 0.06 |

The full confusion matrices for the model and both baselines are in the dashboard's tab.

On the test split the weakest class is fresh apples, with a recall of 81% (3 out of 16 wrong). All three rotten classes have 100% recall, but with so few images per class it is not strong evidence.

On the external photos the model breaks, and not evenly. Bananas still hold up well, with a recall above 89% for both fresh and rotten bananas, while fresh apples and oranges drop to around 55% and are often called fresh bananas. The worst are rotten apples at 37% and rotten oranges at only 5.5%, and many of them are called fresh (87 rotten oranges were called fresh).

The overall number is carried by bananas, since banana rot is large and dark and rotten bananas are the biggest class. The failure that matters most is rotten apples and oranges being called fresh, because that is exactly the €2 mistake, and it happens most for the fruit where the rot is the smallest. The model also mixes up the type of fruit, which suggests it partly learned what the first dataset’s photos look like instead of the fruit itself.

## Cost matrix and operating point

For the cost matrix, a rotten fruit that gets packed which is a false negative, costs €2.00 because the rot spreads in the box and leads to complaints and claims, while a good fruit that gets thrown away, a false positive, costs €0.20, which is 0.2 kg at €0.95/kg. A rotten fruit removed and a good fruit packed both cost €0. I used a base rate of 3% rotten, since a packing-house study found that rot was below 3.5% at every stage.

I couldn’t choose the thresholds on the validation split, because it looks too much like training and the model was almost never unsure there. Instead, I split the external photos into two halves, where the tuning half chooses the thresholds and the check half measures them.

With a single threshold, where the model decides on its own, the lowest cost is at 0.89. On the check half this costs €38.83 per 1 000 fruit compared to €60 for doing nothing, but it still packs about 3 400 rotten fruit a day.

The €2 miss cost is the assumption that changes the answer the most, since it is the figure I am least sure about. If it falls to €0.50 and only 1% of fruit is rotten, doing nothing becomes cheaper than the model with review. Whenever the miss cost or the rot rate goes up, the cheapest plan is to send more fruit to people.


## What a person still has to check

Reviewers are paid for full 8-hour shifts at about €13 an hour, and one check takes about 3 seconds, so one person can check 9 600 fruit per shift on a line that handles about 200 000 fruit a day. The band with the lowest total daily cost on the tuning half is: pass below 0.02, review from 0.02 to 0.99, and fail from 0.99. On the check half it sends 63% of the fruit to a person, which needs 14 reviewers.

This is the cheapest option at €10.51 per 1 000 fruit, and it only packs about 60 rotten fruit a day. Having a person check every fruit costs almost the same (€10.92) but needs 21 reviewers, while the model alone costs €38.83 and doing nothing costs €60. With fewer than 10 reviewers the cost rises quickly, up to about €35 per 1 000 with only one.

I don’t think this is realistic. 14 people on one review station is quite a lot, and the model only saves about 4% compared to having people check every fruit. It also throws away about 2 600 good fruit a day that the model is more than 99% sure are rotten, and no reviewer ever sees those. So I would not ship this model on photos like these. The next step would be to keep checking by hand and collect photos from the real line. 

## Limitations & next steps

The biggest limitation is that the training photos don’t look like ones from a packing line, which is shown by the 56% accuracy on the second dataset. I would address this by photographing and labelling fruit from the actual line, fine-tuning on those and running the cost analysis again. Rotten apples and oranges are also missed the most, especially when the rot is small or early, so I would add more examples of early rot and use a threshold for each class instead of one for all fruit.

Another limitation is that one photo only shows one side of the fruit, so rot on the other side or inside can’t be seen. This could be solved with several cameras and the fruit rotating, or with infrared, which is what commercial sorters already use.

The €2 miss cost is also an assumption I made, and the sensitivity check shows how much the answer depends on it, so I would replace it with the packer’s real data. Lastly, the reviewer is assumed to always be right and the test sets are small, so I would measure how accurate a real reviewer is in 3 seconds and test on more images.

## Generative AI use disclosure

I used Claude in this assignment. It suggested possible problems, helped prepare the dataset and generate the code for the training notebook, baselines, the threshold and review analysis and the Gradio dashboard, which I ran on  Codespaces and had explained to me step by step.

It researched and proposed the cost and operating figures with sources, which I reviewed and accepted. It also suggested testing on a second dataset and tuning the thresholds on half of it, which I agreed to after seeing the first results.

For the report, I then used it to check and reformat what I wrote and to make sure I responded to everything being asked.