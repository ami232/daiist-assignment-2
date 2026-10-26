# Assignment 2 Report
- **Name**: Andrea Saxod
- **Student ID**: 18996
- **Email**: asaxod.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem and the cost of being wrong
* The decision this model supports is whether the system tells a lifeguard if a swimmer looks like they are drowning. Currently, a lifeguard must decide based on watching the water. This model is meant to assist the lifeguard in their duties.
* The model functions in an indoor pool with four cameras, each in a corner of the pool. Each camera takes one picture every 5 seconds, and the model analyzes each image. If a swimmer appears to be drowning, the system alerts the lifeguard to check and proceed.
* In this model, the positive class will be for "drowning". Each picture is one case, and the pool is open from 8:00 to 18:00. This means the system will review 28,800 pictures a day (4 cameras x 720 pictures per hour x 10 hours).

The two mistakes cost very different amounts: 
* A false alarm: The lifeguard looks at a swimmer who is fine, which takes about 30 seconds. This costs €0.117, because a lifeguard costs the pool around €14.05 an hour.
* A missed drowning: This consists of a picture that shows someone in trouble and nobody is told. I price this at €42.50. A death costs around €170,000 in compensation. After research, I estimated that about 0.15% of serious incidents end in a death. An incident lasts about 30 seconds, so it shows up in about 6 pictures, so I split the cost across 6 pictures: 170,000 x 0.0015 / 6 = €42.50.

In short, one miss is about 360 times worse than one false alarm. Drowning, however, is very rare: only about 6 pictures in a million show it. This shapes every choice below: how often it really happens, which score I lead with (drowning recall), where the threshold goes, and how few pictures a person needs to check.


## Dataset
I used a public Roboflow Universe dataset: workspace cuizh, project water-rescue-dataset, version 1 [water-rescue-dataset Model > Overview](https://universe.roboflow.com/cuizh/water-rescue-dataset). This dataset is an object detection dataset, meaning the model draws a box around each person and labels the class.
The data has three classes: drowning, float, and swimmer. One image is a photo or video frame of a water scene, with each person boxed and labeled.
This dataset fits the problem because it includes the class the pool cares about (drowning) alongside people who are fine (floating people and swimmers). The model needs to tell distress apart from normal swimming, and the footage doesn’t look like it comes from a fixed pool camera. The next section explains this in more detail.

## What the dataset's composition implies
The data is split into 3,846 training images, 369 validation images, and 181 test images. The training set consists of the original images, each copied three times with a minor change (around 1,282 different images). The validation and test sets have no copies. We resized every image to 512×512.

| Class | Boxes in train / val / test | Test images with it | Typical box size (share of the image) | Where most images come from |
|---|---|---|---|---|
| drowning | 1,623 / 157 / 80 | 80 | 38% | 1,852 of 1,860 from the web or video |
| float | 1,069 / 143 / 44 | 7 | 0.6% (70% of boxes are under 1%) | 1,156 of 1,256 from drones |
| swimmer | 2,951 / 284 / 144 | 94 | 6.6% | 2,627 of 3,379 from a webcam |
I had to guess the image source from the start of each filename, so this is only a rough guide.
This means how much I can trust my results:
* **Each image only has one class:** The model never observed a drowning person next to swimmers or empty water. The test set has 80 drowning images and 101 non-drowning images.
* **The class and image source go together:** Drowning images are primarily from the web, while float images are from drones, and swimmer images are from webcams. The model may learn partly from the type of pictures, not what the person is doing. The test set follows this same pattern, so it cannot show this problem.
* **Float has very little data **: Only 7 of the test images contain it, and its boxes are tiny. I trust the float class scores the least.
* **Drowning pictures are close-ups**: The typical box covers 38% of the picture, which a pool camera does not see. I haven't had the opportunity to test how well it works on real camera pictures.
* **The test set is too close to the training set**: 49 file names appear in both the train and test sets. I also ran a check that finds near-duplicate pictures. It flagged 95 of the 181 test images as close to a training or validation image, with 35 very close. When I remove these images, the main score (mAP50) drops from 0.878 to 0.874, 0.868, and 0.832 as I count more images as “too close.” A close match does not prove a copy. Some may be different frames from the same scene, which would also make the test unfair. I did not run this check on real photos.
* **The dataset is about 44% drowning pictures**: a real pool almost never has drowning situations. So the precision I measure on the test set will not hold in a real pool.
For these reasons, I estimate the mAP50 range at 0.83 to 0.88. 

## Finetuning and baselines
In my case, I fine-tuned YOLO11n, a small model that detects and labels objects in images. I chose detection because the labels are class-labeled boxes. I trained it on my laptop using an NVIDIA RTX 4050. Settings were: pictures at 512 pixels, 16 pictures at a time, seed 0, up to 100 epochs, and stop if there is no improvement for 10 rounds. In this case, training ran for 56 epochs, with the best at epoch 46 (mAP50 0.840 and mAP50-95 0.360 on validation). It stopped 10 epochs later, as planned, saving the best model to 'weights/best.pt'.

I tested three models on the same 181 test images:
| Model | mAP50 | mAP50-95 |
|:---|---:|---:|
| My finetuned YOLO11n | 0.878 | 0.378 |
| Pretrained COCO model, with no training on my data | 0.004 | 0.001 |
| A model that predicts nothing | 0.000 | 0.000 |
mAP50 is a score from 0 to 1 that counts a box as right if it covers at least half of the real box. mAP50-95 is stricter: it also requires boxes to match almost exactly.
My model beat both baselines by a very large margin. Part of the gap is built in: the COCO model has no drowning class and no float class, so it scores zero on both. The fairest comparison is on swimmers. For this, I treated COCO's "person" as "swimmer". It found only 3 of 144 swimmers and drew 48 wrong boxes. My model, however, scored 0.867 on swimmers. A model that predicts nothing is the lowest bar any useful model can beat.
My first run on 50 epochs scored 0.842. The final run scored 0.878 on the test set. That is higher than its 0.840 on validation, which is a little odd, since we picked the best epoch using validation. This could be caused by test pictures being closer to the training pictures than the validation pictures. Random luck with only 181 test images could have played a role, too. I am not exactly sure which.

## Per-class evaluation and failure modes
In this case, precision is how many alerts were correct. Recall means how many real cases were caught. AP50 is a 0-to-1 score per class. The numbers below use a confidence of 0.25.
| Class | Real boxes in test | AP50 | Precision | Recall | Right (TP) | Wrong alarms (FP) | Missed (FN) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| drowning | 80 | 0.845 | 0.769 | 0.750 | 60 | 18 | 20 |
| float | 44 | 0.921 | 0.851 | 0.909 | 40 | 7 | 4 |
| swimmer | 144 | 0.867 | 0.772 | 0.847 | 122 | 36 | 22 |

Confusion matrix (each row is the truth, each column is what the model said):
| Truth / Model said | drowning | float | swimmer | nothing |
|:---|---:|---:|---:|---:|
| drowning | 60 | 0 | 3 | 17 |
| float | 0 | 40 | 0 | 4 |
| swimmer | 1 | 0 | 122 | 21 |
| empty background | 17 | 7 | 33 | – |
Drowning is the class that matters, so my main number is its recall: 75% (60 of 80 caught), with precision 0.77. The overall 0.878 looks better because float scores 0.921, but float has only 7 test images. Drowning has the lowest recall in every test version: 0.75 on the full test and 0.69 on the cleaned test.

Where the model fails: 
* **Missed drowning is mostly "nothing found".** Of the 20 missed drowning boxes, 17 got no box at all. Only 3 were called swimmer. Only 1 swimmer was called drowning. So the model rarely mixes up drowning and swimmer. Its mistakes are missed boxes and false alarms on an empty background (17 false drowning boxes and 33 false swimmer boxes).
* **Boxes are loose.** mAP50 is 0.878 but the stricter mAP50-95 is only 0.378. The model finds the person, but the box doesn't fit tightly. For an alert, where the box sits matters little, so my cost model only asks, "Is there drowning in this picture?"
* **It gets worse on the cleaned test** (86 images, after removing near copies): mAP50 0.832, drowning recall 0.69, float recall 0.95 (20 boxes), swimmer AP50 0.760 (68 boxes).

## Cost matrix and operating point
One case is one picture. A picture is an alert when the model's highest drowning confidence reaches the threshold. This table shows each cost and where it came from:
| Input | Value | Where it came from |
|:---|:---|:---|
| Pictures per day | 28,800 | 4 cameras, one picture every 5 s, 10 open hours |
| Time between pictures | 5 s | Lifeguards follow a "10/20" rule: notice within 10 seconds, reach the swimmer within 20. Drowning can take as little as 30 seconds. At 5 s, the system gets two looks in the first 10 s. I use this as a guide, not as proof |
| Cost of the monitor per hour | €14.05 | The 2026 pay table for pool workers gives a lifeguard €10.63 an hour. I multiplied by 1.3215 for the employer's social security (32.15%). This pay is a minimum |
| One human check | €0.0195 | €14.05 × 5 s ÷ 3,600. The 5 s is my guess |
| False alarm or correct alert | €0.117 each | €14.05 × 30 s ÷ 3,600. The 30 s is my guess |
| Compensation for one death | €170,000 | Between two real Spanish pool cases: €166,615 paid by insurers (Ripoll, 2017) and €180,000 ordered by a court (Marina d'Or, 2015). Only five cases, mostly children, compensation only |
| Chance a serious incident ends in death | 0.15% | The middle between Hong Kong public pools in 2003 (3 deaths in 96 rescues) and a US water park study (4 deaths in 56,000 rescues) |
| Pictures per incident | 6 | An incident lasts about 30 s, and I take a picture every 5 s |
| Cost of one missed picture | €42.50 | €170,000 × 0.15% ÷ 6 |
| How often it really happens (the base rate) | 5.7 in a million pictures | 10 incidents a year × 6 pictures ÷ 10,512,000 pictures a year |
| Cameras | €1,000 in total, €200 a year | Four 4 MP cameras at about €250 each, spread over 5 years |
| The human checker | always right | An optimistic guess |
I do not put a price on a life. The €42.50 is only a weight to compare choices. Below, I also test a very high cost.

I picked 10 incidents a year because the studies I found report about 2.7 rescues per pool per year in Hong Kong and about 30 per pool per year in four French pools. I used rescues, not deaths, because the model spots people in trouble, and a lifeguard acts on trouble. Deaths alone are about one per thousand pool-years, which would be the wrong rate.

The test set is about 44% drowning, so its precision tells me nothing about a real pool. So I use only two rates from the test set at each threshold: how many drowning pictures we catch (TPR) and how many fine pictures we misclassify as drowning (FPR). Then I weigh them with the real rate of drowning:

> cost per picture = base rate × [TPR × cost of a correct alert + (1 − TPR) × cost of a miss] + (1 − base rate) × [FPR × cost of a false alarm]

I tried every threshold from 0 to 1 in steps of 0.005 and took the cheapest. The cheapest choice is to alert at confidence 0.365, with no review band. At 0.365, the system catches 73.75% of drowning test pictures and raises no alert on any of the 101 other test pictures. It would raise about 0.12 alerts a day (about one every eight days). It costs about €675 a year in losses and work, plus €200 a year for cameras. The search stops at 0.365 because the next lower threshold lets one fine test picture through. One picture out of 101 gives a 1% false-alarm rate, which would be about 285 false alerts a day in a real pool.

The guess that would change my answer most is the cost of a missed picture, which depends on the 0.15% chance of death. The system beats doing nothing once a miss costs more than about €0.12 (counting running costs only), and more than about €4.73 once it also pays for its cameras. That is a 0.017% chance of death. If the cost of a miss is as low as €2.02 (0.007% chance of death) and there are 3 or 10 incidents a year, doing nothing is cheaper. If the cost is higher (€885 or €170,000 per miss), the alert threshold stays at 0.365 and only the review band opens. I tried all 12 combinations of four miss costs and 3, 10, or 30 incidents a year, and the alert threshold was 0.365 every time.

## What a person still has to check
My two thresholds are 0.365 and 0.365. Below it, a picture passes. At or above it, a picture gets an alert. The middle review band is empty. In my main case, no pictures go to a person for checking. Only about 0.12 alerts a day go to the lifeguard. I chose this because it is the cheapest point on the sweep. A review band costs work time for every picture it catches. A band from 0.295 to 0.365 would cost about €2,900 a year in my main case, more than doing nothing, because checking 285 pictures a day costs about €2,030 a year and finds almost no extra drowning.

The band opens when a miss costs more. At 5 seconds a check:
| Case | Review band | Checks per day | Time spent checking | Cost of my system per year | Cost of doing nothing per year |
|:---|:---|---:|:---|---:|---:|
| Main case, 10 incidents a year | none | 0 | none | €875 | €2,550 |
| Costly misses (€885 each), 10 incidents a year | 0.295 to 0.365 | 285 | about 24 minutes a day | €9,542 | €53,125 |
| Costly misses, 30 incidents a year, or the extreme case | 0.03 to 0.365 | about 1,996 | about 2.8 hours a day | about €14,400 | €159,375 or more |
Is this realistic? Yes. Even the biggest load is under 3 hours a day. The lifeguards already on duty can take the alerts and any checks. A person who only watches the screen would cost about €51,282 a year and need 2.06 people to cover the 3,650 open hours. That is far more than the €2,550 a year that doing nothing costs, so I do not recommend it. I assumed the human checker is always right, which is optimistic, so these review numbers are best-case.

My design compared with the three other choices, in my main case:
| Choice | Cost per year |
|:---|---:|
| Do nothing | €2,550 |
| The model decides alone, alert at 0.365 (my design, since the review band is empty) | €875 |
| A person checks every picture | €205,337 |
| Alert on every picture | €1,230,980 |
Checking every picture would take 28,800 × 5 s = 40 hours of checking a day. That means about four people watching at the same time during the 10 open hours. This costs far more than the model.

## Limitations & next steps
* I did not measure the false alarm rate. No fine test picture caused an alert at 0.365. That only tells me the true rate is below about 3% (3 divided by 101), which could still be about 850 false alerts a day. Next: record several hours of real pool video and count the false alerts per hour.
* Class and image source go together, and each image has one class. Next: build a test set from other sources, with mixed scenes and empty water, and see if the score holds.
* The test set is too close to the training set, and I did not check my "almost the same" test on real photos. Next: look at the flagged pairs by eye, then rebuild the splits so no scene is on both sides.
* The data does not look like a real pool camera. The drowning pictures are close-ups, 44% of them show drowning (a real pool is about 6 in a million), and there is no indoor pool footage. Next: collect and label pictures from the real cameras.
* One case is one picture. I did not test several pictures in a row from one incident. A picture also counts as an alert wherever the box is. Next: test on video clips and ask for agreement across several pictures.
* The cost numbers are thin. The compensation comes from five Spanish cases, covers only compensation (not legal fees, closing the pool, insurance or reputation) and only deaths. The death rate and incident rate come from studies outside Spain that define "rescue" differently. Next: ask a pool and its insurer for real incident records and claim costs.
* Some numbers are guesses: a perfect human checker, 5 seconds a check, 30 seconds a response, 365 open days. Next: time real lifeguards, and measure how often a checker gets a 5-second picture wrong.
* The boxes are loose (mAP50-95 is 0.378). Next: look at the labels to see if they are loose before changing the model.

## Generative AI use disclosure
I use AI for boilerplate code such as loading data, saving models, and building the dashboard. All decisions, such as which dataset to use, how to frame the business model, finetuning, and so on, are my own.

