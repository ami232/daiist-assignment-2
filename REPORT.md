# Assignment 2 Report
- **Name**: María Martínez González
- **Student ID**: 18752
- **Email**: mmg.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong
The model looks at a photo from a construction-site camera and decides whether each worker is wearing a hard hat. The site supervisor acts on it: if the model finds a bare head, the supervisor is alerted and goes to correct the situation before the worker keeps working.

A false negative is a worker without a helmet that the model misses. This is the worse mistake, because nobody is alerted and the worker is exposed to a head injury, a regulatory fine and possible stoppage of the work. A false positive is a worker with a helmet that the model flags by mistake. The cost is small: the supervisor walks over, sees the helmet and walks back, which costs a few minutes.

So I treat `head` as the positive class and I will favour recall: I prefer many false alarms over one missed violation. This is why the threshold will be low, and why uncertain cases will go to a person for review and not be decided automatically. The euro values for each error are in the costing section.

## Dataset
- Workspace: iote-7emb7
- Project: hard-hat-7785r
- Version: 1
- Link: https://universe.roboflow.com/iote-7emb7/hard-hat-7785r/dataset/1
- License: CC BY 4.0

Each image is a photo of a construction site or industrial area with one or more workers in it. The dataset has two classes. `helmet` marks a worker who is wearing a hard hat. `head` marks a bare head, so a worker with no hard hat. In my problem `head` is the positive class, because it is the safety violation that a supervisor needs to be told about.

I chose this dataset because the images look like what a site camera would really see: workers at different distances, angles and lighting, indoors and outdoors. The `head` class is exactly the violation I want to catch, so the dataset matches the decision and does not just look similar.

The version I use is v1 with 7,035 images, resized to 640x640 with no augmentations. The split is 4,221 train (60%), 2,111 validation (30%) and 703 test (10%). I did not fork the dataset, so anyone can download the same version.

## What the dataset's composition implies
Counting the boxes in the label files, the train split has 4,131 `head` and 11,717 `helmet` instances, the validation split has 1,838 and 6,044, and the test split has 708 and 1,986. So `head` is about a quarter of all boxes. That is plenty of examples for both classes, so I think the per-class numbers can be trusted. The test split is still small though (703 images, 146 of them with a bare head), so my results could move a few points with different images.

The objects are small. The median `head` box covers only 0.6% of the image and the median `helmet` box 0.8%. This is why I train at 640 pixels and not smaller: at low resolution these boxes would be just a few pixels wide and the model would miss many of them.

The test split has 21% of images with a bare head, which is much higher than what I would expect on a real site (about 5%). That means precision and cost numbers measured on the test set will look worse than in real life, so I correct for it with the base rate in the costing section. Also, many images look like press or stock photos with watermarks, not like fixed cameras, so I can't conclude the model works on a specific camera at a specific site without testing on that site's footage. Finally, I found that some labels are wrong (see the failure section), so the test numbers are a little less reliable than they look.

## Finetuning and baselines
I use object detection, not classification, because one image can contain several workers, some with a helmet and some without, and the supervisor needs to know that at least one bare head is there. I finetune `yolo11n` (the smallest YOLO11 model) starting from its pretrained COCO weights. I chose the small one because it is fast to train and run, and because I wanted to see first whether a small model is already good enough before paying for a bigger one.

I trained on Google Colab with a T4 GPU, because my laptop has no GPU. I ran `python main.py train` with 50 epochs, 640 pixel images, batch size 16, seed 0 and the automatic optimizer of Ultralytics (AdamW). It took 1.0 hours. The weights are in `weights/best.pt`. To measure them again without training, run `SKIP_TRAIN=1 uv run python main.py train`.

I compare three things on the same 703 test images: my finetuned model, the pretrained COCO model used as it comes, and a trivial model that predicts nothing. My model reaches an mAP50 of 0.974 and an mAP50-95 of 0.661. COCO scores exactly 0 on both, because it has no `head` or `helmet` class, so it cannot get a detection right. To be fairer to COCO I also tried its `person` score as an alarm. Its best result catches 99% of the bare-head images but raises a false alarm on 85% of the others, and costs about 2,235 euros per 1,000 images. That is almost the same as flagging every image (2,500 euros). Predicting nothing costs 15,000 euros per 1,000 images, because every bare head is missed. My model with one threshold costs 755 euros.

The gap is bigger than I expected. COCO can find people but cannot tell a helmet from a bare head, and that difference is the whole task. One result surprised me: a person looking at every image costs less than my model alone (see the human review section). This depends on my assumptions about how much a review costs and how reliable a reviewer is.

## Per-class evaluation and failure modes
The positive class is `head`. On the test split, `head` has a precision of 0.935, a recall of 0.917, an AP50 of 0.967 and an mAP50-95 of 0.650. `helmet` has a precision of 0.958, a recall of 0.935, an AP50 of 0.980 and an mAP50-95 of 0.673. Both classes are close, so the overall mAP50 of 0.974 is a fair summary, but it leans on `helmet`, which has almost three times more test examples (1,986 against 708). `head` is the weaker class and also the one that matters for safety: about 8% of bare heads are missed at the detector's own working point.

The supervisor needs one answer per image, so I give each image the highest `head` confidence found in it. An image counts as a bare-head image if its labels contain at least one `head`. The test split has 146 of these and 557 without. With the cheapest single threshold (0.01) the model catches 143 of the 146 and gives 76 false alarms, with 481 correct passes and 3 misses. With a strict threshold of 0.5 it gives only 7 false alarms, but it misses 17 bare-head images.

In the confusion matrix (`outputs/confusion_matrix_normalized.png`, computed on the validation split) the model almost never mixes up the two classes: only 1 or 2% of the time. Its mistakes are other things. It sometimes draws a box where there is nothing, mostly a `helmet`, and it misses about 1% of the real objects.

Looking at the scores in `outputs/test_scores.csv`, the misses are few but they are the ones that cost money. Of the 146 bare-head images, 3 score below 0.01 and are missed even at my lowest threshold (images 006060, 006088 and 004275). Six score below 0.05, 12 below 0.3 and 17 below 0.5. So most bare heads are found with high confidence, and the problem is a short tail of hard cases.

I opened those three images and they were not mainly small or far away, as I had guessed. In one the workers are seen from behind with steel rebar crossing in front of them, in another a man wears a red cap that looks like headwear next to a man in a white hard hat, and in the third several people overlap in a crowded rescue scene. So the model seems to fail when the head is turned away, hidden, or covered by something that is not a hard hat. All three also look like press photos and not like a fixed site camera.

The false alarms have a tail too. Three clean images score above 0.8 for `head` (000805, 005382 and 005150) and 7 score above 0.5. I opened six of the images with the highest scores and found three different things. In four of them (000805, 005382, 005150 and 003084) I can see a person without a hard hat that the labels do not mark, so the model seems right and the label wrong. In one (000918) the bare heads belong to passers-by on a street behind a worker who wears a red helmet, so the model finds real bare heads but of people who are not workers. In one (003455), taken in a dark mine, I cannot tell whether the man at the back wears a helmet. I did not check the other false alarms, so I cannot say how many are label mistakes. It means my false alarm counts are probably too high, but not all of them, because some alerts for passers-by would still send the supervisor for nothing.

## Cost matrix and operating point
All costs are per case, where one case is one camera image. They are my own assumptions, not measured values.

A false positive costs 2.50 euros. The supervisor stops what they are doing, walks over and sees the worker has a helmet. I assumed 5 minutes at a labour cost of 30 euros per hour. A true positive costs the same 2.50 euros, because the walk is identical and the supervisor then fixes the problem. A true negative costs 0.

A false negative costs 300 euros. This is the expected cost of leaving a bare head unaddressed, not the cost of an injury itself. I assumed a serious incident (medical costs, work stoppage, possible fine) costs about 30,000 euros, and that a missed bare head in one image leads to such an incident about 1% of the time. 30,000 x 1% = 300. This is the number I trust least. If the incident cost or the probability falls by a factor of ten, a miss is only worth about 30 euros and the best threshold moves up.

I assume a base rate of 5%: on a real site only about 1 image in 20 shows a worker without a helmet. The test split has 21%, so it is enriched and I use 5% for the costing instead.

A person reviewing one image costs 0.25 euros, about 30 seconds at 30 euros per hour. Because a miss costs about 120 times more than a false alarm, a case should be flagged as soon as the probability of a violation is above roughly 1%. So I expect a low threshold.

I tried 101 thresholds from 0 to 1 on the 703 test images. The cost curve is in the dashboard and the numbers are in `outputs/cost_sweep.csv`. The cheapest single threshold is 0.01, at 755 euros per 1,000 images. The minimum is not sharp: at 0.05 the cost is 873 euros with only 32 false alarms, and at 0.5 it rises to 1,887 euros because 17 bare heads are missed. Some of the false alarms are label mistakes, which makes these costs a bit pessimistic, but some are passers-by who are not workers, which are real false alarms for the supervisor.

I use two thresholds. An image below 0.01 passes, an image at 0.5 or above sends an alert, and anything in between goes to a person. This costs 491 euros per 1,000 images, about a third less than the single threshold. I chose 0.5 as the top because above it the model is almost always right: only 7 of the 557 clean images scored that high. I could go below 0.01 and pay even less, but many more images would go to review (35% at 0.001), so 0.01 is a compromise.

The assumption that would change my answer most is the 300 euros for a miss. If it were ten times smaller, the thresholds would move up and fewer images would go to review. Also, only 146 test images have a bare head, so one more or one less miss changes the cost a lot.

## What a person still has to check
With thresholds 0.01 and 0.5, and my 5% base rate, 12.2% of images go to a person. I assume one site produces about 2,000 images a day, for example one camera taking a picture every 15 seconds during an 8-hour shift. That is about 244 reviews a day. At 30 seconds each it is about 2 hours of work, and with 7 productive hours in a day this is about 0.3 of one person. I think this is realistic: a supervisor or a safety assistant could do it as part of the normal day.

I compare four options, per 1,000 images. Doing nothing costs 15,000 euros, because every bare head is missed. The model deciding alone, with one threshold, costs 755 euros. The model with the human review band costs 491 euros. A person checking every image costs 375 euros: 250 for the reviews and 125 for the walks to fix the real violations, which the other options also pay for. The dashboard shows 250 for this option because it only counts the reviews.

So under my numbers the cheapest option is to have people check everything, which I did not expect. But I assumed the reviewer never makes a mistake, and checking everything would take about 17 hours a day, so about 2.4 people at one site. Nobody looks at 2,000 images a day without getting tired. If a tired reviewer misses more than about 1 bare head in 130 (0.8%), checking everything becomes more expensive than the model plus review. My model alone misses 3 of 146 (2%), and these misses are about two thirds of the 491 euros. So I would ship the model with the review band: it takes about an eighth of the reviewing time and still catches almost all bare heads. If I could prove that people never miss, I would not ship it.

## Limitations & next steps
- **The test split is not like a real site.** It has 21% bare heads against my assumed 5%, and many images are press or stock photos with watermarks. With more time I would test on footage from a real site camera and measure the real base rate.
- **Hard cases are about visibility, not only size.** The most-missed bare heads were seen from behind, covered by a cap or hidden in a crowd. With more time I would collect more of these cases from real site cameras, add them to training, and try a bigger model (`yolo11s`) or a higher image size, and compare.
- **One case is one image.** I flag an image if it has at least one bare head, so I don't track whether a worker is still without a helmet in the next frame. With more time I would combine several frames before alerting.
- **The costs are assumptions.** The 300 euros for a missed head depends on a guessed 1% chance of an incident. I would replace it with real incident and insurance data from the company.
- **The reviewer is assumed perfect.** In the real world a person makes mistakes, especially when tired. I would measure how often people miss a bare head on a sample of images and put that number in the cost.
- **Few positive examples in the test.** With only 146 bare-head images, one miss more or less changes the results. I would test with more images before trusting the exact threshold.
- **Some false alarms are label mistakes.** In four of the six highest-scoring false alarms I checked, I saw a bare head that nobody had labelled. With more time I would check all the false alarms above 0.5, fix the labels and measure again. The real cost per 1,000 images is probably a bit lower than I report.
- **The model finds bare heads, not workers without helmets.** In one image it flagged passers-by on a street behind a worker. On a real site a fixed camera would see visitors and passers-by, and each alert would send the supervisor for nothing. With more time I would limit the camera area to the work zone or add a class for workers, and measure how often this happens on real footage.

## Generative AI use disclosure
I used Claude as an assistant. It helped me set up the repository and Codespace, fix installation errors.