# Assignment 2 Report

- **Name**: Manuela Carreira
- **Student ID**: 18766
- **Email**: mcarreira.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

The goal is to decide whether each piece of fruit should pass, fail, or be reviewed before it is packed. The model predicts the probability that the fruit is rotten. The biggest problem is letting rotten fruit pass, because this costs about €2.00, while throwing away a good piece only costs around €0.25. This means a false negative is about 8 times more expensive than a false positive, so the model should be more careful about letting fruit pass when it is unsure. If needed, the fruit can also be sent for human review, which costs about €0.05 per piece. We assume around 5% of the fruit is rotten, with 20,000 pieces processed per day, and the operator can review up to 2,880 pieces per day.

## Dataset

The dataset comes from **Roboflow Universe**, workspace `fruit-classification-v0e3w`, project `fresh-and-rotten-zj6kl`, **version 6**. The link is: https://universe.roboflow.com/fruit-classification-v0e3w/fresh-and-rotten-zj6kl. Each image shows one single piece of fruit, and the six classes are fresh apples, fresh bananas, fresh oranges, rotten apples, rotten bananas, and rotten oranges. This dataset fits the problem directly because the business decision is also made one fruit at a time from an image, with the main goal of deciding whether that fruit is fresh or rotten before it goes into the pack.

## What the dataset's composition implies

The dataset has **9,457 training images, 2,064 validation images and 2,078 test images**, with between about **1,850 and 2,950 images per class overall**. The smallest test class still has **189 images**, so none of the classes is too small to use. The images are **640×640**, and because each photo shows one fruit by itself, the fruit takes up almost the whole image.

The main issue is that the test split does not really look like a real supermarket production line. Most images have a clean background and controlled lighting, while real images could have shadows, motion blur or several fruits close together. There is also important train/test leakage because many test images are augmented versions of photos that also appear in training. Because of this, the test accuracy may look better than the model would actually perform on new real-world images. I also cannot make strong conclusions about unseen oranges because none of the clean test images are oranges.


## Finetuning and baselines

I used **YOLO11n-cls** for an image classification task. Classification makes more sense than detection because each image already contains one piece of fruit, so the model only needs to decide what type of fruit it is and whether it is fresh or rotten. I used the nano version because it is fast, lightweight and suitable for a packing line.

I finetuned the ImageNet-pretrained model for 10 epochs. On the same test split of **2,078 images**, the finetuned model reached **100% accuracy**, compared with **40.5%** for the pretrained model without finetuning and **16.6%** for the majority-class baseline. The finetuned model also had the lowest estimated cost.

The biggest change after finetuning is that the model learns the difference between **fresh and rotten fruit**, which the generic pretrained model does not really know. The gap is very large, although probably larger than it would be in real production because the dataset has train/test leakage, so the 100% result is likely too optimistic.

## Per-class evaluation and failure modes

For the finetuned model, all six classes have **1.00 precision and 1.00 recall** on the test set, and the confusion matrix is completely diagonal. At the rotten/fresh threshold of 0.08, the model gets **1,203 true positives, 2 false positives, 0 false negatives and 873 true negatives**, so rotten recall is 100% and precision is about 99.8%.

There is not one class dragging the overall result, because all of them perform almost perfectly. The main concern is actually the test set itself. **Oranges are the class I trust the least**, because the only two errors are fresh oranges, and there are no clean orange images in the non-leaked test subset. The closest the model comes to a costly false negative is a rotten apple with a relatively low rotten score, but it is still classified correctly. On the clean test images the model also gets 100% accuracy, although this still only shows that it works on similar studio-style photos, not on real conveyor-belt images.

## Cost matrix and operating point

A false negative costs **€2.00** because a rotten fruit can ruin the whole pack, while a false positive costs only **€0.25** because it means throwing away one good fruit. Human review costs about **€0.05** per image. I assumed that around **5% of fruit is rotten** in production.

I tested thresholds from **0 to 1** and chose **0.08** using the validation set. At this point, the model is very cautious about letting fruit pass, which makes sense because missing rotten fruit is much more expensive than throwing away a good one. The test cost at this threshold is about **€0.54 per 1,000 pieces**.

The assumption that would change the result the most is whether the test error rates really represent production. Since the dataset uses clean studio images and has some train/test leakage, the real-world error rate on a conveyor belt could be higher.

## What a person still has to check

I use two thresholds: **pass below 0.08, review between 0.08 and 0.89, and fail above 0.89**. With this setup, only about **0.22% of the fruit** would need human review, which is around **43 pieces per day**. At 10 seconds per check, that is only about **7 minutes of work per day**, so one operator can easily handle it.

Compared with the alternatives, doing nothing would cost about **€2,000 per day**, while checking everything manually would cost around **€1,000 per day** and require almost 7 full-time workers. Letting the model decide alone costs about **€10.80 per day**, while using the model plus the review band brings the cost down to about **€2.20 per day**. So the model with human review is the cheapest and still very realistic to operate.

## Limitations & next steps

The biggest limitation is the **data leakage**, because many test images are modified copies of training images. I would fix this by splitting the dataset by original photo and testing the model again. There are also **no clean orange images** in the test set, so I would collect new orange photos before trusting those results. Another issue is that the dataset uses **studio photos instead of real conveyor-belt images**, so I would run a pilot on the actual production line and measure performance there. I would also recalibrate the model using the real 5% rotten rate, replace the estimated costs with the retailer’s real numbers, and measure how accurate human reviewers actually are.

## Generative AI use disclosure

I used Claude (Anthropic) with my instructor's authorisation. It wrote the code
(`train.py`, `app.py`, `costing.py`), from the outputs of
my own training run. I reviewed the problem framing, the cost figures, the
thresholds and the conclusions, and I can explain each of them.