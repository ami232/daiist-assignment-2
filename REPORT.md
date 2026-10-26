# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Justo Ruiz Cano
- **Student ID**: 20317
- **Email**: jruizcano.ieu2022@student.ie.edu
- **Group**: [PPLEDBA 5A]

## Business problem, and the cost of being wrong

*The decision this model supports, who or what acts on it, and what happens
when it's wrong in each direction. Be concrete about the asymmetry: which
mistake is worse, roughly by how much, and why. Everything downstream (your
threshold, how much you send to a person, which metric you care about) has to trace
back to this section.*

The business problem for this question is the following: Given a set of tomatoes, can we classify them as ripe, unripe or half-ripe? Depending on the ripeness of a tomato, the business case differs.

For ripe tomatoes: if a tomato is ripe, it means it needs to be consumed in the next few days. This means that it must be sold locally, or delivered with speed. If we misclassify a ripe tomato as half-ripe or unripe, it would spoil if sent a long distance, and possibly contaminate the rest of the shipment. As such, we want to ensure we're not misclassifying this set of tomatoes since they have the highest probability of creating damages.

For half-ripe tomatoes: if a tomato is half-ripe it means that it is fit for consumption, just not immediately. This means that they are ideal for long distance buyers since they will ripen over the shipping period. The mistake here can be twofold. If we classify the tomato as ripe, the company may suffer reputational loss since it is selling misclassified fruit. At the same time, if we classify the fruit as unripe, the company will be incurring a financial loss since it is selling good fruit for a steep discount. 

For unripe tomatoes: For the sake of this assignment, we will assume that all tomatoes have been picked off the stem. The reason I state this is that for some of the pictures, tomatoes are still attached to the stem. This assumption helps simplify the business case. Further, we assume that the unripe tomatoes have been picked from the stem as a mistake. This means that they are worthless and need to be sold at a steep discount because they are not fit for human consumption and instead are to be used purely as fertilizer. Misclassifying the tomatoes as half-ripe or ripe when they are unripe will not cause a shipment to be ruined (unlike the ripe case). Instead it will create reputational damage for the firm as it is selling low quality products. 

The classification here will help automate the process and add a more objective view on the product's quality. A human may use their judgement, but we will assist them by flagging cases where the model is unsure of what to classify the fruit, as such reducing the amount of human oversight during the process. 

Since the cost of misclassifying a ripe tomato as any of the other classes causes the highest financial cost, we'll set a low threshold for classifying a tomato as ripe, so that as few ripe tomatoes as possible are missed, even if this means some half-ripe tomatoes are labelled as ripe. The exact threshold is chosen by the cost sweep below. We'll also assign a large bound where the model is unsure, flagging the need for human intervention, since these fruits will be the most expensive product and we do not want to be selling them at a discount. Half-ripe tomatoes are between the two, they will not cause as much financial harm to the vendors as ripe tomatoes if misclassified. Instead they cause the firm losses if classified as an unripe tomato, or reputational risk if classified as ripe tomatoes. As such, we will need to find a threshold that accurately balances these two costs. 



## Dataset

*Workspace / project / version on Roboflow Universe, and the link. What one
image represents, what the classes are, and why this dataset actually serves
the problem above rather than merely resembling it.*

Dataset: *https://universe.roboflow.com/dataset-w8sqm/tomato-ripeness-jionb/dataset/14*

This dataset has a good amount of varied images for all of the ripe, unripe, half-ripe tomatoes. The following is taken from the EDA notebook: 
- there are 1012 images, but only about 540 original photos (each training photo is copied 3 times)
- the train validate test split is the following, counting original photos: 43/37/20 (235/200/106 photos)
- There are ~4,950 tomatoes all together: 1,315 are half ripe, 1,499 are not ripe, 2,132 are ripe. While it's not perfectly balanced, they are more or less the same magnitude so I expect this won't affect training. 
- One thing to note about the dataset that I did not realise until it was too late was that the dataset has been augmented. This means that there are less original images that I had previously intcipated, with many images having been editted (ie add noise, flipped etc)

## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

The composition is mentioned in the previous bullet point. As mentioned above, there is limited data so the dataset contains augmentations which is not the best. With regards to whether the test split resembles what you'd really see in production, unfortunately I don't know that much about tomato production so I cannot make any reasonable guess. That being said, the amount of real training data is small (235 original photos), which limits how far the results can be trusted.  

## Finetuning and baselines + Per-class evaluation and failure modes
(i've merged the two sections together)

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

This task is detection. I chose the yolo11n due to a few resons. Firstly, my computer, as specified in a few sentences only has 4gb of gpu memory, and yolo11n ran out of memory when setting batches to 16. Taking a larger would make me need to use a smaller batch which would cause training to take even longer. Another consideration was that since the dataset is small, a larger model has a higher chance of memorizing, instead of learning general patterns which could affect test/validation stages. We use detection instead of classification because one photo can hold many tomatoes and that one photo can hold different levels of ripeness. Classfication only gives one label per image. Traing was done with RTX 3050 Ti laptop GPU, PyTorch 2.7.1 + CUDA 11.8, 50 epochs at batch 8 (16 ran out of memory), stopped at epoch 36, best model at epoch 16, 7.5 minutes.


Test split, recall at confidence 0.25

| Model           | mAP50 | AP50 half-ripe | AP50 not_ripe | AP50 ripe | Recall half-ripe | Recall not_ripe | Recall ripe |
|-----------------|------:|---------------:|--------------:|----------:|-----------------:|----------------:|------------:|
| finetuned       | 0.655 |          0.544 |         0.722 |     0.699 |            0.625 |           0.753 |       0.684 |
| coco_zero_shot  | 0.083 |          0.000 |         0.000 |     0.250 |            0.000 |           0.000 |       0.352 |
| predict_nothing | 0.000 |          0.000 |         0.000 |     0.000 |            0.000 |           0.000 |       0.000 |

Our fined tuned model significantly outperforms the coco_zero_shot, by a very large margin.
This makes sense becuase the COCO does not have any tomato class, so we used apple and orange as stand ins. You can have red round tomatoes which are similar to apples, but if you compare them to unripe tomatoes (green) the model completely only labels 1/5 of all the possible tomatoes. Furthermore, the model cannot tell ripeness apart, so the model simply labells each one as ripe. As such it, by design, cannot label half-ripe or not ripe. That's the reason we see 0's in the half-ripe + not_ripe metrics above. 

| True class | Found by COCO (boxed, labelled ripe) | Missed | Share found |
|------------|-------------------------------------:|-------:|------------:|
| half-ripe  |                                  118 |    191 |         38% |
| not_ripe   |                                   96 |    354 |         21% |
| ripe       |                                  145 |    267 |         35% |

So, by design the COCO algorithim couldnt work. This is because it was set to intentially label every tomato as ripe. So the best possible reading here would be a mAP50 of 0.33. However, it only scored 0.083. This wasx expected because the 0.33 would assume that all ripe tomatoes would be found and that it drew no wrong boxes. 
The fined tuned model reached 0.655 mAP50 vs Ultralytics 0.701. 
The gap was large, but expected becuase again, the design of the COCO did not have tomatoes or ripeness as such it could not measure what the model doesn't know. 



Confusion Matrix for fined tuned model 

| True class ↓ / Model said → | half-ripe | not_ripe | ripe | missed | Total |
|-----------------------------|----------:|---------:|-----:|-------:|------:|
| half-ripe                   | 193 (62%) | 26 (8%)   | 44 (14%)  | 46 (15%)  | 309 |
| not_ripe                    | 4 (1%)    | 339 (75%) | 0 (0%)    | 107 (24%) | 450 |
| ripe                        | 54 (13%)  | 1 (<1%)   | 282 (68%) | 75 (18%)  | 412 |
| no labelled tomato          | 115 (23%) | 219 (44%) | 165 (33%) | –         | 499 |
| **Total boxes drawn**       | **366**   | **585**   | **491**   | –         | –   |

Interpretation: 
The model correctly finds and predicts half-ripe 62%, not_ripe 75% and ripe 68% of the time. 

Using the AP50 metic, the not_ripe has the best scoring (0.722) while the worst scoring is the half-ripe (0.544). The hardest for the model to desipher was the half-ripe. If we look at the total share, yes, 62% of them were correctly identified, but we also see that the model is confused in both directions. It misslabled half ripe tomatoes as ripe 14% of the time, and not_ripe tomatoes 8% of the time. Notabley, it missed 15% of tomatoes too. If we look at the ripe classifications , 13% of them were classified as half ripe.

It seems that this is the largest problem with the model, which makes sense because when does a human classify a tomato as ripe/half-ripe. Is it when it's fully red? When it has very little yellow? These inconsistencies may cause worse results by the model. If we contrast this with ripe/not ripe, we see that they are almost never mixed up. This is becuase they are very different colours which the model can learn very well. 

Not_ripe missed the most tomatoes. This makes sense because they may be harder to find considering some of the photos included the tomatoes on the trees and foliage (24% misses). Also not ripe tomatoes are the smallest: small and green --> harder to find. 

Also it's important to note that there are around 500 boxes that don't match a labelled tomato. I asked claude to help me understand this and it stated the following (I was genuinely unsure as to why there were so many). This is a summary: 
- there are 81 different class duplicates, where the model was unsure so it drew two boxes. Ultralytics only removes overlapping boxes of the same class, so both boxes survive. One get's matched to a tomato and the other as a false box. 
- 266 that touch no lablled tomato are split roughly into: 
- 1/3 are real tomatoes that whoever labelled the photos did not box. (tomatoes in piles, green tomatoes on the plant)
- third of them are not tomatoes (flowers, different fruits)
- a third of them are too blurry to tell. 

With regards to the business cost here, looking at the confusion matrix, the model misclassifies the most expensive mistake (a ripe tomato being labelled wrong) 13% of the time. and it misses a further 18% entirely. This is 31% in total which is worrying for something to be placed in production. The misclassification is also one of the more worrying mistakes because it's quite common since ripe/half ripe is a standard confusion. 

For the postive class, we're going to pick rip e since it is the most important one that we need to calculate given our business case. The false negatives are teh ones where we dont catch the ripe tomato aka it spoils in transit. the false postive are the ones which are sold as ripe but are actually half ripe / not ripe which can cause reputational damage. Note the following is for a confidence threshold of 0.25. 

| | Model says ripe | Model says not ripe (or missed it) |
|---|---|---|
| **Really ripe (412)** | 282 | 130 |
| **Really half-ripe or unripe (759)** | 44 | 715 |

## Cost matrix and operating point + What a person still has to check


*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

Again, I don't know exactly what these cost, so the figures below are estimates with the reasoning behind them. 
The cost of a false negative is 0.3 euros: a tomato weighs about 150 g and sells for roughly €2/kg, so a ripe tomato that spoils in transit loses about €0.30. Cost of a false positive will be 0.15 euros: the tomato is still sold, so the loss is reputational (for example a complaint or a partial refund), which we assume is about half the value of the tomato. 
If we scan around 100,000 tomatoes, a day, 3 seconds per tomato, 300,000 seconds. that means that for an 8 hour shift we need 11 people (300,000 seconds is 83.3 hours of checking, and 83.3 / 8 = 10.4). At 13 euros per hour, this would account for about €1,083 per day if a person checked every tomato.

Our base rate is the test set (35% ripe tomatoes)

| Strategy | EUR per 1,000 | EUR per day |
|---|---|---|
| Do nothing | 105.00 | 10,500.00 |
| Person checks every tomato | 10.83 | 1,083.33 |
| Model alone, threshold 0.01 | 26.99 | 2,699.34 |
| Model + review, 0.00–0.81 | 10.22 | 1,022.46 |

The review band sends 90.8% of tomatoes to a person: 9.5 people on 8-hour shifts.

The best option is to score everything 0.81 or higher. This is only around 9% of the tomatoes. Everything else goes to a person. The lower threshold is set to 0.00 so that the model never decided to that a tomato is not ripe on it's own. 

Since a person's check costs around 0.011 per tomato, and a model mistake costs 0.3 or .15, it only pays to let the model decide when it's almost certainly right. 

The model is worth having becuase it does optimize to a certain extent the process. But it only saves around 61 euros compared to all the people checking. However, is it realistic for one human to check one tomator every 3 seconds? Maybe.

The assumption that would change the answer the most waould be the cost of a check. if it took 10 seconds instead of 3, or missed ripe costed less, then the model would take over more work making it more worth it to employ. 
## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

Small set of real training data. As mentioned before there were a lot of copies, so maybe I could merge a few more datasets together. 

Some of the photos dont match the use case since they're photos of tomatoes in a garden not on a sorting line. This could make merging datasets harder. This also affects the business case's validity in theory.  

Some lables are incomplete. But more than that, half ripe tomatoes can be subjective to an extent. Not sure how this could be fixed. 


## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
Wrote the eda boilerplate code with ai. Wrote the dashboard with ai given what was in my report. Used Ai to write the models. Used ai very little in the report to format and to double check key information.