# Assignment 2 Report

- **Name**: Manuela Domínguez
- **Student ID**: 19034
- **Email**: mdominguez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

The railway accident in Adamuz, Spain, this last January 18 of 2026, resulted in 46 fatalities and raised important questions on how railway infrastructure is inspected and monitored. While the official investigation of this case remains ongoin, some preliminary findings point towards a fractured rail which was potentially linked to a defective weld. This incident highlights the importance behind identifying any infrastructure anomalies early and ensuring that any potential warning signs translate immediately into maintenance decisions. 

Hence, this project proposes an AI visual inspection system for railway infrastructure managers such as Adif. So by using images collected from routine spections through drones, I will employ a YOLO model to identify the visible defects such as track cracks, missing bolts or missing rail clips. Hence, our purpose will not be to replace any existing safety procedures, but rather to help inspection teams prioritize those potentially defective sections within the railway network. 

In this case, each image would represent one inspection case with 3 possible outcomes: pass (no additional alert), human review (an inspector should examine uncertain predictions), or fail (potential defect is flagged for priority technical inspection).

Therefore, the economic consequences of incorrect predictions are asymmetric. A false positive leads us to an unnecessary inspection, causing an increase in labour, travel and operational costs, with an incremental cost of 150€ (as an assumption). While a false negative could also lead to leaving a genuine defect unidentified that results in a far more expensive corrective maintainance that additionally could lead to operational disruptions or restrictions of our railway services, with an assumed expected cost of 3,000€. Hence, the central challenge is to minimize the expected costs of missed deffects (without overwhelming maintenance teams with unnecessary inspections) due to the higher consequences of false negatives, since under these illustrative assumptions, missing a defect would be 20 times more expensive than raising a false alarm. 

Still, I highlight that the proposed system is intended to function as an additional support layer rather than a standalone mechanism. 

## Dataset

This project uses the publicly available dataset of Railway Track Fault Detection from Roboflow Universe. 
* Workspace: kerusakan-rel-kereta
* Project: railway-track-fault-detection-fgbu5-eys2z
* Version: 1
* Link: https://universe.roboflow.com/kerusakan-rel-kereta/railway-track-fault-detection-fgbu5-eys2z/dataset/1
* Task: Object detection

The dataset currently contains 1,071 images of railway tracks and their components annotated across 7 classes: fishplate, fishplate_bolthead, fishplate_boltmissing, fishplate_boltnut, track_bolt, track_boltmissing and track_crack. 

In this case, each image represents a single visual inspection of a section of railway infrastructure (which may contain multiple annotated objects). Additionally, an image is considered positive if it contains at least one missing fishplate_boltmissing, track_boltmissing or track_crack. The remaining classes provide contextual information about normal track components.

In this case, the dataset is particularly relevant to the proposed business problem since it includes both normal infrastructure components and visible defects that could justify further inspection.

All this allows the model's object level predictions to be translated into an image level decision: whether a section of the railway track should deffinitely be flagged for further technical inspection or not. Therefore, the dataset supports the proposed maintenance priorization use case rather than simply containing railway related images. 

Lastly, although the dataset is smaller than version 2 (initially considered for the analysis), this one offers a more manageable starting point for developing and evaluating the complete inspection workflor within the project's current computational constraints. 

## What the dataset's composition implies

The dataset consists of 718 training images, 248 validation images and 105 test images. The distribution of annotated objects is as follows (In the following bullet points I am going to show the information as "class: train/val/test):

* Fishplate:  77 / 30 / 16
* Fishplate bolthead: 177 / 64 / 27
* Fishplate bolt missing: 121 / 39 / 16
* Fishplate boltnut: 166 / 66 / 27
* Track bolt: 92 / 35 / 16
* Track bolt missing: 98 / 40 / 15
* Track crack: 118 / 30 / 18

Unlike the analysis I made when considering version 2, we see that in version 1 the training annotations are relatively balanced across all classes. However, we do find that the small test set remains a significant limitation, mainly for safety relevant defects. Imagine there are 18 rail crack annotations and 15 missing track bolt annotations in test, then in this case, a few incorrect predictions could therefore substantially change the reported recall for these classes. 

Additionally, object size also varies considerably across classes. For instance, fishplates occupy a median of 66.86% of the image, while smaller components such as missing fishplae bolts (4.21%), fishplate nuts (7.08%) and track cracks (8.51%) occupy much more less space. This causes that smaller deffects are harder to detect, specially under different lighting conditions or camera angles. Therefore, the overall detection performance should be complemented with per class metrics, mainly for small relevant defects. 

On the other hand, our preliminary dataset checks identified 13 shared source identifiers between the training and test sets, as well as 13 groups of exact duplicate images across data splits. Hence, this suggests data leakage in the dataset, which could inflate the evaluation results by allowing the model to "cheat" by encountering visually identical or closely related images during training and testing. Therefore, the cleaning process removed shared source identifiers between splits, but still this does not guarantee the absence of visually similar images with different file names. For that reason, we will assume for the analysis that the model's reported performance may still be optimistic. 

With all that being said, the limited test size and possible leakage mean that strong peformance in this dataset would not establish reliability in operational railway inspections. Before training, the typical bounding-box size relative to the image will be measured to assess whether small objects create additional detection challenges. 

## Finetuning and baselines

The project employs YOLO11n for object detection, fine tuned from pretrained COCO weights. The object detection was chosen over classification since the model must not only be able to identify defects, but also to locate them within the image. Additionally, specifically YOLO11n was selected fro its lightweight architecture which allowed me to train in GitHub without the GPU access. 

Furthermore, I trained the model for 30 epochs, suing 663 training images, 240 validation images and a size of 416 x 416 and a batch size of 4. So with all three approaches I evaluated them on the same 105 image test split reaching this results (displayed as model: precision / recall / accuracy):
* Fine tuned YOLO11n: 88.68% / 100% / 94.29%
* COCO zero shot: 0% / 0% / 55.24%
* Predict nothing baseline: 0% / 0% / 55.24%

Note also that metrics are calculated at image level and a threshold of 0.01 was used for validation. 

The fine tuned model also achieved a value of 95.49% mAP50 on test. This shows a strong object level detection performance. But the trivial baseline missed all 47 defective images, while fine tuned model was capable of identifying all positive cases while generating six false alarms. 

I saw that the improvement was larger than I was expecting. But still, the comparison still requires some caution. For instance, COCO has no directly equivalent railway defect cases, so its zero shot predictions cannot be mapped to our target labels and are treated as negative. So the gap partially reflects this classification mismatch rather than just fine tuning alone. Additionally, as a reminder, the small test set and potential residual data leakage may inflate the performance. 

## Per-class evaluation and failure modes

Although the fine tuned YOLO11n achieved an overall mAP50 of 95.49%, it is true that performance varies also considerably across classes (information displayed as class: precision / recall / AP50):
* Fishplate: 88.75% / 100% / 99.15%
* Fishplate bolthead: 92.60% / 100% / 95.73% 
* Fishplate bolt missing: 100% / 87.07% / 98.46% 
* Fishplate boltnut: 66.17% / 65.22% / 80.54%
* Track bolt: 97.95% / 100% / 99.50%
* Track bolt missing: 98.04% / 100% / 99.50%
* Track crack: 98.06% / 94.44% / 95.53%

So we can see as the results show, that the headline performance is mainly sipported by fishplate, track bolt and track bolt missing, all achieving AP50 values above 99%, while in contrast we find fishplate boltnut as the weakest class with only 65% recall and 66% precision. 

Additionally, the multiclass confussion matrix suggests us dificulties distinguishing visually similar fastening components. We see that five actual fishplate_boltmissing objects were classified as fishplate_bolthead, while six fishplate_bolthead objects were classified as fishplate_boltnut. And it also shows confusion between track_crack and normal track components. Nonetheless, its unusually high number of background associated predictions makes these patterns indicative rather than definitive.

From a business perspective we see that track crack recall of 94% is particularly relevant, as one of the 18 anotated cracks was only missed at object level. But, nonetheless, it is true that the image-level confusion matrix at the selected 0.01 threshold shows 47 TP, 6 FP, 0 FN and 52 TN, meaning every test image containing a critical defect triggered an alert.

With all that said, these findings show that strong aggregate performance can conceal weakness in individual defect detection. Still, the small test set and remaining uncertainty around the multiclass confusion matrix limit how confidently these results can be generalised to real railway inspections.

## Cost matrix and operating point

Given the potential consequences of undetected railway defects, I decided that the cost model should prioritize missing false negatives over false positives. However, I must state that the following costs are illustrative business assumptions rather than the official Adif figures, that were approximately based on estimated inspection resources or the potential consequences of a delayed maintenance: 
* TP = 150€ = necessary technical inspection of a detected defect
* FP = 150€ = unnecessary technical inspection
* FN = 3,000€ = expected costs of delayed repairs and service disruptions
* TN = 0€ = no additional inspection required. 

I decided to assume that it should be 20 times more expensive than a false positive. So the 3,000€ although it seems high, represents an expected economic impact (not just the cost of an accident). or the initial analysis, the assumed defect base rate is 44.76%, corresponding to 47 positive images out of 105 in the test set, but this is a dataset-derived rate, not a validated estimate of defect prevalence across Adif's network.

Additionally, the model was evaluated across multiple confidence thresholds, calculating the expected cost of each combination of correct and incorrect predictions as follows: 

Total cost = (TP × €150) + (FP × €150) + (FN × €3,000)

So the threshold sweep was performed on the validation set, selecting 0.01 as the minimum cost operating point. And appliying this threshold to the test set produced 47 TP, 6 FP, 0 FN and 52 TN, resulting in an estimated €75,714 per 1,000 images. The low threshold reflects the high assumed cost of missing critical defects.

Hence, the assumption most likely to change this decision is the real world defect prevalence, which could be lower than in the dataset. A lower based rate would change the balance between the unnecessary inspections and the missed defects, potentially favouring a higher threshold. So the relative cost of false negatives is another uncertainty and should be tested in another sensitivity analysis before it is employed. 

## What a person still has to check

The system uses 2 confidence thresholds selected on validation to balance automation and also the human side: 
* PASS (<0.01): No additional inspection.
* HUMAN REVIEW (0.01–0.70): A technician checks uncertain cases.
* FAIL (≥0.70): Flagged for priority technical inspection.

On the test set the the 9.52% of images require human review (10 out of 105). So assuming there are 1,000 images daily and 3 minutes per review, this represents approximately 95 reviews, 4.76 working hours and 0.79 FTE, assuming six productive hours per inspector. This appears manageable with one dedicated technician.

So the cost per 1,000 images is: with no image based inspection 1,342,857€, with fully manual inspection 82.143€, with fully automated model 75,714€ and the cheapest one which is a hybrid model with human review of 68,571€. 

However, these results depend on illustrative costs, a high dataset defect prevalence (44.76%) and the assumption that human reviewers make no mistakes. Therefore, the estimated savings require a more realistic validation.

## Limitations & next steps

The main limitations from the dataset was the small test set (105 images) and particularly small for critical deffects such as the track cracks (18 annotations). Version 2 provided a larger dataset but I could not run it with the resources available on Github.  larger, more representative dataset would be needed to validate performance across different track conditions, lighting and camera angles.

Additionally, as mentioned, the economic model relies on assumed inspection costs, a high defect prevalence (45%) and perfect human review. So this assumptions should be validated in a case of using real Adif operational data before deployment because they may lack realism. 

Also, it is worth mentioning that dataset exploration was another major challenge. During the initial research, I identified potential data leakage in Roboflow datasets and systematically investigated eight candidate datasets, checking shared source identifiers, duplicate images and similarities across train, validation and test splits. Leakage was repeatedly detected, making dataset selection considerably more time-consuming than expected. After discussing the issue with the professor, I decided to proceed with Railway v1 and developed an automated cleaning procedure, removing 63 images with overlapping source identifiers while preserving the original test set. However, this does not rule out visually similar images with different filenames. A future improvement would be to create independent splits grouped by original photograph or inspection session, ideally with temporal separation, to obtain more reliable performance estimates.

## Generative AI use disclosure

I only used ChatGPT as a supproting tool throughout the project, mainly for understanding the code and helping me with data leakage. Also with the assumptions so they could be defined and it had sense. 
