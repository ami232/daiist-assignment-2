# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Shamil Mukhamedov
- **Student ID**: 20352
- **Email**: smukhamedov.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

*The decision this model supports, who or what acts on it, and what happens
when it's wrong in each direction. Be concrete about the asymmetry: which
mistake is worse, roughly by how much, and why. Everything downstream (your
threshold, how much you send to a person, which metric you care about) has to trace
back to this section.*

My project is an image-oriented road-inspection system that detects damaged road surfaces and potholes. Its primary purpose is to help road-maintenance companies identify roads that require inspection and; eventually, potential repair. The system does not aim to replace a human inspector - as we discussed in class with the broke bottel example, that is not the point of image-based computer vision models - instead, the model supports prioritization for a road-inspection company.

The two classes that I decided to keep (more on that in the subsequent section) are "Damaged road" and "Potholes". A false negative takes place when a relevant hazard is not detected, potentially delaying inspection and repair. Per contra, a false positive occurs when the system flags a location that does not contain the predicted object, creating unnecessary inspection, resulting in wasted time, money, and labor for the company.

For the initial cost model, a missed hazard is assigned a cost of €100, a false positive €20, and a human review €10. These are illustrative assumptions for this assignment, not measured financial losses. They preserve the relative cost ratio of 10:2:1, meaning that a missed hazard is assumed to be five times as costly as a false positive and ten times as costly as a human review. Correct decisions incur no error cost. The model therefore prioritizes avoiding missed hazards while controlling false alarms and referring uncertain cases to human reviewers.

My system will be evaluated at various confidence thresholds. In addition, predictions will be routed to automatic acceptance, human review, or no automatic flag according to their confidence. Therefore, the final operating point will be selected using the measured evaluation results and the aforementioned cost assumptions.

## Dataset

*Workspace / project / version on Roboflow Universe, and the link. What one
image represents, what the classes are, and why this dataset actually serves
the problem above rather than merely resembling it.*

The dataset was obtained from Roboflow Universe: Workspace: moh-fzwtt; Project: pothole-detection-hg5ye; Version: 2.

Dataset URL: https://universe.roboflow.com/moh-fzwtt/pothole-detection-hg5ye/dataset/2

The dataset contains 4,388 training images, 250 validation images, and 100 test images. The two classes used in the final model are Damaged road and Potholes; the manhole class was excluded.

The overarching Roboflow dataset had ten classes and more than five datasets, I chose the second one as it initially had three classes and it was fit for a yolov11 model. Not to mention, it has a surplus of images to train the model on. I later found out that the "manhole" class only had four images, which would result in an aggressive class imbalance, so I just removed it entirely using "cp -R PotholeDetection.v2i.yolov11 PotholeDetection.v2i.yolov11_backup" on the bash terminal in my github codepsace. 

Afterwards, I changed the yaml file in the dataset in order to keep the changes consistent across the board to avoid issues in main.py. Later, I also removed the roboflow url portions from the origina yaml file I got while downloading the roboflow dataset since gpt told me that was not necessary to actually do the project properly.

I then created a backup of the original downloaded dataset before modifying the working copy. The working dataset configuration was updated to retain only Damaged road and Potholes, and the four manhole annotations were removed from the training labels. The final model is consequently a two-class object detector.

Each image in the roboflow pothole detection dataset represents a road scene containing zero or more labelled objects. As previously mentioned, the classes represent visible road damage and potholes. The dataset I chose is relevant to the proposed decision because the detected objects can indicate road locations that may need further inspection. However, the dataset alone does not establish the severity of a road defect, its precise location in the road network, or the cost of repairing it, it simply shows whether or not the road has a defect and whether or not the defect is a damaged road or a pothole.

## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

The downloaded dataset contains 4,738 images: 4,388 for training, 250 for validation, and 100 for testing.

After excluding the manhole annotations, the retained object counts are as follows: 

Damaged roads
Training objects: 3346
Validation objects: 192
Test objects: 87

Potholes 
Training objects: 1552
Validation objects: 116
Test objects: 55

Total 
Training: 4898
Validation: 308
Test: 142


The counts above refer to labelled objects and not unique images; therefore, an image may contain more than one object. The training split contains substantially more examples than the validation and test splits, and the Damaged road is more frequent than the Potholes class. On top of that, the test set is relatively small, particularly for per-class evaluation, so performance estimates may be sensitive to individual examples.

The four manhole annotations were confined to the training split. Since no manhole examples appeared in validation or testing, the dataset could not support a reliable evaluation of that class. The limited two-class scope avoids reporting unsupported manhole performance; with that said, the two remaining classes still require inspection for consistent labelling and conceptual overlap.

Object sizes, image conditions, and the representativeness of the test set will be assessed by inspecting the sample images and annotations. Lastly, results should be interpreted as performance on this dataset rather than proof of generalization to every possible road, camera, weather condition, and region - more on that in the limitations section.

## Finetuning and baselines

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

My project uses yolov11 for object detection, with the model fine-tuned on two classes. Object detection is appropriate here because the system needs to locate and classify road defects within an image rather than merely classify the entire image. My final fine-tuned model will be compared on the same test split with both the pretrained COCO - Common Objects in COntext - model without fine-tuning and the trivial baseline that makes no detections at all. The predict-nothing baseline cannot detect labelled objects, so it will have zero recall and zero detection average precision.

Precision is undefined when the model makes no predictions at all. My zero-shot model will be evaluated on the same aforementioned test images, using a documented mapping between its Common Object in COntext - COCO - predictions and the two target classes. Final results will be reported after the evaluations have been made...

Final results after evaluations have been made:

The final evaluation was conducted on the 100-image test set. The fine-tuned YOLO11n model achieved an overall precision of 0.8234, recall of 0.8609, mAP@50 of 0.8782, and mAP@50–95 of 0.8152. These results indicate that the model can identify road defects in the selected dataset, although performance differs substantially between the two classes. The pretrained COCO model and a trivial predict-nothing baseline were also considered to assess the value of fine-tuning for this specific task.

The pretrained COCO model was also evaluated without fine-tuning. Under the image-level decision evaluation, this baseline produced 93 false negatives and 7 true negatives, with a total assumed cost of €9,300 per 100 images. This demonstrates why a general pretrained model without adaptation to the target classes is insufficient for the proposed decision.

A trivial predict-nothing baseline makes no detections by definition. It therefore has zero recall and zero average precision for object detection, while precision is undefined because it makes no positive predictions. At the image level, an always-negative policy incurs the assumed missed-hazard cost for each image containing a hazard.

The fine-tuned model provides a more useful starting point for road-inspection prioritization than these baselines. However, its results should be interpreted in the context of the small test set and the limitations of the dataset rather than as evidence that it is ready to replace professional road inspectors.

## Per-class evaluation and failure modes

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

The evaluation will report per-class precision, recall, AP50, and the confusion matrix where supported by evaluation procedure. Particular attention will need to be paid in regards to whether the model misses potholes, confuses potholes with a simple road damage, or produces false detections on road surfaces and unrelated objects. Results will be added after training and testing. 

Because the test set contains 87 damaged-road objects and 55 pothole objects, per-class results should be interpreted cautiously. False positives and false negatives will be inspected to identify concrete failure modes.

After running model:

The per-class results show that the model performs better on Damaged road than on Potholes. For Damaged road, precision was 0.9542, recall was 0.9581, AP50 was 0.9863, and mAP50–95 was 0.9855. For Potholes, precision was 0.6926, recall was 0.7636, AP50 was 0.7701, and mAP50–95 was 0.6450.

Potholes are therefore the weaker class across all four reported metrics. The lower recall means that some labelled potholes are missed, while the lower precision indicates that a larger proportion of the model's pothole predictions may be incorrect. This matters because a missed pothole could delay inspection and repair, whereas an incorrect detection could waste inspection resources.

The test set contains 87 Damaged road objects and 55 Potholes objects. These relatively small class counts mean that individual examples can materially affect the reported metrics. The results should consequently be treated as estimates of performance on this dataset, not as definitive evidence of performance on roads in general.

The model's object-detection confusion matrix and individual predictions should also be examined to identify examples of missed defects, incorrect detections, and confusion between the two classes. Because Damaged road and Potholes may overlap conceptually, consistent annotation rules and human inspection remain important. The weaker pothole results support retaining a human-review process rather than relying exclusively on automatic decisions.

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

The cost-against-threshold sweep was performed using candidate confidence thresholds from 0.05 to 0.95. Under the assumed costs of €100 per false negative and €20 per false positive, the lowest reported cost for the fully automated policy occurred at a threshold of 0.05, with an estimated cost of €2,480 per 1,000 images. At this threshold, the sweep recorded 31 false positives and no false negatives. As the threshold increased, false positives generally decreased, but false negatives increased substantially. For example, at a threshold of 0.50, there were 11 false positives and 29 false negatives, corresponding to a reported cost of €12,480 per 1,000 images. At a threshold of 0.95, false positives fell to zero, but false negatives increased to 131, producing a cost of €52,400 per 1,000 images.

These results illustrate why maximizing the confidence threshold or minimizing false positives alone would not be appropriate for this business problem. Missing a road hazard is assumed to be five times as costly as a false positive, so the cost model favors detecting potential hazards even when this creates additional inspections. The threshold sweep supports a low threshold for the fully automated policy. The separate model-assisted policy uses a lower threshold of 0.05 and an upper threshold of 0.35, sending intermediate-confidence cases for human review. This distinction is important: the minimum-cost single-threshold policy and the two-threshold human-review policy are different decision strategies, and their costs should be compared separately.

The selected two-threshold policy uses a lower threshold of 0.05 and an upper threshold of 0.35. On the 250-image validation set, it produced 205 true positives, 17 false positives, 3 true negatives, and no false negatives, with 25 images (10%) referred for human review. The total assumed cost was €590, equivalent to €2,360 per 1,000 images. This was lower than the €2,480 per 1,000 images reported for the best fully automated single-threshold policy. The thresholds were selected using validation data and then evaluated separately on the test set.

## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

The dashboard uses two confidence thresholds to divide predictions into automatic acceptance, human review, and no automatic flag. The selected thresholds and review share will be reported from the evaluation results. As mentioned above, per-class results should be checked by a person. The review-capacity estimate will state the assumed number of images processed per day and the average time needed to inspect one image. Then, it estimates the daily review workload and the number of reviewers required. The model-assisted policy will be compared with doing nothing, human review of every image, and fully automated model decisions.

The selected policy uses a lower confidence threshold of 0.05 and an upper threshold of 0.35. On the test set, 7% of images were referred for human review. For a hypothetical workload of 1,000 images per day, this would correspond to approximately 70 images requiring review each day.

If each review takes an assumed average of two minutes, reviewing 70 images would require approximately 140 minutes, or 2 hours and 20 minutes of staff time per day. This is an illustrative capacity estimate; actual review times would need to be measured during deployment.

The model-assisted policy should be compared with three alternatives. First, doing nothing would miss all hazards; based on the 93 positive images represented in the reported image-level baseline, the assumed cost would be €9,300 per 100 images. Second, having a person inspect every image would cost €1,000 per 100 images if each review costs €10 and review is assumed to identify all relevant hazards correctly. Third, a fully automated policy would make decisions without human review, but its cost would depend on the false-positive and false-negative counts at the selected single confidence threshold.

On the held-out test set, the model-assisted policy generated 90 true positives, 3 false positives, no false negatives, and 7 images for human review. Its total assumed cost was €130 per 100 images, or €1,300 per 1,000 images. The pretrained COCO zero-shot baseline incurred an estimated cost of €93,000 per 1,000 images under the same assumptions.

The fully automated single-threshold policy had a validation cost of €2,480 per 1,000 images, while the two-threshold review policy had a validation cost of €2,360 per 1,000 images. The model-assisted policy's test cost was €1,300 per 1,000 images. Because the first two figures are validation results and the last is a test result, they should not be treated as a direct like-for-like comparison. The test results are encouraging, but the sample is small and the costs are illustrative rather than measured operational expenses.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

The test set is small, especially for per-class evaluation. A larger and more diverse held-out dataset would produce more stable performance estimates. The original manhole class was excluded because it had only four training objects and no validation or test objects; consequently, a future version would require a substantially larger and independently evaluated manhole dataset to support that class. For future steps, I would also ensure that this model is applicable across different roads in differen regions under various weather conditions. For instance, where I'm from it tends to snow for nearly 5 months a year, so this model may not work for this since most images in the roboflow dataset I chose have, at most, rainy terrain. 

The labels Damaged road and Potholes have conceptual overlap, which makes sense in the context of the problem for pothole detection. Generally, any type of pothole needs maintenance, but different effort and different laborers for the type of pothole, which is the whole point as mentioned in the business implications. On top of that, sample annotations should be checked for consistency, so a future dataset should define the distinction between general damage and potholes more precisely. The dataset also fails to adequately represent all lighting, camera, and road conditions encountered in real-life deployments.

Lastly, the costs are hypothetical rather than concrete financial estimates. A real deployment needs to estimate the consequences of missed hazards, false alarms, and inspection time using operational data from real financial records from a road-maintenance team in a company.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

Generative AI was used to assist with planning the project, debugging and explaining the implementation, and drafting or editing report sections. The final implementation, experimental results, cost assumptions, and conclusions should be checked and verified by the student before submission.