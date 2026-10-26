# Assignment 2 Report

Name: Adam Baroudi
Student ID: 19379
Email: abaroudi.ieu2022@student.ie.edu
Group: BBADBA 5A

## Business problem, and the cost of being wrong

I am working on a project related to AI and safety in construction.

Construction workers are required to wear protective equipment, especially safety helmets. In normal circumstances it would be the duty of safety supervisors to determine if workers are complying with these rules.

My proposed system will use an artificial intelligence system that will analyze the images taken on the construction site and identify the workers that are not wearing the safety helm. It is designed to assist safety supervisors in making better and more efficient identification of potential violations.

The model will help make three decisions:

PASS: Not enough confidence in the detection of helmet violation to act upon.

FAIL: Model gives high confidence that a helmet violation has been detected, and a safety check is needed.

HUMAN REVIEW: It is uncertain what the model is, and the image is shown to a supervisor for decision making.

Positive class is NO-Hardhat, if a worker is identified who is not wearing a safety helmet.

There are two major types of errors.

False Positive (FP): €10

This occurs when the AI gets it wrong about a worker being in a helmet when he or she is, or vice versa.

I take €10 as the cost as an incorrect alert could take time out of working day and a supervisor would have to do a check.

False Negative (FN): €200

Occurs when the AI is unable to see a worker without a helmet.

I will take €200 as the cost as a risk of getting hurt continues if the safety violation is not identified. This is an example of what the cost of a typical workplace accident is likely to be, not the cost of a real accident.

True Positive (TP): €0 additional classification-error cost

The AI accurately detects a helmet violation. An investigation and/or an intervention is treated as a separate expense.

Three additional classification-error costs: True Negative (TN): €0

The AI identifies an image correctly, even if there is no labelled violation of the helmet laws.

These are not industry costs but assumptions for this assignment.

It is assumed a false negative is 20 times more costly than a false positive.

So, the model needs to focus on identifying safety issues, even if it generates a number of extra alerts. Review by humans can help minimize unnecessary interruptions.

## Dataset

I selected a public dataset from Roboflow Universe.

Document Type: RYG document

Workspace: ppe-detection-e2ly1

Project: ppe-detection-huwcd

Version: 1

Dataset link: https://universe.roboflow.com/ppe-detection-e2ly1/ppe-detection-huwcd

Task: Object Detection

Number of images: 6,747

There are 5 classes in the dataset:

Person

Hardhat

Safety Vest

NO-Hardhat

NO-Safety Vest

These images depict scenes with people and protective gear, possibly including construction workers.

The data set was chosen as it is directly related to the business problem. The NO-Hardhat class enables the model to recognize what the safety violation the company is trying to detect.

The other classes provide additional information about people and protective equipment in the image.

No additional images will be taken, labels will not be altered, and there will be no private forks created using the original public dataset.

In the business decision, I'll consider an image to be positive if there's at least one NO-Hardhat labeled instance in it.

Images that don't have an instance marked in the NO-Hardhat column are considered negative for evaluation. This does not mean that the actual scene is safe, as labels may not be complete.

## What the dataset's composition implies

There are three subsets of 6,747 images in the dataset.

Test set: 1762 images (about 16%)

Test set: 281 images (about 3%)

The test set consists of 326 images (around 5%).

The model is trained with the training set to identify objects. The validation set is for model development, and the test set is used to give a final independent evaluation.

The data set is automatically oriented and images are resized to 640 × 640, with stretching. No augmentations to the dataset.

The amount of annotated examples per class is important because there may be more than one person and object in an image.

The number of annotations exactly: This will be computed before training on the downloaded YOLO labeled annotations. The total on the image cannot be used to make these counts.

Typical object size: To use a median of normalized bounding-box width, height and area.

There are a few salient features in the dataset.

Firstly, about 84% of the images are used for training, providing the model with enough images to learn from.

Second, the test split consists of just 326 images. This makes it more difficult to be precise about the model's performance, especially for less frequent classes.

Third, it's the most important of all classes (NO-Hardhat). The precision and recall of this class can be unstable if there are relatively few instances in the class.

Fourth, the sizes of the objects may be different. Small and/or distant helmets may be less noticeable than large, conspicuous helmets. Sometimes, stretching images to square may also result in distortion of the proportions of the objects.

Lastly, pictures found in Roboflow can be different to actual construction-site camera footage. Thus, the results of this dataset cannot be generalized to all construction sites.

## Finetuning and baselines

We use YOLO11n for object detection in the selected architecture.

The model I selected is called YOLO11n and is a relatively small model that can recognize a number of object classes and their respective locations within images.

The setup for the training is:

Model: YOLO11n

Task: Object Detection

Input image size: 640 × 640

This dataset is provided as an example of a PPE detection.

Number of training epochs: 30 in the beginning, but depending on the actual training run.

Two baselines will be compared to the model on the same test split.

Baseline 2: Transfer learning from another dataset.Baseline 3: Train from scratch with COCO.

The original YOLO11n model will be tested without additional training.

COCO is not defined to have a class for NO-Hardhat; it introduces the Person class. This means it can't identify the exact safety violation without further logic or training.

Baseline 2: Trivial model

This baseline will forecast no helmet violations.

It serves as a baseline to assess the effectiveness of the trained model in detecting unsafe situations.

In the case of trivial baseline, NO-Hardhat recall is zero, which means that all violations are unlabelled. It also has zero image-level false-positives.

The assessment will be based on the comparison of AP50, mAP50-95, precision and recall of the test set.

The final results will be inserted after executing the training and evaluation pipeline. The values in Roboflow's separately published model will not be shown as my own model's values.
## Per-class evaluation and failure modes

The model will be tested independently for each of the five classes.

The persons are Person, Hardhat, Safety Vest, NO-Hardhat or NO-Safety Vest.

The most important class is NO-Hardhat, which directly represents the helmet safety violation.

Recall – The number of actual violations that the model correctly identifies.

Precision is the proportion of true violations out of the total number of predicted violations.

NO-Hardhat recall is of special significance since the cost of not recalling a violation is inferred to be greater than that of a false alarm.

The Final Assessment will consist of:

Precision and recall of each class.

All classes to receive AP50.

Overall AP50 and mAP50-95.

A confusion matrix.

Examples of correct and incorrect predictions.

Potential failure modes include distant workers, small helmets, poor lighting, partial occlusion and confusing background objects.

These are hypotheses to investigate, rather than observed findings. The final report should show failures based on the actual test predictions.

## Cost matrix and operating point

The goal is to pick a value of the confidence level that will minimize the expected business costs.

My assumed cost matrix is:

False Positive: €10

False Negative: €200

Classification-error cost: €0.00

Additional classification-error cost: €0 (True Negative)

Human Review: €1 per reviewed image

The cost of a false positive is €10, despite being an unnecessary investigation or interruption.

The cost of the false negative is €200 – the estimated financial impact if a violation was not discovered.

The €1 review cost is an estimate of the planning effort required which includes the time of the supervisor and administrative overhead.

All costs are assumptions and should be tailored with actual company data prior to deployment.

I'm going to assume a production base rate of 10%, which means that 10% of images have at least one helmet violation.

Based on that, I would predict that I'd have around 100 good images and 900 bad images out of 1000 images of production.

If none of the images are checked, and all violations are missed:

100 false negatives × €200 = €20,000 expected cost per 1,000 images.

This yields the price of inaction within the framework of the simplified assumptions.

The cost of a model without human review is:

Total cost = (False Positives × €10) + (False Negatives × €200)

In the event of a human review model, I will additionally supply €1 per reviewed picture and correct any remaining mistakes after the human review.

The operating threshold is determined by performing the test with various confidence thresholds and choosing that one which best matches the model's test predictions.

The selected threshold will be based on the one with the lowest expected cost and be adjusted to the base rate for the production assumed.

The desired threshold value might be influenced by the desired goal of recall, because a false negative is much more costly than a false positive.

However, the selected confidence threshold is hard to estimate beforehand to get the model predictions.

The sensitivity of the recommendation will be investigated by allowing the violation rate and/or the error costs to vary from the assumptions.

## What a person still has to check

Two confidence thresholds will be used to operate the system.

If the predicted NO-Hardhat confidence falls below the lower threshold, then the image is given the PASS mark.

If the confidence falls in between the two thresholds, the image is labelled HUMAN REVIEW.

When the confidence is equal to or greater than the upper bound, the image is marked as FAIL and a safety response is activated.

The FAIL does not signify that the behavior is actually unsafe; it simply means that the behavior may be unsafe.

For planning the workload, I assume 500 images per day will be processed in the company.

I guess a supervisor takes about 30 seconds to check each image.

Alternative 1: Doing nothing

Assuming a violation rate of 10%:

500 images × 10% = 50 images per day that are in violation.

If all the violations are not caught:

50 × €200 = €10,000 expected daily cost.

Alternative 2: To review manually all the material

500 images × 30 seconds = 15,000 seconds.

This is 250 minutes, or 4 hours/10 minutes per day.

The estimated daily review cost is:

500 × €1 = €500.

This translates into 0.52 full-time-equivalent (FTE) staff members when assuming an 8 hour workday.

The scenario assumes that there are no mistakes by the reviewers, which is an optimistic assumption.

AI takes all decisions:AI decides everything:

This model is used to automatically pass or flag all pictures with no human review.

This eliminates the expense of routine reviews, but there is a cost associated with false positives and false negatives.

As the false negative cost is high, if there are relatively few violations even this can be costlier.

The actual expected cost will be determined once the model has been trained.

To use AI technology, but review it with human eyes.

The model automatically works on clear cases and passes uncertain predictions to a supervisor.

As an illustrative workload: Assume that 20% of images need to be reviewed.

500 images × 20% = 100 reviewed images per day.

100 × 30 seconds = 3,000 seconds.

This translates to 50 minutes of human review a day.

Assuming that there are no other classification errors left, the review cost would be €100 per day.

If an eight-hour worker could see 960 images a day, each for 30 seconds, what is their total capacity?If an 8-hour worker could read 960 images a day, each for 30 seconds, then what is the total capacity that this worker can read in an 8-hour period? This is an illustrative workload that would need about 0.10 FTE positions.

This is an illustrative example of a plan, not an actual plan for review sharing.

The lower threshold, upper threshold, review percentage, staffing requirement and total system cost will be chosen from the predictions evaluated, as actual.

The last recommendation should be compared with the estimated €10000/day doing nothing and the estimated €500/day manual review.

If the hybrid system under the assumptions stated does not perform better than manual review, I will not say that the AI is necessarily superior, I'll just say it's not better.

## Limitations & next steps

The first restriction is that the data set might not be representative of all possible construction sites.

The lighting, camera position, weather and/or movements of workers may vary from the training images.

The second restriction is that helmets can be extremely small or understated. This can make it more difficult to accurately identify them.

The third drawback is the number of images in the test set is limited to 326, limiting the confidence of performance estimates in less common classes.

The fourth limitation is that the economic costs and assumed 10% violation rate is illustrative and not measured company data.

The fifth restriction is that the analysis assumes that there is a correct resolution of uncertain cases, which the human reviewers would make. Human reviewers do make mistakes, too.

One more restriction is that not everybody has a helmet on does not mean they don't have one on.

If I had more time I would try to apply the model to some more pictures from the construction site and compare the results in different lighting conditions and test the model with more challenging inputs.

Also, I would get data from the companies concerning how long it takes for them to review and how many safety violations and operating expenses they incur.

The critical next step is to fill in the actual confusion matrix prior to selecting the final operating thresholds.

## Generative AI use disclosure

I used ChatGPT to help understand the assignment requirements, organize the project, explain technical concepts, and draft parts of the report.

<!-- BEGIN MEASURED RESULTS -->
## Measured results from my training run

These results come from my trained model and test predictions.

### Dataset composition
- train: 5646 images; 1251 contain NO-Hardhat.
- val: 775 images; 242 contain NO-Hardhat.
- test: 326 images; 47 contain NO-Hardhat.

Annotated objects per class:
- Hardhat: 6813
- NO-Hardhat: 2531
- NO-Safety Vest: 1487
- Person: 7168
- Safety Vest: 8019
Median NO-Hardhat box: 12.6% of image width, 11.7% of height, and 1.59% of image area.

### Model evaluation
- Training epochs: 15
- Input size: 640
- Training device: 0
- Overall AP50: 0.6777
- Overall mAP50-95: 0.4846

Per-class performance:
- Hardhat: precision 0.954, recall 0.713, AP50 0.874, mAP50-95 0.603.
- NO-Hardhat: precision 0.308, recall 0.448, AP50 0.273, mAP50-95 0.128.
- NO-Safety Vest: precision 0.645, recall 0.444, AP50 0.477, mAP50-95 0.254.
- Person: precision 0.846, recall 0.756, AP50 0.828, mAP50-95 0.666.
- Safety Vest: precision 0.903, recall 0.896, AP50 0.936, mAP50-95 0.772.

The pretrained COCO baseline has no NO-Hardhat class, so its direct zero-shot recall for this violation is zero. The trivial predict-nothing baseline also has zero violation recall. Neither provides a directly comparable five-class AP score.

### Cost and thresholds
- Production violation base rate assumed: 10%.
- False positive: EUR 10.
- False negative: EUR 200.
- Human review: EUR 1 per image.
- Validation-selected single threshold: 0.020.
- Lower review threshold: 0.010.
- Upper review threshold: 0.470.
- Hybrid test cost per 1,000: EUR 1266.37.
- AI-alone test cost per 1,000: EUR 3960.19.
- Doing nothing: EUR 20,000 per 1,000 cases.
- Reviewing every case: EUR 1,000 per 1,000 cases.

### Human review and confusion matrix
- True positives (automatic FAIL): 9
- False positives (automatic FAIL): 3
- True negatives (automatic PASS): 201
- False negatives (automatic PASS): 2
- Reviewed positives: 36
- Reviewed negatives: 75
- Test review share, production-adjusted: 31.9%
- Expected images reviewed daily: 159.27
- Expected review minutes daily: 79.63
- Estimated reviewer staffing (FTE): 0.1659

Thresholds were selected using validation predictions and evaluated separately on test predictions. Costs are weighted to the assumed production base rate. Human reviewers are assumed to resolve reviewed cases correctly.

The measured results do not establish performance on real construction-site cameras. Individual prediction errors still require visual inspection.
<!-- END MEASURED RESULTS -->
