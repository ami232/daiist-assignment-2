# Assignment 2 Report

- **Name**: Beatriz Cruces Alvaro
- **Student ID**: 20082
- **Email**: bcruces.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Business problem, and the cost of being wrong
My project supports quality control in the textile industry. Before fabric is cut and sewn into clothes, an inspector looks at each metre and decides whether to accept it, scrap it or check it more closely. My model supports that decision. One image represents one metre of denim, and a "positive" means that the metre has at least one defect. I assume that the inspector scraps any metre with a visible defect.

There are two kinds of mistakes:

A missed defect costs about €30. The metre reaches the cutting stage, so I lose the fabric (€10), the work already done (roughly €3–6) and I may get a customer complaint (my own assumption). Scrapping a good metre costs about €10, the value of the metre.

For each case, a missed defect is three times worse. But if only 5% of the metres are defective, good metres are much more common, so false positives add up to more. In my results, about €365 of the €423 per 1,000 metres comes from scrapping good metres, and only about €58 from missed defects. That is why I measure success in euros and not in accuracy.

## Dataset

I used the Textile Defects dataset from Roboflow Universe: workspace deepti-raj-7uqio, project textile-defects-tqim4, version 2 (https://universe.roboflow.com/deepti-raj-7uqio/textile-defects-tqim4/dataset/2). The license is CC BY 4.0, and I used it as it is, without forking it.

The images show dark blue denim and grey fabric, with the defects marked by boxes. Version 2 has three classes: Hole, Knot and Stain. I chose it because version 1 contains flipped copies of images, which could end up in different splits and make the test score look better than it is.

It suits my problem because it has both defective and defect-free fabric (about 30% of the images have no defect), so I can measure how often the model scraps good material. I treat each image as one metre, but I cannot show that the images come from a real production line, so the results would need to be checked in a real factory.

## What the dataset's composition implies

There are 1,258 training images, 363 validation images and 182 test images. The test set has 52 images without defects and 204 boxes: 95 Knot, 72 Hole and 37 Stain. A typical defect covers about 5% of the image in the test set.

About 70% of my test images have a defect, but I assume that only 5% of the metres in a real factory do. For this reason I reweight the errors with 5% when I calculate costs. 

There are three more limitations:

72 of the 182 test images come from the same original photo as a training or validation image, so my test scores are probably too optimistic.

With only 52 defect-free images, I cannot measure the false-positive rate precisely.

Stain has only 37 test instances, so its results can change a lot with one mistake.

## Finetuning and baselines
I fine-tuned YOLO11n, a small object detection model pretrained on COCO. I chose detection and not classification because the dataset has boxes, and I need them to know if a metre has any defect. I trained it for 60 epochs on a MacBook Air with an Apple M2 chip, which took about 2.2 hours (image size 640, batch size 16, seed 0).

On the test set, the model reaches an AP50 of 0.79. At metre level, with a threshold of 0.22, it finds 96% of the defective metres, and 98% of what it flags is really defective. Its cost is €423 per 1,000 metres.

I compared it with two baselines on the same test set. The COCO model without fine-tuning costs €1,373, even with its best threshold, and doing nothing costs €1,500. So my model is about three times cheaper. I expected COCO to do badly, because it was never trained on fabric defects. I do not compare them by accuracy, because predicting "defect" for every metre already gets 71% right on my test set.

## Per-class evaluation and failure modes

The model performs differently depending on the type of defect. Hole is the best-performing class, with an AP50 of 0.89, a precision of 0.90 and a recall of 0.88. Knot also achieves good results, with an AP50 of 0.83. However, Stain is the weakest class, with an AP50 of 0.66, a precision of 0.72 and a recall of 0.68. This means that the model has more difficulty identifying stains than holes or knots. One possible explanation is that Stain has fewer examples in the test set, with only 37 instances compared with 72 for Hole and 95 for Knot.

Overall, the model achieves an AP50 of 0.79 and a recall of 0.77. This indicates that it detects most defects, although some are still missed. Since the business decision is whether a metre of fabric contains any defect, missing a defect is particularly important. The next step is to inspect the images where the model failed and analyse the confusion matrix to understand which defects are being confused. This would help identify the main causes of errors and determine how the model could be improved.

After checking the 7 images, I think some of the errors come from the labels, and some come from the model reading normal parts of the fabric as defects.

Missed defects (images 1 to 5). The labelled boxes were very thin or very small, and in several of them I could not see a clear defect inside the box. So either the label is doubtful, or the defect is too faint for the model to find.

Good metres that were scrapped (images 6 and 7). In image 7, the model drew a Hole box on a mark on the fabric, but the dataset has no label there. This is a false box (a box where no defect is labelled) and not a confusion between classes, which agrees with the confusion matrix. [In image 6 and 3] 

I only checked 7 images, so these are observations and not proof. A textile expert should review the labels.

## Cost matrix and operating point

The cost matrix is based on the financial consequences of making incorrect decisions about fabric quality. A false positive, which occurs when the model rejects a metre of good fabric, costs €10, based on the approximate value of one metre of denim (€9.70). A false negative, which occurs when the model fails to detect a defect, costs an estimated €30. This includes the fabric cost, work already completed and potential customer complaints. Therefore, missing a defect is considered three times more expensive than rejecting good fabric. Correctly accepting a good metre has a cost of €0.

The model assumes that 5% of fabric metres contain defects. This is an assumption rather than a figure obtained from a real factory. Human inspection is estimated to cost €0.12 per case, based on a labour cost of €14.29 per hour and 30 seconds per inspection. The human inspector is assumed to detect 90% of defects and incorrectly reject 2% of good fabric.

To choose the best operating point, different confidence thresholds are compared by calculating the expected cost per 1,000 metres. The proposed system accepts fabric when the defect score is below 0.12, sends cases with scores between 0.12 and 0.37 for human inspection, and rejects fabric when the score is at least 0.37. The estimated cost is €432 per 1,000 metres on the test set. These results should be interpreted cautiously because the defect rate and several cost estimates are assumptions. In particular, collecting real factory data on defect frequency would help determine whether the selected thresholds minimise costs under actual production conditions.

The most important assumption is that 5% of the fabric in a real factory has defects. I chose this number, but I do not have a source to confirm it. My dataset cannot tell me the real defect rate because around 70% of its images contain defects. This matters because it can change whether human review is worth it. At a 5% defect rate, the model alone costs €423 per 1,000 metres, compared with €432 when human review is included. However, if the defect rate increases to 10%, the cost of using the model alone rises to €785, while the model with human review costs €457. This shows that the best option depends on the actual defect rate, so I would need data from a real factory to make a more reliable decision.

## What a person still has to check

The proposed system combines automatic detection with human inspection to reduce the risk of costly mistakes. Fabric with a defect score below 0.12 is accepted, fabric with a score between 0.12 and 0.37 is sent to a human inspector, and fabric with a score of 0.37 or higher is rejected. Approximately 3.9% of metres require human inspection. Assuming a factory processes 20,000 metres per day, this represents around 785 inspections per day. If each inspection takes 30 seconds, the total workload is approximately 6.5 hours per day, equivalent to around 0.8 full-time employees working eight-hour shifts.

The estimated cost of the model with human review is €8,632 per day. By comparison, doing nothing costs approximately €30,000 per day, having a person inspect everything costs €9,182 per day, and allowing the model to decide on its own costs €8,462 per day. Although the model alone has the lowest estimated cost under the base assumptions, human review provides an additional safeguard against expensive errors. The final choice depends on whether the factory prioritises the lowest estimated cost or additional protection against defective fabric passing through production.

I would keep the human-review band because it provides extra protection against costly mistakes. In the base case, it costs €432 per 1,000 metres, compared with €423 for the model alone, and requires around 0.8 people per day. However, if 20% of the fabric has defects or missing a defect becomes much more expensive, human review can reduce costs by about half. I also think it is useful because the model may perform worse in a real factory than it does on my test set. Therefore, even though the review band is slightly more expensive in the base case, I would keep it to reduce the risk of more costly errors.

## Limitations & next steps

The project has several limitations that should be addressed before implementing the model in a real factory. First, some test images share an origin with images in the training or validation sets, which may make the results look better than they really are. This could be addressed by splitting the dataset according to the original photographs so that related images cannot appear in different sets. Second, the test set contains only 52 defect-free images and 37 Stain instances. Collecting more examples of these cases would improve the reliability of the evaluation. The class labels should also be reviewed with a textile expert because some Knot and Stain annotations may be incorrect.

Another limitation is that the defect rate, error costs, daily production volume and human inspection performance are assumptions rather than verified factory measurements. These values should be collected from real production data and used to recalculate the cost matrix and decision thresholds. The model should also be tested on images captured directly from an industrial production line, using the actual fabric types and lighting conditions. Finally, future experiments could compare different YOLO model sizes and repeat training with different random seeds. These improvements would help determine whether the model can deliver reliable financial benefits outside the experimental dataset.

## Generative AI use disclosure

### AI Disclosure

I used Claude to help develop the code for both the model training (`train.py`) and the application (`app.py`). Throughout the process, I worked interactively with the AI, discussing different approaches and making the key decisions myself. I chose the business problem, selected the dataset, decided on the modelling approach and cost assumptions, and interpreted the results. I also used ChatGPT to help structure the report and translate my explanations and ideas, into as clear structures English. The main ideas, decisions and conclusions are my own, while AI tools supported the coding process and the communication of my work.
