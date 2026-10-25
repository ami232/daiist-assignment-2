Cost-Sensitive Fire and Smoke Detection for Forest Surveillance

Name: Santiago Llorca
Student ID: 19952
Email: sllorca.ieu2022@student.ie.edu
Group: BBADBA 5A

1. Business Problem

Forest fires must be detected as early as possible to reduce damage to ecosystems, infrastructure and human life. This project investigates a computer vision system that analyses images from outdoor surveillance cameras and selects one of three actions for each frame: take no action, send the frame for human review, or trigger a priority alert.

The system is designed as a decision-support tool for a forest surveillance control centre. When a frame is sent for review, an operator examines it and decides whether further action is required. A priority alert is intended to receive immediate attention from the operator, who may notify the fire brigade or a forestry response team if necessary. Neither action automatically confirms a fire or dispatches emergency services.

The main positive class is defined as the presence of fire or smoke in a frame. Both classes are relevant indicators of a potential forest fire, although neither necessarily proves that an uncontrolled forest fire is occurring. Fire and Smoke are also evaluated separately because their visual characteristics and detection difficulties differ.

The main objective is to minimise the expected total cost of missed detections, false alarms and human review while respecting the limited capacity of operators.

The initial operational scenario assumes ten cameras, each analysed once every five minutes. This produces 288 frames per camera per day, or 2,880 frames in total. Two operators are assumed to provide four effective hours of review per day each. At two minutes per case, their combined capacity is 240 reviews per day.

The cost of a missed fire incident is assumed to be €1,000, while human review costs €0.60 per case, based on two minutes of work at €18 per hour. The missed-incident cost is therefore approximately 1,667 times the cost of a single review. This difference reflects the potentially serious consequences of delayed fire detection compared with the limited labour cost of examining an image. However, the €1,000 figure is a modelling assumption rather than a measured estimate of the average economic damage caused by a missed forest fire.

The three-action policy is intended to balance these competing costs while preventing the review workload from exceeding the available human capacity.

2. Dataset

The project uses the Fire and Smoke dataset published on Roboflow Universe.

Property	Description
Workspace	object-detection-yxxz2
Project	fire-smoke-lwjte
Version	1
Number of images	4,000
Image dimensions	640 × 640 pixels
Resize method	Stretch
Licence	MIT
Fork	No

Dataset URL: https://universe.roboflow.com/object-detection-yxxz2/fire-smoke-lwjte/dataset/1

The dataset contains images annotated with bounding boxes for Fire and Smoke. It is suitable for an initial object-detection experiment because it provides examples of both target classes and identifies their locations within images. Bounding-box annotations allow the model to learn where the relevant objects appear, rather than simply assigning a class to an entire image.

However, the dataset does not fully represent the intended deployment environment. It contains internet-sourced images, including stock photographs with watermarks, candles, fireplaces or chimneys, and scenes from roads, ports and hangars. Some images may therefore contain flames or smoke without depicting a forest fire. The model could learn visual patterns that do not reliably indicate an uncontrolled wildfire.

An alternative dataset, mediiot/fire-smoke-znprn, version 10, was considered but rejected. It contained 8,687 images across seven classes. A similarity analysis indicated that approximately 81% of its test images had an almost identical counterpart in the training set, while approximately 99% were highly similar to training images. Such extensive overlap would make it difficult to establish whether the test results reflected genuine generalisation to unseen images.

The selected dataset was therefore used for the experiment, although its own potential overlap between splits and limited representativeness remain important limitations.

3. Dataset Composition and Leakage Analysis

The selected dataset is divided into training, validation and test sets.

Split	Images	Background images	Fire instances	Smoke instances
Training	3,202	0	5,078	4,309
Validation	403	0	591	549
Test	395	0	660	631
Total	4,000	0	6,329	5,489

The dataset contains 11,818 annotated object instances. The number of instances exceeds the number of images because a single image may contain multiple Fire or Smoke bounding boxes.

The absence of background-only images is a significant limitation. Every image contains at least one annotated positive object, so the dataset does not represent the large number of ordinary surveillance frames in which neither fire nor smoke is present. Consequently, it cannot provide a reliable direct estimate of the false-alarm rate in empty scenes.

Bounding-box size also differs between the two classes. Fire has a median bounding-box area equal to 1.7% of the image, with the 10th and 90th percentiles at 0.31% and 15.2%, respectively. Smoke has a median area of 7.2%, with corresponding percentiles of 1.82% and 22.4%.

The small median area of Fire indicates that flames frequently occupy only a small part of an image. This can make detection difficult, especially when the fire is distant, partially obscured or visually similar to the background. Smoke generally occupies a larger area, but its diffuse boundaries and similarity to clouds, fog and haze introduce different challenges.

A similarity analysis identified potential data leakage between the splits. Approximately 17% of test images were almost identical to training images, and 29% were highly similar. The median reported image distance was 14. In addition, five original images were shared with the test set and 121 with the validation set.

These findings suggest that some evaluation images may not be fully independent of the training data. This may lead to optimistic performance estimates, although the extent of its effect cannot be established from similarity statistics alone.

Validation performance must therefore be interpreted cautiously. The validation set is affected by potential overlap and does not contain background-only images. It can support model development, but it cannot establish a reliable operating threshold for false alarms in realistic surveillance conditions.

Finally, several visual elements in the dataset, including watermarks, indoor flames, roads, ports and hangars, would not necessarily appear in the same form in a forest observation tower. Real surveillance images would introduce additional challenges such as compression artefacts, changing illumination, weather conditions, camera movement and distant smoke.

4. Fine-Tuning and Baselines

The selected model was YOLO11s, a relatively small object-detection model that provides a practical balance between detection capability and computational requirements. Its size makes it a reasonable starting point for a potential surveillance application, where inference speed and deployment resources matter.

The model was fine-tuned using the following configuration:

Parameter	Setting
Model	YOLO11s
Training epochs	50
Batch size	16
Image size	640 × 640
Random seed	0
Environment	Google Colab with NVIDIA T4
Optimiser	AdamW, selected by Ultralytics
Ultralytics version	8.4.187
Completed run duration	0.687 hours

Training was run on Google Colab (Ultralytics 8.4.187). Evaluation, the cost analysis and the dashboard were run locally on CPU with Ultralytics 8.4.173.

The first training run was interrupted by a Colab disconnection at around epoch 39 and has no recorded duration. A second run was started from epoch zero and completed all 50 epochs. The reported duration of 0.687 hours corresponds to this completed run, which produced the weights used in the evaluation. Therefore, the reported duration does not include the interrupted run.

The following baselines were considered on the test set:

Model or baseline	Reported result	Interpretation
Fine-tuned YOLO11s	mAP50 = 0.519; mAP50–95 = 0.293	Detects Fire and Smoke, but performance remains limited
COCO zero-shot	mAP50 = 0, by construction	The value was assigned in the evaluation code because the generic COCO model does not provide the required Fire and Smoke classes
Predict nothing	mAP50 = 0, by definition	The value was assigned in the code because the baseline produces no detections
Always alert at frame level	Accuracy = 100%	All 395 test frames contain annotated Fire or Smoke

The COCO zero-shot baseline did not provide the required target classes. Only the loosely related fire hydrant class was available, and it was detected in one of the 395 test frames. Its reported mAP50 of zero was assigned by construction in the evaluation code; it was not obtained from a standard evaluation of the target classes. Similarly, the zero mAP50 for the predict-nothing baseline was assigned by definition because it produces no detections.

The always-alert baseline requires particular caution. Its 100% frame-level accuracy does not demonstrate a useful detection system. It results from the absence of negative frames in the test set. In a realistic surveillance environment, where most frames are expected to contain neither fire nor smoke, always alerting would generate an excessive number of false alarms.

The fine-tuned model achieved an AP50 of 0.727 on validation and 0.519 on test, a decrease of 0.208. This gap may indicate overfitting or differences between the validation and test distributions. Potential data leakage, annotation inconsistencies, small fire objects and differences in visual conditions may also contribute. However, the available results do not establish which factor is responsible, and further experiments would be needed to identify the cause.

Overall, fine-tuning improves the model's ability to detect the target classes compared with the chosen baselines, but the test results do not establish reliable real-world performance.

5. Per-Class Performance and Failure Analysis

The fine-tuned model was evaluated separately for Fire and Smoke using precision, recall, AP50 and mAP50–95.

Test class	Precision	Recall	AP50	mAP50–95
Fire	0.571	0.501	0.547	0.311
Smoke	0.647	0.403	0.491	0.275
All classes	0.609	0.452	0.519	0.293

Precision and recall in this table correspond to the best-F1 operating point. At a confidence threshold of 0.25, Fire precision and recall are 0.587 and 0.500, respectively, while Smoke precision and recall are 0.654 and 0.395. The recall values computed directly from the confusion matrix below (about 0.52 for Fire and 0.42 for Smoke) differ slightly because the two procedures match predictions to annotations differently.

Fire has the higher AP50 and recall, while Smoke has the higher precision. These results do not establish that either class is categorically easier or more reliable to detect. Instead, they indicate different trade-offs: the model retrieves a larger proportion of annotated Fire instances, while its Smoke detections are more precise at the reported operating points.

Validation results were more optimistic. Fire achieved an AP50 of 0.867, compared with 0.586 for Smoke. On test, the corresponding values were 0.547 and 0.491. The particularly large reduction in Fire performance may reflect differences between the splits, overfitting, small or distant flames, annotation variability or potential leakage. These are plausible explanations rather than proven causes.

The reported precision and recall values must also be interpreted in the context of object detection. They measure bounding-box predictions against annotated objects rather than simply classifying each complete frame as positive or negative. A missed bounding box is not necessarily equivalent to a completely missed fire incident, and multiple boxes in one image may correspond to the same event.

At a confidence threshold of 0.25, the dominant error is missed detection: 686 of the 1,291 annotated test objects (53%) are not detected, including 320 of 660 Fire boxes (48%) and 366 of 631 Smoke boxes (58%). Fire and Smoke are almost never confused with each other (one Smoke object was predicted as Fire), so the main weakness is failing to detect objects rather than mislabelling them. The matrix also shows 221 Fire and 117 Smoke predicted boxes that match no annotation; since every test frame contains annotations, these are unmatched boxes inside positive frames, not false alarms on empty scenes. At frame level, 47 of the 395 test frames contain no predicted box with confidence of at least 0.25.

To examine extreme cases, the eight test frames with the lowest maximum confidence were inspected. Five appear to show the same scene (a balcony with a faint plume of white smoke); the other three are elevated or aerial views of diffuse smoke over buildings or open land. They therefore represent roughly four distinct scenes rather than eight independent cases, and they are a deliberately extreme sample, not a representative one. The labelled objects appear to be faint or diffuse smoke rather than visible flames. Faint, distant smoke is the kind of early-warning signal that matters for forest surveillance, so these cases may be more relevant to the operational challenge than the candles or indoor flames found elsewhere in the dataset, although they say little about how the model would behave on real tower footage. Among the cases inspected, weak, low-contrast smoke is the clearest candidate for future work, but Fire also loses almost half of its annotated boxes and was not inspected.

The confusion matrix generated at confidence 0.25 is a box-level matrix, not a frame-level operational confusion matrix. Its values describe the correspondence between predicted and annotated objects and should not be interpreted directly as the number of frames correctly or incorrectly alerted.

For the main project objective, the combined Fire-or-Smoke category remains the most relevant positive event definition. Per-class results are retained to explain the model's behaviour, while the combined category aligns with the operational objective of identifying possible fire incidents.

6. Cost Model and Operating Point

The model's predictions are used to investigate a cost-sensitive decision policy with three actions: take no action, request human review, or trigger a priority alert.

The initial cost model includes the following assumptions:

Parameter	Value	Rationale
Cost of a missed fire incident	€1,000	Modelling assumption without empirical source; chosen so that a missed incident is about three orders of magnitude costlier than a review
Frames per incident	12	Twelve frames at one frame every five minutes equal one hour of a visible incident
Cost per missed positive frame	€83.33	€1,000 divided by 12
Human review cost	€0.60	Two minutes at €18 per hour
Additional cost of a false alert	€5	Assumption without empirical source; roughly 17 minutes of operator time at €18 per hour
Cost of dispatch without operator verification	€300	Assumption without empirical source; used only in the model-decides-alone comparison
Delay-related cost	10% of the missed-fire cost	Assumption without empirical source; a queued review is assumed to add a tenth of the cost of a missed frame
Base rate	0.0001	At 2,880 frames per day this implies about 0.29 positive frames per day, roughly one 12-frame incident every 42 days across the ten cameras

The €1,000 incident cost and €0.60 review cost provide the starting point for the economic comparison. The €5 false-alert penalty, €300 dispatch cost, 10% delay penalty and base rate were introduced as modelling proposals rather than measured operational quantities. They are retained as explicit assumptions for the simulation and should be validated before any real deployment.

Dividing the incident penalty by 12 frames gives €83.33 per missed positive frame. This simplifies the cost calculation but assumes that the incident cost can be allocated linearly across frames. In reality, missed frames are correlated, and the economic consequences depend on how quickly an incident is detected, the fire's development and whether other detection systems identify it.

Because the missed-fire cost is charged per frame, the ratio that drives the threshold is €83.33 to €0.60 (about 139 times) against a review, and about 15 times against a false alert (€0.60 plus €5). The 1,667 figure applies to the whole incident, not to a single frame.

The cost comparison per 1,000 analysed frames is:

Strategy	Expected cost per 1,000 frames
Take no action	€8.33
Review every frame	€600.00
Model decides alone	€6.18
Three-action policy	€6.20

Under the stated assumptions, the three-action policy has an estimated cost of €6.20 per 1,000 frames, compared with €8.33 for the no-action baseline. This represents an approximate 26% reduction in expected cost. However, the model-only policy is marginally cheaper, at €6.18 per 1,000 frames, so the three-action policy does not provide the lowest estimated cost under the initial assumptions.

The selected thresholds are 0.73 for both decision boundaries. Under this configuration, no frames are sent for human review. Of the 395 test frames containing Fire or Smoke, 293 do not trigger an alert and 102 do. This illustrates that the review band is empty at the chosen thresholds, so the nominal three-action policy effectively operates as a two-action policy.

The threshold of 0.73 also coincides with the point at which the final unmatched predicted box disappears from the false-alarm proxy used in the simulation. This helps explain why the selected operating point produces no human reviews at the base rate of 0.0001. However, the disappearance of an unmatched box in this proxy does not demonstrate that the detector would generate no false alarms on genuinely empty forest scenes. The threshold is therefore a property of the current dataset and proxy, not a validated operational threshold.

The threshold also has a substantial effect on the number of frames triggering an alert. At a confidence threshold of 0.25, 348 of the 395 test frames trigger an alert, corresponding to approximately 88%. At a threshold of 0.73, this falls to 102 frames, or approximately 26%. Although the higher threshold substantially reduces the alert volume, it also means that many frames containing annotated fire or smoke do not trigger an alert. The operating point must therefore balance alert workload against the risk of missing early signs of fire, and the current positive-only test set cannot establish whether the reduction in alerts is acceptable in practice. A frame counts as alerted here if any predicted box reaches the threshold, whether or not that box matches an annotation, so these figures are an optimistic measure of frame-level detection.

A sensitivity analysis shows how the assumed base rate changes the cost comparison:

Base rate	Three-action policy (€ / 1,000 frames)	Model only (€ / 1,000 frames)	No action (€ / 1,000 frames)	Reviews per day	Thresholds
0.00001	€0.62	€0.62	€0.83	0	0.73 / 0.73
0.0001	€6.20	€6.18	€8.33	0	0.73 / 0.73
0.001	€56.93	€61.81	€83.33	30	0.57 / 0.73
0.01	€461.76	€618.14	€833.33	217	0.48 / 0.73

The comparison with the model-only policy shows that human review provides no meaningful economic benefit at a base rate of 0.0001: the three-action policy costs €6.20 per 1,000 frames, compared with €6.18 for the model-only policy, and sends no cases for review. At a base rate of 0.001, the three-action policy reduces the estimated cost from €61.81 to €56.93, a reduction of approximately 8%. At a base rate of 0.01, the cost decreases from €618.14 to €461.76, a reduction of approximately 25%. These improvements are conditional on the cost model and, in particular, on the proxy used to estimate false alarms. They should not be interpreted as measured savings in a real forest surveillance system.

The base rate has a major influence because it determines how frequently positive events occur relative to all analysed frames. When positive events are rare, false alarms can dominate the workload even when the detector performs reasonably well on positive examples. As prevalence increases, missed detections become more prominent in the total expected cost, and human review may become more valuable.

The estimated cost advantage over doing nothing persists across the displayed scenarios. Nevertheless, the comparison is conditional on the assumed costs, prevalence and false-alarm behaviour. The test set contains no background-only frames, so the false-alarm rate in empty scenes cannot be measured directly. Instead, the simulation uses the highest confidence of a predicted box that does not match an annotated object as a proxy for false alarms.

This proxy is not equivalent to a measured false-positive rate on genuinely empty frames. The estimated threshold of 0.73 and the corresponding €6.20 expected cost per 1,000 frames are therefore not reliable estimates of real-world performance. In particular, scenes containing clouds, haze or other smoke-like structures could alter the optimal operating point substantially.

7. Human Review Capacity

The assumed surveillance system contains ten cameras, each producing 288 analysed frames per day. The total workload is therefore 2,880 frames per day.

Two operators, each contributing four effective hours per day, provide eight hours of human review in total. At two minutes per case, this corresponds to a maximum of 240 reviews per day, or approximately 8.3% of all analysed frames.

The cost-sensitive simulation produces the following review workloads:

At a base rate of 0.00001, the selected thresholds produce zero reviews per day.
At a base rate of 0.0001, the selected thresholds also produce zero reviews per day.
At a base rate of 0.001, the policy produces approximately 30 reviews per day.
At a base rate of 0.01, the policy produces approximately 217 reviews per day.

The highest scenario would require approximately 434 minutes of review per day, equivalent to 7.2 hours. At four effective hours per operator, this represents approximately 1.8 operators. Thus, two operators would nominally be sufficient, but the available spare capacity would be limited.

Alerts also consume operator time. At a base rate of 0.01 about 7 frames per day trigger an alert (2,880 × 0.01 × 0.258), bringing the total to roughly 224 cases against a capacity of 240. Note also that a base rate of 0.01 equals about 29 positive frames per day on ten cameras, roughly 2.4 incidents per day, which is implausibly high for forest surveillance; the 217-review scenario should be read as a stress test.

The three-action policy offers a way to allocate human attention selectively instead of reviewing every frame. Reviewing all 2,880 daily frames would require 96 hours of work at two minutes per frame, or 24 operators providing four effective hours each. By comparison, the model-only and no-action strategies do not create a human review queue in the cost model, but they also lack the selective verification that a review process can provide.

The capacity constraint is important because the real base rate may be higher than assumed, or the model may generate more uncertain detections than expected. If the number of cases exceeds 240 per day, the queue will grow unless the system reduces the review rate, prioritises cases more effectively or adds staff. Longer waiting times could delay the investigation of genuine fires and undermine the purpose of early detection.

The current policy does not establish that the workload will remain within capacity in real operation. Its review volumes are simulation outputs based on provisional assumptions, and the lack of negative test frames prevents reliable estimation of the actual arrival rate of false alerts. A more realistic system would also group consecutive frames into incidents, prioritise persistent or rapidly developing fire evidence and monitor queue length over time.

8. Limitations and Future Work

Several limitations restrict the conclusions that can be drawn from this experiment.

1. Internet images rather than surveillance video. The dataset consists of internet-sourced images, including stock photographs and watermarks, rather than continuous video captured by forest surveillance cameras. The visual distribution may therefore differ substantially from deployment conditions. Future work should collect representative footage from fixed outdoor cameras under different weather, lighting and distance conditions.

2. No background-only images in the test set. All 395 test images contain annotated Fire or Smoke. Consequently, the experiment cannot directly measure false alarms in scenes without either class. A dedicated negative test set should include ordinary forest scenes, clouds, fog, haze, sunlight, roads and other potential sources of confusion.

3. An imperfect false-alarm proxy. The cost simulation uses the highest confidence of a predicted box that does not match an annotation as a substitute for false alarms on empty frames. This does not accurately reproduce the behaviour of the detector on genuinely negative images. The threshold of 0.73 coincides with the disappearance of the final unmatched predicted box used by this proxy, but this does not demonstrate that the model would produce no false alarms on empty forest scenes. Future work should evaluate the detector on a dedicated negative dataset and recalibrate the policy using a measured false-positive rate.

4. Potential data leakage. Approximately 17% of test images were almost identical to training images, and 29% were highly similar. In addition, five original images were shared with the test set and 121 with the validation set. Five of the eight lowest-confidence test frames also appear to be near-duplicates of one another, so the 395 test frames are not 395 independent samples. These overlaps may inflate some performance estimates or complicate comparisons between splits. Future experiments should group duplicate and near-duplicate images before splitting the data.

5. Small Fire bounding boxes. Fire has a median box area of only 1.7% of the image, and the lower percentile is smaller still. Distant flames may be difficult to distinguish from background objects. Future work should examine higher-resolution inputs, small-object detection methods and performance stratified by object size.

6. Ambiguous class labels. Some examples contain candles, fireplaces or chimneys rather than forest fires. Although these objects may correctly represent flames, they do not necessarily represent the target operational event. The annotation policy should distinguish the presence of flame from evidence of an uncontrolled outdoor fire, or the system's intended scope should be explicitly limited to visual fire and smoke detection.

7. Simulated costs and prevalence. The €1,000 missed-incident penalty, €5 false-alert penalty, €300 dispatch cost, 10% delay penalty and base rate are assumptions rather than empirical estimates. The linear allocation of incident cost across 12 frames is also a simplification. Future work should use operational data to estimate costs and examine a wider range of scenarios.

8. No temporal evaluation. The model is evaluated on individual images rather than continuous video. The experiment does not measure time to detection, persistence across frames, incident-level recall or duplicate alerts for the same event. Future work should evaluate video sequences and introduce temporal aggregation to combine evidence across consecutive frames.

9. Limited evidence about real-world generalisation. The difference between validation AP50 of 0.727 and test AP50 of 0.519 indicates a substantial performance gap. Potential causes include distribution shift, overfitting, leakage and annotation differences, but further experiments are needed to determine their contributions. Independent testing on footage from different cameras and locations would provide stronger evidence.

10. The no-action baseline is incomplete. The cost model treats taking no action as having no immediate review cost, but real forest surveillance systems may already use human observation, other camera systems, public reports or other detection technologies. The relevant comparison should therefore be against the existing operational process rather than an assumption that no other detection mechanism exists.

11. Limited interpretation of the always-alert baseline. Its 100% frame-level accuracy is a consequence of a test set containing only positive frames. It does not indicate good precision or an acceptable operational alert rate. A balanced evaluation with representative negative scenes is required before meaningful operational comparisons can be made.

12. Operational costs are not fully modelled. The present framework simplifies the consequences of missed fires, false alerts, human review and delayed responses. It does not fully model the time-dependent growth of a fire, correlations between consecutive frames, operator fatigue or the cost of queue overload. These factors should be incorporated into a more realistic decision model.

9. Conclusion

This project demonstrates how an object detector can be combined with an explicit cost model to compare different decision policies for forest-fire surveillance. The fine-tuned YOLO11s model achieves a test AP50 of 0.519, but the dataset and evaluation design are insufficient to establish reliable real-world performance. The dominant detection error is missing annotated objects, with qualitative inspection of eight extreme failure cases highlighting faint or diffuse smoke, in several cases seen from aerial views, as a particularly important failure case.

The cost analysis shows that the three-action policy provides no meaningful advantage over the model-only policy at a base rate of 0.0001, but reduces the estimated cost by approximately 8% at 0.001 and 25% at 0.01. These savings depend on the assumed costs and the imperfect false-alarm proxy, so they should be treated as simulation results rather than demonstrated operational benefits. The reduction in alerted frames from 348 at a threshold of 0.25 to 102 at 0.73 (counting any predicted box, not only correct ones) further illustrates the trade-off between workload and missed detections.

The next steps are to add representative negative surveillance images, remove potential data leakage, validate the cost assumptions and evaluate continuous video before drawing conclusions about deployment in real forest surveillance systems.

10. AI Use Disclosure

AI tools were used to support different stages of this project, including technical assistance, dataset research, evaluation support, cost modelling, dashboard development, project organisation and report preparation.

Claude was used to assist with dataset searches and comparisons, the development and review of scripts (training, evaluation, costing.py, failure analysis and the Gradio dashboard), model evaluation, the organisation of the project structure and the interpretation of experimental results. Claude also reviewed the report, checked its figures against the outputs, and drafted or edited parts of its text, including the failure-analysis paragraphs in section 5 and several corrections in sections 6 to 8.

Several initial modelling assumptions, including the false-alert penalty, dispatch cost, delay-related penalty and base rate, originated from proposals made during AI-assisted discussions. They were treated as assumptions for the cost simulation, not as empirically established operational costs.

Gemini was used to assist with restarting the Google Colab training process after the initial run was interrupted by a disconnection.

The project framing, the cost figures I confirmed, the choice of thresholds and the conclusions drawn are my own decisions.