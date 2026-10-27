# Assignment 2 Report

Name: Jacobo Galindo Sanz

Student ID: 19738

Email: jgalindo.ieu2022@student.ie.edu

Group: BBA + BDBA "A"

## Business problem, and the cost of being wrong

This project examines how a hypothetical solar maintenance company could prioritize photographs showing possible cracks, dust, or bird droppings. Each photograph receives one of three outcomes. Pass means no immediate review while routine maintenance continues. Human review means a technician examines the photograph and creates an incident if an anomaly is confirmed. Automatic incident means creating a case directly for verification before physical intervention. The dashboard simulates these actions without creating external tickets.

The decision unit is one photograph, regardless of how many objects it contains. A positive photograph has at least one annotated cracked, dusty, or bird_drop object. The panel label identifies a panel and does not establish that it is healthy. Photographs without anomaly annotations are operational negatives, subject to annotation completeness.

The consequences of errors differ. An unnecessary review consumes staff time, while a false incident creates additional handling work. Missing an anomaly can delay investigation. The scenario assigns €0.20 to photo review, €1 to incident handling, and €5 to an unaddressed positive. The last figure represents a hypothetical €0.10 daily delay cost over 50 days. These are explicit assumptions, not measured industry costs or energy losses inferred from photographs. The system supports triage rather than certifying electrical performance or safety.

## Dataset

The project uses Susan's public Panel Solar dataset on Roboflow Universe, licensed under CC BY 4.0. The workspace is susan-ifblr, the project is panel-solar-bw945, and the version is 1. The source is https://universe.roboflow.com/susan-ifblr/panel-solar-bw945/dataset/1. The original public version was downloaded without creating a Roboflow dataset fork.

Its four classes are bird_drop, cracked, dusty, and panel. The version applies auto-orientation and stretching to 640 × 640 pixels, with no export-time augmentation. A photograph can contain multiple panel and anomaly annotations. Inspected examples include small panels photographed indoors, so the dataset supports a visual screening experiment but does not establish suitability for outdoor installations or aerial inspection.

The export mixes rectangular boxes and polygons. A separate detection-format copy retained existing boxes and converted polygons into enclosing axis-aligned boxes, preserving images, classes, and splits. This converted 440 training polygons, 113 validation polygons, and 61 test polygons. The resulting annotation totals were 11,031, 3,167, and 1,593. Original files remained unchanged. Conversion loses contour detail and may include background, but provides a consistent detection representation without manual relabeling.

## What the dataset's composition implies

The dataset has 6,493 images: 4,546 training, 1,299 validation, and 648 test. Training contains 57 bird_drop, 6,298 cracked, 251 dusty, and 4,425 panel objects. Validation contains 24, 1,796, 68, and 1,279 objects respectively. Test contains zero bird_drop, 946 cracked, 32 dusty, and 615 panel objects.

Independent image support is more limited than object counts suggest. Bird_drop appears in only six training photographs and three validation photographs, with none in test. Its test recall and AP are therefore unavailable. Dust appears in only 16 test photographs. Cracks dominate the anomaly data, making broad claims about the rarer classes unjustified.

At image level, training has 2,724 photographs with an annotated anomaly and 1,822 panel-only photographs. Validation has 748 and 551, while test has 387 and 261. No missing or empty label files were found. The positive proportions of approximately 57.6% in validation and 59.7% in test are not assumed to represent production.

Median bounding-box areas in training, validation, and test are respectively 21.95%, 23.00%, and 23.72% of image area for panels; 0.72%, 0.78%, and 0.64% for cracks; and 9.34%, 4.23%, and 15.62% for dust. Bird_drop medians are 1.00% in training and 0.94% in validation. Polygon measurements refer to enclosing boxes. Small crack regions motivate checking performance by object size, but do not establish the cause of individual errors.

Exact-pixel hashing found one image shared by training and test, and no other exact duplicates within or across splits. Original-split results were retained, with a supplementary policy evaluation excluding that image. Similar-looking examples suggest that near-duplicates or shared panels may remain. Consequently, the test results do not demonstrate independence at panel or acquisition-session level.

## Finetuning and baselines

YOLO11n was chosen as a compact detector suitable for the available Google Colab Tesla T4. It localizes anomaly types before detections are aggregated into a photograph-level decision.

The completed run started from pretrained yolo11n.pt and trained for 30 epochs at image size 640, batch size 16, seed 42, early-stopping patience 10, and two data-loader workers. Automatic optimizer selection chose AdamW with initial learning rate 0.00125. Training used Ultralytics 8.4.190, Python 3.13.16, and PyTorch 2.11.0+cu130, taking approximately 36 minutes. Validation selected the best checkpoint, which was copied to weights/best.pt. Test was reserved for assessment.

On the same original test split, fine-tuned YOLO11n achieved AP50 0.9429 and mAP50–95 0.7887. The predict-nothing baseline has zero recall and AP for classes present in test, with undefined precision because it produces no detections.

The unchanged pretrained COCO detector was run on the same 648 photographs and its predictions were saved. None of the target categories exists in its vocabulary. No arbitrary mapping between COCO and solar class IDs was used. Restricted to semantically matching target categories, it therefore produces no target outputs and coincides with predicting nothing. Its target-class recall and AP are zero for supported classes, and precision is undefined. The improvement reflects both fine-tuning and this vocabulary mismatch; it is not a comparison against another specialist solar detector.

The integrated training script automates download, conversion, training, evaluation, baseline inference, threshold selection, and artifact export. An end-to-end integration check was subsequently run in an isolated copy on an Apple GPU using the complete dataset, one training epoch, image size 160, and batch size 16. This reduced check exercises the pipeline rather than reproducing the final model quality; it does not replace the completed 30-epoch checkpoint. The submitted candidate checkpoint and the results reported here come from the first completed notebook run.

## Per-class evaluation and failure modes

For cracked, test precision is 0.9161, recall 0.7283, AP50 0.8981, and AP50–95 0.6876. Dusty achieves 0.8799, 0.9063, 0.9478, and 0.7688 respectively. Panel achieves 0.9615, 0.9561, 0.9830, and 0.9098. Bird_drop metrics are unavailable because test contains no ground-truth examples.

Overall evaluator precision is 0.9192 and recall is 0.8636. These summaries use the evaluator-selected confidence operating point. AP evaluation retains predictions from confidence 0.001. Overall mAP covers the three classes with test support, not demonstrated performance on all four categories. Panel has the highest AP, while cracks have the weakest AP and recall. The headline average therefore should not be read as crack-detection performance. Dust results require caution because of their small sample.

The object-level confusion matrix saved at reference confidence 0.25 shows 850 correctly matched cracks, seven classified as panels, and 89 unmatched ground-truth cracks. This gives recall of approximately 89.85% under that matrix's criteria. There are also 189 unmatched crack predictions and 17 panel objects predicted as cracks. Dust has 25 correct detections, five objects classified as panels, and two unmatched objects. These are object counts, and their confidence and matching criteria differ from the evaluator summaries above.

At photograph level, the frozen policy passes two of the 387 positive test photographs. Their anomaly scores are approximately 0.0063 and 0.0377. Ten photographs without annotated anomalies receive review and three generate automatic incidents. A visual review found that the first has a crack label covering approximately 38.4% of the image without a clearly localized crack at the displayed resolution. The second shows a dark panel with fine line texture and a crack box covering 2.34% of the image. These cases suggest label-granularity and visual-contrast difficulties, but do not prove incorrect labels or a causal explanation for the misses. The dashboard and saved evaluation artifacts provide the confusion matrices.

## Cost matrix and operating point

The scenario assumes 1,000 photographs daily, 10% production anomaly prevalence, 30 seconds per photo review, and a fully loaded labour rate of €24/hour. Review therefore costs €0.20. Incident handling costs €1, an unaddressed positive costs €5, and human review is assumed correct. Productive staff capacity is six hours per person daily.

The final cost matrix includes all three actions. Passing a positive costs €5; passing a negative costs nothing. Reviewing a positive costs €1.20, including the incident subsequently created, while reviewing a negative costs €0.20. An automatic incident costs €1 whether justified or unnecessary. Repairs, travel, physical inspection, and software overhead are excluded. This consistent accounting matters: reviewed positives must incur incident costs just as automatically escalated positives do.

The photograph score is the maximum retained confidence among cracked, dusty, and bird_drop detections, or zero if none is retained. Scoring uses image size 640, confidence floor 0.001, NMS IoU 0.7, and at most 300 detections per image. Confidence is a ranking signal, not a calibrated probability.

Both thresholds were selected by searching observed validation score boundaries, including extreme policies. Positive and negative conditional action rates were reweighted to 10% production prevalence. Expected cost equals the review share multiplied by €0.20, plus the total incident share multiplied by €1, plus the missed-positive share multiplied by €5. Total incidents include automatic incidents and reviewer-confirmed positives.

The chosen lower threshold is approximately 0.048029 and the upper threshold approximately 0.377467. Scores below the lower threshold pass; scores from the lower threshold to below the upper threshold receive review; scores at or above the upper threshold generate incidents. Full precision is preserved in the saved policy to avoid boundary changes from rounding.

Validation cost is €112.17 per 1,000 photographs. With both thresholds frozen, test cost is €120.86. Removing the known duplicate gives €120.87 on 647 photographs. This duplicate has little influence on that estimate, although near-duplicate leakage remains unresolved. The dashboard displays the validation cost curve and marks the selected policy; exported artifacts retain the underlying calculations.

Sensitivity analysis includes incident costs. At 5% prevalence and €5 missed-positive cost, the optimum lower threshold rises to approximately 0.0678, with cost €61.08 and 25.5 reviews per 1,000. At 20% prevalence and €20 missed-positive cost, it falls to approximately 0.0094, with cost €218.24 and 69.4 reviews. These scenarios change both expected prevalence and consequences; their costs do not measure changes in model quality.

When omission costs only €1, minimum costs equal doing nothing: €50, €100, and €200 per 1,000 at 5%, 10%, and 20% prevalence. Multiple policies can tie because correctly handling a positive costs as much as omitting it, while review adds expense. Thus the model has no monetary advantage over doing nothing under those assumptions. At baseline prevalence and omission cost, changing review time from 15 to 60 seconds changes minimum cost from €109.07 to €118.20, with €112.17 at 30 seconds.

The most consequential uncertainty is the missed-anomaly cost relative to incident cost. The €5 assumption and other operating inputs require measurement before deployment. Sensitivity supports a conditional decision, not a universally optimal threshold.

## What a person still has to check

A technician reviews uncertain photographs and verifies incidents before intervention. The uncertainty band is therefore only part of the human workload.

At the assumed 1,000 photos/day, validation estimates approximately 31 photo reviews and 94.9 automatic incidents, with 104.6 total incidents including reviewed positives. Test estimates 42.2 reviews, 102.1 automatic incidents, and 109.8 total incidents. Photo review alone takes approximately 0.35 hours/day on the test estimate, or 0.059 staff equivalent. The policy escalates 99.48% of positive test photographs. After production-prevalence weighting, missed positives average 0.52 per 1,000 photos.

The actual test counts are distinct from those production estimates. Among 261 photographs without annotated anomalies, 248 pass, ten receive review, and three generate automatic incidents. Among 387 positive photographs, two pass, 30 receive review, and 355 generate automatic incidents.

To include incident staffing, an additional interpretation treats the entire €1 incident cost as labour at €24/hour. This corresponds to 150 seconds per incident, including handling and verification. At 109.8 incidents/day, this requires approximately 4.58 hours. Adding photo review gives 4.93 productive hours/day, or 0.82 staff equivalent. One technician with six productive hours could accommodate the average modeled workload, leaving approximately 1.07 hours of capacity. This is a planning assumption, not an observed service-time estimate or a guarantee against demand peaks. If verification takes longer or incident cost includes non-labour expenses, staffing must be recalculated. Physical inspection and repair require separate capacity.

Doing nothing costs €500 per 1,000. Reviewing every photo costs €300, including reviews and expected true incidents. Under the labour interpretation above, review-all requires 12.5 productive hours/day, or 2.08 staff equivalents. The model-only automatic pass/fail policy costs €126.11 on validation and €141.03 on test, using its validation-selected threshold. The hybrid costs €112.17 and €120.86 respectively. Its modeled test savings are €179.14 relative to review-all and €20.17 relative to automatic pass/fail, before software overhead. Model-only describes photo routing, not unverified physical repairs.

The Gradio dashboard loads saved weights and artifacts without training. It supports image upload, detections, both policy thresholds, the image-level action matrix, expected cost, review share, per-class metrics, and policy comparisons. Validation, original test, and test excluding the duplicate are identified separately. The official local check passed key verification, weight loading, and HTTP startup. Six additional numerical and annotation-conversion tests passed, and the dashboard was tested interactively. The reduced integration check additionally exercised download, preparation, training, evaluation, baseline inference, decision analysis, and artifact export. The submitted model and reported performance remain those of the completed 30-epoch Colab run.

## Limitations & next steps

Rare-class support and split independence are the main data limitations. Bird_drop lacks test examples, dust has few independent photographs, and near-duplicates may remain beyond the known exact duplicate. Further evaluation should use adequate class support and independent panel or acquisition groups. Representative outdoor images are needed before transferring conclusions to operational solar farms.

Annotation completeness is also uncertain. Panel-only labels do not certify health, while polygon-to-box conversion loses contour information. The two missed-positive photographs were visually inspected, but their failure causes remain uncertain. Representative negatives and apparent false positives need further annotation auditing, without tuning the submitted policy against test cases.

The economic model assumes correct reviewers and hypothetical costs. A pilot should measure prevalence, review accuracy, review and incident times, and consequences of delayed action. It should incorporate workload peaks, software overhead, and physical-inspection capacity. Confidence calibration and alternative photograph-level aggregation rules also require validation. Panel identifiers would prevent multiple photos of one asset creating duplicate incidents.

Finally, the integration check used a reduced training configuration; a second full 30-epoch execution of the integrated script was not completed. The script retains its 30-epoch, 640-pixel production configuration, while the delivered weights, metrics, and dashboard artifacts consistently refer to the original completed run. Current evidence supports a prototype and controlled pilot under explicit assumptions rather than unrestricted deployment.

## Generative AI use disclosure

OpenAI ChatGPT/Codex assisted with problem and dataset selection, illustrative economic assumptions, notebook organization, code generation and debugging, interpretation, sensitivity analysis, dashboard development, and report drafting. Assistance included troubleshooting annotation formats, memory errors, and setup.

Dataset counts and model results come from executed code and exported artifacts from the completed notebook run. Economic inputs are hypothetical rather than independently measured estimates. This disclosure includes assistance with analysis and writing, not only formatting. The integration check used one epoch at reduced resolution; the delivered model was trained for 30 epochs at 640 pixels.
