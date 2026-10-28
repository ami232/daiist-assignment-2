# Short explanation guide — Assignment 2

Use these study notes to understand the work and explain it in your own words.

## 1. What is the business decision?

Flag a returned-car photo for closer damage inspection. Missing real damage
was assumed to cost EUR 200, versus EUR 5 for an unnecessary inspection.
A photo does not establish who caused damage or what a customer should pay.

## 2. What did the dataset contain?

418 car photos: 250 training, 84 validation and 84 test. Its two original
classes are Damage and Undamage. We retained the published version and split.
Unlike the hourly bike-rental data, it has no usable time sequence for a
chronological split. We cannot confirm that its split separates vehicle identities.

## 3. What is finetuning?

Start with YOLO11n's pretrained visual features, then train its weights on
our labelled damage photos. The original COCO model knows objects such as cars,
but it has no Damage or Undamage label.

## 4. What did we try changing?

The input size: 320 versus 480 pixels, with the same model and training settings.
480 had a slightly lower validation business cost, so it was selected. The
improvement was small; it does not prove that larger images always work better.

## 5. How did we avoid choosing the best-looking test result?

We selected checkpoints and model settings using validation. Then we reported
test results. However, we had already seen the first model's test results before
trying the second size, so the report admits that the test was not completely blind.

## 6. Which class is harder?

Damage: its test AP50 is 0.419, versus
0.846 for Undamage. The average hides that gap.
A badly located box can fail the detection metric even when the photo is flagged.

## 7. Why not use confidence 0.5 automatically?

Wrong decisions have different costs. We swept thresholds on validation and
picked the cheapest under the stated assumptions. A confidence score is not
a calibrated probability, so 0.06 does not mean a 6% chance of damage.

## 8. What are our two thresholds?

The cost-minimum pair is 0.00 and 0.06. At the lower value
of zero, nothing passes automatically: low-score photos go to human review,
and scores at or above 0.06 trigger closer inspection. The dashboard
also compares a policy that uses all three outcomes, including automatic passes.

## 9. How much human work remains?

Under the 10% production-damage assumption, about 87% of photos
need review: 87 a day at 5 minutes each. That is about
1.21 staff equivalents for photo review, or
1.75 including closer inspections, assuming
360 productive minutes per person/day. It is not a large staffing reduction.

## 10. Why is that review percentage different from the test matrix?

The test dataset has about 60% Damage photos, while the hypothetical branch
assumes 10%. We reweight the error rates to that production assumption.
Reweighting does not fix different photos, different damage or bad labels.

## 11. Is the model the best business solution?

The scenario calculation gives roughly EUR 4,679/1,000 photos
for the model alone, EUR 4,200 for human
review of all photos and EUR 3,413 for the hybrid.
These are conditional estimates, not achieved savings. Human accuracy was
assumed, and some supplied labels visibly conflict with the photos.

## 12. What is the recommendation?

Keep normal staff inspections. Use this as an offline prototype and collect
audited rental-return photos before a live pilot. Better pixels cannot fix
unreliable labels. We need real damage rates, staff timings, costs and human
accuracy before trusting the projected savings.
