# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Cloe Chapotot
- **Student ID**: 18549
- **Email**: cchapotot.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Business problem, and the cost of being wrong

*The decision this model supports, who or what acts on it, and what happens
when it's wrong in each direction. Be concrete about the asymmetry: which
mistake is worse, roughly by how much, and why. Everything downstream (your
threshold, how much you send to a person, which metric you care about) has to trace
back to this section.*

I wanted to work with a problem related to food or to food waste. After searching for datasets in Roboflow, I stumbled upon a dataset about damaged eggs. My business problem is about identifying (computationally) if the eggs packaged into cartons are good or damaged before they exit a factory (the place they package the cartons). In this context, if an egg is found to be damaged, it would be flagged and removed before the carton is packaged for distribution. If it is uncertain it has to be cheked by a quality control employee which costs labor time but not that much money (like 40 seconds of a labor hour). If there is a false negative, so a missed cracked egg, it can cause reputational damage for the egg brand and for the super market that is selling it. It would lead to a customer complaint and the revenue for a whole carton of eggs is dismissed (around 4.50€). Hence, it is more expensive to have a false negative.

## Dataset

*Workspace / project / version on Roboflow Universe, and the link. What one
image represents, what the classes are, and why this dataset actually serves
the problem above rather than merely resembling it.
*

title = { egg-quality-detection Dataset },  type = { Open Source Dataset }, author = { anbies Workspace }, version: v5  url = { https://universe.roboflow.com/anbies-workspace/egg-quality-detection-nvhwd }

One image has an object, some of them are random objects that have round/oval shape, others are eggs in good and damaged conditions. The classes are "good" and "cracked". It serves the purpose of my business case because it is exacly what my model needs to detect if the eggs are in good conditions. 

## What the dataset's composition implies

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

The train part has good 4208, and cracked 4197. Validation has cracked 307, good 294. Test has cracked 297, good 303. Boxes are large, the median good egg covers about 19% of the image and cracked about 43%. Test has 297 cracked and 303 good, so neither class is thin. One missed cracked egg moves recall by about 0.3 points. The thin part is background. Only 20 of 620 test images have no label, so false positives on empty conveyor belt or other objects are barely tested. Yes, the test images resemble a lot what we would see in the factory because despite the images being individual eggs, the camera would capture the eggs in a conveyor belt for example.

## Finetuning and baselines

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

## Per-class evaluation and failure modes

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
