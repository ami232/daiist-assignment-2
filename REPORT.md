# Assignment 2 Report

- **Name**: Ignacio Fierro Botin
- **Student ID**: 20206
- **Email**: ifierro.ieu2022@student.ie.edu
- **Group**: BBADBA A



## Business problem, and the cost of being wrong

In a greenhouse where tomatoes, strawberries and beans grow, the scouts take a picture of every plant that doesnt look right and the head of the crops decides wether the bed where the fruit or vegetable is growing gets a treatment with a fungicide or not. These crops witb fungal infections need the treatment quick because the disease spreads in days. Spider mites and bacterial angular leaf spot dont need fungicide as it does nothing to them, they require another treatment. 

One case is one photo and what the model does is answer the question "Does this crop show signs of a fungal infection or not?" and the crop bed either goes into the list of beds that need the fungicide (fail) or not (pass) or it goes through a manual review by the head of the crops (review). The head acts based on the list it recieves, nothing else gets automated.

There exists two possible errors, the both cost a different amount:
- If the model doesnt detect an infection, it costs around 60€ as the bed is not treated until the next scouting which happens once a week. In that week they lose around 10kg of fruit at a price of about 4€/kg plus the price of removal labour.
- If the model incorrectly classifies spider mites or the bacterial angular leaf spot as a fungal infection, the cost is about 12€. This is deducted by calculating that around 5€ of product plus 15min of crew time which is around 4€ and the real problem without treatment which is around 3€.

A correct treatment costs around 9€ in total plus time saved and of course not treating the plants that we dont need to treat is free. A faulty detection is around 5 times more costly than an unnecessary treatment which is the key here. The threshold is way below 0.5, the mnetric that really matters is the recall on the fungal classes. The review classification exists exactly for that purpose, to double check manually if the model has its doubts.



## Dataset

For the dataset I picked the 5th version of the Roboflow Universe workspace `artificial-intelligence-82oex` project `detecting-diseases` which has 5,493 images that I split like this: 2,904 | 1,416 | 1,173 (train | validation | test). The link to the dataset is https://universe.roboflow.com/artificial-intelligence-82oex/detecting-diseases.

As explained, one image is a photo of a leaf, flower or fruit with a boxed area showing where the symptom is, the photos looks like they were taken with a phone by one of the scouts. I decided to make 12 detection types, which are the types of pathogen that the crop can have. There are fungal ones (positive) which are bean angular leaf spot, bean rust, strawberry anthracnose fruit rot, blossom blight, gray mold, leaf spot, powdery mildew (fruit and leaf), tomato blight and tomato leaf mold. Then there are non-fungal ones (negative) which are strawberry angular leaf spot (bacterial) and tomato spider mites (pest). There are two angular leaf spots that are in different classes as they are completely different as when they are on strawberries they are not fungal but on beans they are fungal, this is why I had to do the grouping by hand. As indicated, a photo is labeles as positive if it has a fungal box in it, then its score is the highest confidence of detecting a fungal infection.



## What the dataset's composition implies

The three splits are similar as in each one around 80% of the images contain a fungal type, they are also similar as each image has at least one box and the mean box takes up around 12% of the image. The test instances per class run from 698 in the strawberry leaf spot to 31-41 in the three tomato classes, this means a few things:

- There is no "healthy" class which means that the model cant really learn whether a crop is sick or not, it can help the decision of what to do when a crop is sick, wether to use fungicide or not, which is the decision that I have framed.
- The base rate of the test is not the one we would have in a real production scenario as in the dataset we have 80% of crops that are fungally infected which isnt realistic. That is why I took the recall and the rate of unnecessary treatments and y reweight them by the assumed base rate.
- The tomato clases are too scarce to be confident as we onlt have 31-41 instances, which is why I made it so that one single wrong classification moves its recall 3 points rather than one.
- The model is made so that it decets different pathogens so images from far away will cause the model to be overconfident in detecting a pathogen.
- There are validaiton leaks as tomato classes are scoring 0.98-0.99 AP50 on validation but getting much lower scores on the test (0.63-0.77) meaning that there are crop images in the validation set that are also being used in the training set. The numbers that are shown below are all from the test set.



## Finetuning and baselines

I fine tuned the yolo11n for detecting from the COCO checkpoint and then I retrained it using the 5,493 photos of the crops. I used detection rather than clasification as the photos returned several boxes of symptoms and the box shows where the head of the crops should look. I also used the yolo nano as the model would have to run on the mobile device of the scout or in a relatively cheap device in the greenhouse rather than a server with a strong GPU. I trained the model on my own PC at home which has Windows 11 and an RTX 5080, I ran 30 epochs with an image size of 416, batch 8, seed 0 and default augmentation and it took around 10 minutes. The weights are stored in the weights folder in the repo under the title best.pt

Cross level results on the test split are the following (positive = photo shows a fungal disease):

| model                                          | precision | recall    | F1    | TP  | FP  | FN  | TN  |
|------------------------------------------------|-----------|-----------|-------|-----|-----|-----|-----|
| Trivial: predict nothing                       | 0.000     | 0.000     | 0.000 | 0   | 0   | 933 | 240 |
| Trivial: majority class (always fungal)        | 0.795     | 1.000     | 0.886 | 933 | 240 | 0   | 0   |
| COCO yolo11n zero-shot ("potted plant" ≥ 0.25) | 0.895     | 0.018     | 0.036 | 17  | 2   | 916 | 238 |
| Finetuned yolo11n (fungal ≥ 0.24)              | 0.981     | 0.974     | 0.977 | 909 | 18  | 24  | 222 |

The majority class baselaine only looks good because of what we mentioned of 80% of the test having fungal cases, in a real life scenario that would mean spraying all of the mite and bacterial cases which of course wouldnt be the ideal case. The COCO model doesnt have any type of pathogens so its AP over them is 0 by default, the only thing it can do is finding a plant in a pot which it manages to do in 19 cases. The adjusted model reached an AUC of 0,956, the difference is as big as I expected as COCO has never seen a sick leaf and the model makes more mistakes about what fungus it is rather than making mistakes about wether a fungus is present or not.



## Per-class evaluation and failure modes

| class                            | group      | precision | recall | AP50  | mAP50-95 |
|----------------------------------|------------|-----------|--------|-------|----------|
| Beans Angular LeafSpot           | fungal     | 0.795     | 0.748  | 0.803 | 0.569    |
| Beans Rust                       | fungal     | 0.848     | 0.765  | 0.882 | 0.606    |
| Strawberry Angular LeafSpot      | non-fungal | 0.887     | 0.641  | 0.754 | 0.553    |
| Strawberry Anthracnose Fruit Rot | fungal     | 0.636     | 0.500  | 0.441 | 0.306    |
| Strawberry Blossom Blight        | fungal     | 0.919     | 0.536  | 0.527 | 0.432    |
| Strawberry Gray Mold             | fungal     | 0.921     | 0.446  | 0.599 | 0.355    |
| Strawberry Leaf Spot             | fungal     | 0.846     | 0.840  | 0.897 | 0.816    |
| Strawberry Powdery Mildew Fruit  | fungal     | 0.515     | 0.896  | 0.774 | 0.587    |
| Strawberry Powdery Mildew Leaf   | fungal     | 0.824     | 0.635  | 0.785 | 0.644    |
| Tomato Blight                    | fungal     | 0.758     | 0.991  | 0.772 | 0.732    |
| Tomato Leaf Mold                 | fungal     | 0.663     | 0.971  | 0.725 | 0.632    |
| Tomato Spider Mites              | non-fungal | 0.573     | 0.968  | 0.626 | 0.561    |

On the test split overall the results were a precision of 0.765, a recall of 0.745, a mAP50 of 0.715 abd a mAP50-95 of 0.566. The headline is held by the strawberry leaf spot and the bean rust pathogens which are the two largeest classes. The most important figure for the decision is the recall of the fungal group at the operating point which is 97.4% with a 7.9% of unnecessary treatments. The confusion matrix can be found in the artifacts folder of the repo under the name confusion_matrix.png and confusion_matrix.json. Having seen it, we can see that the model has some issues:

- The biggest weak point of the model are the fruit and flower diseases that are all fungal. For example, the blossom blight is not detectes in 97 of 211 boxes and the anthracnose is missed in 58 of 132 boxes.
- Gray mold isnt missed but just detected wrong sometimes but it doesnt cost anything as it is still detected. Only 44 of 213 are gray modl and they get wrongfully labelled as anthracnose or powdery mildew which are all fungal.
- 155 leaf spot and 50 powdery mildew boxes are called bacterial angular leaf spot as they look similar and 86 fungal boxes are confused with spider mites.
- Non fungal cases are called fungal in some instances, 99 out of 295 bacterial angular boxes and 25 of 31 spider mite boxes are called fungal. There is a clear issue with the spider mite boxes.



## Cost matrix and operating point

The costs are the ones mentioned at the beginning of the report, 60€ per missed detection, 12€ per unnecessary treatment, 9€ per correct treatment anbd 0€ for correct non-treatment. There are three assumptions more, the first one being that the scout's revision is around 1.50€ per photo (calculates as 2 minutes at 45€/h) and is right 95% of the time. The second one is that the base rate is 40% because scouts only photograph sick plants and in a real case scenario I'm assuming that only around 4 out of 10 plants have some sort of pathogen (rather than 80%), this is the figure I am the least sure about. The last assumption is that the greenhouse produces around 300 photos a day, thanks to two scouts looking over 2 hectares of greenhouse, with a head of crops that can spend 1h a day reviewing cases. For each confidence threshold the test split gives us recall and the rate of unnecessary treatments which are properties of the model so they transfer. The base rate isnt a property of the model so the expected cost per 1,000 photos is calculated with the 40% assumption. That would leave us with the following formula:

cost = 100 x [ 0.4 x (recall x 9 + (1 - recall) x 60) + 0.6 x unnecessary treatment rate x 12]

Sweeping the threshold from 0.01 to 0.99, the minimum is at 0.24 which leaves us with 4,665€ per 1,000 photos, with a recall score of 0.974 and an unnecessary treatment rate of 0.079. The curve is flat between 0.15 and 0.50 and increases strongly above 0.6 which is where every recall point lost costs 60€ over 400 photos per 1,000 potos. The assumption that would change the answer the most is the 60€ cost for missed detection. Repeating a sweep with an error at 15€, 30€, 60€ and 150€ and a base rate of 20%, 40% and 70€, the threshold stays between 0.20-0.24 as long as a miss costs 30€ or more. It only changes to 0.62 if a miss costs as little as a unnecessary treatment which I dont believe for the gray mold, the base rate barely moves it. The cost sweeps are in the repo in the artifacts folder under the cost_sweep.csv and the cost_vs_threshold.png names.



## What a person still has to check

After sweeping all the thresholds I picked the cheapest one that the agronomist could cover whcih was a pass below 0.20, review between 0.20 and 0.62 and a fail at 0.62 or above. That will leave us with only 3.2% of the photos being sent to the head of the crops for review, which is only about 10 photos and 20 minutes a day, a third of the hour that he is available for so the need for personnel is minimal and not restrictive. Per 1,000 photos, once the head of crops has reviewed the images, we would get 390 true positives, 10 misses, 33 unnecessary treatments and 567 correct non treatments, leavinf the recall on fungal infections at 97.5%.

| design                               | cost / 1,000 photos | share reviewed | people needed | cost / day |
|--------------------------------------|---------------------|----------------|---------------|------------|
| Do nothing (never act on the photo)  | 24,000€             | 0%             | 0             | 7,200€     |
| Head of crop checks every photo      | 6,480€              | 100%           | 10            | 1,944€     |
| Model alone (threshold 0.24)         | 4,665€              | 0%             | 0             | 1,399€     |
| Model + review band (0.20, 0.62)     | 4,565€              | 3.2%           | 0.32          | 1,370€     |

Not doing anything costs 1,000 x 40% x 60€, if we neede the head of the crop to check every image, we would need ten of them and it would still be worse than the model becasue a 5% human error on 400 images is 20 mistakes. The review band only adds 100€ per 1,000 because the model is correct in 97% of images. It is realistic as 20 minutes a day over 10 photos is less than the actual situation right now, which is that someone checks 300 and around one crop bed in every 12 on the list does not need any treatment. 



## Limitations & next steps

- No healthy class so adding a few photos of healthy plants would help retrain the model and then we could re sweep.
- Half of the blossom blight and anthracnose boxes arent detected so if we evaluate the images at a bigger size (for example 640) we could try the yolo11s and add some fruit and flower examples which are the smallest fungal classes.
- Validation leaks are a limitation that can be solved by re splitting by plant or source folder and retraining the model.
- We should evaluate on real photos from scouts and if the recall falls on photos that are from too far away we cna make the app force scouts to take an image from closer.



## Generative AI use disclosure

I used Claude (Anthropic) throughout the project to draft `train.py`, `app.py` and `costing.py`, and propose the framing and cost figures. I chose the dataset and the class grouping and I also set up the environment and wwrote the report. I trained the model on my own machine, ran and verified the pipeline and dashboard which claude helped me make, I also reviewed every section done by claude. I can explain every outcome of the project and understand its weaknesses, implications and results.