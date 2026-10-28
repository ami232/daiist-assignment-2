# Assignment 2 Report

- **Name**: Nour Farhat
- **Student ID**: 19299 
- **Email**: nfarhat.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 


## Business problem, and the cost of being wrong

In many regional hospitals, for example outside the big cities in Tunisia, there is no neuroradiologist on site. Brain MRIs are read by a general radiologist, who is not a specialist in brain tumours. The decision I support is what happens to each scan after it is taken. The model gives every scan a tumour score, and the scan is sent to one of three outcomes:

| Outcome | What happens | Who acts on it |
|---|---|---|
| **pass** | normal report | the local radiologist |
| **review** | sent to a remote neuroradiologist (teleradiology, paid per read) | the remote neuroradiologist |
| **fail** | urgent referral to neurosurgery / oncology | the referring doctor |

The two mistakes do not cost the same:
- **False negative (a tumour is passed as normal):** the diagnosis is delayed, the tumour keeps growing and the treatment becomes harder and more expensive. The hospital is also exposed to a liability claim. I estimate this at **€30,000**.
- **False positive (a healthy scan is sent for urgent referral):** the patient has an unnecessary specialist consultation and extra imaging. I estimate this at **€600**.
- **Review:** one remote read costs about **€50**, and I assume the neuroradiologist gets the scan right.

A missed tumour is about **50 times worse** than an unnecessary referral and 600 times more expensive than a review. This asymmetry drives the rest of the pipeline:
- the score is the probability of *any* tumour, `1 − P(notumor)`, because the decision is "tumour or not", not which type;
- I judge the model on tumour recall and expected cost per 1,000 scans, not on accuracy;
- the thresholds come from a cost sweep on the validation set, and they are much lower than 0.5, because missing a tumour is so much more expensive;
- the review band exists because a €50 remote read is far cheaper than either mistake.

## Dataset

I used the **Brain Tumour Detection** dataset on Roboflow Universe:
- **Workspace / project / version:** `kaplan-business-school-gqxef` / `brain-tumour-detection-svrlh` / `1`
- **Link:** https://universe.roboflow.com/kaplan-business-school-gqxef/brain-tumour-detection-svrlh/dataset/1
- **Licence:** CC BY 4.0

One image is one 2D slice of a brain MRI, resized to 640×640. There are 5,505 images in four classes: three tumour types (`glioma`, `meningioma`, `pituitary`) and `notumor`. The uploader does not say where the images come from, but they match the public Kaggle *Brain Tumor MRI Dataset*, which combines three earlier public sources.

I chose this version over the other brain tumour datasets on Roboflow because it was the only one that fit the task properly:
- it has a real **validation split** (3,858 / 1,099 / 548), so I can choose my thresholds on validation and keep the test set for the final result. The largest alternative had no validation split at all;
- it has **no augmentation added**. Two other versions had rotated and upside-down copies of each training image, which inflates the counts;
- it has a **`notumor` class**. Without healthy scans I could not count false positives, and half the cost matrix would disappear.

The dataset serves the decision and does not just resemble it: the hospital's question is "does this scan show a tumour?", and that is exactly what `1 − P(notumor)` answers. The tumour types are not used for routing, but they let me see where the model gets confused.

## What the dataset's composition implies

| Class | Train | Val | Test | Total | Share |
|---|---|---|---|---|---|
| glioma | 894 | 281 | 146 | 1,321 | 24.0% |
| meningioma | 928 | 281 | 124 | 1,333 | 24.2% |
| notumor | 1,003 | 248 | 155 | 1,406 | 25.5% |
| pituitary | 1,033 | 289 | 123 | 1,445 | 26.2% |
| **Total** | **3,858** | **1,099** | **548** | **5,505** | |

The four classes are balanced (24–26% each), and every class is split about 70/20/10, so the validation and test sets have the same mix as training. I also checked for exact duplicate images across the splits (MD5 hash of each file) and found none: 0 test images appear in train, 0 validation images appear in train, and there are 0 duplicates inside train. So the test score is not inflated by images the model has already seen.

Since this is classification, there are no bounding boxes. A tumour usually covers only a small part of the slice, and I train at 224×224 instead of the original 640×640, so very small tumours lose detail.

What this means for my results:
- **The tumour rate is far from reality.** 74.5% of the scans in the dataset show a tumour, but a regional hospital sees far fewer; I assume 5%. This is why I compute every cost at the hospital's base rate, not the dataset's.
- **Per-class numbers are thin.** Each test class has only 123–155 images, so one image moves a class's recall by about 0.7 points. A difference of 1–2 points between two classes is not meaningful.
- **The false-positive side rests on 155 healthy test scans.** Every "healthy scan sent to review or referral" figure in the cost section comes from these 155 images.
- **The test set is cleaner than real life.** Each image is one well-chosen slice from a curated public dataset, whereas the hospital would send full MRI series from its own scanner and its own patients.

So I cannot conclude that the model would reach the same accuracy in a Tunisian regional hospital. My results show the model separates tumour from no-tumour very well **on this kind of image**, and my cost analysis shows how the routing would work **if** it performed similarly on local scans.

## Finetuning and baselines

I finetuned **`yolo11n-cls`**, the nano classification model pretrained on ImageNet. I chose classification and not detection because the hospital's decision is made once per scan ("is there a tumour?"), not per object, so I only need one score per image. I chose the nano model because the dataset is small (3,858 training images) and the task is simple enough, and because a small model is fast and cheap to run on a hospital computer without a GPU.

**Training:** Google Colab, T4 GPU, 20 epochs, image size 224, `seed=0`, `patience=10`, the other settings at the Ultralytics defaults. Training took about 12 minutes, and validation accuracy reached 0.986 by the last epoch, so 20 epochs was enough.

All three models are scored on the same 548 test images:

| Model | 4-class accuracy | Tumour vs no-tumour AUC | Best cost / 1,000 scans, model alone |
|---|---|---|---|
| **Finetuned yolo11n-cls** | **0.976** | **1.000** (0.99987) | **€3,817** |
| COCO yolo11n, zero-shot | n/a (no tumour classes) | 0.538 | €570,000 |
| Trivial: always "pituitary" (the largest training class) | 0.224 | 0.500 | €1,500,000 |

- **COCO zero-shot** has no tumour class at all, so it cannot name a tumour type. I used its highest box confidence as a "something is in the image" score. Its AUC of 0.538 is almost random, which shows that a general object detector sees nothing medical in an MRI.
- **Trivial baseline:** always predicting the largest class gets 22.4% accuracy, and for routing it means passing every scan, which misses every tumour.
- **The cost column** uses the best threshold picked on the test set itself, so it is optimistic and only for comparing the models. The threshold I actually ship is chosen on validation (see below).

The gap was larger than I expected. The finetuned model almost perfectly ranks tumour scans above healthy ones: only 1 tumour scan sits among the lowest-scoring healthy ones. This is very high for medical imaging, which makes me think the dataset is easier than real hospital scans (see limitations).

## Per-class evaluation and failure modes

| Class | Test images | Precision | Recall |
|---|---|---|---|
| glioma | 146 | 0.986 | **0.959** |
| meningioma | 124 | **0.931** | 0.976 |
| notumor | 155 | 0.993 | 0.981 |
| pituitary | 123 | 0.992 | 0.992 |

Confusion matrix on test (rows = true class, columns = predicted class):

| | glioma | meningioma | notumor | pituitary |
|---|---|---|---|---|
| **glioma** | 140 | 6 | 0 | 0 |
| **meningioma** | 2 | 121 | 1 | 0 |
| **notumor** | 0 | 2 | 152 | 1 |
| **pituitary** | 0 | 1 | 0 | 122 |

- **Pituitary carries the headline number** (0.992 precision and recall). Pituitary tumours sit in one fixed place at the base of the brain, so they are easy to recognise.
- **Glioma drags it down**, with the lowest recall (0.959). All 6 of its errors were predicted as meningioma.
- **Meningioma has the lowest precision** (0.931), because it absorbs 6 gliomas, 2 healthy scans and 1 pituitary. It is the class the model falls back on when it is unsure.

The main confusion, glioma ↔ meningioma, matters less for my decision. Both are tumours, so both get a high tumour score and are still referred; the specialist decides the type afterwards. The errors that cost money are between tumour and no tumour:
- **1 of 393 tumour scans** (a meningioma) was predicted as healthy. This is the expensive mistake, a €30,000 missed tumour.
- **3 of 155 healthy scans** were predicted as a tumour. These cost a referral or a review.

So the 97.6% accuracy hides where it breaks: the one case that really matters is the meningioma predicted as healthy, and it scores so low that my thresholds still pass it (see next section).

## Cost matrix and operating point

| Item | Cost | Where it comes from |
|---|---|---|
| Missed tumour (FN) | €30,000 | delayed diagnosis: more advanced treatment, worse prognosis, and liability exposure for the hospital |
| Unnecessary urgent referral (FP) | €600 | one specialist consultation plus extra imaging |
| One remote neuroradiologist read | €50 | a pay-per-read teleradiology fee |
| Base rate of tumours | 5% | my assumption for brain MRIs in a regional hospital; far below the dataset's 74.5% |

These are illustrative assumptions, not measured prices; the order of magnitude matters more than the exact number. For each scan, the expected cost per 1,000 scans is:

`1,000 × [5% × (P(pass | tumour) × €30,000 + P(review | tumour) × €50) + 95% × (P(review | healthy) × €50 + P(fail | healthy) × €600)]`

I assume the neuroradiologist gets reviewed scans right, and correct outcomes cost nothing extra.

I swept the thresholds from 0 to 1 in steps of 0.01 on the **validation** set and kept the cheapest. I used the test set only once, to report the result.
- **Model alone (one threshold):** 0.63, which costs **€7,494** per 1,000 test scans.
- **With a review band:** pass below **0.10**, review between 0.10 and 0.77, fail at **0.77** or above. This costs **€5,349** per 1,000 scans.

The €5,349 breaks down into the one missed tumour (€3,817) and the remote reads (€1,532); no healthy scan is referred by mistake. The model alone costs more because it misses the same tumour **and** sends one healthy scan for an unnecessary referral. The cost-against-threshold curve is saved in `outputs/cost_curve.png` and is shown live in the dashboard with my operating point marked.

**Sensitivity:** I re-chose the band on validation for other base rates and miss costs.

| Base rate | Cost of a miss | t_low | t_high | Share reviewed |
|---|---|---|---|---|
| 1% | €30,000 | 0.63 | 1.00 | 1.6% |
| 2% | €30,000 | 0.10 | 1.00 | 5.2% |
| 5% | €5,000 | 0.63 | 0.77 | 0.6% |
| **5%** | **€30,000** | **0.10** | **0.77** | **3.1%** |
| 10–20% | €30,000 | 0.10 | 0.77 | 2.6–2.9% |

The assumption that would most change my answer is the **base rate**. At 1–2%, t_high goes to 1.00, so no scan is referred straight to urgent care and every suspicious scan goes through the neuroradiologist first: when tumours are rare, most high scores are false alarms. The miss cost mainly moves t_low: once a miss costs about €15,000 or more, t_low drops to 0.10 so that borderline scans are reviewed instead of passed. Above a 5% base rate, the band stays at 0.10–0.77, so my choice is stable for most realistic hospitals.

## What a person still has to check

My two thresholds are **t_low = 0.10** and **t_high = 0.77**. At a 5% base rate, they send **3.1%** of scans to the remote neuroradiologist.

**Workload:**
- I assume the hospital does about **20 brain MRIs a day**, which gives **0.6 reviews a day**, or about 4 a week.
- At **15 minutes a read** and 6 reading hours a day, that is **0.03** of one neuroradiologist's day.

This is realistic: the hospital does not need to hire anyone. A pay-per-read teleradiology contract covers it easily, and the remote reader has time to look properly at each scan.

My design priced against the three alternatives, on the test set:

| Option | Cost / 1,000 scans | Share sent to a person |
|---|---|---|
| Do nothing (pass everything) | €1,500,000 | 0% |
| A person checks everything | €50,000 | 100% |
| The model decides alone (t = 0.63) | €7,494 | 0% |
| **Model + review band (0.10–0.77)** | **€5,349** | **3.1%** |

- Doing nothing is by far the worst, because every tumour is missed.
- Sending everything to a neuroradiologist is safe, but it costs about 9 times more and needs 20 remote reads a day, which a hospital without a specialist cannot easily get.
- Letting the model decide alone is cheaper than both, but it has no safety net for the uncertain scans.

The band is the cheapest option, and it sends only the uncertain cases to a person. This is the design I would ship, together with monitoring of the passed scans, because the one tumour it still misses scores very low.

## Limitations & next steps

- **One curated 2D slice, not a full scan.** In a hospital, a brain MRI is a series of many slices, and the tumour may only be visible on some of them. *Next step:* score every slice of a series, take the highest tumour score, and validate on full scans from the target hospital.
- **The images probably come from a few public sources, with the healthy scans mostly from one.** The model might partly learn the source of an image instead of the tumour, which could explain the almost perfect AUC. *Next step:* test on scans from a new hospital and scanner before deploying, and check the model's performance per source.
- **A small test set.** With 123–155 images per class and only 155 healthy scans, small differences are not meaningful, and the whole missed-tumour cost comes from a single image. *Next step:* collect a larger local test set, or report bootstrap confidence intervals.
- **The costs and the base rate are assumptions.** The thresholds depend on them, especially the base rate. *Next step:* replace them with the hospital's real tumour rate, its teleradiology contract price, and the real cost of a delayed diagnosis, then re-run the sweep.
- **The missed tumour scored below 0.10.** No threshold can catch it without reviewing many more healthy scans. *Next step:* look at that image to understand why the model missed it (tumour size, contrast, or slice position), and add similar cases to training.

## Generative AI use disclosure

I used Claude (Anthropic) to compare candidate datasets on Roboflow Universe, write the code in `train.ipynb`, `costs.py` and `app.py`, debug my Colab setup, explain the methods and results, and draft and format this report from my decisions and results. The business framing (triage for a regional hospital with no neuroradiologist), the choice of dataset, the cost assumptions, the thresholds I would ship, and the conclusions are my own decisions. I reviewed and can explain all of the code and every number in this report.
