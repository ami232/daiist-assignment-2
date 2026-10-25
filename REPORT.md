# Assignment 2 Report

- **Name**: Adolfo Gutierrez de GAndarilla Dominguez
- **Student ID**: 19003
- **Email**: fgutierrezde.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

> This is a university exercise. The model is not a medical device and must
> not be used to diagnose anyone.

**The decision.** A teledermatology service receives photos of skin lesions
(moles, spots) from patients who are worried about them. Today a
dermatologist looks at every photo and decides whether the patient needs an
in-person appointment. The model looks at the photo first and gives each
case one of three outcomes:

| Outcome | What happens | Who acts |
|---|---|---|
| 🟢 **Pass** | Probably benign. The patient gets advice to watch the lesion and send a new photo if it changes. No appointment. | Automatic message |
| 🟡 **Review** | The model isn't sure. A dermatologist looks at the photo (about 2 minutes) and decides. | Dermatologist |
| 🔴 **Fail** | Probably malignant. The patient gets a priority in-person appointment. | Clinic booking |

**Positive class = "malignant or pre-malignant"**: melanoma (MEL), basal
cell carcinoma (BCC), squamous cell carcinoma (SCC) and actinic keratosis
(AK, pre-cancerous). These are the lesions a dermatologist must see in
person. The other four classes are benign: nevus/mole (NV), benign keratosis
(BKL), dermatofibroma (DF) and vascular lesion (VASC). The model's score for
a case is the sum of the probabilities of the four positive classes.

**Why the two mistakes cost very different amounts:**

- **False negative (cancer sent home).** The diagnosis is delayed until the
  patient comes back, often months later. For melanoma this is the
  difference between removing a thin lesion in a short outpatient procedure
  and treating a melanoma that has spread, which can mean surgery, scans and
  years of immunotherapy, with a much lower survival rate. The other cancers
  grow more slowly, so a delay costs less, but it still means bigger surgery.
- **False positive (benign lesion sent to an appointment).** One unnecessary
  dermatologist visit and some worry for the patient. That is roughly the
  price of a consultation.

So a missed melanoma is **hundreds of times** more expensive than an
unnecessary appointment, and the cost of a miss also depends on **which**
cancer is missed. That is why the costs are set per class (section *Cost
matrix and operating point*), and why melanoma gets its own analysis even
though it is grouped with the other positives.

**What this framing decides later:**

1. **The metric that matters is recall on the positive class, above all on
   melanoma**, not accuracy. Accuracy treats a missed melanoma and an
   unnecessary visit as the same mistake.
2. **The thresholds are chosen by minimising expected cost**, which will push
   them low: the model should send a case on whenever there is a real chance
   it is cancer.
3. **The base rate must be the real one, not the dataset's.** The dataset has
   far more cancers than a real teledermatology inbox (see *Dataset*), so
   costs are re-weighted to a realistic share of positives before choosing a
   threshold.
4. **The review band is limited by dermatologist time.** A threshold that
   sends half the photos to a person is not a solution if there aren't
   enough dermatologists to look at them.

## Dataset

- **Workspace / project / version:** `riteshs-workspace-qkstg` /
  `skin-lesion-classification-sh4sk` / **version 2**
- **Link:** <https://universe.roboflow.com/riteshs-workspace-qkstg/skin-lesion-classification-sh4sk/dataset/2>
- **Licence:** CC BY 4.0, public, used as it is (not forked).
- **Task:** image classification, 8 classes, colour images resized to 224×224.
- **One image** is a close-up of a single skin lesion. The class names
  (AK, BCC, BKL, DF, MEL, NV, SCC, VASC) are those of the ISIC 2019
  challenge, the main public collection of labelled skin-lesion images.

**Why this dataset and not another one.** I compared the public skin-cancer
datasets on Roboflow Universe:

- The largest one (HAM10000, 9,900 images) is stored in **grayscale**. Colour
  is one of the main signs a dermatologist uses (the "C" in the ABCDE rule:
  several colours inside one lesion), so I rejected it.
- Others had fewer than 2,000 images, or mixed in an "Unlabeled" class.
- **Version 2** of this project has **no augmentation**, so every image is a
  real one. Version 1 adds flipped/rotated copies to the training set, which
  makes the dataset look bigger than it is.

**Does it serve the problem or just resemble it?** It covers exactly the
decision: the classes include the cancers that need an appointment and the
benign lesions that don't. But there is one important mismatch, discussed in
the next section: these are mostly **dermoscopic** images (taken through a
magnifying lens in a clinic), while patients would send **phone photos**.

## What the dataset's composition implies

I downloaded version 2 and counted the images myself (one image = one
lesion = one label, since this is classification).

| Class | Malignant? | Train | Valid | Test | Total | Share |
|---|---|---|---|---|---|---|
| AK – actinic keratosis | 🟠 pre-malignant | 455 | 130 | 66 | 651 | 12.9% |
| BCC – basal cell carcinoma | 🔴 yes | 562 | 185 | 74 | 821 | 16.3% |
| MEL – melanoma | 🔴 yes | 582 | 170 | 83 | 835 | 16.6% |
| SCC – squamous cell carcinoma | 🔴 yes | 405 | 112 | 62 | 579 | 11.5% |
| NV – nevus (mole) | 🟢 no | 590 | 171 | 95 | 856 | 17.0% |
| BKL – benign keratosis | 🟢 no | 570 | 151 | 85 | 806 | 16.0% |
| DF – dermatofibroma | 🟢 no | 156 | 54 | 29 | 239 | 4.7% |
| VASC – vascular lesion | 🟢 no | 197 | 33 | 23 | 253 | 5.0% |
| **Total** | | **3,517** | **1,006** | **517** | **5,040** | |

- **Split:** 70% / 20% / 10%. Every class appears in every split.
- **Image size:** all 224×224 pixels, colour.
- **Object size:** this is classification, so there are no boxes. Every image
  is a close-up centred on one lesion, which fills most of the frame. The
  catch is the resolution: at 224×224, fine details a dermatologist looks for
  (small dots, thin lines) are partly lost.

**What this implies**

1. **Positives are far over-represented.** 57% of the images (and 55% of the
   test set: 285 of 517) are malignant or pre-malignant. In a real
   teledermatology inbox most photos are of harmless moles, and only a small
   share are cancer. The dataset was clearly balanced on purpose (NV, the most
   common lesion in real life, has about the same count as melanoma). So
   **costs measured directly on the test set would be wrong**: before choosing
   thresholds I re-weight the test cases to a realistic base rate (see *Cost
   matrix and operating point*).
2. **Some per-class numbers are too thin to trust.** The test set has only
   **23 VASC and 29 DF** images. One more or one fewer mistake moves their
   recall by 3–4 points, so I won't draw conclusions about those two classes.
   Melanoma (83 test images) is enough for a rough number, but its recall
   still has an uncertainty of several points.
3. **Duplicates across splits.** File names never repeat across splits, but
   comparing the centre of every image I found **6 near-identical pairs in
   different splits**, the same lesion saved under two ISIC numbers. **5 of
   them are test images that also appear in training** (4 SCC, 1 BKL). That is
   1% of the test set, so it inflates the results only slightly, but SCC's
   test number is the most affected (4 of its 62 test images were seen in
   training).
4. **A possible shortcut: the dermoscope frame.** A third of the images (1,685)
   have the round black border of a dermoscope. It is much more common in the
   cancer classes than in the benign ones:

   | AK | BCC | SCC | MEL | DF | BKL | VASC | NV |
   |---|---|---|---|---|---|---|---|
   | 53% | 45% | 40% | 37% | 27% | 26% | 19% | 12% |

   A model could learn "black frame → cancer" instead of looking at the
   lesion. I check this after training (*Per-class evaluation and failure
   modes*).
5. **The test set doesn't look like production.** These are dermoscopic
   images taken by clinicians: lit evenly, in focus, lesion centred. Patients
   would send phone photos, with shadows, blur, hair and different skin
   tones. **So the test results are an upper bound** on how well the model
   would really work, and I can't conclude it is ready for patient photos.

## Finetuning and baselines

**Task: image classification**, because each photo shows one lesion and the
decision is about the whole photo (does this lesion need a dermatologist?).
There is nothing to locate, so boxes would add work without adding
information. **Model: `yolo11n-cls`**, the smallest YOLO11 classifier
(1.5 million parameters), finetuned from its ImageNet-pretrained checkpoint.
A small model is enough for 3,517 training images and trains quickly.

**Where and how I trained it.** On my laptop (MacBook, Apple M3, 16 GB),
using the Mac GPU (PyTorch MPS), about 14 minutes per model:
`epochs=40, patience=10, imgsz=224, batch=64, seed=0, flipud=0.5`, with all
other settings at the Ultralytics defaults. I turned on vertical flips
because a photo of a lesion has no "up", so a flipped lesion is still a
realistic example.

**Two models, same settings.** After finding the dermoscope-frame shortcut
(see *What the dataset's composition implies*), I trained a second model on
a copy of the images where the black frame is **painted over with the
photo's own skin colour** (`preprocess.py`: dark pixels connected to the
edge of the image are replaced with the median colour of the rest). The same
step is applied to every photo the dashboard receives. I kept the model with
the **lower business cost on the validation set**, not the higher accuracy:

| | Original images | **Frame removed (kept)** |
|---|---|---|
| Validation accuracy | 65.4% | 62.9% |
| **Validation cost per 1,000 photos** | €63,276 | **€54,184** |
| Test accuracy (8 classes) | 66.2% | 63.1% |
| Test AUC, malignant vs benign | 0.897 | 0.894 |
| **Test melanoma recall** | 51.8% | **57.8%** |
| Lowest score given to a test melanoma | 0.0005 | 0.033 |
| Test cost per 1,000 photos (own thresholds) | €67,035 | **€49,043** |

The frame-free model is slightly **less accurate overall but better where it
matters**: it finds more melanomas, and it no longer gives any melanoma a
near-zero score. That makes it possible to send the lowest-scoring photos
home safely, which is why its cost is lower. Its weights are
`weights/skin_noframe.pt` (the original model is kept as
`weights/skin_original.pt` for comparison).

**Results on the same test split (517 images), kept model:**

| Model | 8-class accuracy | Malignant vs benign AUC | What it does |
|---|---|---|---|
| **Finetuned yolo11n-cls (frame removed)** | **63.1%** | **0.894** | Learns the 8 lesion types |
| Trivial baseline: always "NV" | 18.4% | 0.5 | Calls everything a mole (the most common training class), so it **refers nobody** |
| Pretrained model, zero-shot | 0% | — | Its 1,000 ImageNet classes contain no skin lesions. It called 210 of the 517 lesions a **"tick"**, 75 "face powder" and 52 "velvet" |

**Was the gap as large as expected?** The gap to the baselines is huge, as
expected: the zero-shot model has no way to name a lesion, and the trivial
one would send every cancer home. What surprised me is that accuracy is
**lower than I hoped** (63–66% on 8 classes): validation accuracy stopped
improving after 22–37 epochs. The limits are probably the small 224-pixel
images and classes that genuinely look alike (next section).

## Per-class evaluation and failure modes

**Positive class:** malignant or pre-malignant (AK + BCC + MEL + SCC).
All numbers are for the kept model (frame removed) on the test set.

| Class | Malignant? | Test images | Precision | Recall |
|---|---|---|---|---|
| VASC | no | 23 | 0.92 | **1.00** |
| SCC | yes | 62 | 0.51 | 0.71 |
| NV | no | 95 | **0.76** | 0.70 |
| BKL | no | 85 | 0.66 | 0.62 |
| DF | no | 29 | 0.67 | 0.62 |
| BCC | yes | 74 | 0.56 | 0.59 |
| **MEL** | **yes** | **83** | 0.62 | **0.58** |
| AK | yes | 66 | 0.55 | **0.45** |

**Confusion matrix (rows = true class, columns = predicted):**

| true \ pred | AK | BCC | BKL | DF | MEL | NV | SCC | VASC |
|---|---|---|---|---|---|---|---|---|
| AK | **30** | 14 | 5 | 0 | 2 | 0 | 15 | 0 |
| BCC | 9 | **44** | 4 | 0 | 3 | 2 | 11 | 1 |
| BKL | 2 | 7 | **53** | 1 | 7 | 6 | 8 | 1 |
| DF | 2 | 4 | 0 | **18** | 1 | 1 | 3 | 0 |
| MEL | 6 | 3 | **9** | 1 | **48** | **11** | 5 | 0 |
| NV | 0 | 2 | 7 | 4 | 15 | **66** | 1 | 0 |
| SCC | 6 | 5 | 2 | 3 | 1 | 1 | **44** | 0 |
| VASC | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **23** |

**Which class carries the headline number.** VASC (100% recall: vascular
lesions are red/purple and easy to tell apart) and NV (the highest
precision). But VASC has only 23 test images, so its perfect score is not
reliable.

**Which classes drag it down.**

- **Melanoma, the one that matters most.** Recall is **58%**. Of 83
  melanomas, **21 are called benign** (11 mole, 9 benign keratosis, 1
  dermatofibroma). Melanoma, moles and keratoses are all brown, irregular
  spots, and at 224×224 pixels the details that tell them apart are partly
  lost. Still, the lowest score any test melanoma gets is 0.033, above the
  "pass" threshold, so **no test melanoma is sent home by the model**.
- **AK (45% recall)** is mostly confused with BCC and SCC (29 of 66).
- **Mistakes that don't matter for the decision.** Most errors among the
  malignant classes are between themselves (AK, BCC and SCC confused with
  each other in 60 cases). For the business they are harmless: all of them
  lead to an appointment. For the binary decision at a 0.5 cut-off, the
  model catches **86.7% of malignant lesions** and correctly clears **77.2%
  of benign ones**.

**The dermoscope-frame shortcut: removing the frame did not remove it.**

| Benign lesions flagged as malignant (score ≥ 0.5) | Original model | Frame-removed model |
|---|---|---|
| Photos that originally had a frame | 51% | 54% |
| Photos without a frame | 14% | 17% |

Even after painting over the black border, benign lesions that **came from
a dermoscope with a frame** are still flagged as cancer about 3 times more
often. So the shortcut is not just the black pixels: those photos also
differ in other ways the model can see (the round outline, the lighting,
the camera, probably the hospital they came from). The frame removal helped
the business result (better melanoma recall, no near-zero melanoma scores)
but **did not fix the bias**. This is a real limitation: on phone photos
the model's false alarms could behave quite differently.

## Cost matrix and operating point

**Costs per case (my assumptions):**

| Outcome | Benign lesion | Malignant lesion |
|---|---|---|
| 🟢 Pass (sent home) | €0 (correct) | **missed cancer**: MEL €20,000 · SCC €3,000 · BCC €1,000 · AK €300 |
| 🟡 Review (dermatologist looks at the photo) | €4 | €4 + 90% × €100 appointment + **10% × missed-cancer cost** |
| 🔴 Fail (appointment) | €100 (wasted visit) | €100 |

Where each figure comes from:

- **€100 appointment:** roughly the price of one private dermatology
  consultation in Spain. For a correctly referred cancer I don't count the
  treatment, because it would be needed anyway.
- **€4 review:** 2 minutes of a dermatologist's time at about €120/hour.
- **Missed cancer:** the extra cost of finding it months later instead of now.
  Melanoma is by far the highest: caught early, it is a short excision;
  caught late, it can mean major surgery, scans and immunotherapy, and a
  much lower chance of survival. SCC can spread but rarely does; BCC grows
  slowly and almost never spreads (a later diagnosis mostly means a bigger
  excision); AK is a pre-cancer that is cheap to treat. I chose round
  numbers that keep this order and size, not exact prices.
- **The dermatologist misses 10% of cancers when judging from a photo.** A
  photo is not an in-person exam (no touch, no dermoscope, one angle), so
  even a specialist misses some cancers. Those patients are sent home and
  cost the missed-cancer amount. An in-person appointment ("fail") is
  assumed to catch the cancer.

**Base rate: 10%** of the photos the service receives are malignant or
pre-malignant (melanoma about 3%, since it is 29% of the positives in the
test set). The test set is 55% malignant, so every test image is
**re-weighted** so that positives count for 10% of the total, and costs are
given per 1,000 photos.

**The sweep.** For one threshold (model alone: below → home, above →
appointment) I tried every threshold from 0 to 1 on the **validation** set
and kept the cheapest. For two thresholds I tried every pair on validation
and kept the cheapest one that fits the dermatologists available (next
section), then measured it **once** on the test set. The grid uses steps of
0.005 below 0.1 (where the model's confident "benign" cases sit) and 0.01
above. The dashboard shows the cost curve with the chosen point marked.

**What the sweep found:**

- **Model alone:** the cheapest single threshold is **0.015**: 29% of photos
  are sent home, 71% get an appointment, costing **€70,905 per 1,000
  photos**.
- **Chosen operating point: pass below 0.015, appointment from 0.54, review
  in between.** On the test set it costs **€49,043 per 1,000 photos**:
  **29% pass, 43% review, 28% appointment**, and **no test melanoma is sent
  home by the model**.

**The assumption that would most change the answer is the dermatologist's
miss rate on photos.** I re-ran the whole choice with other values:

| Change | Thresholds (pass / appointment) | Our design | Dermatologist checks all |
|---|---|---|---|
| **As assumed** | 0.015 / 0.54 | **€49,043** | €81,063 |
| Dermatologist misses 0% | 0.015 / 0.58 | €30,374 | **€14,000** |
| Dermatologist misses 3% | 0.015 / 0.58 | €35,880 | **€34,119** |
| Dermatologist misses 4% | 0.015 / 0.58 | **€37,716** | €40,825 |
| Dermatologist misses 20% | 0.005 / 0.21 | **€59,876** | €148,126 |
| MEL miss €5,000 | 0.05 / 0.81 (more pass; 2 melanomas home) | **€35,450** | €37,379 |
| MEL miss €50,000 | **nobody passes** / 0.07 | **€68,048** | €168,432 |
| Base rate 5% | 0.015 / 0.54 | **€36,426** | €42,532 |
| Base rate 20% | **nobody passes** / 0.07 | **€71,160** | €158,126 |
| Review €10 / €20 | 0.015 / 0.54 (no change) | **€51,598 / €55,858** | €87,063 / €97,063 |

So: **the design beats a dermatologist checking every photo as long as
dermatologists miss more than about 3–4% of cancers from a photo.** Below
that, paying a person to look at everything is cheaper. Separately, if a
missed melanoma costs €50,000+ or more than 20% of photos are cancers, the
model should **never send anyone home** and only decide who gets an
appointment directly. The price of a review hardly matters.

## What a person still has to check

**Thresholds: pass below 0.015, appointment from 0.54, review in between.**

- **Volume:** I assume the service receives **500 photos a day**.
- **Review share:** **42.6%** of photos → about **213 photos a day**.
- **Time:** about **2 minutes** per photo → 7.1 hours of dermatologist time a day.
- **People:** if each dermatologist spends 4 hours a day on photo review
  (120 photos), that needs **1.8 dermatologists**.
- **Is it realistic? Yes.** The service has **2 dermatologists** for review.
  I only let the thresholds use **90%** of their capacity (216 photos a day)
  to leave room for busy days, and the chosen design needs 213 on the test
  set, so it fits. Their full capacity is 240 reviews a day, so a day with
  about 13% more photos than usual would go over; the team would need a rule
  for busy days (for example, sending the highest-scoring reviews straight
  to appointments).

**Compared with the alternatives (test set, 10% base rate, per 1,000 photos):**

| Strategy | Cost | Home | Review | Appointment | Melanomas sent home by the model (of 83) |
|---|---|---|---|---|---|
| Do nothing (everyone home) | €680,632 | 100% | 0% | 0% | 83 |
| Dermatologist checks every photo | €81,063 | 0% | 100% | 0% | 0 (but ~10% missed in review) |
| Model alone (threshold 0.015) | €70,905 | 29% | 0% | 71% | 0 |
| **Model + review band (chosen)** | **€49,043** | **29%** | **43%** | **28%** | **0** |

**What this means.** The design is the cheapest option: **40% cheaper than
a dermatologist looking at every photo** and 93% cheaper than doing
nothing. It works because the model is good at the two ends: it can safely
send home the 29% of photos it is most sure are benign, and send the 28% it
is most sure are cancer **straight to an appointment**, where the cancer
can't be missed in a photo review. The dermatologists only see the
uncertain middle.

**What extra dermatologists are worth** (thresholds re-chosen on validation
for each staffing level):

| Dermatologists | Thresholds | Cost per 1,000 | Review |
|---|---|---|---|
| 1 | 0.015 / 0.16 | €58,038 | 18% |
| **2 (chosen)** | **0.015 / 0.54** | **€49,043** | **43%** |
| 3 | nobody passes / 0.40 | €50,532 | 65% |
| 4–5 | nobody passes / 0.54 | €50,206 | 72% |

Going from 1 to 2 dermatologists saves about €9,000 per 1,000 photos. **A
third one adds nothing**: once the uncertain cases are covered, sending more
photos to review instead of straight to an appointment doesn't help,
because the review itself misses 10% of cancers. (The 3-dermatologist line
is even slightly worse on test than 2, because its thresholds were chosen on
validation: a sign of how noisy these numbers are.) **So 2 reviewing
dermatologists is the right staffing.**

## Limitations & next steps

- **The frame bias is still there.** Painting over the frame improved
  melanoma recall but benign lesions from framed dermoscopes are still
  flagged ~3× more often. *Next step:* train only on frame-free photos, or
  balance the classes within each image source so the source can't predict
  the label.
- **Melanoma recall is only 58%.** 21 of 83 melanomas are called benign
  (though none is sent home, because their scores are above 0.015). *Next
  step:* train at a higher resolution from the original ISIC images
  (384–512 pixels), try `yolo11s-cls`, and give melanoma a higher weight in
  the training loss.
- **Dermoscope images, not phone photos.** The test set doesn't look like
  what patients send, so all results are an upper bound. *Next step:* before
  any real use, collect a few hundred phone photos with a confirmed
  diagnosis and measure the model on those.
- **The key assumption is the dermatologist's miss rate.** The design only
  beats full review if dermatologists miss more than ~3–4% of cancers from a
  photo. *Next step:* measure it: have dermatologists review a set of photos
  whose diagnosis is known from biopsy.
- **The costs and the base rate are my assumptions.** If a missed melanoma
  costs €50,000+ or more than 20% of photos are cancers, nobody should be
  sent home. *Next step:* get the real share of cancers among the service's
  photos, and the real cost of late vs early treatment.
- **"Pass" is decided by very few images.** The pass threshold (0.015) sits
  where only a handful of test cancers are, so one unlucky melanoma could
  change the decision. *Next step:* cross-validation and confidence
  intervals on the number of cancers sent home before shipping the pass
  band.
- **Small classes and duplicates.** DF (29) and VASC (23) test images are too
  few to trust, and 5 test images also appear in training (4 SCC, 1 BKL).
  *Next step:* remove the near-duplicates and report both numbers.

## Generative AI use disclosure

I used **Claude (Anthropic), through Claude Code**, throughout this
assignment, as allowed by the professor for this assignment. It helped me
compare and choose the Roboflow dataset, analyse the dataset (class counts,
duplicates and the dermoscope-frame check), write `train.py`, `preprocess.py`
and `app.py`, run the training and the cost sweeps, test the dashboard, and
draft this report. I reviewed the framing, the cost assumptions, the choice
of 2 reviewing dermatologists and the conclusions, and I can explain them.
