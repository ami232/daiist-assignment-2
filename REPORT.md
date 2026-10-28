# Assignment 2 Report

- **Name**: Paul Khairallah
- **Student ID**: 18966
- **Email**: pkhairallah.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

**The decision.** A Dutch packhouse washes table potatoes and packs them into
3 kg retail bags. Today, as at most packers, people at a **manual picking
table** pull rotten or blemished potatoes and stones off the belt: R S
Cockerill, one of the UK's largest independent packers, ran **4–8 workers on
manual inspection per line** at 2–8 t/h before automating
([Food Engineering, 2023](https://www.foodengineeringmag.com/articles/100894-potato-packer-improves-performance-with-automated-optical-sorting)).
The model would take over that job on one ~6 t/h lane after destoning and
washing: for every object it decides **pass**, **eject** (air jet), or, if a
review band is used, **divert** to one person at a side station. The line
supervisor owns the reject setting; quality staff own release criteria. The
unit of decision is one object on the belt, not one image.

So the question is not "model or nothing" but **"model or today's picking
table"**. I price both, plus doing nothing, on the same terms.

**What each mistake costs** (EUR per object):

| Outcome | Cost | Construction | Source / status |
|---|---|---|---|
| Missed defect (FN) | **€1.50** | 1 in 20 bags with a bad potato comes back as a complaint × (€3.85 refund + £22.50 ≈ €26.25 retailer charge). Assumes the shopper notices; the share who would is tested below | Complaint propensity: only ~1 in 20 dissatisfied consumers complains (TARP 1986, cited in [Chebat & Davidow, 2005](https://chaireomerdesserres.hec.ca/wp-content/uploads/2020/05/Silent-Voices.pdf)). Charge: UK retailers charge suppliers **£0–45 per consumer complaint** ([Groceries Code Adjudicator, 2015](https://www.gov.uk/government/news/survey-shows-groceries-code-adjudicator-is-making-a-difference)); mid-point used. Refund: AH 3 kg bag, €3.85 |
| Missed stone (FN) | **€4.61** | 1 in 20 × (€3.85 refund + £45 ≈ €52.49 + 45 min QA investigation at €47.90/h) | Same GCA page: all retailers charge for product-safety failings, so the top of the range. Injury or recall not priced |
| Good potato rejected (FP) | **€0.060** | 0.200 kg × (€0.34/kg − €0.04/kg salvage) | LNCN Middenmeer firm-cooking 40/65, 2026 week 37, €33–35/100 kg ([LNCN](https://www.landbouwbeurs.com/beursnoteringen/tafelaardappelen)); salvage assumed |
| One human review | **€0.050** | €47.90/h × 3 s ÷ (3,600 × 0.80 utilisation) | Netherlands 2025 hourly labour cost, €47.90 ([Eurostat](https://ec.europa.eu/eurostat/en/web/products-eurostat-news/w/ddn-20260331-2)) |
| Picking-table labour | **€6.39 per 1,000 objects** | 4 pickers × €47.90/h ÷ 30,000 objects/h | Cockerill's lower bound of 4 per line |
| Correct reject / pass | €0 | | |

£ converted at HMRC's October 2026 rate of €1.1665
([HMRC](https://www.trade-tariff.service.gov.uk/exchange_rates/view/2026-10)).
A missed defect costs **25×** a false reject; a missed stone **77×**.

**People are not perfect sorters either.** In a study of 82 professional
inspectors, people caught **85% of defective items** and wrongly rejected a
large share of good ones ([See, 2015, *Human Factors*](https://pubmed.ncbi.nlm.nih.gov/26342002/)).
For potatoes specifically, a sorting-equipment vendor reports that human final
inspectors **discard 10–20% of acceptable potatoes**
([SiftAI/KPM, via Michigan Farm News, 2024](https://www.michiganfarmnews.com/ai-powered-potato-sorter-is-more-accurate-than-human-inspector));
that is a vendor claim, so I use its low end (10%) and test 2–20%. The same
85% / 10% apply to a reviewer at a side station.

**Base rates.** **1.0% of objects are defective potatoes**: that is what the
test belt shows (31 of 2,953 potatoes) for this camera's defect types. For
context, Swiss wholesalers reject 12–24% of table potatoes at sorting, for all
reasons combined
([ETH Zurich via ScienceDaily, 2015](https://www.sciencedaily.com/releases/2015/10/151023105848.htm)).
**0.05% are stones** (one per 2,000 objects) after destoning; I found no
published residual rate, so this is an assumption, and the decision turns out
not to depend on it. Quality rules frame what is acceptable: UNECE FFV-52
allows ware potatoes up to 6% defective tubers by weight, of which at most 1%
rotten, and **packages must be free of all foreign matter**
([UNECE, 2017](https://www.unece.org/fileadmin/DAM/trade/agr/standard/standard/fresh/FFV-Std/English/52_EarlyAandWarePotatoes.pdf)).

**Line.** 30,000 objects/h (~6 t/h at 200 g) for one 8-hour shift: 240,000
objects and ~16,000 bags a day. At most one person is available for a review
station.

**The asymmetry, and what it implies.** Misses cost far more than false
rejects, so recall on `bad` and `stone` matters more than precision and the
threshold should sit low. But a missed defect only costs money when a shopper
complains, so throwing away good potatoes is not free either; the threshold
is where those balance.

## Dataset

- **Roboflow Universe:** workspace `ms-workspace-m1gci`, project
  `potato-sorting-on-conveyor`, **version 9**
  (https://universe.roboflow.com/ms-workspace-m1gci/potato-sorting-on-conveyor/dataset/9),
  CC BY 4.0. Downloaded as is in YOLOv11 format; not forked.
- **What one image is:** a 512×512 top-down frame of potatoes on a roller
  conveyor, ~30 objects per frame. Two conveyor cameras (file prefixes
  `kartoffel_auto` and `BELANA_auto`) plus ~50 web/phone photos in train.
- **Classes:** `potato` (every potato), `bad` (a box around a rot/blemish spot
  **on** a potato; the potato itself is also boxed as `potato`), `stone`, and
  `cut` (a cut-open potato; almost all from web photos).
- **Why it serves the problem:** it is the actual sorting scene, the actual
  camera geometry, and the two things a sorter removes (defects and stones),
  rather than studio photos of single potatoes. Because `bad` marks the spot, not
  the potato, I turn detections into one decision per object: a potato is
  "defective" if a `bad` box sits inside it, and its reject score is the
  confidence of the most confident predicted `bad`/`cut` box inside it (or of an
  overlapping predicted `stone` box).

## What the dataset's composition implies

| Split | Images | potato | bad (images with) | stone (images with) | cut |
|---|---|---|---|---|---|
| train | 3,012 (≈1,004 originals × 3 augmented copies) | 84,514 | 1,160 (772) | 2,075 (279) | 176 |
| valid | 121 | 3,763 | 58 (30) | 43 (9) | 10 |
| test | 99 | 2,953 | 29 (21) | 93 (8) | 3 |

Object size (side of the box ÷ side of the image, median): potato 0.10–0.11
(~55 px), stone ~0.09, but **bad only 0.026–0.033 (~13 px)**, and the smallest
10% of blemishes are under 6 px. Version 9 is pre-processed (contrast
stretching, 512×512 stretch) and its training split carries Roboflow's
augmentations (±15% brightness, 90° rotations, ±15° rotation).

What this implies:

1. **The positive classes are the thin ones.** 29 `bad` instances and 3 `cut`
   instances in test. A recall measured on 29 objects has a 95% interval about
   ±17 points wide, so I report Wilson intervals and do not trust any
   difference of a few points. Nothing at all can be concluded about `cut`.
2. **Stones are clustered.** 93 test stones in 8 frames. Stone recall is really
   measured on 8 scenes, not 93 independent events.
3. **Blemishes are tiny.** At ~13 px they sit at the edge of what a YOLO
   detector resolves at 512 px input, which is why I train at 640 (upsampling
   gives the spots more pixels) and why I expect `bad` to be the class that
   drags.
4. **The test split is not an honest stand-in for production.** The conveyor
   frames carry capture timestamps: **94 of the 99 test frames were taken
   within 2 seconds of a training frame** (median gap 0.56 s), on the same two
   days. They show the same potatoes on the same belt from almost the same
   position. Test scores will therefore be optimistic about a new day, a new
   batch or a new camera. The assignment asks me to use the split as published,
   so I keep it, but every number below should be read as "on frames very like
   training", not as a production estimate.

## Finetuning and baselines

**Model and task.** Object detection, `yolo11s` finetuned from the COCO
checkpoint. Detection rather than classification because a frame holds ~30
objects and the sorter has to act on each one, and because the defect label is a
small spot on a potato, which a whole-image classifier could not localise.
`yolo11s` rather than `yolo11n` because the bottleneck is ~13 px blemishes and
the extra capacity is cheap on a GPU.

**Where and how it was trained.** Google Colab, one Tesla T4, via
`colab/train_on_colab.ipynb`, which clones this repo and runs
`python main.py train` unchanged. Settings (all in `TRAIN_CFG` in `train.py`):
60 epochs, imgsz 640, batch 32, optimizer auto (AdamW, lr 0.00125), mosaic
off for the last 10 epochs, extra vertical flips (top-down camera), seed 0,
deterministic. 60 epochs took 47 minutes. The best checkpoint (epoch 48, val
mAP50 0.662 / mAP50-95 0.552) is committed as `weights/potato_yolo11s.pt`. The
evaluation below was re-run on CPU from those committed weights and reproduces
the test numbers Colab printed (mAP50 0.617, mAP50-95 0.501).

**Results on the same test split (99 images).**

| Model | mAP50 | mAP50-95 | Class-agnostic AP50 (finds the objects at all?) | As a sorter at its best threshold, € per 1,000 objects |
|---|---|---|---|---|
| Finetuned yolo11s | **0.617** | **0.501** | 0.995 | **€4.94** (t = 0.02) |
| COCO yolo11s zero-shot | 0.000 | 0.000 | 0.441 | €17.35 (never rejects anything) |
| Predict nothing | 0.000 | 0.000 | 0.000 | €17.35 |

The COCO model scores zero on every class by construction: it has no potato,
stone or blemish class. To give it a fair hearing I also scored it
class-agnostically. It does find 94% of the objects at some confidence, but at
conf 0.25 it calls them `elephant` (11), `hot dog` (7), `knife` (5) and `spoon`
(1). It has no label that means "remove this", so as a sorter it is identical to
doing nothing. The finetuned model beats both baselines by a wide margin, which I
expected; what I did not expect was how badly the blemish class does at box
level (next section) while still being useful for the decision.

## Per-class evaluation and failure modes

Positive class for the decision: **`bad` and `stone`** (things to remove);
`potato` is the negative. `cut` is also a reject reason but has 3 test
instances, so I report it and draw no conclusion from it.

| Class | Test instances | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| potato | 2,953 | 0.989 | 0.998 | 0.995 | 0.971 |
| stone | 93 (8 images) | 0.912 | 0.968 | 0.972 | 0.884 |
| bad | 29 (21 images) | 0.289 | 0.103 | 0.120 | 0.033 |
| cut | 3 | 0.565 | 0.333 | 0.382 | 0.118 |
| **all (mean)** | | 0.689 | 0.601 | **0.617** | **0.501** |

(P and R are Ultralytics' values at the confidence that maximises F1.)

Box-level confusion matrix at conf 0.25 (rows predicted, columns true):

| | true bad | true cut | true potato | true stone | true background |
|---|---|---|---|---|---|
| pred bad | 3 | 0 | 0 | 0 | 6 |
| pred cut | 0 | 1 | 0 | 0 | 0 |
| pred potato | 1 | 1 | 2,947 | 2 | 29 |
| pred stone | 0 | 0 | 0 | 88 | 10 |
| pred background (missed) | 25 | 1 | 6 | 3 | — |

**Who carries the headline number.** The 0.617 mAP50 is an average of potato
(0.995) and stone (0.972) being nearly solved and `bad` (0.120) being close to
useless *as a box detector*: at conf 0.25 it finds 3 of 29 blemishes. Averaging
hides exactly the class the business case depends on.

**Why it still works as a sorter.** The blemish boxes exist, they just come
with low confidence. Scoring each potato by its most confident blemish box, the 31
defective potatoes on the test belt have a median score of 0.048, while 2,251 of
the 2,922 good potatoes score exactly 0 and 90% score below 0.007. The ranking is
good even though the confidences are low (low scores rank well; they are not
calibrated probabilities), which is why the operating threshold
ends up at 0.02 rather than anywhere near Ultralytics' default 0.25. At 0.25 the
sorter would catch only 16% of defects and cost €12.84 per 1,000, more than two and a half times the cost at 0.02.

**Where it breaks.**
- **Blemishes.** `bad` boxes are ~13 px and the model is never confident about
  them. At the operating point (t = 0.02), 4 of 31 defective units still pass.
  The 3 missed blemishes are only slightly smaller than the 26 caught (median
  13 px vs 15 px), and 3 cases are too few to say what drives the misses.
- **Stray reject boxes.** 109 low-confidence blemish/stone boxes land on no
  labelled object. The ejector would still fire, so they count as false
  ejections in the costing. At conf 0.25, 10 stone boxes and 29 potato boxes
  also match no labelled object.
- **Stones** are the strong class: 88 of 93 found, 3 missed, 2 called potato.
  But the 93 come from 8 frames, so this is closer to 8 observations than 93.
- **`cut`** is untestable (3 instances) and comes mostly from web photos that
  look nothing like the belt.

## Cost matrix and operating point

**Method.** Each object on the test belt has one reject score. Recall and
false-reject rates are measured per group (31 defective potatoes/cut pieces,
93 stones, 2,922 good potatoes, plus 109 stray reject boxes) and re-weighted
to the production base rates, not the test split's clustered mix. Stray boxes
are extra ejector events, costed like false rejects (a convention; ejector
logs would show how many really knock a potato off).

`cost per 1,000 = 1,000 × [π_d (1 − R_d(t)) C_d + π_s (1 − R_s(t)) C_s + π_g F(t) C_fp]`

where F(t) = (good + stray scoring ≥ t) / 2,922. The sweep covers 104
thresholds from 0.001 to 0.99 using every object's score (`costing.py`,
`artifacts/cost_sweep.csv`). A separate standard-library script working from
the 24-row count table (`analysis/recompute.py`) gives the same optimum and
the same cost to four decimals.

The 3,049 labelled objects (2,953 potatoes + 93 stones + 3 cut) become 3,046
decision units because each of the 3 cut pieces is also boxed as a potato and
is counted once.

![cost against threshold](artifacts/cost_vs_threshold.png)

| Threshold | € / 1,000 | Defect recall | Stone recall | Good potatoes rejected (incl. strays) |
|---|---|---|---|---|
| 0.001 | 15.85 | 100% | 100% | 26.7% |
| 0.01 | 6.02 | 93.5% | 97.8% | 8.4% |
| **0.02** | **4.94** | **87.1%** | **96.8%** | **4.9%** |
| 0.03 | 5.08 | 80.6% | 96.8% | 3.5% |
| 0.05 | 9.18 | 48.4% | 96.8% | 2.3% |
| 0.25 | 12.84 | 16.1% | 96.8% | 0.2% |
| 0.50 | 15.26 | 0% | 92.5% | 0.1% |

**Operating point: reject at a score of 0.02. €4.94 per 1,000 objects,
€1,186 per day.**

| Component | Calculation | € / 1,000 |
|---|---|---|
| Missed defects | 1,000 × 0.010 × 4/31 × €1.505 | 1.942 |
| Missed stones | 1,000 × 0.0005 × 3/93 × €4.613 | 0.074 |
| False rejects | 1,000 × 0.9895 × 144/2,922 × €0.06 | 2.926 |
| **Total** | | **4.942** |

At that point the model catches **87% of defective potatoes**, about the same
as a person (85%), and throws away **4.9% of good potatoes**, about half the
low end of what human inspectors bin (10%). Defect recall has a 95% Wilson
interval of 71%–95% (27 of 31); stone recall 91%–99%, but the 93 stones sit in
8 frames, so that interval is too narrow.

**It is not a large-margin winner.** t = 0.03 costs €5.08 per 1,000, €33 a day
more. Just above 0.03 the curve climbs fast: at 0.05 defect recall halves and
cost nearly doubles.

**Sensitivity** (one input moved, all else fixed, full sweep and band search
repeated each time):

| Input (base value) | Range that keeps t = 0.02, no review | Outside that range |
|---|---|---|
| Missed defect C_d (€1.505) | €1.29–3.01 | below → 0.03 (below €0.36 → 0.12, below €0.29 → 0.21); above → 0.0075 |
| False reject C_fp (€0.060) | €0.030–0.070 | below → 0.0075; above → review band [0.02, 0.03) |
| Defect rate π_d (1.0%) | 0.86%–1.98% | below → 0.03; above → 0.0075 |
| Missed stone C_s (€4.61) | up to €140 | above → 0.0075 |
| Stone rate π_s (0.05%) | up to 1.48% | above → 0.0075 |
| Review cost C_r (€0.050) | ≥ €0.042 | below → review band [0.02, 0.03) |
| Reviewer miss rate h_m (15%) | ≥ 2.6% | below → review band |

**The assumption that matters most is how often a missed blemish actually
costs money.** The €1.505 assumes every potato the dataset marks `bad` would
bother a shopper. If only a share *s* of them would:

| Share of missed blemishes that would bother a shopper | Best threshold | Model € / 1,000 | Picking table € / 1,000 |
|---|---|---|---|
| 100% (base) | 0.02 | 4.94 | 14.93 |
| 50% | 0.03 | 3.62 | 13.80 |
| 25% | 0.03 | 2.90 | 13.23 |
| 10% | 0.21 | 1.39 | 12.90 |

So the threshold sits at 0.02–0.03 unless most blemishes are cosmetic, in which
case it should rise to ~0.21. Either way the model costs far less than the
picking table. The share is the first thing to measure: grade a sample of the
potatoes the model rejects and passes.

## What a person still has to check

**Recommendation: no review station, at these costs.** Searching all
threshold pairs (pass below L, review from L to H, reject at H and above), the
cheapest policy is an **empty band, L = H = 0.02**, with or without a staffing
cap. The cheapest non-empty band, **[0.02, 0.03)**, sends 1.45% of objects to a
person (436/h, 3,487 a day, **0.45 of one person**, within the one available
reviewer's 960/h). But it costs **€5.06 per 1,000**, more than the model alone:
the reviewer's own errors (15% misses, 10% good binned) and €0.73 per 1,000 of
labour outweigh what the review saves. Review would only pay if a check cost
under **€0.042** (about €40/h at 3 s and 80% utilisation), a good potato were
worth more than **€0.07**, or reviewers missed fewer than **2.6%** of defects.

**Priced against the alternatives** (€ per 1,000 objects and per 8-hour day of
240,000 objects):

| Policy | € / 1,000 | € / day | Missed defects / day | Good potatoes binned / day | People |
|---|---|---|---|---|---|
| Do nothing (also the COCO and predict-nothing baselines) | 17.35 | 4,165 | 2,400 | 0 | 0 |
| **A person checks everything: today's picking table** | **14.93** | **3,582** | 360 | ~23,700 | 4 |
| **Model alone, t = 0.02** | **4.94** | **1,186** | 310 | ~11,700 | 0 |
| Model + best non-empty review band [0.02, 0.03) | 5.06 | 1,215 | 333 | ~8,700 | 0.45 |

(Inspecting each object individually at 3 s would need ~31 people at once, so
it is not a real option; the picking table is how "a person checks everything"
is done in practice.)

The picking table costs **€8.54 per 1,000 in errors** (mostly good potatoes
binned) **plus €6.39 in labour**. The model's error cost is **€4.94**, and it
needs no pickers. **Saving: about €2,400 a day, or ~€600,000 a year** at 250
one-shift days. That is the most the camera, compute, ejector, installation
and maintenance could cost before the case disappears. I found no published
price for such a system, so I state the break-even rather than a payback;
vendor case studies report payback in about two seasons
([Raytec, 2026](https://www.raytecvision.com/case-studies/thorland-farms-raytec-potato-optical-sorter-case-study/))
and a 20–25% throughput gain after automating
([Cockerill](https://www.foodengineeringmag.com/articles/100894-potato-packer-improves-performance-with-automated-optical-sorting)).

**How robust is that?** The saving rests mostly on two uncertain inputs: how
many good potatoes people bin, and the cost of the pickers. Varying them:

| Human false-reject rate | 2 pickers, €30/h | 4 pickers, €47.90/h (base) | 8 pickers, €47.90/h |
|---|---|---|---|
| 2% | €204 / day | €1,256 / day | €2,789 / day |
| 10% (base) | €1,344 / day | €2,396 / day | €3,929 / day |
| 20% | €2,768 / day | €3,821 / day | €5,354 / day |

The model comes out cheaper in all 24 combinations tested (2–20% false
rejects, 2–8 pickers, €30 or €47.90/h), from €204 to €5,354 a day. **The case
does not depend on any one optimistic number.** In the most pessimistic
combination it is purely a labour saving (the model's error cost is then
slightly higher than the people's); in the central one, about two-thirds is
labour and one-third fewer good potatoes thrown away.

**Sanity check on complaints.** With every `bad` blemish treated as
complaint-worthy, the model lets ~310 defective potatoes a day reach bags,
which becomes **~15 complaints a day** (picking table ~18; doing nothing
~120). That is high for one lane (~970 per million bags), which suggests many
`bad` marks are cosmetic. That is the share *s* in the table above, and it
lowers every policy's cost without changing the ranking.

## Limitations & next steps

1. **Near-duplicate test split.** 94 of 99 test frames were taken within 2 s of
   a training frame (median 0.56 s), all on the same two days, so the model's
   87% defect recall is optimistic for a new day, lot or camera. Since the
   model only matches a person on defects, a real drop would erode the
   quality side of the case (the labour saving stays). Next: new capture
   sessions across days, lots, cultivars and lighting; keep each session in one
   split; tune on a development set and evaluate once on a grouped holdout.
2. **Thresholds tuned on the test counts, from 31 defects.** Each defect is 3.2
   points of recall. Next: re-run the sweep on a few hundred defective potatoes
   from production, ideally in shadow mode next to the existing picking table.
3. **How many blemishes actually matter is unknown.** The dataset's `bad` mixes
   rot with small cosmetic marks. Next: grade a sample of the model's passes
   and rejects against the retailer's spec to measure the share *s*.
4. **Human error rates partly rest on a vendor claim** (10–20% of good potatoes
   binned). The independent study (85% caught) is on manufactured parts, not
   potatoes. Next: time and score the current picking table for a few shifts.
5. **Complaint economics.** The 1-in-20 complaint rate is a general consumer
   figure and the £0–45 charge range is UK (2015); a Dutch retailer's terms may
   differ. Loss of a contract after repeated complaints, the real risk behind
   picking tables, is not priced. Next: the packer's own complaint and claims
   records.
6. **System cost not priced.** No published price for a vision sorter was
   found; the report gives a break-even (~€2,400/day) instead. Next: quotes
   for camera, compute, ejector and integration.
7. **Ground-truth-assisted scoring.** Object scores attach predicted blemish
   boxes to labelled potato boxes; a real line needs detection, tracking and
   ejector timing, which add failures. Next: end-to-end trial with ejector logs.
8. **Stones and safety.** The €4.61 is a routine complaint cost; injury or
   recall would be far larger, and UNECE requires packs free of foreign
   matter. Economic optimisation cannot set an acceptable stone escape rate;
   the destoner and final checks stay.
9. **`cut` is not testable** (3 test instances, mostly web photos).

## Generative AI use disclosure

I used Claude throughout this assignment. It helped with writing the code
and formatting of the report.
