# Notes

AI-generated working log, written by Claude Code during the build from Javier's prompts. The analysis and conclusions are in REPORT.md.

Running log of decisions, bugs and numbers. Bullets, newest section last.

## Setup (2026-10-23)

- Environment: Claude cloud container, no GPU. `uv sync` pins the CPU build of
  PyTorch (2.11.0+cpu). Ultralytics 8.4.173, Gradio 6.29.1.
- First `uv sync` failed: the container's network policy refused
  `download-r2.pytorch.org`. Fixed by allowlisting the PyTorch, Roboflow and
  GitHub release hosts in the environment settings. Sync now succeeds.
- Roboflow key is supplied as the `ROBOFLOW_API_KEY` environment variable in the
  cloud environment settings, not via `.env`. `api_key.py` checks the variable
  first. Verified: Roboflow accepts it, workspace `javialonsov03-gmail-com`.
- A `.env` with the placeholder value exists locally and is git-ignored. The
  environment variable takes precedence, so it is inert.
- Repo ships both `train.py` and `train.ipynb`, and both `app.py` and
  `app.ipynb`. `main.py` refuses to run a stage while both exist. The notebooks
  must be deleted before either stage can run. Not yet done.
- `check.py` loads every `*.pt` under `weights/` and launches `main.py app` on
  port 7860. `weights/` currently holds only `.gitkeep`.
- Training plan: `train.py` is the real pipeline. Training itself runs on
  Google Colab (T4) because this container has no GPU. Weights are committed
  from Colab as `weights/best.pt`, then pulled here for the sweep, review band
  and dashboard.
- Branching: all work happens on `assignment-2`. Before the final
  PR to ami232/daiist-assignment-2, merge that branch into the fork's `main`
  and open the PR from `main`. To be done at the end, on instruction.

## Dataset survey (2026-10-23)

- `scripts/survey_datasets.py` downloads each candidate's latest version in
  YOLOv11 format to `data/candidates/<workspace>__<project>/` and computes
  stats from the label files. Report: `data/candidates/SURVEY.md` (git-ignored
  with the rest of `data/`). Seed 42 for the preview picks. Reruns reuse
  existing downloads unless `--refresh` is passed.
- Skipped `project-33xgh/damaged-box-detection`: it is a classification
  project, not object detection.
- Five object-detection candidates surveyed. Headline facts from the labels:
  - `dms-8pnzm/package-kglu1-we3hv` v2: 1169 images, 1023/73/73 split, classes
    Damaged (1402 instances) and package (1175). Damaged boxes are tiny: median
    0.96% of image area, p10 0.23%. The exported images are grayscale, so the
    version has a grayscale preprocessing step baked in. No zero-label images.
  - `infocomm-project-fpj1z/damaged-packages-6snht` v2: 1303 images,
    1149/108/46 split. "damaged package" 3099 instances vs "undamaged" 517, so
    the negative class is the minority. Test split is small (46 images, 121
    instances, 19 undamaged). Median box area 11.7% / 7.6%.
  - `nani-tmzf6/package-detection-5ozpr` v2: 2406 images, 1679/476/251 split.
    "damaged" 2051 vs "no damaged" 386. Boxes cover most of the image (median
    72% / 58%), with roughly one box per image, so it behaves like
    classification with boxes. 9 images with zero labels.
  - `roboflow-ngkro/package-detection-e1ssd` v6: 1200 images, 840/240/120
    split, 4 classes (Box 448, Box_broken 304, Open_package 308, Package 433).
    Balanced, mid-size boxes (median 26 to 46%). Test split per class is 29 to
    42 instances. Mix of boxes and mailer envelopes.
  - `damages-shoeg/damaged_boxes-3oeks` v3: 219 images, train split only, no
    valid or test. Frames from a handful of videos of one product (Nescafe
    cartons), so images are near-duplicates. Classes buckled 8, open 10, tear
    149, well 54.
- Bug fixed during the run: preview boxes were drawn before resizing, so on
  high-resolution video frames the lines and labels vanished. Now draws after
  resizing.
- Dataset choice is the owner's. No recommendation recorded.

## Dataset pinned (2026-10-23)

- Chosen by the owner: Roboflow Universe `roboflow-ngkro/package-detection-e1ssd/6`
  (workspace `roboflow-ngkro`, project `package-detection-e1ssd`, version 6,
  YOLOv11 export). Pinned in `src/data.py`; `get_dataset()` downloads to
  `data/raw/` once and returns `data.yaml`.
- The other candidate downloads were deleted from `data/candidates/`; only the
  chosen one and its preview remain there.
- Decision classes, set by the owner: positive ("compromised") = Box_broken or
  Open_package; negative ("intact") = Box or Package. An image is positive if it
  holds at least one positive box. Constants live in `src/data.py`.
- Project is public on Universe (API `public: True`), but the export's
  `data.yaml` license field says `Private`. Worth stating in the report.

## Dataset description (2026-10-23)

- `scripts/describe_dataset.py` writes `outputs/dataset_stats.json`,
  `outputs/dataset_stats.md` and `outputs/box_area_hist.png`.
- Roboflow version 6 metadata: preprocessing = auto-orient plus resize to
  640x640 stretch; augmentation field empty, so no augmentation. All 1200
  images are 640x640 on disk. No field says how the split was made; the exact
  70/20/10 ratio (840/240/120) matches Roboflow's default, so a Roboflow random
  split is the likely explanation. Inference, not fact.
- Per split: train 840 images / 1052 boxes, valid 240 / 305, test 120 / 136.
  Instances per class (train/valid/test): Box 314/103/31, Box_broken 206/64/34,
  Open_package 225/54/29, Package 307/84/42.
- Image-level balance: train 424 positive vs 416 negative, valid 118 vs 122,
  test 62 vs 58. Roughly 50/50 in every split, so the dataset base rate is not
  a production base rate. Only 5 images mix positive and negative boxes.
- Zero-label images: none. Boxes per image: mean about 1.2, max 11.
- Box area: medians Box 25.6%, Box_broken 29.6%, Open_package 38.7%, Package
  45.6% of image area. p10 is 8 to 19%, so objects are large; not a small-object
  problem.
- Near-duplicates (pHash, Hamming <= 4): 24 pairs, 14 of them exact hash
  matches. Within train 11, none within valid or test. Across splits: 13 pairs
  (train/valid 7, test/valid 4, test/train 2). So some test images have a
  near-identical twin in train or valid, which inflates test scores slightly.
  Example pairs with filenames are in `outputs/dataset_stats.md`.
- Duplicate check verified at pixel level: 18 of the 24 flagged pairs differ by
  a mean of under 5 grey levels, so they are re-uploads, not pHash
  false positives. The twins carry different label files (re-annotated copies).
  6 of the 120 test images have a twin in train or valid. Keep this in mind
  when reading test metrics; dropping them is an option the owner decides.

## Business figures and scoring (2026-10-23)

- `src/config.py` holds every owner-supplied figure. Costs in DOP per unit:
  FN 7000 (replacement 4400 + two-way freight 2500, claim window closed), FP 40
  (10-min unpack at RD$3/min), human review 40 (same unpack), model-FAIL
  confirmation 10 (3-min glance, photo is the claim evidence). Wage RD$3/min.
  Base rate 1%, sweep [0.5%, 1%, 3%, 5%, 10%, 15%]. Containers 20ft 200,
  40ft_low 400, 40ft_high 600 units; receiving is per container, no daily
  flow. Staff 2, 7 productive hours/day, 7-day claim window. DOP per EUR
  68.67 (poundsterlinglive.com, 2026-10-23).
- The class lists moved from `src/data.py` into `src/config.py`; `data.py`
  re-exports them so the describe script is unchanged.
- `src/scoring.py`: `image_score` = max confidence over positive-class
  detections, resolved by class name via `results.names`, 0.0 if none.
  `decide(score, t_low, t_high)`: `score >= t_high` fail, `score >= t_low`
  review, else pass. Ties go to the stricter outcome; `t_low == t_high`
  gives no review band. `image_label` reads a YOLO label file and returns
  True if any box is a positive class.
- `tests/test_scoring.py`: 14 tests, all passing. Covers the five required
  cases plus name-based resolution with a shuffled index map, bad threshold
  order, and that the config class lists are disjoint and cover the dataset.

## Training pipeline and evaluation code (2026-10-23)

- Branch renamed from `claude/funny-rubin-b746w8` to `assignment-2` locally and on
  origin. The old remote branch could not be deleted from this session (the push
  proxy returns 403 on deletions); the owner deletes it on GitHub.
- `train.ipynb` deleted. `train.py` is the pipeline: `ensure_dataset()`, finetune
  `yolo11n.pt` with imgsz 640, epochs 50, batch 16, seed 42, deterministic,
  patience 10, project `runs`, name `finetune`; copy best checkpoint to
  `weights/best.pt`; then `src.evaluate`. Device, versions, settings and
  wall-clock go to `outputs/train_log.json` and are appended to NOTES.md by
  the run itself. The log is written before evaluation so a failure there
  cannot lose it.
- `src/data.py` gained `ensure_dataset()` (old `get_dataset` kept as alias) and
  `normalise_data_yaml()`: Roboflow writes `train: ../train/images`, which
  Ultralytics resolves against its own datasets directory, not the yaml, so
  the yaml now carries an absolute `path` and split folders relative to it.
- `src/evaluate.py`: detection metrics via `model.val` twice per subset, conf
  0.001 for mAP and conf 0.25 for the confusion matrix, because in
  ultralytics 8.4 the matrix uses the validator's own conf and a 0.001 matrix
  would flag everything. Image scores at conf 0.001 via `src.scoring` to
  `outputs/test_scores.csv` with filename, score, label, main-box class and a
  `dup_in_trainval` flag. TPR/FPR at 0.25 and 0.50 with FPR split by Box vs
  Package. Everything computed on all 120 test images and on the 114 without a
  twin in train/valid, taking the pair list from `outputs/dataset_stats.json`.
  Baselines: COCO yolo11n zero-shot (class histogram at conf 0.25, every image
  scored 0.0, no class mapping invented) and trivial all-pass / all-fail, to
  `outputs/baselines.json`.
- Not yet exercised end to end: a smoke test of `evaluate.py` with a
  random-weight 4-class checkpoint was prepared but not run. First real run
  is the Colab training; `train.py` keeps the weights even if evaluation
  fails.
- `notebooks/train_colab.ipynb` (outside the repo root so `main.py` ignores
  it): GPU check, clone the fork at `assignment-2`, pip install
  ultralytics==8.4.173 and roboflow, key from Colab secrets, `python main.py
  train`, zip `weights/best.pt` + `outputs/` and download.
- `.gitignore`: root-level `*.pt` ignored (the `yolo11n.pt` Ultralytics
  downloads); `weights/best.pt` stays committed.
- Smoke test of `src/evaluate.py` done with a random-weight 4-class checkpoint
  built in the scratchpad (class head biased toward Box_broken so positive
  scores appear). No training. Ran end to end on CPU in about 70 s: all-120
  and dedup-114 paths, both confusion matrices, `test_scores.csv` with the
  right columns and 6 `dup_in_trainval` rows, and both baselines.
  Two bugs fixed: (1) ultralytics 8.4 only fills the confusion matrix when
  `plots=True`, so the matrix pass now sets it (its PNGs go to `runs/`,
  git-ignored); (2) CSV writers used CRLF line endings, now LF. The scratch
  checkpoint and every file it produced were deleted; nothing from random
  weights is committed.

## Colab run 1 failure and fixes (2026-10-25)

- Bug: `train.py` hard-coded the run directory as `runs/finetune/`, but with
  `project="runs"` Ultralytics nested it under `runs/detect/runs/finetune/`,
  so after a complete 50-epoch Colab run the copy to `weights/best.pt` failed.
  Fix: take the location from the trainer's return value
  (`results.save_dir / "weights" / "best.pt"`), with `model.trainer.save_dir`
  as fallback; the run directory is now logged in `train_log.json`.
  `tests/test_train.py` fakes a nested save_dir and checks the copy.
- Ultralytics' AMP check on GPU downloads `yolo26n.pt` into `weights/`.
  `check.py` loads every `.pt` there, so `train.py` now deletes
  `weights/yolo26n.pt` (and `yolo11n.pt`, same mechanism in older versions)
  after training, logs what it removed, and both are git-ignored.
- Evaluation confidences, stated for the report: per-class P/R/AP and mAP
  come from a validation pass at conf 0.001 (standard for mAP); the confusion
  matrix comes from a second pass at conf 0.25; image scores keep every
  detection at conf 0.001. The two passes therefore report different numbers
  by design. `test_metrics_finetuned.json` records `map_conf`,
  `confusion_matrix_conf` and a `conf_note` at top level and per subset.

## Training run (2026-10-25)

- Dataset `roboflow-ngkro/package-detection-e1ssd/6`, base weights `yolo11n.pt`.
- Device: Tesla T4 (0). ultralytics 8.4.173, torch 2.11.0+cu130, python 3.13.16.
- Settings: imgsz=640, epochs=50, batch=16, seed=42, deterministic=True, patience=10, project=runs, name=finetune, exist_ok=True.
- Epochs run: 50 of 50 (patience 10). Best epoch: 50.
- Wall-clock: training 652s, evaluation 19s, total 681s.
- Run directory: `/content/daiist-assignment-2/runs/detect/runs/finetune`. Best checkpoint copied to `weights/best.pt` (5.5 MB).

## Colab run 2: results landed and reproduced (2026-10-25)

- The Colab artifacts were uploaded through the GitHub web UI onto the fork's
  `main` (two "Add files via upload" commits: `weights/best.pt`, then
  `outputs/`), not onto `assignment-2`. Brought onto `assignment-2` with
  `git checkout origin/main -- weights/best.pt outputs/`. `main` was left as is.
- `weights/best.pt` is 5,474,010 bytes; `outputs/` holds 12 files including
  `train_log.json`. Training: Tesla T4, 50 of 50 epochs in 652 s, best epoch
  50, so early stopping never triggered and the model was still improving at
  the last epoch. Worth stating in the report; a longer run is an option the
  owner decides. The nested `runs/detect/runs/finetune` path and the
  `yolo26n.pt` removal both behaved as fixed.
- Headline test numbers (GPU run, conf 0.001 for mAP): all 120 images mAP50
  0.9678, mAP50-95 0.8884; dedup 114 images mAP50 0.965, mAP50-95 0.8836.
  Image level: TPR 0.9516 / FPR 0.069 at 0.25, TPR 0.9355 / FPR 0.0517 at
  0.50 (all); TPR 0.9483 / FPR 0.0714 at 0.25, TPR 0.931 / FPR 0.0536 at 0.50
  (dedup).
- CPU reproduction of `src.evaluate` against the committed GPU outputs:
  `test_metrics_finetuned.json` identical except the `device` field;
  `baselines.json` identical; both confusion-matrix CSVs identical;
  `test_scores.csv` identical in labels and classes, 12 of 120 scores differ
  in the sixth decimal (max difference 0.000002), no pass/fail flip at 0.25
  or 0.50; `baseline_coco_detections.csv` has the same 91 detections with
  identical confidences, only row order within an image differs. The
  committed GPU files stay as the canonical outputs.
- `check.py`: key check and weights check pass (4 classes load). Step 3 fails
  because `app.py` and `app.ipynb` both still exist and `app.py` is the
  template; expected until the dashboard is built.

## Single-threshold cost sweep (2026-10-25)

- `scripts/sweep.py` reads `outputs/test_scores.csv` and `src/config.py`;
  thresholds 0.00 to 1.00 in 0.01 steps, flag if score >= t. Test TPR/FPR
  rescaled to 1,000 units: FN = 1000 x base_rate x (1 - TPR),
  FP = 1000 x (1 - base_rate) x FPR, cost = FN x 7000 + FP x 40 DOP.
  Outputs: `sweep.csv`, `sweep_summary.json`, `cost_curve.png`,
  `options_single_threshold.csv`.
- At base rate 1%: minimum 6,118 DOP per 1,000 (89.09 EUR), reached at every
  t from 0.20 to 0.34 (TPR 0.9516, FPR 0.0690; 0.48 missed and 68.3 false
  alarms per 1,000). Flat band within one test positive of the minimum
  (tolerance 1,129 DOP = 1000 x 0.01 x 1/62 x 7000): t = 0.01 to 0.85. The
  curve is flat because 62 test positives make each TPR step worth 1,129 DOP
  and 58 negatives make each FPR step worth 683 DOP, so only a handful of
  images separate neighbouring thresholds.
- Dedup check (114 images): same minimum threshold 0.20 and same band
  0.01 to 0.85; minimum 6,449 DOP, higher because the 6 removed test images
  included easy cases.
- Minimum-cost threshold moves with base rate: 0.65 at 0.5%, 0.20 at 1%,
  0.02 at 3% to 10%, 0.00 (flag everything) at 15%. Above about 3% the cost
  of a miss dominates so far that the model should flag almost anything.
- Break-even: the model beats "person checks everything" (40,000 DOP per
  1,000) at every t from 0.00 up to 0.94; from t = 0.95 it no longer does
  (42,903 DOP) because misses pile up.
- Five single-threshold options at base rate 1%, DOP per 1,000 units:
  do nothing 70,000 (1,019 EUR); person checks everything 40,000 (583 EUR);
  model alone at t = 0.20: 6,118 (89 EUR); COCO zero-shot 70,000, identical
  to do nothing because every image scores 0.0; flag everything 39,600
  (577 EUR). Note flag-everything is below person-checks-everything only
  because the formula excludes the handling cost of true positives; the
  review-band costing adds those.
- Worst test cases (`scripts/failures.py`, `outputs/failures/`):
  - `5ded88cs-960_jpg...`: GT Box_broken, score 0.0000; model saw Package 0.79.
    A tape-wrapped parcel with a customs form and a crumpled top; the damage
    is subtle and the parcel looks like the Package class.
  - `670055d7-233Open_boo_jpg...`: GT Box_broken, score 0.0015; model saw
    Box 0.72 and Box 0.50. A heavily taped box with a torn, dented end; the
    model read the whole box as intact Box.
  - `76c8973b-164Packageeees_jpg...`: GT Package x2, score 0.9338; model saw
    Open_package 0.93. The costliest kind of false alarm: a confident one.
  - `21926138-50Booooox_jpg...`: GT Box, score 0.8984; model saw Box_broken
    0.90 and Box 0.45.
  - `1packaaaageeeees_jpg...`: GT Package, score 0.6410; model saw
    Open_package 0.64 and Package 0.38.
  The two misses score essentially zero, so no threshold recovers them; the
  false alarms sit at 0.64 to 0.93, inside the flat band, which is why the
  band is wide rather than why it is cheap.

## Review band (two-threshold) sweep (2026-10-25)

- `scripts/review_band.py`: 0.02 grid over (t_low, t_high), outcome costs per
  unit: pass 0 if intact / 7000 if damaged; review 40 either way (person
  catches it); fail 10 either way (3-min confirmation, intact unit released).
  Rescaled to 1,000 units at base rate 1% as in sweep.py. Outputs:
  `review_grid.csv`, `review_grid.png`, `review_summary.json`,
  `review_load.csv`, `options.csv`.
- Result: the minimum is 3,379 DOP per 1,000 (49.21 EUR) at
  (t_low, t_high) = (0.02, 0.02), i.e. no review band. Ten pairs tie exactly,
  all with t_low in {0.02..0.08} and t_high up to 0.08, but no test image
  scores between 0.02 and 0.08, so those bands are empty. 133 pairs sit within
  one test positive (1,129 DOP) of the minimum: t_low 0.02 to 0.34, t_high
  0.02 to 0.38. The dedup subset picks the same pair (3,571 DOP).
- Why no band: with these figures "fail" strictly dominates "review". Both
  put a person on the unit and both are assumed to catch every damaged one,
  but fail costs 10 (a 3-minute glance at the photo) and review costs 40 (a
  10-minute unpack). So whenever the model is unsure, the cheapest thing is
  to fail the unit, never to review it. The review band can only earn its
  place if the owner decides a 3-minute glance does not catch everything, or
  that failing an intact unit has a cost beyond the glance (supplier
  relations, a wrong claim). Owner's call; config unchanged.
- At (0.02, 0.02): 60 of 62 test positives fail, 2 pass (the two near-zero
  misses), 6 of 58 negatives fail. Per 1,000 units: 0.32 missed, 112 failed,
  0 reviewed. The cheapest pair that actually uses a review band is
  (0.02, 0.10) at 3,384 DOP, 5 DOP above the minimum and well inside the flat
  band, sending 0.16 units per 1,000 to review (one test negative). Note this optimum (0.02) differs from sweep.py's 0.20
  because sweep.py charged a false alarm the full 40-DOP unpack and nothing
  for a true positive; here a fail costs 10 regardless.
- Review load at (0.02, 0.02), 2 staff, 7 h/day, 7-day claim window:
  20ft (200 units): 22.4 failed, 0 reviewed, 1.1 person-hours, 0.08 days.
  40ft_low (400): 44.8 failed, 2.2 person-hours, 0.16 days.
  40ft_high (600): 67.3 failed, 3.4 person-hours, 0.24 days. All fit the
  window with a wide margin; expected missed damaged units per container
  0.06 / 0.13 / 0.19.
- Options at base rate 1% (per 1,000 units; per 400-box container):
  do nothing 70,000 DOP (28,000); person checks everything 40,000 (16,000);
  model alone at t = 0.20: 4,165 (1,666); model at the chosen pair
  (0.02, 0.02): 3,379 (1,352). In EUR per container: 408 / 233 / 24 / 20.

## Review band, attempt 2: the false-claim cost (2026-10-25)

- Why attempt 1 collapsed: it priced a FAIL at 10 DOP whether the unit was
  damaged or intact, and assumed the 3-minute glance catches everything. That
  made FAIL a strictly cheaper version of REVIEW (40 DOP), so the band could
  never win and the optimum became "fail anything above 0.02". But a FAIL is
  a claim filed on the photo without unpacking; on an intact unit that is a
  false claim with admin and credibility costs. Attempt 1 is kept as
  `outputs/review_attempt1_*` and is reproducible with
  `scripts/review_band.py --false-claim 0`.
- New figures in `src/config.py`: `COST_FALSE_CLAIM_DOP = 500` (about 1 h of
  admin at ~RD$250/h to file and reverse a claim, plus credibility with the
  supplier/forwarder; owner to confirm) and `COST_FALSE_CLAIM_SWEEP =
  [100, 500, 2000]`. Outcome costs now: pass 0 intact / 7000 damaged; review
  40 either way; fail 10 damaged / 10 + false claim intact.
- Result, identical at all three false-claim costs: chosen
  (t_low, t_high) = (0.34, 0.94), 6,310 DOP per 1,000 (91.89 EUR). Eight
  pairs tie exactly (t_low 0.20 to 0.34, t_high 0.94); the tie-break takes
  the narrowest band. Near-optimal band within one test positive (1,129
  DOP): t_low 0.02 to 0.84, t_high 0.94 to 1.00 (138 pairs at 500 and 2000,
  146 at 100 where t_high 0.90 also qualifies). No single-threshold pair is
  inside the band at any false-claim cost; the best single threshold at 500
  is t = 0.90 at 17,825 DOP. Dedup check: same pair, 6,637 DOP.
- Why the false-claim cost does not move the optimum: at t_high = 0.94 no
  test negative fails (the highest negative score is 0.934), so the
  false-claim term is zero there. It does shape everything else: one step
  down to t_high = 0.92 lets that one negative fail and the cost jumps to
  7,447 / 14,274 / 39,878 DOP at 100 / 500 / 2000. So t_high is pinned by a
  single test negative; the safe reading is "t_high high enough that no
  intact unit is ever failed on the photo", which 0.94 to 1.00 all satisfy
  within 190 DOP of each other. At the chosen pair: 39 of 62 test positives
  fail, 20 review, 3 pass; 54 of 58 negatives pass, 4 review, 0 fail.
- Per 1,000 units at (0.34, 0.94): 0.48 missed, 71.5 reviewed, 6.3 failed,
  0 false claims. Model alone at sweep.py's t = 0.20 now costs 38,303 DOP
  (68 false claims per 1,000 at 500 each), barely better than checking
  everything; the band is what makes the model usable under this costing.
- Review load at (0.34, 0.94), 2 staff, 7 h/day, 7-day claim window:
  20ft (200): 14.3 review, 1.3 fail, 2.5 person-hours, 0.18 days;
  40ft_low (400): 28.6 review, 2.5 fail, 4.9 person-hours, 0.35 days;
  40ft_high (600): 42.9 review, 3.8 fail, 7.3 person-hours, 0.52 days.
  All fit the window; expected missed damaged units 0.10 / 0.19 / 0.29.
- Options at 500 (per 1,000; per 400-box container): do nothing 70,000
  (28,000); person checks everything 40,000 (16,000); model alone at 0.20:
  38,303 (15,321); model + review band (0.34, 0.94): 6,310 (2,524). EUR per
  container: 408 / 233 / 223 / 37.

## Chosen operating point (2026-10-25)

- `src/config.py`: `CHOSEN_T_LOW = 0.20`, `CHOSEN_T_HIGH = 0.94`. Reasoning,
  as recorded in the config comment: t_high sits just above the highest
  score of any intact test image (0.934), so no intact unit is failed on
  the photo. t_low: the symmetric rule, just below the lowest detected
  damaged score (0.088), gives 0.08, but that sends 10.5% of boxes to
  review to catch one extra damaged box in 62, which the test set cannot
  resolve; 0.20 is the low end of the exact-minimum plateau and halves the
  extra load. Both edges rest on one test image each; re-derive from dock
  data.
- `scripts/review_band.py` now reports the chosen pair beside the argmin
  (0.34, 0.94): same cost 6,310 DOP per 1,000 (no test image scores between
  0.20 and 0.34, so the two pairs are indistinguishable on this test set),
  same 2x3 outcome table (intact: 54 pass / 4 review / 0 fail; damaged: 3
  pass / 20 review / 39 fail), same load (7.2% to review, 0.6% failed).
  New file `outputs/outcome_table.csv`; `review_load.csv` and `options.csv`
  carry both pairs. The cost model moved into `src/costing.py` so the
  report scripts and the dashboard share one implementation; the fc500
  grid is byte-identical before and after the refactor.

## Dashboard (2026-10-25)

- `app.ipynb` deleted; `app.py` is the Gradio app (`uv run python main.py
  app`, port 7860). It loads `weights/best.pt` once at import and reads
  everything else from `outputs/` and `src/config.py`. No training, no
  downloads; boxes are drawn with OpenCV rather than Ultralytics' plotter,
  which would fetch a font file on first use.
- Tabs: Inspect (upload, boxes at a confidence slider, image score at conf
  0.001, pass/review/fail at the current thresholds, cost of that decision if
  intact vs damaged); Thresholds (t_low, t_high sliders with t_high clamped to
  t_low, base-rate and false-claim dropdowns, 2x3 outcome table per 1,000,
  cost in DOP and EUR, review and fail shares, load per container); Cost curve
  (sweep.csv curve at the selected base rate with both thresholds and the
  near-optimal band, the two-threshold heatmap, options.csv); Per class
  (per-class P/R/AP table, model comparison incl. COCO zero-shot and trivial,
  confusion matrix, image-level TPR/FPR at the current thresholds with FPR
  by Box vs Package, the five failure images).
- The app and the report scripts share `src/costing.py`; a test asserts the
  Thresholds tab at the defaults reproduces review_band.py's chosen-pair
  cost. `tests/test_app.py` also asserts importing the app creates nothing
  under `runs/` or `weights/`, loads the model once, and that the source has
  no training or download calls. `check.py` passes all three steps.
- Image-level at the chosen thresholds: fail only (score >= 0.94) TPR 0.629,
  FPR 0.000; review or fail (score >= 0.20) TPR 0.952, FPR 0.069 (Box 0.080,
  Package 0.061).

## Final checks (2026-10-25)

- `.github/workflows/ci.yml`: on every push and pull request, `uv sync`,
  `pytest tests/`, then `check.py` if the repository secret
  `ROBOFLOW_API_KEY` exists (check.py's first step verifies the key, so the
  secret is required for that step only; without it the step is skipped with
  a notice and the tests still run). A last step asserts no `runs/` folder
  exists, so CI can never have trained. The manual `verify.yml` is unchanged.
- `scripts/survey_datasets.py` deleted: nothing imports it and the candidate
  survey is finished; its findings stay in the dataset-survey notes above.
  README.md carries no reference to `data/candidates/` (it is the upstream
  brief and was never edited).
- Verified: `.env` never appears in git history on any branch; `.env` and
  `data/` are untracked and ignored; `weights/best.pt`, `outputs/` (36 files)
  and `notebooks/train_colab.ipynb` are tracked. The repo root has
  `api_key.py`, `app.py`, `check.py`, `main.py`, `train.py` and no notebook,
  so `main.py train` resolves to `train.py` and `main.py app` to `app.py`.
