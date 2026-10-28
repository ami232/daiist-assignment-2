# Run and review the project

This is a local project for Assignment 2. The report discloses AI contributions
to its analysis and recommendation. Nothing in this folder is submitted to
Blackboard automatically.

## What each file does

- `REPORT.md`: business framing, dataset, results and decisions.
- `train.py`: downloads the original dataset, audits it, tries YOLO11n at 320
  and 480 pixels, selects using validation cost, and evaluates the selected model.
- `app.py`: opens the dashboard using saved weights and evaluation outputs.
  It does not train or download a dataset at startup.
- `weights/best.pt`: the selected, actually finetuned model.
- `artifacts/`: actual metrics, prediction caches, experiment comparison and plots.
- `policy.py`: turns observed error rates and explicit business assumptions into
  expected costs and review workload.
- `RESEARCH_NOTES.md`: published benchmarks, scenario assumptions and development history.
- `.env`: private Roboflow key. It is ignored by Git and must be excluded from the backup.

## Dashboard

When it is running locally, open <http://127.0.0.1:7861/> for the preview started
in this chat. That link works on this computer while the app is running.

1. **Inspect a photo**: upload an image and click Inspect photo. Two confidence
   thresholds assign pass, human photo review or fail / hold for closer inspection.
2. **Cost and human review**: change thresholds or business assumptions. The
   test confusion matrices, expected cost, review load and comparison update.
3. **Model results and dataset**: see per-class metrics, the baselines and
   examples of the original data.

The dashboard distinguishes real test counts from expected production counts.
Human accuracy is a scenario assumption. A photo's damage score is not a
calibrated probability and does not prove that damage is new.

## Reproduce from a fresh clone

Install uv, open a terminal in the project folder, then run:

```bash
uv sync --locked
```

For dataset access and the professor's checker, put your own Roboflow private
key in `.env`, using `.env.example` as the format. Keep `.env` local.

To run the saved dashboard without training:

```bash
uv run python main.py app
```

The default port is 7860; set `APP_PORT=7861` to use the preview port instead.
The committed saved model and artifacts are sufficient for the app.

To reproduce both training experiments and the evaluation:

```bash
uv run python main.py train
```

This really trains two models and takes longer than starting the dashboard.
It retains the original public Roboflow split and labels. The optional
`TRAIN_REUSE_320=1` continuation switch only reuses a completed local first
experiment when that checkpoint exists. A normal fresh run trains both.

To check the submission without retraining:

```bash
uv run python check.py
uv run python -m pytest -q
```

## Submission after review

1. Review REPORT.md, the dashboard and the saved results.
2. Push the finished branch to Javier's fork.
3. Create a pull request to `ami232/daiist-assignment-2`, comparing the finished
   branch in Javier's fork against the original repository's main branch.
4. Submit that pull-request URL on Blackboard.
5. Upload the backup ZIP on Blackboard too. The ZIP should contain code,
   report, weights and artifacts, but no `.env`, virtual environment, dataset
   downloads, Git history or temporary training runs.

The professor's original instructions remain in README.md and SETUP.md.
