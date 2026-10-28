# Car Damage Inspection

Read **REPORT.md** for the business problem, dataset, model comparison, evaluation,
cost assumptions, human review policy, limitations and AI disclosure.

## Run the dashboard

From this folder, with `uv` installed:

```bash
uv sync
uv run python main.py app
```

Open http://127.0.0.1:7860. The dashboard loads saved results and weights; it does
not train or require a downloaded dataset or Roboflow key at startup. It supports
photo uploads, both decision thresholds, confusion matrices, cost curves,
per-class metrics and the comparison of all three trained models.

## Reproduce training and evaluation

Follow **SETUP.md** to put your own Roboflow key in a local `.env` file, then run:

```bash
uv run python main.py train
```

This downloads the original public dataset, trains YOLO11s for 20 epochs,
selects thresholds using validation, evaluates the fixed policy on test and
exports dashboard results. It replaces the active results with the new run.
Training time depends on the available hardware.

The submitted YOLO11s weights come from the completed 20-epoch Colab run, with
the checkpoint from epoch 19. A separate one-epoch run verified the integrated
workflow; it was not a fresh 20-epoch reproduction. Its verification summary is
in `results/pipeline_verification/summary.json`.

The professor's optional self-check is:

```bash
uv run python check.py
```

It requires a valid Roboflow key and does not retrain the model.

## Submission contents

- `REPORT.md`, `README.md` and `SETUP.md`: report, assignment and setup instructions.
- Root Python files and configuration: the training, evaluation and dashboard pipeline.
- `weights/`: the three models used in the dashboard comparison.
- `audit/`: dataset statistics and evidence of related images across splits.
- `results/`: training summaries, saved predictions, threshold analysis, final
  test results and the two missed-damage examples.
- The two Colab notebooks and `tools/run_validation_experiment.py`: records of
  the earlier Colab experiments, retained as supporting material. Their old
  transfer archives are not included; use `main.py train` for reproduction.

The API key, virtual environment, downloaded dataset, temporary files and
development-only outputs are excluded. The test has been used previously;
the report explains this limitation and the ideal human-review assumption.

## Submit

Commit and push this folder to your fork. Open a pull request to
https://github.com/ami232/daiist-assignment-2, then submit the pull-request link
and the ZIP backup on Blackboard, as required by README.md.
