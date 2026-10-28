# Project rules

Assignment 2: finetune a YOLO model, cost its mistakes, pick a threshold, decide
what goes to a person. README.md is the brief, SETUP.md the environment. These
rules apply to every change in this repo.

## Weights and startup

- Never retrain or download weights at app startup.
- `app.py` loads `weights/best.pt` and nothing else. Not `runs/`, not a hub
  download, not a fresh training call.

## Business numbers

- Every business figure lives in one file: `src/config.py`. That means costs
  per outcome, the base rate, container sizes, minutes per review, and any
  other number that feeds the cost model or the review-load estimate.
- Nothing else hard-codes such a figure. Training, sweep, app and notes all
  import from `src/config.py`.
- The owner supplies every figure. If one is missing, ask for it. Never guess,
  never fill in a placeholder.

## Outputs

- Every analysis result goes to `outputs/` as CSV or JSON plus a PNG.
- The report and the dashboard read those same files. No result is computed
  one way for the report and another way for the app.

## Determinism

- Fix `seed=42` everywhere: training, evaluation, sweeps, any sampling.
- If a step cannot be made deterministic, say so in NOTES.md.

## Git

- `data/` and `.env` are git-ignored and stay that way.
- `weights/best.pt` and `outputs/` are committed.

## Notes and reporting

- Keep a running `NOTES.md` of every decision, bug and number worth reporting,
  in bullet form, updated as work happens, not at the end.
- The owner writes REPORT.md. Do not write or edit it.
- Report back in short lists. No code dumps unless asked.
