# Setup

Environment, credentials, and how to check your work. The assignment itself is
in README.md.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS
```
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

## Fork and clone

Fork this repo (https://github.com/ami232/daiist-assignment-2), then clone
**your fork**:

```bash
git clone https://github.com/<your-github-username>/<your-fork-name>.git
cd <your-fork-name>
uv sync
```

## Roboflow API key

Make a free Roboflow account and copy your key from the dashboard, under
Account > Roboflow Keys.

Put it in a `.env` file rather than typing it into your terminal each session:

```bash
cp .env.example .env
```

Paste your key into `.env`, then check it works:

```bash
uv run python api_key.py
```

That reports where it found your key and whether Roboflow accepts it. Do this
before writing any training code. An unusable key produces a confusing error
much later.

In your own code, read the key through `api_key.py` so it works from either
source.

### Keeping your key out of the repo

`.env` is in `.gitignore` and must stay there. A key pushed to a public fork is
a key you have to revoke. `.env.example` is the file that gets committed, and it
never holds a real key.

A `ROBOFLOW_API_KEY` environment variable also works and takes precedence over
`.env`. That is how the key reaches GitHub, which has no `.env` file.

## Training the real model

Where you train the model you actually submit is up to you: your own machine, a
GPU you have access to, or a hosted notebook. Commit the resulting weights to
`weights/`.

The uv environment in this repo pins a CPU build of PyTorch. It exists so
`main.py` runs on your laptop, not to train your real model. If you want to
train locally on a GPU, install the matching PyTorch build yourself.

### Google Colab, if you do not have a GPU

Colab gives you a free GPU: Runtime > Change runtime type > T4. Colab brings its
own GPU build of PyTorch, so do not run `uv sync` there. Install what you need
with pip. This is a convenience, not a requirement.

## Adding dependencies

The starting dependencies cover the assignment. If you want something else:

```bash
uv add <package-name>
```

Commit both `pyproject.toml` and `uv.lock` so `uv sync` reproduces your
environment for anyone else.

## Checking your submission

`check.py` checks what has to be true of the repo you hand in:

1. Your Roboflow API key is findable and Roboflow accepts it.
2. `weights/` holds a model that actually loads.
3. Your app launches and responds, serving those weights.

It does not retrain anything, because a real finetune does not fit in a check.

If you also want to confirm your training stage runs end to end, ask for it
explicitly, on a machine where training is practical:

```bash
uv run python check.py --train
```

That runs `main.py train` first, with no time limit.

### Checking on your computer

```bash
uv run python check.py
```

It reads your key from `.env`, so there is nothing else to set. Set `APP_PORT`
if your app uses a port other than Gradio's default. Run this before you submit.
A non-zero exit is something a grader will hit too.

### Checking on GitHub

The same check exists as a workflow. It is manual on purpose and never runs by
itself:

1. Push your work to your fork.
2. Open the **Actions** tab on your fork. The first time, click through the "I
   understand my workflows" banner, because GitHub disables workflows on forks
   by default.
3. Add your key as a repository secret named exactly `ROBOFLOW_API_KEY`, under
   Settings > Secrets and variables > Actions.
4. Select **Verify Assignment 2**, then **Run workflow**.

It has to be manual because GitHub refuses to give secrets to a workflow
triggered by a pull request from a fork. An automatic check on your PR would
have no key and could verify nothing. Your own fork is the only place your key
exists.

This route is optional and proves nothing extra. It runs
`uv run python check.py`, the same thing you can run locally. Use it if you want
a clean-machine check, or if something works for you and you suspect your
environment is why.

Neither route is a grading mechanism. Both are sanity checks that your
submission runs.
