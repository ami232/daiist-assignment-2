# Training on Google Colab

Training needs a GPU; this machine has none. The pipeline is the same `main.py train`
that runs anywhere: these cells only set up the environment around it.

Before starting, push your latest code to your fork (Colab clones it from GitHub).

**Runtime > Change runtime type > T4 GPU**, then run each cell in order.

## 1. Mount Google Drive (runs survive a disconnect)

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Clone the fork

```python
!git clone https://github.com/Angelbenito1/daiist-assignment-2.git /content/a2
%cd /content/a2
```

## 3. Install dependencies

Colab ships its own CUDA build of PyTorch, so do not run `uv sync` here.

```python
!pip install -q "ultralytics>=8.4,<8.5" "roboflow>=1.7"
```

## 4. Roboflow API key

Add it once under the key icon in Colab's left sidebar (Secrets), named
`ROBOFLOW_API_KEY`, with notebook access on. Never paste it into a cell.

```python
import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
!python api_key.py
```

## 5. Train

Runs go to Drive. If Colab disconnects, re-run cells 1–5: each model resumes from its `last.pt`,
and a model that already finished is not retrained.

```python
os.environ["RUNS_DIR"] = "/content/drive/MyDrive/daiist-a2/runs"
!python main.py train
```

## 6. Save the results to Drive

```python
!mkdir -p /content/drive/MyDrive/daiist-a2/out
!cp -r weights artifacts /content/drive/MyDrive/daiist-a2/out/
```

Then download `daiist-a2/out/` (and `daiist-a2/runs/`, for the training curves) from Drive
into the repo on your computer.
