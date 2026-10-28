"""Shared, explicit settings and public dataset preparation."""
import contextlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets' / 'car-damage-v1'
ARTIFACTS = ROOT / 'artifacts'
WORKSPACE = 'celebal-wlpui'
PROJECT = 'car-damage-zxcdc'
VERSION = 1
DATA_URL = f'https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}'
SEED = 19654
IMAGE_SIZE = 320
EPOCHS = 40
WEIGHTS = ROOT / 'weights' / 'best.pt'


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare_dataset():
    """Download the original published version; preserve its labels and splits."""
    from api_key import load_api_key, verify
    import yaml

    key = load_api_key()
    ok, _ = verify(key)
    if not ok:
        raise RuntimeError('Roboflow did not accept the key. Check the private .env file.')
    if not (DATA / 'data.yaml').exists():
        from roboflow import Roboflow
        print('Downloading the original public Roboflow dataset version 1 ...', flush=True)
        # The SDK can print signed download links; keep those out of public logs.
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rf = Roboflow(api_key=key)
                rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
                    'yolov8', location=str(DATA))
        except Exception as error:
            raise RuntimeError('Roboflow dataset download failed; check account access and connection.') from None
    raw = yaml.safe_load((DATA / 'data.yaml').read_text())
    names = raw['names']
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    if sorted(names) != ['Damage', 'Undamage']:
        raise ValueError(f'Unexpected classes: {names}')
    config = {'path': str(DATA), 'train': 'train/images', 'val': 'valid/images',
              'test': 'test/images', 'names': names}
    local_yaml = DATA / 'local.yaml'
    local_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return local_yaml, names


def configure_runtime():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs' / 'matplotlib'))
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs' / 'ultralytics-settings'))
    import torch
    torch.set_num_threads(min(6, os.cpu_count() or 2))
    ARTIFACTS.mkdir(exist_ok=True)


def image_files(split):
    return sorted(p for p in (DATA / split / 'images').iterdir()
                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})


def labels_for(image):
    label = image.parent.parent / 'labels' / (image.stem + '.txt')
    return [[float(x) for x in line.split()] for line in label.read_text().splitlines()
            if line.strip()] if label.exists() else []
