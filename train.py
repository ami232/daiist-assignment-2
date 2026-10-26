"""Assignment 2: training pipeline. Se ejecuta con: uv run python main.py train

Descarga la versión citada en REPORT.md, construye un test sin solape con
train, entrena yolo11n, evalúa contra los dos baselines y guarda todo lo que
necesita app.py (pesos, métricas, puntuaciones por imagen).
"""
import csv
import json
import shutil
from pathlib import Path

import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

# --- Dataset (debe coincidir con REPORT.md) ---
WORKSPACE = "pcb-defect-detection-emmts"
PROJECT = "pcb-defect-detection-datasets-whwcy"
VERSION = 5

# --- Ajustes de entrenamiento (anótalos en REPORT.md) ---
MODEL = "yolo11n.pt"
EPOCHS = 8
IMGSZ = 640
BATCH = 16
FRACTION = 1.0  # baja a 0.3 para una prueba rápida
SEED = 0
IMAGE_THR = 0.25  # umbral solo para las métricas de resumen a nivel de placa

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pcb_v5"
OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights"


def download():
    if (DATA / "data.yaml").exists():
        return
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov8", location=str(DATA), overwrite=False
    )


def base(name):
    """Identidad de la placa: quita sufijos de Roboflow y '_rotated'."""
    return name.split(".rf.")[0].replace("_rotated", "")


def build_clean_test():
    """Test sin las imágenes cuya placa base aparece en train."""
    train_bases = {base(p.name) for p in (DATA / "train" / "images").iterdir()}
    dst = DATA / "test_clean"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    total = kept = 0
    for img in sorted((DATA / "test" / "images").iterdir()):
        total += 1
        if base(img.name) in train_bases:
            continue
        shutil.copy2(img, dst / "images" / img.name)
        lab = DATA / "test" / "labels" / (img.stem + ".txt")
        if lab.exists():
            shutil.copy2(lab, dst / "labels" / lab.name)
        kept += 1
    return total, kept


def class_names():
    names = yaml.safe_load((DATA / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def write_eval_yaml(names):
    cfg = {
        "path": str(DATA.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test_clean/images",
        "names": names,
    }
    path = DATA / "eval.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def gt_count(label_path):
    if not label_path.exists():
        return 0
    return sum(1 for ln in label_path.read_text().splitlines() if ln.strip())


def instances_per_class(names):
    counts = {n: 0 for n in names}
    for f in (DATA / "test_clean" / "labels").glob("*.txt"):
        for ln in f.read_text().splitlines():
            if ln.strip():
                counts[names[int(ln.split()[0])]] += 1
    return counts


def image_scores(weights, tag, chunk=64):
    """Puntuación por placa = confianza máxima de cualquier detección.
    Positivo (real) = la placa tiene al menos un defecto etiquetado.
    Se procesa en trozos para no agotar la RAM."""
    import gc
    model = YOLO(weights)
    imgs = sorted((DATA / "test_clean" / "images").iterdir())
    rows = []
    for i in range(0, len(imgs), chunk):
        batch = [str(p) for p in imgs[i:i + chunk]]
        results = model.predict(batch, conf=0.001, imgsz=IMGSZ, max_det=50,
                                verbose=False, save=False)
        for res in results:
            name = Path(res.path).name
            n_gt = gt_count(DATA / "test_clean" / "labels" / (Path(name).stem + ".txt"))
            score = float(res.boxes.conf.max()) if len(res.boxes) else 0.0
            rows.append((name, int(n_gt > 0), n_gt, round(score, 5)))
        del results
        gc.collect()
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"scores_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "has_defect", "n_gt", "score"])
        w.writerows(rows)
    return rows


def image_level(rows, thr):
    tp = sum(1 for _, y, _, s in rows if y and s >= thr)
    fp = sum(1 for _, y, _, s in rows if not y and s >= thr)
    fn = sum(1 for _, y, _, s in rows if y and s < thr)
    tn = sum(1 for _, y, _, s in rows if not y and s < thr)
    n = len(rows)
    return {
        "threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def per_class_metrics(val, names, inst):
    box = val.box
    out = {}
    for i, c in enumerate(box.ap_class_index):
        n = names[int(c)]
        out[n] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
            "instances_test_clean": inst[n],
        }
    return out


def main():
    download()
    total, kept = build_clean_test()
    names = class_names()
    eval_yaml = write_eval_yaml(names)
    print(f"Test limpio: {kept} de {total} imágenes")

    # 1) Finetune
    model = YOLO(MODEL)
    model.train(data=str(eval_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                fraction=FRACTION, seed=SEED, project=str(ROOT / "runs"),
                name="pcb", exist_ok=True, plots=True)
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy2(model.trainer.best, WEIGHTS / "best.pt")

    # 2) Evaluación en el test limpio (por detección y por clase)
    OUT.mkdir(exist_ok=True)
    val = YOLO(str(WEIGHTS / "best.pt")).val(
        data=str(eval_yaml), split="test", imgsz=IMGSZ, conf=0.001, plots=True,
        project=str(OUT), name="val_finetuned", exist_ok=True)
    inst = instances_per_class(names)
    per_class = per_class_metrics(val, names, inst)
    (OUT / "per_class.json").write_text(json.dumps(per_class, indent=2))
    (OUT / "confusion_matrix.json").write_text(json.dumps(
        {"labels": names + ["background"],
         "matrix": val.confusion_matrix.matrix.tolist()}))

    # 3) Puntuaciones por placa: modelo propio y COCO zero-shot
    ft_rows = image_scores(str(WEIGHTS / "best.pt"), "finetuned")
    coco_rows = image_scores("yolo11n.pt", "coco_zeroshot")

    # 4) Baselines triviales a nivel de placa
    n = len(ft_rows)
    n_def = sum(y for _, y, _, _ in ft_rows)
    trivial = {
        "predict_nothing": {"accuracy": (n - n_def) / n, "recall": 0.0},
        "predict_all_defective": {"accuracy": n_def / n, "precision": n_def / n, "recall": 1.0},
    }

    metrics = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "test_total": total, "test_clean": kept,
                    "clean_images_with_defect": n_def,
                    "base_rate_in_clean_test": n_def / n},
        "settings": {"model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ,
                     "batch": BATCH, "fraction": FRACTION, "seed": SEED},
        "detector": {"map50": float(val.box.map50), "map50_95": float(val.box.map)},
        "image_level": {
            "finetuned": image_level(ft_rows, IMAGE_THR),
            "coco_zeroshot": image_level(coco_rows, IMAGE_THR),
            "trivial": trivial,
        },
        "notes": "COCO no tiene clases de defectos de placa: solo se evalúa a nivel de placa.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Listo. Pesos en weights/best.pt, resultados en outputs/")


if __name__ == "__main__":
    main()