"""
Assignment 2: pipeline de entrenamiento (detector de stock en expendedoras).

Se ejecuta con:  uv run python main.py train

Pasos, de principio a fin y sin intervención manual:
1. Descarga el dataset de Roboflow Universe (amits-pwfut / vending-niqv5 / v8).
2. Describe el dataset (instancias por clase, tamaño de los splits, tamaño de las cajas).
3. Hace fine-tuning de YOLO11 desde el checkpoint preentrenado en COCO.
4. Evalúa en test el modelo y los dos baselines (COCO zero-shot y "no predecir nada").
5. Barre el umbral de confianza, calcula el coste en € y elige los umbrales en VALID.
6. Guarda en outputs/ todo lo que necesita app.py y copia los pesos a weights/best.pt.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

# ----------------------------------------------------------------------------
# CONFIG: lo único que se cambia entre runs
# ----------------------------------------------------------------------------
RUN_NAME = "run2"
BASE_MODEL = "yolo11n.pt"
# Fase 1: fine-tuning desde COCO a 640 px (es el Run 1; si ya existe, se reutiliza)
STAGE1_EPOCHS, STAGE1_IMGSZ = 50, 640
# Fase 2: seguir entrenando los pesos de la fase 1 a más resolución (productos diminutos)
EPOCHS = 20          # 0 = no hay fase 2
IMGSZ = 800          # resolución final: la usan la fase 2, la evaluación y la app

# Dataset (Roboflow Universe)
WORKSPACE, PROJECT, VERSION = "amits-pwfut", "vending-niqv5", 8

# Costes (decididos en REPORT.md). Positivo = "hay un producto en el hueco".
COST_FP = 6.00       # producto fantasma: hueco vacío que no se repone -> ventas perdidas
COST_FN = 0.50       # producto no visto: el reponedor comprueba un hueco que estaba lleno
COST_REVIEW = 0.0125 # 3 s de una persona a 15 €/h para revisar una caja dudosa
BASE_RATE = 0.80     # % de huecos ocupados cuando se hace la foto

# Operación (para dimensionar la revisión humana)
SLOTS_PER_DAY = 200 * 40      # 200 máquinas x 40 huecos, 1 foto al día
SECONDS_PER_REVIEW = 3
REVIEW_HOURS_AVAILABLE = 4    # 1 persona a media jornada

IOU_MATCH = 0.5                                # una predicción acierta si IoU >= 0.5
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
# Clases COCO que cuentan como "producto" en el baseline zero-shot
COCO_PRODUCT_CLASSES = [39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]  # bottle, wine glass, cup, comida

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "vending"
OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------------------
# 1. Descarga
# ----------------------------------------------------------------------------
def download_dataset():
    if not (DATA_DIR / "test").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR), overwrite=True
        )
    # Escribimos nuestro propio data.yaml con rutas absolutas (el de Roboflow usa rutas relativas)
    import yaml
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    data = {"path": str(DATA_DIR), "train": "train/images", "val": "valid/images",
            "test": "test/images", "names": names}
    yaml_path = DATA_DIR / "data_abs.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    return yaml_path, names


# ----------------------------------------------------------------------------
# 2. Descripción del dataset
# ----------------------------------------------------------------------------
def read_labels(split):
    """Devuelve {nombre_imagen: array de cajas [clase, xc, yc, w, h]} (coordenadas 0-1)."""
    labels = {}
    for img in sorted((DATA_DIR / split / "images").iterdir()):
        txt = DATA_DIR / split / "labels" / (img.stem + ".txt")
        rows = np.loadtxt(txt, ndmin=2) if txt.exists() and txt.stat().st_size else np.zeros((0, 5))
        labels[img.name] = rows[:, :5]
    return labels


def describe_dataset(names):
    stats = {}
    for split in ["train", "valid", "test"]:
        boxes = np.concatenate(list(read_labels(split).values()))
        counts = {names[i]: int((boxes[:, 0] == i).sum()) for i in range(len(names))}
        area = boxes[:, 3] * boxes[:, 4]  # área de la caja relativa a la imagen
        stats[split] = {
            "images": len(read_labels(split)),
            "instances": int(len(boxes)),
            "instances_per_class": counts,
            "median_box_area_pct": round(float(np.median(area)) * 100, 2),
            "instances_per_image": round(len(boxes) / len(read_labels(split)), 1),
        }
    print(json.dumps(stats, indent=2))
    return stats


# ----------------------------------------------------------------------------
# 4. Predicciones + matching (para baselines y para el barrido de coste)
# ----------------------------------------------------------------------------
def iou(box, boxes):
    """IoU entre una caja y varias, todas en formato xyxy."""
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def predict_and_match(model, split, keep_classes=None):
    """Predice en un split y marca cada caja como acierto (TP) o error (FP).

    Es class-agnostic: solo importa si hay un producto, no cuál. Esa es la
    decisión de negocio (hueco ocupado o vacío). Las métricas por clase salen
    de model.val() más abajo.
    """
    labels = read_labels(split)
    rows, n_gt = [], 0
    for name, gt in labels.items():
        img_path = DATA_DIR / split / "images" / name
        r = model.predict(img_path, conf=0.01, iou=0.6, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        h, w = r.orig_shape
        gt_xyxy = np.column_stack([(gt[:, 1] - gt[:, 3] / 2) * w, (gt[:, 2] - gt[:, 4] / 2) * h,
                                   (gt[:, 1] + gt[:, 3] / 2) * w, (gt[:, 2] + gt[:, 4] / 2) * h])
        n_gt += len(gt)
        pred = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy()
        if keep_classes is not None:
            mask = np.isin(cls, keep_classes); pred, conf, cls = pred[mask], conf[mask], cls[mask]
        used = np.zeros(len(gt), bool)
        for i in np.argsort(-conf):  # de más a menos confianza: cada producto real solo cuenta una vez
            match = False
            if len(gt):
                ious = iou(pred[i], gt_xyxy); ious[used] = 0
                j = int(np.argmax(ious))
                if ious[j] >= IOU_MATCH:
                    used[j] = True; match = True
            rows.append({"image": name, "conf": float(conf[i]), "cls": int(cls[i]), "tp": match})
    return pd.DataFrame(rows, columns=["image", "conf", "cls", "tp"]), n_gt


def confusion_at(model, split, conf, names):
    """Matriz de confusión por clase al umbral elegido. Filas = clase real, columnas = predicha.

    La última fila y columna es "background": FP (caja donde no hay producto)
    y FN (producto que no se detecta).
    """
    n = len(names)
    cm = np.zeros((n + 1, n + 1), int)
    for name, gt in read_labels(split).items():
        r = model.predict(DATA_DIR / split / "images" / name, conf=conf, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        h, w = r.orig_shape
        gt_xyxy = np.column_stack([(gt[:, 1] - gt[:, 3] / 2) * w, (gt[:, 2] - gt[:, 4] / 2) * h,
                                   (gt[:, 1] + gt[:, 3] / 2) * w, (gt[:, 2] + gt[:, 4] / 2) * h])
        used = np.zeros(len(gt), bool)
        for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            ious = iou(box, gt_xyxy) if len(gt) else np.zeros(0)
            if len(gt):
                ious[used] = 0
            if len(gt) and ious.max() >= IOU_MATCH:
                j = int(np.argmax(ious)); used[j] = True
                cm[int(gt[j, 0]), c] += 1   # producto encontrado (bien o mal clasificado)
            else:
                cm[n, c] += 1               # FP: caja sin producto real
        for j in np.where(~used)[0]:
            cm[int(gt[j, 0]), n] += 1       # FN: producto no detectado
    return cm


def plot_confusion(cm, names, conf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = names + ["background"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión en test (conf ≥ {conf})")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def agnostic_ap50(preds, n_gt):
    """AP50 class-agnostic: área bajo la curva precisión-recall."""
    if len(preds) == 0 or n_gt == 0:
        return 0.0
    tp = preds.sort_values("conf", ascending=False)["tp"].to_numpy()
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0], recall])) * precision))


# ----------------------------------------------------------------------------
# 5. Coste
# ----------------------------------------------------------------------------
def cost_two_thresholds(preds, n_gt, t_low, t_high):
    """Coste con tres resultados: conf >= t_high acepta, conf < t_low descarta y lo de en medio va a una persona.

    La persona acierta siempre: se queda los TP y quita los FP. Los huecos
    totales salen de la base rate: huecos = productos reales / BASE_RATE.
    """
    slots = n_gt / BASE_RATE
    auto = preds[preds.conf >= t_high]
    band = preds[(preds.conf >= t_low) & (preds.conf < t_high)]
    tp = int(auto.tp.sum() + band.tp.sum())
    fp = int((~auto.tp).sum())
    fn = n_gt - tp
    reviewed = len(band)
    cost = fp * COST_FP + fn * COST_FN + reviewed * COST_REVIEW
    return {"t_low": t_low, "t_high": t_high, "tp": tp, "fp": fp, "fn": fn,
            "tn": max(int(slots - n_gt - fp), 0), "reviewed": reviewed,
            "review_share": reviewed / slots, "cost_per_1000": 1000 * cost / slots}


def sweep(preds, n_gt):
    """Un solo umbral (el modelo decide solo, sin revisión): t_low = t_high = t."""
    return pd.DataFrame([cost_two_thresholds(preds, n_gt, t, t) for t in THRESHOLDS])


def best_band(preds, n_gt):
    """Busca el par (t_low, t_high) más barato cuya revisión cabe en las horas disponibles."""
    best = None
    for lo in THRESHOLDS:
        for hi in THRESHOLDS[THRESHOLDS >= lo]:
            r = cost_two_thresholds(preds, n_gt, lo, hi)
            hours = r["review_share"] * SLOTS_PER_DAY * SECONDS_PER_REVIEW / 3600
            if hours <= REVIEW_HOURS_AVAILABLE and (best is None or r["cost_per_1000"] < best["cost_per_1000"]):
                best = {**r, "review_hours_per_day": hours}
    return best


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def main():
    global COST_FP  # se cambia temporalmente en el análisis de sensibilidad
    OUT.mkdir(exist_ok=True); WEIGHTS.mkdir(exist_ok=True)
    data_yaml, names = download_dataset()
    stats = describe_dataset(names)

    # 3. Fine-tuning en dos fases
    best_pt = ROOT / "runs" / "run1" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"\nFase 1: {BASE_MODEL} en {DEVICE} ({STAGE1_EPOCHS} épocas, imgsz {STAGE1_IMGSZ})")
        YOLO(BASE_MODEL).train(data=str(data_yaml), epochs=STAGE1_EPOCHS, imgsz=STAGE1_IMGSZ, device=DEVICE,
                               seed=0, project=str(ROOT / "runs"), name="run1", exist_ok=True, plots=True)
    if EPOCHS > 0:
        print(f"\nFase 2: pesos de la fase 1 en {DEVICE} ({EPOCHS} épocas, imgsz {IMGSZ})")
        YOLO(best_pt).train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, device=DEVICE, seed=0,
                            project=str(ROOT / "runs"), name=RUN_NAME, exist_ok=True, plots=True)
        best_pt = ROOT / "runs" / RUN_NAME / "weights" / "best.pt"
    model = YOLO(best_pt)

    # 4a. Métricas por clase en test (las calcula Ultralytics)
    m = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, device=DEVICE,
                  project=str(ROOT / "runs"), name=f"{RUN_NAME}_test", exist_ok=True)
    per_class = [{"class": names[c], "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                  "ap50": float(m.box.ap50[i]), "instances": stats["test"]["instances_per_class"][names[c]]}
                 for i, c in enumerate(m.box.ap_class_index)]

    # 4b. Predicciones class-agnostic: modelo fine-tuneado y COCO zero-shot
    val_preds, val_gt = predict_and_match(model, "valid")
    test_preds, test_gt = predict_and_match(model, "test")
    coco_preds, _ = predict_and_match(YOLO("yolo11n.pt"), "test", keep_classes=COCO_PRODUCT_CLASSES)
    val_preds.to_csv(OUT / "val_preds.csv", index=False)
    test_preds.to_csv(OUT / "test_preds.csv", index=False)

    # 5. Barrido de umbral: se ELIGE en valid y se REPORTA en test
    val_sweep, test_sweep = sweep(val_preds, val_gt), sweep(test_preds, test_gt)
    t_star = float(val_sweep.loc[val_sweep.cost_per_1000.idxmin(), "t_low"])
    pd.concat([val_sweep.assign(split="valid"), test_sweep.assign(split="test")]).to_csv(OUT / "sweep.csv", index=False)
    band = best_band(val_preds, val_gt)
    cm = confusion_at(model, "test", t_star, names)
    plot_confusion(cm, names, t_star, OUT / "confusion_matrix.png")

    # 6. Comparación con baselines y alternativas (todo en test)
    model_alone = cost_two_thresholds(test_preds, test_gt, t_star, t_star)
    coco_sweep = sweep(coco_preds, test_gt)
    t_coco = float(coco_sweep.loc[coco_sweep.cost_per_1000.idxmin(), "t_low"])
    empty_slots = test_gt / BASE_RATE - test_gt
    test_band = cost_two_thresholds(test_preds, test_gt, band["t_low"], band["t_high"])
    test_band["review_hours_per_day"] = test_band["review_share"] * SLOTS_PER_DAY * SECONDS_PER_REVIEW / 3600
    alternatives = {
        # Sin sistema: ningún hueco vacío se detecta, así que todos se quedan sin reponer (como un FP)
        "do_nothing": 1000 * empty_slots * COST_FP / (test_gt / BASE_RATE),
        # Una persona mira todos los huecos y acierta siempre
        "human_checks_all": 1000 * COST_REVIEW,
        "model_alone": model_alone["cost_per_1000"],
        "model_plus_review": test_band["cost_per_1000"],
    }
    baselines = {
        "finetuned": {"ap50_agnostic": agnostic_ap50(test_preds, test_gt), "map50": float(m.box.map50),
                      "map50_95": float(m.box.map), **model_alone},
        "coco_zero_shot": {"ap50_agnostic": agnostic_ap50(coco_preds, test_gt),
                           **cost_two_thresholds(coco_preds, test_gt, t_coco, t_coco)},
        "predict_nothing": {"ap50_agnostic": 0.0, **cost_two_thresholds(test_preds.iloc[0:0], test_gt, 0.5, 0.5)},
    }
    # Sensibilidad: ¿cambia el umbral óptimo si el coste del FP es otro?
    # (la base rate solo cambia la escala del coste, no dónde está el mínimo)
    sensitivity, original = {}, COST_FP
    for c in [1, 2, 3, 6, 10]:
        COST_FP = c
        s = sweep(val_preds, val_gt)
        sensitivity[str(c)] = float(s.loc[s.cost_per_1000.idxmin(), "t_low"])
    COST_FP = original

    results = {
        "run": RUN_NAME, "base_model": BASE_MODEL, "epochs": f"{STAGE1_EPOCHS}@{STAGE1_IMGSZ}+{EPOCHS}@{IMGSZ}", "imgsz": IMGSZ, "device": DEVICE,
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "stats": stats},
        "names": names, "per_class": per_class, "confusion_matrix": cm.tolist(),
        "baselines": baselines, "chosen_threshold": t_star, "chosen_band": band,
        "test_band": test_band, "alternatives_cost_per_1000": alternatives,
        "threshold_sensitivity_to_cost_fp": sensitivity,
        "costs": {"fp": COST_FP, "fn": COST_FN, "review": COST_REVIEW, "base_rate": BASE_RATE,
                  "slots_per_day": SLOTS_PER_DAY, "seconds_per_review": SECONDS_PER_REVIEW,
                  "review_hours_available": REVIEW_HOURS_AVAILABLE},
        "test_gt": test_gt, "val_gt": val_gt,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    history = OUT / "history"; history.mkdir(exist_ok=True)  # copia de cada run para compararlos
    shutil.copy(OUT / "metrics.json", history / f"{RUN_NAME}_metrics.json")
    shutil.copy(OUT / "confusion_matrix.png", history / f"{RUN_NAME}_confusion_matrix.png")

    # Registro de runs para comparar run1, run2 y run3
    log_path = OUT / "runs_log.csv"
    row = pd.DataFrame([{"run": RUN_NAME, "model": BASE_MODEL, "epochs": f"{STAGE1_EPOCHS}@{STAGE1_IMGSZ}+{EPOCHS}@{IMGSZ}", "imgsz": IMGSZ,
                         "map50": m.box.map50, "map50_95": m.box.map, "t_star": t_star,
                         "test_cost_per_1000": model_alone["cost_per_1000"],
                         "band_cost_per_1000": test_band["cost_per_1000"]}])
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    log = pd.concat([log[log.run != RUN_NAME] if len(log) else log, row])
    log.to_csv(log_path, index=False)

    # Pesos que usa la app y un par de imágenes de test de ejemplo
    shutil.copy(best_pt, WEIGHTS / "best.pt")
    examples = OUT / "examples"; examples.mkdir(exist_ok=True)
    for img in sorted((DATA_DIR / "test" / "images").iterdir())[:3]:
        shutil.copy(img, examples / img.name)

    print(json.dumps({"map50": m.box.map50, "t_star": t_star, "band": band,
                      "alternatives": alternatives}, indent=2, default=float))


if __name__ == "__main__":
    main()
