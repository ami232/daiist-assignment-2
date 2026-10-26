"""Dashboard Gradio. Se ejecuta con: uv run python main.py app

No entrena nada: carga los pesos de weights/ y los resultados de outputs/
que dejó train.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
from ultralytics import YOLO

import costing as C

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

model = YOLO(str(ROOT / "weights" / "best.pt"))
scores = C.load_scores()
metrics = json.loads((OUT / "metrics.json").read_text())
per_class = json.loads((OUT / "per_class.json").read_text())
cm = json.loads((OUT / "confusion_matrix.json").read_text())

T_BEST, _ = C.best_single(scores)
LO_BEST, HI_BEST, _ = C.best_pair(scores)
N_DEF = int((scores.has_defect == 1).sum())
N_GOOD = int((scores.has_defect == 0).sum())


def cost_figure(t_low, t_high):
    ts, costs = C.sweep(scores)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ts, costs, label="Coste (un solo umbral)")
    ax.axvline(T_BEST, color="green", ls="--", label=f"Umbral elegido {T_BEST:.2f}")
    ax.axvspan(t_low, t_high, color="orange", alpha=0.25, label="Zona de revisión actual")
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylabel("EUR por 1000 placas")
    ax.legend()
    fig.tight_layout()
    return fig


def update(t_low, t_high):
    t_high = max(t_high, t_low)
    cost = C.cost_per_1000(scores, t_low, t_high)
    share, per_day, people = C.review_load(scores, t_low, t_high)
    banner = "**COSTES PROVISIONALES: no son cifras definitivas.**\n\n" if C.PROVISIONAL else ""
    text = (
        f"{banner}"
        f"- Coste esperado: **{cost:,.0f} EUR por 1000 placas**\n"
        f"- Enviadas a revisión: **{100 * share:.1f} %** "
        f"(≈ {per_day:,.0f} placas/día, ≈ {people:.1f} personas)\n"
        f"- Tasa base asumida: {100 * C.BASE_RATE:.1f} % de placas defectuosas. "
        f"Estimado con {N_DEF} placas defectuosas y solo **{N_GOOD} buenas** del test limpio: "
        f"las tasas de falsos positivos son poco fiables."
    )
    return C.confusion_per_1000(scores, t_low, t_high), text, cost_figure(t_low, t_high)


def predict(image, t_low, t_high):
    if image is None:
        return None, "Sube una imagen."
    t_high = max(t_high, t_low)
    raw = model.predict(image, conf=0.001, max_det=50, verbose=False)[0]
    score = float(raw.boxes.conf.max()) if len(raw.boxes) else 0.0
    shown = model.predict(image, conf=max(t_low, 0.001), max_det=50, verbose=False)[0]
    annotated = shown.plot()[..., ::-1]
    if score < t_low:
        verdict = "PASA"
    elif score < t_high:
        verdict = "REVISIÓN HUMANA"
    else:
        verdict = "FALLA"
    return annotated, f"Puntuación de la placa: {score:.2f} → **{verdict}**"


def per_class_table():
    rows = []
    for name, m in per_class.items():
        n = m["instances_test_clean"]
        rows.append({
            "Clase": name, "Instancias (test limpio)": n,
            "Precisión": round(m["precision"], 3), "Recall": round(m["recall"], 3),
            "AP50": round(m["ap50"], 3), "mAP50-95": round(m["ap50_95"], 3),
            "Fiable": "sí" if n >= 100 else "NO (<100 casos)",
        })
    return pd.DataFrame(rows)


def baseline_table():
    il = metrics["image_level"]
    thr = il["finetuned"]["threshold"]
    f = lambda d, k: round(d[k], 3) if k in d else "—"
    rows = [
        ("Modelo finetuneado", il["finetuned"]),
        ("COCO zero-shot", il["coco_zeroshot"]),
        ("Trivial: no detectar nada", il["trivial"]["predict_nothing"]),
        ("Trivial: todo defectuoso", il["trivial"]["predict_all_defective"]),
    ]
    return pd.DataFrame([
        {"Modelo": n, f"Accuracy (umbral {thr})": f(d, "accuracy"),
         "Precisión": f(d, "precision"), "Recall": f(d, "recall")}
        for n, d in rows
    ])


def confusion_figure():
    labels, mat = cm["labels"], np.array(cm["matrix"])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Clase real")
    ax.set_ylabel("Clase predicha")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    return fig


with gr.Blocks(title="Inspección de placas electrónicas") as demo:
    gr.Markdown("# Inspección de placas: pasa / revisión humana / falla")
    t_low = gr.Slider(0, 1, value=LO_BEST, step=0.01, label="Umbral inferior (por debajo: PASA)")
    t_high = gr.Slider(0, 1, value=HI_BEST, step=0.01, label="Umbral superior (por encima: FALLA)")

    with gr.Tab("Imagen"):
        img = gr.Image(type="numpy", label="Sube una placa")
        btn = gr.Button("Analizar")
        out_img = gr.Image(label="Detecciones")
        verdict = gr.Markdown()
    with gr.Tab("Matriz y coste"):
        conf_df = gr.Dataframe(label="Placas esperadas por cada 1000 (con la tasa base asumida)")
        summary = gr.Markdown()
        curve = gr.Plot(label="Coste frente a umbral")
    with gr.Tab("Resultados por clase"):
        gr.Markdown(
            f"mAP50 = {metrics['detector']['map50']:.3f} | "
            f"mAP50-95 = {metrics['detector']['map50_95']:.3f}. "
            "Precisión y recall por clase están en el punto de mejor F1, no en tu umbral."
        )
        gr.Dataframe(per_class_table(), label="Por clase")
        gr.Dataframe(baseline_table(), label="Modelo frente a baselines (nivel de placa)")
        gr.Plot(confusion_figure(), label="Matriz de confusión (filas: predicha, columnas: real)")
    with gr.Tab("Políticas"):
        gr.Dataframe(C.policy_comparison(scores), label="Coste por 1000 placas")

    btn.click(predict, [img, t_low, t_high], [out_img, verdict])
    for s in (t_low, t_high):
        s.change(update, [t_low, t_high], [conf_df, summary, curve])
    demo.load(update, [t_low, t_high], [conf_df, summary, curve])

if __name__ == "__main__":
    demo.launch()