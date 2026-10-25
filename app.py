"""
Assignment 2: dashboard Gradio del detector de stock en expendedoras.

Se ejecuta con:  uv run python main.py app

No entrena nada: carga weights/best.pt y los resultados que train.py dejó en outputs/.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

from train import cost_two_thresholds, SECONDS_PER_REVIEW, SLOTS_PER_DAY, REVIEW_HOURS_AVAILABLE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

model = YOLO(ROOT / "weights" / "best.pt")
metrics = json.loads((OUT / "metrics.json").read_text())
test_preds = pd.read_csv(OUT / "test_preds.csv")
sweep = pd.read_csv(OUT / "sweep.csv")
TEST_GT = metrics["test_gt"]
T_STAR = metrics["chosen_threshold"]
BAND = metrics["chosen_band"]


# --- Pestaña 1: predicción sobre una foto ----------------------------------
def predict(image, threshold, capacity):
    if image is None:
        return None, pd.DataFrame(), "Sube una foto de una máquina expendedora."
    r = model.predict(image[:, :, ::-1], conf=threshold, imgsz=metrics["imgsz"], verbose=False)[0]  # Gradio da RGB, YOLO espera BGR
    names = [model.names[int(c)] for c in r.boxes.cls]
    counts = pd.Series(names, dtype=str).value_counts().rename_axis("producto").reset_index(name="huecos")
    occupied = len(names)
    free = max(int(capacity) - occupied, 0)
    summary = (f"**Huecos ocupados:** {occupied}  \n**Huecos libres (estimado):** {free} de {int(capacity)}  \n"
               f"Umbral de confianza: {threshold:.2f}")
    return r.plot()[:, :, ::-1], counts, summary  # plot() devuelve BGR, Gradio espera RGB


# --- Pestaña 2: los dos umbrales (pass / revisión / fail) -------------------
def band_view(t_low, t_high):
    if t_low > t_high:
        t_low, t_high = t_high, t_low
    r = cost_two_thresholds(test_preds, TEST_GT, t_low, t_high)
    cm = pd.DataFrame({"": ["Real: producto", "Real: hueco vacío"],
                       "Pred: producto": [r["tp"], r["fp"]],
                       "Pred: vacío": [r["fn"], r["tn"]]})
    hours = r["review_share"] * SLOTS_PER_DAY * SECONDS_PER_REVIEW / 3600
    alt = metrics["alternatives_cost_per_1000"]
    text = (f"**Coste esperado:** {r['cost_per_1000']:.1f} € por 1.000 huecos  \n"
            f"**Enviado a revisión:** {r['review_share']:.1%} de los huecos ({r['reviewed']} cajas en test)  \n"
            f"**Carga de revisión:** {hours:.2f} h/día con {SLOTS_PER_DAY} huecos/día "
            f"→ {hours / REVIEW_HOURS_AVAILABLE:.2f} personas a media jornada  \n\n"
            f"Comparación (€ / 1.000 huecos): no hacer nada {alt['do_nothing']:.1f} · "
            f"persona revisa todo {alt['human_checks_all']:.1f} · "
            f"modelo solo (t={T_STAR}) {alt['model_alone']:.1f}")
    return cm, text


# --- Pestaña 3: curva de coste ---------------------------------------------
def cost_curve():
    fig = go.Figure()
    for split in ["valid", "test"]:
        s = sweep[sweep.split == split]
        fig.add_trace(go.Scatter(x=s.t_low, y=s.cost_per_1000, mode="lines+markers", name=split))
    fig.add_vline(x=T_STAR, line_dash="dash", annotation_text=f"umbral elegido = {T_STAR}")
    fig.update_layout(xaxis_title="Umbral de confianza", yaxis_title="€ por 1.000 huecos",
                      title="Coste frente a umbral (elegido en valid)")
    return fig


# --- Pestaña 4: resultados por clase y baselines ---------------------------
per_class = pd.DataFrame(metrics["per_class"]).round(3)
baselines = pd.DataFrame(metrics["baselines"]).T.reset_index(names="modelo")
baselines = baselines[["modelo", "ap50_agnostic", "t_low", "tp", "fp", "fn", "cost_per_1000"]]
baselines = baselines.rename(columns={"t_low": "umbral"})
baselines[["ap50_agnostic", "cost_per_1000"]] = baselines[["ap50_agnostic", "cost_per_1000"]].astype(float).round(3)


with gr.Blocks(title="Stock en expendedoras") as demo:
    gr.Markdown(f"# Detector de stock en máquinas expendedoras\n"
                f"Modelo: {metrics['base_model']} · run `{metrics['run']}` · "
                f"mAP50 test = {metrics['baselines']['finetuned']['map50']:.3f}")
    with gr.Tab("1. Predicción"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="numpy", label="Foto de la máquina")
                thr = gr.Slider(0.05, 0.95, value=T_STAR, step=0.05, label="Umbral de confianza")
                cap = gr.Number(value=40, label="Capacidad de la máquina (nº de huecos)")
                gr.Examples([str(p) for p in sorted((OUT / "examples").glob("*"))], inputs=img_in)
            with gr.Column():
                img_out = gr.Image(label="Detecciones")
                summary = gr.Markdown()
                counts = gr.Dataframe(label="Productos por categoría")
        for comp in [img_in, thr, cap]:
            comp.change(predict, [img_in, thr, cap], [img_out, counts, summary])

    with gr.Tab("2. Umbrales y revisión humana"):
        gr.Markdown("conf ≥ t_alto → producto aceptado · conf < t_bajo → hueco vacío · "
                    "entre los dos → revisión humana. Calculado sobre el split de test.")
        lo = gr.Slider(0.05, 0.95, value=BAND["t_low"], step=0.05, label="t_bajo")
        hi = gr.Slider(0.05, 0.95, value=BAND["t_high"], step=0.05, label="t_alto")
        cm_out = gr.Dataframe(label="Matriz de confusión (huecos)")
        band_text = gr.Markdown()
        for s in [lo, hi]:
            s.change(band_view, [lo, hi], [cm_out, band_text])
        demo.load(band_view, [lo, hi], [cm_out, band_text])

    with gr.Tab("3. Curva de coste"):
        gr.Plot(cost_curve())

    with gr.Tab("4. Resultados por clase"):
        gr.Markdown("### Precisión, recall y AP50 por clase (test)")
        gr.Dataframe(per_class)
        gr.Markdown("### Modelo frente a baselines (test, producto sí/no)")
        gr.Dataframe(baselines)
        gr.Image(str(OUT / "confusion_matrix.png"), label="Matriz de confusión por clase")

if __name__ == "__main__":
    demo.launch()
