"""Gradio validation workbench. Loads saved artifacts; never trains at startup.

Run using main.py app. Sliders explore validation; the final test snapshot is
displayed separately and never used to retune the operating policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/.matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/.ultralytics"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyse_validation import model_inputs
from costing import alternatives, capacity_limit, evaluate, photo_workflow
from photo_predictions import model_labels

OUT = ROOT / "results/validation_analysis"
BUNDLE = json.loads((OUT / "photo_predictions.json").read_text())
ANALYSIS = json.loads((OUT / "analysis.json").read_text())
SCENARIO = json.loads((ROOT / "scenario.json").read_text())
CAPACITY = capacity_limit(SCENARIO)
PHOTO_ONLY = photo_workflow(SCENARIO)
HIGH_ROUTE = "Incidencia provisional" if PHOTO_ONLY else "Comprobación física"
assert BUNDLE["evaluation_split"] == ANALYSIS["evaluation_split"] == "valid"
assert not BUNDLE["test_evaluated"] and not ANALYSIS["test_evaluated"]
assert SCENARIO == ANALYSIS["assumptions"], "Regenerate analysis after changing scenario.json."
LABELS = model_labels(BUNDLE)
assert set(LABELS.values()) == set(ANALYSIS["models"])
MODELS = {}
FINAL = None
FINAL_SCENARIO = None
final_pointer = ROOT / "results/final_evaluation.json"
if final_pointer.exists():
    pointer = json.loads(final_pointer.read_text())
    if pointer["policy_sha256"] == hashlib.sha256((ROOT / "operating_policy.json").read_bytes()).hexdigest():
        FINAL = json.loads((ROOT / pointer["summary"]).read_text())
        FINAL_SCENARIO = json.loads((ROOT / pointer["summary"]).with_name("scenario.json").read_text())
FINAL_MATCHES_SCENARIO = FINAL is not None and FINAL["policy"]["scenario_sha256"] == hashlib.sha256((ROOT / "scenario.json").read_bytes()).hexdigest()


def workload_description(minutes, scenario):
    limit = capacity_limit(scenario)
    if limit is not None:
        status = "Dentro" if minutes <= limit + 1e-9 else "FUERA"
        return f"{minutes:.2f} min/día · {status} del presupuesto de {limit:g} min. "
    manual = alternatives(scenario)["physical_check_everything"]["minutes_per_day"]
    saved = manual - minutes
    change = "Tiempo liberado" if saved >= 0 else "Trabajo adicional"
    return (f"{minutes:.2f} min/día ({minutes/60:.2f} horas acumuladas de una persona). "
            f"{change} estimado frente a inspeccionar todos físicamente: {abs(saved):.2f} min/día. ")


def scenario_table(rows, scenario, title="Alternativa"):
    manual = alternatives(scenario)["physical_check_everything"]["minutes_per_day"]
    limit = capacity_limit(scenario)
    return pd.DataFrame([
        [name, value["cost_per_day"], value["minutes_per_day"],
         manual - value["minutes_per_day"] if limit is None else value["capacity_feasible"]]
        for name, value in rows], columns=[title, "€/día", "Minutos/día",
        "Tiempo liberado · min/día" if limit is None else f"Cabe en {limit:g} min"]).round(2)


def figures(label, lower, upper):
    key = LABELS[label]
    scores, truth = model_inputs(BUNDLE, key)
    v = evaluate(scores, truth, lower, upper, SCENARIO)
    matrix = np.array(v["expected_confusion_per_1000"])
    confusion = go.Figure(go.Heatmap(z=matrix, x=[HIGH_ROUTE, "Pasar"],
                                    y=["Daño real", "Sin daño real"], text=np.round(matrix, 1),
                                    texttemplate="%{text}", colorscale="Blues", showscale=False))
    confusion.update_layout(title="Resultado esperado tras la revisión de fotos · por 1.000 casos",
                            height=330, margin=dict(l=50,r=10,t=70,b=40))
    curve = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Variar límite inferior · superior fijo {upper:.4f}",
        f"Variar límite superior · inferior fijo {lower:.4f}"))
    for col, grid in ((1, np.unique(np.r_[np.linspace(0, upper, 101),lower])),
                       (2, np.unique(np.r_[np.linspace(lower, 1, 101),upper]))):
        values = [evaluate(scores, truth, t if col == 1 else lower,
                           upper if col == 1 else t, SCENARIO) for t in grid]
        curve.add_trace(go.Scatter(x=grid, y=[p["cost_per_1000"] for p in values],
                                  name="Coste", line=dict(color="#2468a2"), showlegend=col==1), 1, col)
        if CAPACITY is not None:
            infeasible = [i for i,p in enumerate(values) if p["capacity_feasible"] is False]
            curve.add_trace(go.Scatter(x=grid[infeasible], y=[values[i]["cost_per_1000"] for i in infeasible],
                                      mode="markers", marker=dict(color="#cd6b35",size=4),
                                      name=f"Supera {CAPACITY:g} min/día", showlegend=col==1), 1, col)
        curve.add_trace(go.Scatter(x=[lower if col==1 else upper], y=[v["cost_per_1000"]],
                                  mode="markers", marker=dict(color="#b52b46",size=12,symbol="star"),
                                  name="Valores actuales", showlegend=col==1), 1, col)
    curve.update_xaxes(title_text="Umbral de puntuación")
    curve.update_yaxes(title_text="Euros / 1.000 casos")
    curve.update_layout(height=390, margin=dict(l=55,r=10,t=60,b=60), legend=dict(orientation="h",y=-.2))
    summary = (f"**{v['cost_per_1000']:,.2f} € / 1.000 casos** · {v['cost_per_day']:.2f} €/día  \n"
               +workload_description(v["minutes_per_day"], SCENARIO)+"  \n"+
               f"Revisión de foto: **{v['photo_review_rate']:.1%}** ({v['photo_reviews_per_day']:.2f}/día). "+
               (f"Incidencias: **{v['incidents_per_day']:.2f}/día**, de ellas **{v['false_incidents_per_day']:.2f} falsas/día**.  \n" if PHOTO_ONLY else f"Comprobación física: **{v['physical_checks_per_day']:.2f}/día**.  \n")+
               f"Daños que pasarían sin detectar: **{v['conditional_damage_miss_rate']:.1%} de los casos con daño** "
               f"({v['missed_damage_cases_per_day']:.3f}/día en el escenario).  \n"
               f"Coste de tiempo: {v['labour_per_case']*SCENARIO['cases_per_day']:.2f} €/día; pérdida por daños omitidos: "
               f"{v['missed_damage_loss_per_case']*SCENARIO['cases_per_day']:.2f} €/día; "
               f"otros costes de falsas incidencias: {v['false_incident_loss_per_case']*SCENARIO['cases_per_day']:.2f} €/día.")
    routing = pd.DataFrame(v["routing_counts"], columns=["Con daño", "Sin daño"])
    routing.insert(0, "Ruta inicial", ["Pasar", "Revisar foto", HIGH_ROUTE])
    per_class = pd.DataFrame([{"Clase": c["Class"], "Precisión": c["Box-P"], "Recall": c["Box-R"],
                                "AP50": c["mAP50"], "AP50–95": c["mAP50-95"]}
                               for c in BUNDLE["models"][key]["per_class"]])
    return summary, confusion, curve, routing, per_class


def refresh(label, lower, upper):
    # A numeric field is briefly empty while the user replaces its contents.
    if lower is None or upper is None:
        return (gr.skip(),)*5
    if not 0 <= lower <= upper <= 1:
        return ("Ajusta los límites: 0 ≤ inferior ≤ superior ≤ 1. Los gráficos conservan el último cálculo válido.",
                gr.skip(),gr.skip(),gr.skip(),gr.skip())
    return figures(label, lower, upper)


def proposal(label):
    v = ANALYSIS["models"][LABELS[label]]["candidate_feasible"]
    return v["lower"], v["upper"], *figures(label, v["lower"], v["upper"])


@lru_cache(maxsize=4)
def predict_boxes(path, key, image_sha256):
    """Cache inference by file content/model; slider movement only redraws boxes."""
    from PIL import Image
    from train import choose_device
    from ultralytics import YOLO
    metadata = BUNDLE["models"][key]
    if key not in MODELS:
        weights = ROOT / metadata["weights"]
        if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["weights_sha256"]:
            raise gr.Error("Los pesos no coinciden con los evaluados.")
        MODELS[key] = YOLO(str(weights))
    device = choose_device(["cuda", "mps", "cpu"])
    result = MODELS[key].predict(source=path, imgsz=640, batch=1, rect=False, conf=.001,
                                iou=.7, max_det=300, device=device, verbose=False)[0]
    boxes = [{"class": int(c), "score": float(s), "xyxy": b} for c,s,b in zip(
        result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist())]
    return Image.fromarray(result.orig_img[..., ::-1]).convert("RGB"), boxes, result.names


def render_prediction(prediction, label, lower, upper):
    if lower is None or upper is None:
        return gr.skip(), gr.skip()
    if prediction is None:
        return None, ""
    if not 0 <= lower <= upper <= 1:
        return gr.skip(), gr.skip()
    from PIL import ImageDraw, ImageFont
    original, boxes, names = prediction
    score = max((b["score"] for b in boxes if b["class"] == 0), default=0.)
    route = HIGH_ROUTE if score >= upper else "Revisión de foto" if score >= lower else "Pasar"
    picture = original.copy()
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=max(14,round(picture.width/55)))
    for box in boxes:
        if box["score"] < lower:
            continue
        color = "#d14343" if box["class"] == 0 else "#147d58"
        draw.rectangle(box["xyxy"], outline=color, width=3)
        x,y = box["xyxy"][:2]
        draw.text((x+3,y+3), f"{names[box['class']]} {box['score']:.3f}", fill=color, font=font, stroke_width=1, stroke_fill="white")
    text = (f"**Ruta propuesta: {route}.** Puntuación máxima de daño: **{score:.5f}**.  \n"
            f"Se muestran cajas con puntuación ≥ {lower:.4f}. La decisión usa la mayor puntuación de daño. "
            "Esta puntuación no es una probabilidad calibrada; una foto que pasa puede contener daño. "
            "La subida no añade datos a la evaluación.")
    return picture, text


def infer(path, label, lower, upper):
    if path is None:
        raise gr.Error("Sube una fotografía primero.")
    if not 0 <= lower <= upper <= 1:
        raise gr.Error("El límite inferior debe ser menor o igual al superior.")
    prediction = predict_boxes(str(path), LABELS[label], hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return *render_prediction(prediction,label,lower,upper), prediction


def optional_infer(path, label, lower, upper):
    return infer(path,label,lower,upper) if path else (None,"",None)


def comparison(label):
    """Three operational scenarios for the selected model, at saved minima."""
    rows = []
    manual = ANALYSIS["alternatives"]["physical_check_everything"]
    rows.append(("1. Revisión física de todos los vehículos",manual))
    for policy,title in (("candidate_feasible", "2. Modelo + revisión de fotos dudosas"),
                         ("single_threshold_feasible", "3. Modelo solo")):
        v = ANALYSIS["models"][LABELS[label]][policy]
        rows.append((title+" · "+label,v))
    return scenario_table(rows, SCENARIO)


def model_comparison():
    rows = []
    for label, key in LABELS.items():
        model = BUNDLE["models"][key]
        damage = next(c for c in model["per_class"] if c["Class"] == "Damage-detected")
        candidate = ANALYSIS["models"][key]["candidate_feasible"]
        rows.append([label, model["detection_metrics"]["metrics/mAP50(B)"],
                     model["detection_metrics"]["metrics/mAP50-95(B)"], damage["mAP50"],
                     candidate["cost_per_day"], candidate["minutes_per_day"],
                     candidate["photo_reviews_per_day"]])
    table = pd.DataFrame(rows, columns=["Modelo", "mAP50", "mAP50–95", "AP50 daño",
                        "€/día · híbrido", "Min/día · híbrido", "Fotos revisadas/día"])
    return table.round({"mAP50": 4, "mAP50–95": 4, "AP50 daño": 4,
                        "€/día · híbrido": 2, "Min/día · híbrido": 2, "Fotos revisadas/día": 2})


def build_demo():
    explore_key = FINAL["policy"]["model_key"] if FINAL_MATCHES_SCENARIO else min(
        ANALYSIS["models"],key=lambda key: ANALYSIS["models"][key]["candidate_feasible"]["cost_per_case"])
    default = next(label for label,key in LABELS.items() if key == explore_key)
    v = ANALYSIS["models"][LABELS[default]]["candidate_feasible"]
    initial = figures(default,v["lower"],v["upper"])
    with gr.Blocks(title="Car Damage Inspection · Assignment 2") as demo:
        stage = (f"Política elegida: {default} · evaluación secundaria de test disponible más abajo."
                 if FINAL_MATCHES_SCENARIO and FINAL.get("evaluation_context") == "secondary_reuse_of_previously_evaluated_test"
                 else f"Política fijada: {default} · test final disponible más abajo."
                 if FINAL_MATCHES_SCENARIO else "Exploración del escenario actual en validación · la evaluación original de test se conserva como histórico.")
        gr.Markdown("# Car Damage Inspection\n**"+stage+"** "
                    "365 fotos con condición conocida; 1 foto sin etiqueta excluida de costes. "
                    "Las métricas de detección conservan las 366 fotos.")
        gr.Markdown(f"Escenario simulado: {SCENARIO['cases_per_day']:g} devoluciones/día; "
                    f"{SCENARIO['damage_prevalence']:.0%} con daño; {SCENARIO['missed_damage_euros']:g} € por daño omitido; "
                    f"{SCENARIO['staff_euros_per_hour']:g} €/hora; revisión de foto {SCENARIO['photo_review_minutes']:g} min. "+
                    (f"Tramitar cada incidencia, cierta o falsa: {SCENARIO['incident_handling_minutes']:g} min; otros costes por falsa incidencia: {SCENARIO['false_incident_extra_euros']:g} €. Sin comprobación física. " if PHOTO_ONLY else "Comprobación física 3 min; informe de daño 5 min extra. ")+
                    f"Empleado de fotos: sensibilidad supuesta {SCENARIO['photo_reviewer_sensitivity']:.0%}; especificidad supuesta {SCENARIO['photo_reviewer_specificity']:.0%}. "
                    +( "Sin límite diario fijo: el tiempo se valora como coste y se compara con la inspección física de todos. " if CAPACITY is None else f"Capacidad acumulada: {CAPACITY:g} min/día para revisión y tramitación. ")+
                    "Son hipótesis académicas, no resultados medidos con empleados.")
        if SCENARIO.get("scope_description_es"):
            gr.Markdown(SCENARIO["scope_description_es"])
        gr.Markdown("## Comparación de modelos · validación\n"
                    "La detección mide clases y colocación de cajas. Los costes y tiempos usan las decisiones por foto "
                    "y el mínimo de cada modelo bajo los mismos supuestos. Un mAP de 0,53 no significa acertar el 53 % de las fotos.")
        gr.Dataframe(model_comparison(), interactive=False)
        model = gr.Dropdown(list(LABELS),value=default,label="Modelo para explorar (no fija la elección final)")
        with gr.Row():
            lower = gr.Slider(0,1,value=v["lower"],step=.000001,label="Inferior: pasar → revisar foto")
            upper = gr.Slider(0,1,value=v["upper"],step=.000001,label="Superior: revisar foto → "+HIGH_ROUTE.lower())
        gr.Markdown("Puntuación < inferior: pasar. Inferior ≤ puntuación < superior: revisar foto. "
                    +( "Puntuación ≥ superior: generar incidencia provisional automáticamente. Una revisión humana positiva también genera incidencia. La tramitación administrativa no es otra revisión de la foto. " if PHOTO_ONLY else "Puntuación ≥ superior: comprobar físicamente. Una revisión de foto puede acabar en comprobación física. ")+
                    "No se decide cobrar al cliente. La persona solo corrige las fotos que llegan a su banda de revisión.")
        restore = gr.Button("Cargar mínimo numérico de validación para este modelo")
        summary = gr.Markdown(initial[0])
        confusion = gr.Plot(initial[1])
        curve = gr.Plot(initial[2])
        gr.Markdown("La matriz muestra cantidades **esperadas** con prevalencia del 5 % y revisión humana simulada; "
                    "puede contener decimales. "+
                    ("Un falso positivo significa una incidencia innecesaria. " if PHOTO_ONLY else "Un falso positivo significa una comprobación física innecesaria. ")+
                    "Las curvas son cortes del problema de dos umbrales; cada una mantiene fijo el otro límite.")
        routing = gr.Dataframe(initial[3],interactive=False,label="Rutas iniciales reales · 365 fotos de validación")
        gr.Markdown("## Comparación de tres escenarios operativos\n"
                    "Inspección física de todos los vehículos, modelo con revisión de fotos dudosas y modelo solo. "
                    "Para el modelo seleccionado se muestran los mínimos guardados de validación de cada política, "
                    "no los valores exploratorios de los controles. El modelo solo utiliza su propio umbral de mínimo coste. "
                    "La inspección física conserva el supuesto de confirmación ideal e incluye informes. "
                    "El tiempo liberado se puede dedicar a otras tareas; no implica automáticamente una reducción de nómina.")
        comparison_table=gr.Dataframe(comparison(default),interactive=False)
        model.change(comparison,[model],comparison_table,api_name="compare_scenarios")
        gr.Markdown("## Detección por clase\nP/R por caja en el punto F1 del evaluador de cada modelo, "
                    "distinto de los dos umbrales operativos. No cambian al mover la banda de revisión.")
        per_class = gr.Dataframe(initial[4],interactive=False)
        gr.Markdown("Baselines en validación: COCO zero-shot y predecir ninguna caja obtienen AP50 y AP50–95 = 0 "
                    "para estas clases. COCO no contiene etiquetas de estado del coche: `car` no significa `Undamaged`. "
                    "La precisión sin predicciones positivas es indefinida. Este baseline mide una limitación del vocabulario.")
        if FINAL:
            final_label = next(label for label,key in LABELS.items() if key == FINAL["policy"]["model_key"])
            historical = "Evaluación original de test · escenario anterior" if not FINAL_MATCHES_SCENARIO else "Evaluación final en test · política fijada antes de medir"
            if FINAL_MATCHES_SCENARIO and FINAL.get("evaluation_context")=="secondary_reuse_of_previously_evaluated_test":
                historical="Evaluación secundaria · test ya utilizado anteriormente"
            gr.Markdown("## "+historical+"\n"
                        f"**Revisor de fotos en este resultado: sensibilidad {FINAL_SCENARIO['photo_reviewer_sensitivity']:.0%} y especificidad {FINAL_SCENARIO['photo_reviewer_specificity']:.0%}.** "
                        +( "Este resultado corresponde a los supuestos anteriores y no representa el nuevo escenario. " if not FINAL_MATCHES_SCENARIO else "")+
                        "Este apartado permanece fijo al mover los controles de validación. "
                        "Este modelo y sus límites se fijaron con validación antes de esta evaluación. "
                        +( "Este test ya se había utilizado con otro modelo; no constituye una nueva prueba independiente. " if FINAL.get("evaluation_context") == "secondary_reuse_of_previously_evaluated_test" else "")+
                        f"{FINAL['detection_images']} fotos para detección; {FINAL['known_condition_images']} con condición conocida para costes.")
            f = FINAL["frozen_policy_scenario"]
            final_workload=(f"{f['incidents_per_day']:.2f} incidencias/día" if photo_workflow(FINAL_SCENARIO) else f"{f['physical_checks_per_day']:.2f} comprobaciones físicas/día")
            gr.Markdown(f"**Política:** inferior {FINAL['policy']['lower']:.6f}; superior {FINAL['policy']['upper']:.6f}.  \n"
                        f"**Escenario calculado con tasas de test:** {f['cost_per_day']:.2f} €/día; "
                        f"{f['minutes_per_day']:.2f} min/día; {f['photo_reviews_per_day']:.2f} revisiones de foto/día; "
                        +final_workload+".  \n"+
                        f"Daños omitidos tras revisión simulada: **{f['conditional_damage_miss_rate']:.1%} de los casos con daño**. "
                        +workload_description(f["minutes_per_day"], FINAL_SCENARIO))
            if FINAL.get("model_only_scenario"):
                manual = FINAL["alternatives"]["physical_check_everything"]
                model_only = FINAL["model_only_scenario"]
                gr.Markdown("### Tres escenarios calculados con tasas de test\n"
                            "El híbrido y el modelo solo conservan sus límites elegidos en validación; no se reajustan en test.")
                gr.Dataframe(scenario_table([
                    ("1. Revisión física de todos los vehículos", manual),
                    ("2. Modelo + revisión de fotos dudosas", f),
                    ("3. Modelo solo", model_only)], FINAL_SCENARIO, "Alternativa · tasas de test"), interactive=False)
            gr.Dataframe(pd.DataFrame([
                [final_label,FINAL['metrics']['metrics/mAP50(B)'],FINAL['metrics']['metrics/mAP50-95(B)']],
                ["COCO zero-shot",0.,0.],["Predecir ninguna caja",0.,0.]],
                columns=["Modelo · mismo test","AP50 global","mAP50–95"]).round(4),interactive=False)
            gr.Dataframe(pd.DataFrame(FINAL["per_class"]),interactive=False,label="Resultados por clase · test")
            gr.Dataframe(pd.DataFrame(f["routing_counts"],index=f["routing_rows"],columns=["Con daño","Sin daño"]).reset_index(),
                         interactive=False,label="Rutas iniciales observadas · test")
            gr.Dataframe(pd.DataFrame(FINAL["box_confusion"]["values"],columns=FINAL["box_confusion"]["classes"],
                                     index=FINAL["box_confusion"]["classes"]).reset_index(),interactive=False,
                         label="Confusión por caja · test · filas predichas, columnas reales; background = sin pareja")
            gr.Markdown("La matriz por caja usa IoU de emparejamiento 0,45 y confianza igual al límite superior. "
                        "P/R de la tabla por clase corresponden al punto F1 del evaluador; AP recorre sus umbrales. "
                        "Los tiempos y pérdidas económicas siguen siendo una simulación, no una prueba con empleados.")
        with gr.Accordion("Sensibilidad a los supuestos y limitaciones",open=False):
            rows=[]
            for label,key in LABELS.items():
                for entry in ANALYSIS["models"][key]["sensitivity_analysis"]:
                    a=entry["nominal_policy_held_fixed"]
                    b=entry["reoptimised_candidate"]
                    row = [label,str(entry["changed"]),a["cost_per_day"],a["minutes_per_day"]]
                    if CAPACITY is not None:
                        row.append(a["capacity_feasible"])
                    rows.append(row+[b["cost_per_day"] if b else None])
            columns=["Modelo","Supuesto cambiado","€/día con límites fijos", "Min/día fijos"]
            if CAPACITY is not None:
                columns.append(f"Cabe en {CAPACITY:g} min")
            gr.Dataframe(pd.DataFrame(rows,columns=columns+["€/día tras nuevo barrido"]).round(2),interactive=False)
            gr.Markdown("Los límites fijos permiten comprobar la robustez de la propuesta; el nuevo barrido responde a otro escenario. "
                        "Los mínimos se han buscado en validación y pueden ser optimistas. "
                        "La carga es una media acumulada; no garantiza atender picos de llegadas. "
                        "Una foto no cubre todo el vehículo y no determina cuándo ocurrió un golpe. "
                        "Hay tres pares de fotos relacionadas entre entrenamiento y validación; independencia no demostrada.")
        gr.Markdown("## Probar una fotografía\nInferencia local con pesos guardados; no se entrena ningún modelo.")
        with gr.Row():
            image = gr.Image(type="filepath",label="Fotografía",sources=["upload"])
            predicted = gr.Image(type="pil",label="Predicciones")
        button = gr.Button("Analizar fotografía",variant="primary")
        decision = gr.Markdown()
        prediction_state = gr.State(None)
        outputs=[summary,confusion,curve,routing,per_class]
        changed_model = model.change(proposal,[model],[lower,upper,*outputs],api_name="load_proposal")
        restored = restore.click(proposal,[model],[lower,upper,*outputs],api_name=False)
        lower.input(refresh,[model,lower,upper],outputs,api_name="update_policy",trigger_mode="always_last",preprocess=False)
        upper.input(refresh,[model,lower,upper],outputs,api_name=False,trigger_mode="always_last",preprocess=False)
        button.click(infer,[image,model,lower,upper],[predicted,decision,prediction_state],concurrency_limit=1,
                     concurrency_id="photo_inference",api_name="predict_photo")
        # A boundary change redraws saved predictions, without another model call.
        for component in (lower,upper):
            component.input(render_prediction,[prediction_state,model,lower,upper],[predicted,decision],
                            api_name=False,trigger_mode="always_last",preprocess=False)
        for event in (changed_model,restored):
            event.then(optional_infer,[image,model,lower,upper],[predicted,decision,prediction_state],concurrency_limit=1,
                       concurrency_id="photo_inference",api_name=False)
        image.change(lambda: (None,"",None),None,[predicted,decision,prediction_state],api_name=False)
    return demo


if __name__ == "__main__":
    print("Preparing validation dashboard from saved artifacts...", flush=True)
    demo = build_demo()
    print("Dashboard ready; starting local server...", flush=True)
    demo.launch(server_name="127.0.0.1",server_port=int(os.environ.get("GRADIO_SERVER_PORT",7860)),
                share=False,inbrowser=False)
