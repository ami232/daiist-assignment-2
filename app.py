"""Run with python main.py app. Loads saved artifacts; never trains."""
import json
import os
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
(ROOT / '.runtime').mkdir(exist_ok=True)

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO
from decision import ACTIONS, comparisons, evaluate, optimize

ARTIFACTS = ROOT / 'artifacts'
DECISIONS = ARTIFACTS / 'decision_analysis'
POLICY = json.loads((DECISIONS / 'frozen_policy.json').read_text())
ASSUMPTIONS = POLICY['assumptions']
SCORES = {name: pd.read_csv(DECISIONS / f'{name}_scores.csv', float_precision='round_trip') for name in ['validation', 'test']}
POLICY['automatic_lower_threshold'] = optimize(SCORES['validation'], ASSUMPTIONS)['automatic_lower_threshold']
PER_CLASS = pd.read_csv(ARTIFACTS / 'evaluation/finetuned_test/per_class_metrics.csv')
OVERALL = json.loads((ARTIFACTS / 'evaluation/finetuned_test/overall_metrics.json').read_text())
MODEL = YOLO(str(ROOT / 'weights/best.pt'))
MODEL_LOCK = Lock()
ANOMALIES = {i for i, name in MODEL.names.items() if name in {'cracked', 'dusty', 'bird_drop'}}
DUPLICATE = 'Screenshot-2023-04-02-002436_png_jpg.rf.ff762fb15d12276b24f64f2a12789ba5.jpg'


def selected_scores(split):
    if split == 'Test without known duplicate':
        frame = SCORES['test']
        return frame.loc[~frame.image.map(lambda x: Path(x).name == DUPLICATE)]
    return SCORES['validation' if split == 'Validation' else 'test']


def update_policy(lower, upper, split):
    if lower > upper:
        raise gr.Error('The lower threshold must not exceed the upper threshold.')
    frame = selected_scores(split)
    r = evaluate(frame, lower, upper, ASSUMPTIONS)
    daily = ASSUMPTIONS['daily_images']
    reviews = daily * r['review_share']
    hours = reviews * ASSUMPTIONS['review_seconds'] / 3600
    metrics = (
        f"### €{r['cost_per_1000']:.2f} per 1,000 photographs\n"
        f"**Photo review: {100*r['review_share']:.2f}%** · Automatic incidents: {100*r['automatic_share']:.2f}% · "
        f"Positive cases escalated: {100*r['positive_escalation']:.2f}%\n\n"
        f"At {daily:,} photos/day: **{reviews:.1f} photo reviews**, **{daily*r['automatic_share']:.1f} automatic incidents**, "
        f"{hours:.2f} photo-review hours ({hours/ASSUMPTIONS['productive_hours_per_person']:.3f} staff equivalent). "
        f"Total incidents, including reviewer-confirmed cases: {daily*r['incident_share']:.1f}/day.\n\n"
        'Incident handling is priced at €1 each. Its staffing hours are not included in photo-review hours.'
    )
    matrix = go.Figure(go.Heatmap(
        z=r['counts'], x=ACTIONS, y=['No annotated anomaly', 'Annotated anomaly'],
        text=r['counts'], texttemplate='%{text}', colorscale='Blues',
        hovertemplate='Actual: %{y}<br>Action: %{x}<br>Images: %{z}<extra></extra>',
    ))
    matrix.update_layout(title=f'Image-level action matrix — {len(frame)} actual images',
                         xaxis_title='Policy action', yaxis_title='Annotated state', height=340)
    weighted = pd.DataFrame(r['weighted_counts_per_1000'], columns=ACTIONS,
                            index=['No annotated anomaly', 'Annotated anomaly'])
    weighted = weighted.rename_axis('Actual state').reset_index().round(3)
    values = np.unique(np.r_[np.linspace(0, upper, 101), lower])
    costs = [evaluate(SCORES['validation'], t, upper, ASSUMPTIONS)['cost_per_1000'] for t in values]
    current = evaluate(SCORES['validation'], lower, upper, ASSUMPTIONS)
    frozen = evaluate(SCORES['validation'], POLICY['lower_threshold'], POLICY['upper_threshold'], ASSUMPTIONS)
    curve = go.Figure(go.Scatter(x=values, y=costs, mode='lines', name='Current upper threshold'))
    curve.add_trace(go.Scatter(x=[lower], y=[current['cost_per_1000']], mode='markers',
                              marker=dict(size=11), name='Current policy'))
    curve.add_trace(go.Scatter(x=[POLICY['lower_threshold']], y=[frozen['cost_per_1000']], mode='markers',
                              marker=dict(symbol='star', size=14), name='Frozen chosen policy'))
    curve.update_layout(title=f'Validation cost vs lower threshold (upper held at {upper:.6f})',
                        xaxis_title='Lower threshold', yaxis_title='Expected cost / 1,000 (€)', height=360)
    return metrics, matrix, weighted, curve


def predict_image(image, lower, upper, display_conf):
    if image is None:
        raise gr.Error('Upload a photograph first.')
    if lower > upper:
        raise gr.Error('The lower threshold must not exceed the upper threshold.')
    with MODEL_LOCK:
        result = MODEL.predict(np.ascontiguousarray(image[:, :, ::-1]), device='cpu', imgsz=640, conf=0.001,
                               iou=0.7, max_det=300, verbose=False)[0]
    score = max((float(b.conf.item()) for b in result.boxes if int(b.cls.item()) in ANOMALIES), default=0.0)
    action = ACTIONS[0] if score < lower else ACTIONS[1] if score < upper else ACTIONS[2]
    table = [{'Class': MODEL.names[int(b.cls.item())], 'Confidence': float(b.conf.item())}
             for b in result.boxes if float(b.conf.item()) >= display_conf]
    shown = result[result.boxes.conf >= display_conf]
    annotated = shown.plot()[:, :, ::-1]
    message = (f'### {action}\nAnomaly confidence score: **{score:.6f}**. Confidence is not a calibrated probability.\n\n'
               'An automatic incident is a recommendation for verification before intervention. '
               'This dashboard does not create tickets in an external service.')
    return annotated, message, pd.DataFrame(table, columns=['Class', 'Confidence'])


def predict_if_uploaded(image, lower, upper, display_conf):
    return predict_image(image, lower, upper, display_conf) if image is not None else (gr.skip(), gr.skip(), gr.skip())


def build_demo():
    lo, hi = POLICY['lower_threshold'], POLICY['upper_threshold']
    initial = update_policy(lo, hi, 'Validation')
    with gr.Blocks(title='Solar Panel Inspection Decisions') as demo:
        gr.Markdown('# Solar Panel Inspection Decisions\nExplore the trained YOLO11n model and the cost of pass, photo review, and automatic incident decisions.')
        gr.Markdown('**Illustrative scenario:** 1,000 photos/day · 10% anomaly prevalence · 30 seconds/review · '
                    '€24/hour · €5 per missed positive · €1/incident. Human review is assumed correct. '
                    'Pass means no immediate review, not a safety certification.')
        with gr.Row():
            lower = gr.Slider(0, 1, value=lo, step=0.000001, label='Lower threshold: pass → review')
            upper = gr.Slider(0, 1, value=hi, step=0.000001, label='Upper threshold: review → automatic incident')
        reset = gr.Button('Restore validation-selected thresholds')
        gr.Markdown(f'Frozen choices: **{lo:.9f} / {hi:.9f}**. Moving controls explores alternatives; '
                    'it does not replace the saved policy. Select thresholds using validation, then assess them on test.')
        with gr.Tab('Decision policy'):
            split = gr.Dropdown(['Validation', 'Original test', 'Test without known duplicate'], value='Validation', label='Evaluation images')
            metrics = gr.Markdown(initial[0])
            matrix = gr.Plot(initial[1])
            weighted = gr.Dataframe(initial[2], interactive=False, label='Expected counts / 1,000 at 10% prevalence')
            curve = gr.Plot(initial[3])
            gr.Markdown('The matrix uses actual image counts. Expected costs and workload reweight rates to 10% prevalence. '
                        'The star marks the saved two-threshold policy; when the upper slider changes it need not lie on the displayed curve.')
        with gr.Tab('Upload a photograph'):
            with gr.Row():
                upload = gr.Image(type='numpy', label='Photograph')
                annotated = gr.Image(label='Detected objects', interactive=False)
            display_conf = gr.Slider(0.001, 1, value=lo, step=0.001, label='Display boxes with confidence at least')
            gr.Markdown('Display confidence filters visible boxes only. Policy thresholds determine the action using all retained anomaly scores.')
            predict = gr.Button('Analyze photograph', variant='primary')
            prediction_message = gr.Markdown()
            prediction_table = gr.Dataframe(interactive=False, label='Visible detections')
            predict.click(predict_image, [upload, lower, upper, display_conf], [annotated, prediction_message, prediction_table])
            for control in [lower, upper, display_conf]:
                control.release(predict_if_uploaded, [upload, lower, upper, display_conf], [annotated, prediction_message, prediction_table])
        with gr.Tab('Model evaluation'):
            gr.Markdown(f"**Original test, 648 images:** AP50 {OVERALL['metrics/mAP50(B)']:.4f} · "
                        f"mAP50–95 {OVERALL['metrics/mAP50-95(B)']:.4f}. These measure object detection, not image-level decision accuracy. "
                        'Per-class precision/recall are evaluator summaries; the matrix below uses confidence 0.25.')
            gr.Dataframe(PER_CLASS.fillna('N/A'), interactive=False, label='Per-class test results')
            gr.Image(str(ARTIFACTS/'evaluation/finetuned_test_conf025/confusion_matrix.png'), label='Object-level matrix at confidence 0.25', interactive=False)
            gr.Dataframe(pd.read_csv(ARTIFACTS/'evaluation/comparison/all_models_per_class.csv').fillna('N/A'),
                         interactive=False, label='Detection baselines')
            gr.Markdown('**Limitations:** no bird_drop objects in test; only 3 validation photos contain bird_drop; '
                        '32 dusty objects in 16 test photos; one exact training–test duplicate; near-duplicates remain possible. '
                        'Panel labels also occur with defects. Photos without anomaly labels are not verified healthy panels. '
                        'Sampled photos include indoor scenes, so performance in solar farms is unverified.')
        with gr.Tab('Frozen policy comparisons'):
            for name, frame in [('Validation', SCORES['validation']), ('Original test', SCORES['test'])]:
                gr.Markdown(f'### {name}\nAll model thresholds were selected on validation. Costs include incident handling.')
                gr.Dataframe(comparisons(frame, ASSUMPTIONS, POLICY).round(4), interactive=False)
            gr.Markdown('COCO has no semantic match for these target classes. Its direct target-class baseline coincides with predicting nothing. '
                        'No arbitrary COCO class-ID mapping is used.')
        outputs = [metrics, matrix, weighted, curve]
        for control in [lower, upper]:
            control.change(update_policy, [lower, upper, split], outputs)
        split.change(update_policy, [lower, upper, split], outputs)
        reset.click(lambda: (lo, hi), outputs=[lower, upper]).then(update_policy, [lower, upper, split], outputs)
    return demo


if __name__ == '__main__':
    demo = build_demo()
    demo.launch(server_name='127.0.0.1', server_port=int(os.environ.get('APP_PORT', '7860')),
                share=os.environ.get('GRADIO_SHARE', '0') == '1', inbrowser=False,
                allowed_paths=[str(ARTIFACTS)])
