"""A small Gradio dashboard using saved weights and cached evaluation only."""
import json
import os
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw
import gradio as gr
from project import ROOT, ARTIFACTS, WEIGHTS, IMAGE_SIZE, configure_runtime
from policy import Scenario, evaluate_policy, alternatives

configure_runtime()
required=[WEIGHTS,ARTIFACTS/'operating_point.json',ARTIFACTS/'test_predictions.json']
if any(not p.exists() for p in required):
    raise RuntimeError('Saved training outputs are missing. Run main.py train first; the dashboard never trains.')
from ultralytics import YOLO
MODEL=YOLO(WEIGHTS)
INFERENCE_SIZE=json.loads((ARTIFACTS/'training_run.json').read_text())['settings']['imgsz']
LOCK=threading.Lock()
CHOICE=json.loads((ARTIFACTS/'operating_point.json').read_text())
TEST=json.loads((ARTIFACTS/'test_predictions.json').read_text())
VALID=json.loads((ARTIFACTS/'valid_predictions.json').read_text())
AUDIT=json.loads((ARTIFACTS/'dataset_audit.json').read_text())
SCORES=[r['score'] for r in TEST]; TRUTH=[r['positive'] for r in TEST]
VS=[r['score'] for r in VALID]; VY=[r['positive'] for r in VALID]
LOWER=CHOICE['triage']['lower']; UPPER=CHOICE['triage']['upper']
OPTION_PATH=ARTIFACTS/'three_outcome_option.json'
THREE=json.loads(OPTION_PATH.read_text()) if OPTION_PATH.exists() else None


def figure_matrix(matrix,x,y,title):
    z=np.asarray(matrix)
    fig=go.Figure(go.Heatmap(z=z,x=x,y=y,text=z,texttemplate='%{text:.1f}',
                            colorscale='Blues',showscale=False))
    fig.update_layout(title=title,height=340,margin=dict(l=85,r=20,t=55,b=55),
                      xaxis_title='Predicted / final action',yaxis_title='Actual photo label')
    return fig


def update_policy(lower,upper,prevalence,missed,inspection,review_minutes,hourly,volume,sensitivity,specificity):
    if lower>upper:
        raise gr.Error('Keep the lower threshold at or below the upper threshold. Move the upper slider first.')
    cfg=Scenario(prevalence=prevalence/100,missed_cost=missed,inspection_cost=inspection,
                 review_minutes=review_minutes,hourly_cost=hourly,cases_per_day=int(volume),
                 human_sensitivity=sensitivity/100,human_specificity=specificity/100)
    result=evaluate_policy(SCORES,TRUTH,lower,upper,cfg)
    summary=(f"### EUR {result['cost_per_1000']:,.0f} per 1,000 photos\n"
        f"**{result['review_share']:.1%} sent for photo review** · "
        f"{result['reviews_per_day']:.1f} reviews/day · "
        f"{result['review_minutes_per_day']:.0f} review minutes/day · "
        f"{result['review_staff_fte']:.2f} review staff equivalents.\n\n"
        f"Closer inspection referrals: **{result['closer_inspections_per_day']:.1f}/day**. "
        f"Including 10 minutes per closer inspection: "
        f"**{result['total_inspection_staff_fte']:.2f} total staff equivalents** "
        f"at 360 productive minutes per person/day.\n\n"
        "Daily figures assume one selected photo per return; a complete vehicle inspection is outside this test.")
    actual=figure_matrix(result['actual_triage_counts'],['Pass','Human review','Fail / hold'],
                        ['No damage label','Damage label'],'Actual held-out test counts: 84 photos')
    final=figure_matrix(result['expected_binary_per_1000'],['Pass','Hold for inspection'],
                       ['No damage','Damage'],'Expected final confusion per 1,000 production photos')
    curve=[evaluate_policy(VS,VY,t,t,cfg)['cost_per_1000'] for t in np.linspace(0,1,101)]
    fig=go.Figure(go.Scatter(x=np.linspace(0,1,101),y=curve,mode='lines',name='Model alone'))
    fig.add_vline(x=CHOICE['single']['lower'],line_dash='dash',line_color='#bd582c',
                  annotation_text='Saved validation choice')
    fig.add_vrect(x0=lower,x1=upper,fillcolor='#3a8577',opacity=.15,line_width=0)
    fig.update_layout(title='Validation cost sweep: model alone; shaded band = current review region',
                      xaxis_title='Damage confidence threshold',yaxis_title='Expected EUR per 1,000 photos',
                      height=360,margin=dict(l=65,r=20,t=65,b=50))
    current=dict(CHOICE,triage={'lower':lower,'upper':upper})
    prices=alternatives(SCORES,TRUTH,current,cfg)
    table=pd.DataFrame([{'Approach':k,'Expected EUR / 1,000 photos':round(v,2)} for k,v in prices.items()])
    return summary,actual,final,fig,table


def inspect_photo(image,lower,upper):
    if image is None:
        return None,'Upload one vehicle photo to inspect it.'
    if lower>upper:
        raise gr.Error('The lower threshold must not exceed the upper threshold.')
    with LOCK:
        result=MODEL.predict(image,imgsz=INFERENCE_SIZE,conf=.001,iou=.7,device='cpu',verbose=False,max_det=50)[0]
    damage_id=next(i for i,n in MODEL.names.items() if n=='Damage')
    score=max((float(s) for c,s in zip(result.boxes.cls,result.boxes.conf) if int(c)==damage_id),default=0.)
    action='PASS — photo passes triage' if score<lower else (
           'HUMAN REVIEW — ask a person to check the photo' if score<upper else
           'FAIL / HOLD — refer for a closer inspection')
    canvas=image.convert('RGB').copy(); draw=ImageDraw.Draw(canvas)
    for box,cls,confidence in zip(result.boxes.xyxy,result.boxes.cls,result.boxes.conf):
        if float(confidence)<upper:
            continue
        color='#d95339' if int(cls)==damage_id else '#238878'
        coords=box.cpu().tolist()
        draw.rectangle(coords,outline=color,width=3)
        draw.text((coords[0]+3,coords[1]+3),f'{MODEL.names[int(cls)]} {float(confidence):.2f}',fill=color)
    return canvas,f'**{action}**\n\nDamage score: **{score:.3f}**. Boxes shown at the current fail / hold threshold, confidence ≥ {upper:.2f}. This is a model score, not a calibrated probability or proof that damage is new.'


with gr.Blocks(title='Rental return photo triage',analytics_enabled=False) as demo:
    gr.Markdown('# Rental return photo triage\nA class-project prototype: flag visible damage for staff inspection. A photo cannot establish who caused damage or whether a whole car is safe.')
    with gr.Row():
        lower=gr.Slider(0,1,value=LOWER,step=.01,label='Pass → human review threshold')
        upper=gr.Slider(0,1,value=UPPER,step=.01,label='Human review → fail / hold threshold')
    gr.Markdown(f'Saved choices were selected on validation: model alone **{CHOICE["single"]["lower"]:.2f}**; triage **{LOWER:.2f} / {UPPER:.2f}**. Slider changes explore alternatives; they do not retrain or change the saved choice.')
    if LOWER==0:
        gr.Markdown('**The cost minimum disables automatic passing:** at the saved lower threshold of 0.00, every photo gets human review or a closer inspection. A higher lower threshold enables automatic passes and can miss damage.')
    with gr.Row():
        restore=gr.Button('Restore chosen thresholds')
        if THREE:
            compare_three=gr.Button('Compare a policy using all three outcomes')
    with gr.Tab('Inspect a photo'):
        with gr.Row():
            photo=gr.Image(type='pil',label='Upload a returned-car photo')
            annotated=gr.Image(type='pil',label='Predicted boxes',interactive=False)
        inspect=gr.Button('Inspect photo',variant='primary')
        decision=gr.Markdown('Upload one vehicle photo to inspect it.')
    with gr.Tab('Cost and human review'):
        gr.Markdown('Costs use held-out test error rates, reweighted to your assumed production damage rate. Human accuracy and business costs below are scenario assumptions, not measured results.')
        gr.Markdown('**Label quality limits this estimate:** some visibly damaged cars have empty labels. Costs calculated against these annotations do not establish real rental-branch savings.')
        with gr.Accordion('Business assumptions — change these to test the decision',open=False):
            with gr.Row():
                prevalence=gr.Slider(0,100,value=10,step=1,label='Production photos with damage (%)')
                volume=gr.Number(value=100,minimum=0,precision=0,label='Selected photos / returns per day (1 photo each)')
            with gr.Row():
                missed=gr.Number(value=200,minimum=0,label='Additional missed-damage loss (EUR)')
                inspection=gr.Number(value=5,minimum=0,label='Closer inspection cost (EUR; TP and FP)')
                review_minutes=gr.Number(value=5,minimum=0,label='Minutes per photo review')
                hourly=gr.Number(value=30,minimum=0,label='Employer staff cost (EUR/hour)')
            with gr.Row():
                sensitivity=gr.Slider(0,100,value=95,step=1,label='Assumed human sensitivity (%)')
                specificity=gr.Slider(0,100,value=95,step=1,label='Assumed human specificity (%)')
        summary=gr.Markdown()
        with gr.Row():
            triage_plot=gr.Plot(); final_plot=gr.Plot()
        gr.Markdown('The left matrix contains observed test counts. The right matrix is an expectation after assumed human decisions; fractional counts are expected values.')
        cost_plot=gr.Plot()
        price_table=gr.Dataframe(interactive=False,label='Same test split, costs and production base rate')
    with gr.Tab('Model results and dataset'):
        gr.Markdown('Detection metrics assess both original classes and box location. Business costs assess whether a photo has any Damage annotation. These are different questions.')
        gr.Markdown(f'Two real experiments changed input size only. The **{INFERENCE_SIZE}-pixel version** was selected using validation triage cost.')
        gr.Dataframe(pd.read_csv(ARTIFACTS/'experiments.csv').round(3),interactive=False,label='Image-size experiments; validation results')
        gr.Dataframe(pd.read_csv(ARTIFACTS/'baselines.csv')[['model','AP50','mAP50_95']].round(3),interactive=False,label='Test baselines')
        gr.Markdown('COCO has neither Damage nor Undamage, so it makes no predictions in the target label vocabulary. Detecting a car does not establish its condition. Its zero AP is a vocabulary limitation, not a strong same-task baseline.')
        gr.Dataframe(pd.read_csv(ARTIFACTS/'per_class.csv')[['class','precision','recall','AP50','mAP50_95']].round(3),interactive=False,label='Per-class test results; precision/recall at saved model-only threshold, IoU ≥ 0.50')
        gr.Image(str(ARTIFACTS/'detection_confusion.png'),interactive=False,label='Detection confusion; predicted rows, true columns')
        gr.Markdown(f'Dataset: [original public version 1]({AUDIT["source"]}). 250 train / 84 validation / 84 test images. Original labels and splits retained. The data includes severe crash scenes, stock-style clean-car photos and questionable labels; rental-return generalisation is unproven.')
        gr.Image(str(ARTIFACTS/'training_examples.png'),interactive=False,label='Training examples with original labels')
        if (ARTIFACTS/'failure_examples.png').exists():
            gr.Image(str(ARTIFACTS/'failure_examples.png'),interactive=False,label='Model-only errors against the original labels')
    controls=[lower,upper,prevalence,missed,inspection,review_minutes,hourly,volume,sensitivity,specificity]
    outputs=[summary,triage_plot,final_plot,cost_plot,price_table]
    for control in controls:
        control.change(update_policy,controls,outputs,api_name=False)
    demo.load(update_policy,controls,outputs,api_name='policy')
    restore.click(lambda:(LOWER,UPPER),outputs=[lower,upper],api_name=False)
    if THREE:
        compare_three.click(lambda:(THREE['validation']['lower'],THREE['validation']['upper']),outputs=[lower,upper],api_name=False)
    inspect.click(inspect_photo,[photo,lower,upper],[annotated,decision],api_name='inspect_photo')
    for control in [lower,upper]:
        control.change(inspect_photo,[photo,lower,upper],[annotated,decision],api_name=False)

if __name__=='__main__':
    demo.launch(server_name='127.0.0.1',server_port=int(os.environ.get('APP_PORT','7860')),
                share=False,inbrowser=False,show_error=True,theme=gr.themes.Soft(),
                allowed_paths=[str(ARTIFACTS)],blocked_paths=[str(ROOT/'.env'),str(ROOT/'.git')])
