"""Local Gradio review dashboard. Loads saved weights; never trains at startup."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from train import safe_weights
from decision_policy import Costs,policy_metrics
from feedback import add_feedback,diagnose

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'

def read(name):return json.loads((ART/name).read_text())

def build_demo():
    if not (ROOT/'weights/best.pt').exists() or not (ART/'chosen-policy.json').exists():
        raise RuntimeError('Real trained weights and evaluation outputs are missing. Run main.py train first; the dashboard will not invent results.')
    cfg=json.loads((ROOT/'config.json').read_text());s=cfg['scenario'];policy=read('chosen-policy.json')
    test=read('test-predictions.json');val=read('validation-predictions.json')
    det=read('detection-results.json');health=read('dataset-health.json')
    model=YOLO(safe_weights(ROOT/'weights/best.pt'));positive=set(cfg['positive_classes'])
    inference_size=read('run-manifest.json').get('inference_imgsz',cfg['model']['imgsz'])
    low=policy['review_policy']['low'];high=policy['review_policy']['high']

    def update(lo,hi,prevalence,fn,fp,hourly,seconds,hsens,hspec):
        if lo>hi:raise gr.Error('The pass threshold cannot exceed the fail threshold.')
        c=Costs(fp,fn,s['true_positive'],s['true_negative'],hourly*seconds/3600)
        kw=dict(human_sensitivity=hsens,human_specificity=hspec,daily_volume=s['daily_volume'],
                review_seconds=seconds,productive_seconds_per_person=s['productive_hours_per_person']*3600)
        y=np.array([r['y'] for r in test]);p=np.array([r['score'] for r in test]);forced=~np.array([r['covered'] for r in test])
        result=policy_metrics(y,p,lo,hi,c,prevalence,force_review=forced,**kw)
        cm=result['expected_confusion_per_1000']
        matrix=pd.DataFrame([[cm['TN'],cm['FP']],[cm['FN'],cm['TP']]],
                            index=['Actually no visible damage','Actually damaged'],columns=['Predicted pass','Predicted fail'])
        vy=np.array([r['y'] for r in val]);vp=np.array([r['score'] for r in val])
        curve=[]
        for t in np.r_[np.linspace(0,1,51),1.000001]:
            curve.append(policy_metrics(vy,vp,t,t,c,prevalence,**kw)['cost_eur_per_1000'])
        fig,ax=plt.subplots(figsize=(8,4));ax.plot(np.r_[np.linspace(0,1,51),1.000001],curve,color='#167d8d')
        ax.axvline(policy['model_only_policy']['low'],color='#b4533e',ls='--',label='Saved validation choice')
        ax.set(xlabel='Damage confidence threshold',ylabel='Expected EUR / 1,000 images',title='Model-only validation cost curve')
        ax.legend();fig.tight_layout()
        capacity=s['review_capacity_hours_day']*3600/seconds
        text=(f"**Expected cost: €{result['cost_eur_per_1000']:.2f} per 1,000 inspection images.** "
              f"Review: {result['review_share']:.1%}, {result['review_cases_per_day']:.0f} images/day, "
              f"{result['review_hours_per_day']:.2f} hours/day, {result['people_required_fte']:.2f} FTE. "
              f"Capacity: {capacity:.0f} reviews/day. "
              +('Capacity exceeded.' if result['review_cases_per_day']>capacity else 'Within assumed capacity.'))
        return text,matrix.reset_index(names='Actual / predicted'),fig

    def infer(image,lo,hi):
        if image is None:raise gr.Error('Choose an image first.')
        if lo>hi:raise gr.Error('Pass threshold must not exceed fail threshold.')
        result=model.predict(image,conf=.001,imgsz=inference_size,verbose=False)[0]
        scores=[float(v) for k,v in zip(result.boxes.cls,result.boxes.conf) if result.names[int(k)] in positive]
        score=max(scores,default=0.)
        covered=bool(any(float(v)>=.25 for v in result.boxes.conf))
        outcome='HUMAN REVIEW' if not covered or lo<=score<hi else ('HOLD FOR INSPECTION' if score>=hi else 'PASS')
        visible=model.predict(image,conf=max(.001,float(lo)),imgsz=inference_size,verbose=False)[0]
        annotated=Image.fromarray(visible.plot()[...,::-1])
        return annotated,f'**{outcome}** · Damage score {score:.3f}. '+('No recognised box at 0.25 confidence; review required.' if not covered else ''),{'score':score,'covered':covered,'low':lo,'high':hi}

    def feedback(image,state,label,reviewer,lo,hi):
        if not state:raise gr.Error('Run this image through the model before saving feedback.')
        try:case=add_feedback(image,state['score'],state['covered'],label,reviewer,state['low'],state['high'])
        except ValueError as e:raise gr.Error(str(e))
        return f'Saved verified case {case}. No labels or weights were changed.',diagnose()

    with gr.Blocks(title='Parcel Damage Inspection') as demo:
        gr.Markdown('# Parcel damage inspection\nSaved model outputs. Costs and staffing are scenario assumptions. No training occurs at startup.')
        with gr.Row():
            lo=gr.Slider(0,1.000001,value=low,step=.02,label='Pass below this damage score')
            hi=gr.Slider(0,1.000001,value=high,step=.02,label='Hold at or above this damage score')
        with gr.Accordion('Cost and staffing assumptions',open=False):
            with gr.Row():
                prev=gr.Slider(.001,.5,value=s['base_rate'],step=.001,label='Assumed damage prevalence')
                fn=gr.Number(s['false_negative'],label='Missed damage cost EUR')
                fp=gr.Number(s['false_positive'],label='Unnecessary hold cost EUR')
            with gr.Row():
                hourly=gr.Number(s['labour_eur_hour'],label='Labour EUR per effective hour')
                seconds=gr.Number(s['review_seconds'],minimum=1,label='Seconds per review')
                hsens=gr.Slider(0,1,value=s['human_sensitivity'],label='Assumed human sensitivity')
                hspec=gr.Slider(0,1,value=s['human_specificity'],label='Assumed human specificity')
        metrics=gr.Markdown();matrix=gr.Dataframe(label='Expected binary confusion matrix per 1,000 images, including assumed human errors',interactive=False)
        curve=gr.Plot()
        inputs=[lo,hi,prev,fn,fp,hourly,seconds,hsens,hspec]
        for widget in inputs:widget.change(update,inputs,[metrics,matrix,curve])
        demo.load(update,inputs,[metrics,matrix,curve])
        gr.Markdown('Thresholds were selected on validation data. Slider changes explore test sensitivity; do not use this view to retune on the test split.')
        with gr.Tab('Inspect an image'):
            image=gr.Image(type='pil',label='Local inspection image');button=gr.Button('Inspect image')
            annotated=gr.Image(type='pil',label='Detections at current display threshold');decision=gr.Markdown();state=gr.State()
            image.change(lambda:None,outputs=[state])
            button.click(infer,[image,lo,hi],[annotated,decision,state])
            gr.Markdown('Verified feedback stays local and separate from the coursework dataset.')
            actual=gr.Radio(['Damage present','No visible damage'],label='Human-verified image label')
            reviewer=gr.Textbox(label='Reviewer identifier');save=gr.Button('Save verified feedback')
            saved=gr.Markdown();diagnosis=gr.JSON(label='Feedback agents: error audit and repair plan')
            save.click(feedback,[image,state,actual,reviewer,lo,hi],[saved,diagnosis])
        with gr.Tab('Per-class results'):
            gr.Dataframe(pd.DataFrame(det['finetuned']['per_class']),interactive=False,label='Held-out detection metrics by class')
            gr.Dataframe(pd.DataFrame(det['finetuned']['confusion_matrix'],columns=det['finetuned']['matrix_labels']),interactive=False,label='Detection confusion matrix: predicted rows, true columns, background included')
            gr.JSON(det,label='Fine-tuned model and both baselines')
        with gr.Tab('Dataset and alternatives'):
            gr.Dataframe(pd.DataFrame(health['class_split_counts']),interactive=False,label='Public dataset composition')
            gr.JSON(read('test-policy-comparison.json'),label='Frozen policy versus model alone, do nothing and human review of everything')
    return demo

if __name__=='__main__':build_demo().launch(server_name='127.0.0.1',share=False)
