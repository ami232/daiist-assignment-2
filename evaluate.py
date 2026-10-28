"""Detection metrics and image-level business evaluation on their proper splits."""
import json
import shutil
import numpy as np
import pandas as pd
from project import (ROOT, ARTIFACTS, WEIGHTS, IMAGE_SIZE, DATA_URL, image_files,
                     labels_for, write_json, configure_runtime, prepare_dataset)
from policy import Scenario, choose_policy, evaluate_policy, alternatives


def cache_predictions(model, split, names, image_size=IMAGE_SIZE, save=True):
    files=image_files(split)
    damage_id=names.index('Damage')
    records=[]
    predictions=model.predict(source=[str(p) for p in files],stream=True,
                              conf=.001,iou=.7,imgsz=image_size,device='cpu',
                              batch=16,max_det=50,verbose=False)
    for p,result in zip(files,predictions):
        if p.name != ROOT.joinpath(result.path).name:
            raise ValueError('Prediction order differs from the image list; refusing to misalign labels.')
        boxes=result.boxes
        detected=[{'class':int(c),'confidence':float(s),'xyxy':b.tolist()}
                  for c,s,b in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(),
                                   boxes.xyxyn.cpu().numpy())]
        gt=labels_for(p)
        records.append({'file':p.name,'truth':gt,
                        'positive':any(int(r[0])==damage_id for r in gt),
                        'score':max((d['confidence'] for d in detected if d['class']==damage_id),default=0.),
                        'detections':detected})
    if len(records)!=len(files):
        raise ValueError('Prediction count differs from the number of input images.')
    if save:
        write_json(ARTIFACTS/f'{split}_predictions.json',records)
    return records


def box_iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection
    return float(intersection/union) if union>0 else 0.


def ground_truth_boxes(record):
    return [(int(c),[x-w/2,y-h/2,x+w/2,y+h/2]) for c,x,y,w,h in record['truth']]


def fixed_threshold_metrics(records,names,threshold):
    """Class-aware P/R with greedy confidence-ordered IoU>=0.50 matching."""
    counts=np.zeros((len(names),3),dtype=int)  # TP, FP, FN
    for rec in records:
        gt=ground_truth_boxes(rec)
        for cls in range(len(names)):
            targets=[b for c,b in gt if c==cls]
            matched=set()
            preds=sorted((d for d in rec['detections'] if d['class']==cls and d['confidence']>=threshold),
                         key=lambda d:-d['confidence'])
            for pred in preds:
                candidates=[(box_iou(pred['xyxy'],b),i) for i,b in enumerate(targets) if i not in matched]
                overlap,index=max(candidates,default=(0.,-1))
                if overlap>=.5:
                    counts[cls,0]+=1; matched.add(index)
                else:
                    counts[cls,1]+=1
            counts[cls,2]+=len(targets)-len(matched)
    return [{'class':name,'precision':float(tp/(tp+fp)) if tp+fp else 0.,
             'recall':float(tp/(tp+fn)) if tp+fn else 0.,
             'true_positives':int(tp),'false_positives':int(fp),'false_negatives':int(fn),
             'confidence_threshold':float(threshold),'matching_iou':.5}
            for name,(tp,fp,fn) in zip(names,counts)]


def detection_confusion(records,names,threshold):
    """Match geometry first so wrong-class detections appear as confusions."""
    cm=np.zeros((len(names)+1,len(names)+1),dtype=int)
    bg=len(names)
    for rec in records:
        gt=ground_truth_boxes(rec); used=set()
        for d in sorted(rec['detections'],key=lambda d:-d['confidence']):
            if d['confidence']<threshold:
                continue
            candidates=[(box_iou(d['xyxy'],b),j) for j,(_,b) in enumerate(gt) if j not in used]
            overlap,j=max(candidates,default=(0.,-1))
            if overlap>=.5:
                cm[d['class'],gt[j][0]]+=1; used.add(j)
            else:
                cm[d['class'],bg]+=1
        for j,(cls,_) in enumerate(gt):
            if j not in used:
                cm[bg,cls]+=1
    return cm


def plot_matrix(matrix,labels,title,path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5)); ax.imshow(matrix,cmap='Blues')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center')
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,
           yticklabels=labels,xlabel='True class',ylabel='Predicted class',title=title)
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def evaluate_models(data_yaml,names,image_size=IMAGE_SIZE):
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model=YOLO(WEIGHTS)
    valid=cache_predictions(model,'valid',names,image_size)
    vs=[r['score'] for r in valid]; vy=[r['positive'] for r in valid]
    choice=choose_policy(vs,vy)
    write_json(ARTIFACTS/'operating_point.json',choice)
    # Threshold choices are now frozen. The test split is opened only for evaluation.
    test=cache_predictions(model,'test',names,image_size)
    ts=[r['score'] for r in test]; ty=[r['positive'] for r in test]
    metrics=model.val(data=str(data_yaml),split='test',imgsz=image_size,device='cpu',
                      batch=16,workers=0,conf=.001,iou=.7,max_det=50,plots=True,
                      project=str(ROOT/'runs'/'evaluation'),name='finetuned',verbose=False)
    per_class=fixed_threshold_metrics(test,names,choice['single']['lower'])
    for cls,ap50,ap in zip(metrics.box.ap_class_index,metrics.box.ap50,metrics.box.ap):
        per_class[int(cls)].update(AP50=float(ap50),mAP50_95=float(ap))
    pd.DataFrame(per_class).to_csv(ARTIFACTS/'per_class.csv',index=False)
    matrix=detection_confusion(test,names,choice['single']['lower'])
    write_json(ARTIFACTS/'detection_confusion.json',{'labels':names+['background'],
        'matrix_predicted_rows_true_columns':matrix.tolist(),
        'confidence':choice['single']['lower'],'iou':.5})
    plot_matrix(matrix,names+['background'],'Detection confusion on test (IoU >= 0.50)',
                ARTIFACTS/'detection_confusion.png')
    # Run the actual unchanged COCO model. Never equate a detected car with an undamaged car.
    coco=YOLO('yolo11n.pt')
    overlap={i:names.index(name) for i,name in coco.names.items() if name.casefold() in
             [n.casefold() for n in names]}
    if overlap:
        raise ValueError('The documented no-shared-label COCO baseline assumption has changed.')
    coco_results=coco.predict(source=[str(p) for p in image_files('test')],stream=True,
                             conf=.001,iou=.7,imgsz=image_size,device='cpu',batch=16,verbose=False,max_det=50)
    coco_count=sum(len(r.boxes) for r in coco_results)
    baseline_note='COCO has neither Damage nor Undamage; no target-label predictions. A car box does not establish condition.'
    comparison=[{'model':'Finetuned YOLO11n','AP50':float(metrics.box.map50),
                 'mAP50_95':float(metrics.box.map),'note':'Both original dataset classes; frozen test split'},
                {'model':'COCO YOLO11n zero-shot','AP50':0.,'mAP50_95':0.,'note':baseline_note},
                {'model':'Predict no detections','AP50':0.,'mAP50_95':0.,'note':'No boxes predicted'}]
    pd.DataFrame(comparison).to_csv(ARTIFACTS/'baselines.csv',index=False)
    write_json(ARTIFACTS/'baseline_details.json',{'coco_raw_detections_on_test':coco_count,
        'shared_labels':overlap,'note':baseline_note,'test_images':len(test)})
    triage=evaluate_policy(ts,ty,choice['triage']['lower'],choice['triage']['upper'])
    costs=alternatives(ts,ty,choice)
    write_json(ARTIFACTS/'test_policy.json',{'triage':triage,'alternatives_cost_per_1000':costs,
        'actual_damage_prevalence':float(np.mean(ty)), 'production_prevalence_assumed':.1,
        'unit':'one dataset photo; one photo per return is only a workload proxy'})
    sweep=pd.DataFrame([{'threshold':r['lower'],'validation_cost_per_1000':r['cost_per_1000']}
                        for r in choice['curve']])
    sweep.to_csv(ARTIFACTS/'cost_sweep.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(sweep.threshold,sweep.validation_cost_per_1000,label='Validation, reweighted to 10% damage')
    ax.axvline(choice['single']['lower'],color='#b34d26',linestyle='--',label='Validation minimum')
    ax.set(xlabel='Damage confidence threshold',ylabel='Expected EUR per 1,000 photos',
           title='Model-only cost sweep; selection uses validation')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'cost_curve.png',dpi=140); plt.close(fig)
    sensitivity=[]
    for prevalence in [.02,.05,.1,.2]:
        for missed in [50.,100.,200.,400.]:
            cfg=Scenario(prevalence=prevalence,missed_cost=missed)
            new=choose_policy(vs,vy,cfg)
            result=evaluate_policy(ts,ty,choice['triage']['lower'],choice['triage']['upper'],cfg)
            sensitivity.append({'prevalence':prevalence,'missed_cost':missed,
                'validation_optimal_single':new['single']['lower'],
                'validation_optimal_lower':new['triage']['lower'],
                'validation_optimal_upper':new['triage']['upper'],
                'frozen_policy_test_cost_per_1000':result['cost_per_1000']})
    pd.DataFrame(sensitivity).to_csv(ARTIFACTS/'sensitivity.csv',index=False)
    from build_figures import build_figures
    build_figures()
    print('Test model comparison:',comparison,flush=True)
    print('Validation-selected thresholds:',choice['single']['lower'],
          choice['triage']['lower'],choice['triage']['upper'],flush=True)
    print('Test expected costs per 1,000 photos:',costs,flush=True)


if __name__=='__main__':
    configure_runtime()
    yaml_path,class_names=prepare_dataset()
    run=json.loads((ARTIFACTS/'training_run.json').read_text())
    evaluate_models(yaml_path,class_names,run['settings']['imgsz'])
