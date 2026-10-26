"""Download the public version unchanged, train YOLO11n and save real evaluations."""
from pathlib import Path
import contextlib,io,json,shutil,hashlib,os
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import torch
from ultralytics import YOLO
from roboflow import Roboflow
from api_key import load_api_key
from decision_policy import Costs,policy_metrics,choose_on_validation

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
RUNS=Path(tempfile.gettempdir())/('jimena-parcel-'+hashlib.sha256(str(ROOT).encode()).hexdigest()[:12])

def safe_weights(path):
    # Ultralytics strips apostrophes from checkpoint strings; stage a local copy.
    RUNS.mkdir(parents=True,exist_ok=True)
    target=RUNS/Path(path).name
    if Path(path).resolve()!=target.resolve(): shutil.copy2(path,target)
    return str(target)
EXT={'.jpg','.jpeg','.png','.bmp','.webp'}

def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))

def load_config():return json.loads((ROOT/'config.json').read_text())

def get_device(config):
    if config['model']['device']!='auto':return config['model']['device']
    return 'mps' if torch.backends.mps.is_available() else ('0' if torch.cuda.is_available() else 'cpu')

def dataset(config):
    location=ROOT/'data'/'public-parcels'
    if not (location/'data.yaml').exists():
        key=load_api_key()
        # The SDK can print signed download URLs. Keep them out of logs/artifacts.
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                rf=Roboflow(api_key=key)
                info=config['dataset']
                rf.workspace(info['workspace']).project(info['project']).version(info['version']).download(
                    info['format'],location=str(location),overwrite=False)
        except Exception:
            raise RuntimeError('Roboflow dataset download failed. Check the private key, project access and network; no credential was logged.') from None
    raw=yaml.safe_load((location/'data.yaml').read_text())
    names=raw['names'];names=list(names.values()) if isinstance(names,dict) else names
    if len(names)<2:raise ValueError('Detection assignment requires at least two classes.')
    # Resolve only the supplied split directories; do not reshuffle, relabel or filter images.
    splits={}
    for split,folder in [('train','train'),('val','valid'),('test','test')]:
        path=location/folder/'images'
        if not path.is_dir():raise ValueError(f'Missing public {split} image split.')
        splits[split]=sorted(p for p in path.iterdir() if p.suffix.lower() in EXT)
        if not splits[split]:raise ValueError(f'Empty public {split} split.')
    runtime={**raw,'path':str(location),'train':str(location/'train/images'),
             'val':str(location/'valid/images'),'test':str(location/'test/images'),'names':names}
    run_yaml=ART/'runtime-data.yaml'
    run_yaml.write_text(yaml.safe_dump(runtime))
    return location,names,splits,run_yaml

def label_rows(image):
    p=image.parent.parent/'labels'/(image.stem+'.txt')
    if not p.exists():return []
    return [list(map(float,line.split())) for line in p.read_text().splitlines() if line.strip()]

def describe(names,splits):
    rows=[];hashes={};cross_duplicates=[]
    for split,paths in splits.items():
        count=np.zeros(len(names),dtype=int);areas=[[] for _ in names];multi=0
        for image in paths:
            labels=label_rows(image);multi+=len(labels)>1
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest]!=split:cross_duplicates.append([hashes[digest],split,image.name])
            hashes[digest]=split
            for row in labels:
                cls=int(row[0]);count[cls]+=1;areas[cls].append(row[3]*row[4])
        for cls,name in enumerate(names):
            rows.append({'split':split,'class':name,'images':len(paths),'instances':int(count[cls]),
                         'median_box_area_fraction':float(np.median(areas[cls])) if areas[cls] else None,
                         'images_with_multiple_boxes':int(multi)})
    result={'class_split_counts':rows,'exact_cross_split_duplicates':cross_duplicates,
            'split_images':{s:len(v) for s,v in splits.items()},'names':names}
    save_json(ART/'dataset-health.json',result)
    pd.DataFrame(rows).to_csv(ART/'dataset-health.csv',index=False)
    return result

def predictions(model,paths,names,positive,device,imgsz,*,coco=False):
    rows=[]
    # A COCO checkpoint does not contain box-damage classes. Map by exact name,
    # never by class index; class zero in COCO is person, not the dataset's box.
    for path,r in zip(paths,model.predict(source=[str(p) for p in paths],stream=True,conf=.001,
                                         imgsz=imgsz,device=device,verbose=False)):
        boxes=[]
        for cls,conf,xyxy in zip(r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.xyxy.cpu().tolist()):
            name=r.names[int(cls)]
            if name in names:boxes.append({'class':name,'confidence':float(conf),'xyxy':xyxy})
        risk=max([b['confidence'] for b in boxes if b['class'] in positive],default=0.)
        covered=any(b['confidence']>=.25 for b in boxes)
        truth=any(names[int(row[0])] in positive for row in label_rows(path))
        rows.append({'image':path.name,'relative_path':str(path.relative_to(ROOT)),
                     'y':int(truth),'score':risk,'covered':covered,'detections':boxes})
    return rows

def policy_kwargs(cfg):
    s=cfg['scenario']
    return dict(human_sensitivity=s['human_sensitivity'],human_specificity=s['human_specificity'],
                daily_volume=s['daily_volume'],review_seconds=s['review_seconds'],
                productive_seconds_per_person=s['productive_hours_per_person']*3600)

def costs(cfg):
    s=cfg['scenario']
    return Costs(s['false_positive'],s['false_negative'],s['true_positive'],s['true_negative'],
                 s['labour_eur_hour']*s['review_seconds']/3600)

def choose_model_only(y,scores,c,prevalence,**kw):
    sweep=[policy_metrics(y,scores,t,t,c,prevalence,**kw) for t in np.r_[np.linspace(0,1,51),1.000001]]
    return min(sweep,key=lambda r:(r['cost_eur_per_1000'],r['low'])),sweep

def detection_metrics(model,data,split,device,imgsz,name,names):
    m=model.val(data=str(data),split=split,device=device,imgsz=imgsz,conf=.001,plots=True,
                project=str(RUNS),name=name,exist_ok=True,verbose=False)
    per=[]
    indices=list(m.box.ap_class_index)
    for i,cls in enumerate(indices):
        p,r,ap50,ap=m.box.class_result(i)
        per.append({'class':names[int(cls)],'precision_max_f1':float(p),'recall_max_f1':float(r),
                    'AP50':float(ap50),'mAP50_95':float(ap)})
    # Missing classes are explicit, not silently assigned the overall score.
    for cls in names:
        if not any(row['class']==cls for row in per):
            per.append({'class':cls,'precision_max_f1':None,'recall_max_f1':None,'AP50':None,'mAP50_95':None})
    matrix=m.confusion_matrix.matrix.tolist()
    return {'AP50':float(m.box.map50),'mAP50_95':float(m.box.map),'per_class':per,
            'confusion_matrix':matrix,'matrix_labels':names+['background'],
            'precision_recall_note':'Ultralytics summary precision/recall use the maximum-F1 point; the matrix uses the documented validation matching settings.'}

def offline_feedback_repair(model,val,cfg,names,splits,data,device):
    """One bounded agent cycle using public validation labels, never test labels.

    Observer finds actual misses; critic records the failure; planner tries a
    higher-resolution candidate; evaluator keeps it only if cost improves
    without reducing damage recall. Failure to improve is retained as a result.
    This is local model selection, not approval for real-world deployment.
    """
    s=cfg['scenario'];kw=policy_kwargs(cfg);c=costs(cfg)
    y=np.array([r['y'] for r in val]);scores=np.array([r['score'] for r in val])
    chosen,_=choose_model_only(y,scores,c,s['base_rate'],**kw)
    threshold=chosen['low']
    missed=[r['image'] for r in val if r['y']==1 and r['score']<threshold]
    false_holds=[r['image'] for r in val if r['y']==0 and r['score']>=threshold]
    log={'evidence_split':'validation','test_inspected':False,'new_feedback_added_to_training':False,
         'agents':['validation observer','error critic','bounded repair planner','candidate trainer','cost and recall evaluator'],
         'missed_damage_images':missed,'false_hold_images':false_holds,'champion_before':chosen,
         'repair_hypothesis':'Test whether higher resolution and additional fine-tuning improve damage evidence; this is not a proven causal explanation.',
         'candidate_accepted':False,'deployment_approved':False}
    if not cfg['feedback'].get('offline_auto_repair',False) or (not missed and not false_holds):
        log['status']='No bounded repair triggered.'
        save_json(ART/'offline-feedback-cycle.json',log)
        return model,val,cfg['model']['imgsz']
    if min(np.sum(y==0),np.sum(y==1))<5:
        log['status']='Insufficient validation examples per binary class for an automatic repair comparison.'
        save_json(ART/'offline-feedback-cycle.json',log)
        return model,val,cfg['model']['imgsz']
    candidate=YOLO(safe_weights(ROOT/'weights/best.pt'))
    size=max(512,cfg['model']['imgsz'])
    print('Feedback agents: testing a higher-resolution repair candidate on validation evidence.',flush=True)
    candidate.train(data=str(data),device=device,workers=0,imgsz=size,
        epochs=cfg['feedback'].get('repair_epochs',10),batch=cfg['model']['batch'],seed=cfg['model']['seed'],
        patience=5,project=str(RUNS),name='feedback-candidate',exist_ok=True,verbose=False)
    candidate_path=ROOT/'weights/candidate.pt';shutil.copy2(candidate.trainer.best,candidate_path)
    candidate=YOLO(safe_weights(candidate_path))
    candidate_val=predictions(candidate,splits['val'],names,cfg['positive_classes'],device,size)
    cp=np.array([r['score'] for r in candidate_val])
    candidate_choice,_=choose_model_only(y,cp,c,s['base_rate'],**kw)
    log['candidate']=candidate_choice
    log['candidate_settings']={'imgsz':size,'additional_epochs':cfg['feedback'].get('repair_epochs',10)}
    old_recall=chosen['sensitivity'] or 0
    new_recall=candidate_choice['sensitivity'] or 0
    accept=(candidate_choice['cost_eur_per_1000']<chosen['cost_eur_per_1000']-1e-9
            and new_recall>=old_recall)
    if accept:
        shutil.copy2(ROOT/'weights/best.pt',ROOT/'weights/rollback.pt')
        shutil.copy2(candidate_path,ROOT/'weights/best.pt')
        log['candidate_accepted']=True
        log['status']='Selected candidate for the coursework pipeline using validation only. No operational deployment.'
        save_json(ART/'offline-feedback-cycle.json',log)
        return candidate,candidate_val,size
    log['status']='Candidate did not pass the cost/recall gate; retained the original model.'
    save_json(ART/'offline-feedback-cycle.json',log)
    return model,val,cfg['model']['imgsz']

def main():
    os.chdir(ROOT);ART.mkdir(exist_ok=True)
    cfg=load_config();device=get_device(cfg)
    print('Device:',device,flush=True)
    location,names,splits,data=dataset(cfg)
    if not set(cfg['positive_classes'])<=set(names):raise ValueError('Positive classes absent from dataset.')
    health=describe(names,splits)
    print('Public split sizes:',health['split_images'],flush=True)
    pretrained=YOLO(cfg['model']['checkpoint'])
    settings={k:v for k,v in cfg['model'].items() if k not in ['checkpoint','device']}
    weights=ROOT/'weights';weights.mkdir(exist_ok=True)
    recovery=ROOT/'runs/parcel-training/weights/best.pt'
    if os.environ.get('PARCEL_RECOVER_COMPLETED_TRAINING')=='1' and recovery.exists():
        print('Recovering checkpoint from completed 27-epoch early-stopped run; rerunning evaluation.',flush=True)
        shutil.copy2(recovery,weights/'best.pt')
    else:
        pretrained.train(data=str(data),device=device,workers=0,project=str(RUNS),name='parcel-training',
                         exist_ok=True,verbose=False,**settings)
        shutil.copy2(pretrained.trainer.best,weights/'best.pt')
    model=YOLO(safe_weights(weights/'best.pt'))
    pos=cfg['positive_classes'];imgsz=cfg['model']['imgsz']
    val=predictions(model,splits['val'],names,pos,device,imgsz)
    model,val,imgsz=offline_feedback_repair(model,val,cfg,names,splits,data,device)
    # Held-out test predictions are produced only after the bounded validation cycle.
    test=predictions(model,splits['test'],names,pos,device,imgsz)
    save_json(ART/'validation-predictions.json',val);save_json(ART/'test-predictions.json',test)
    y=np.array([r['y'] for r in val]);score=np.array([r['score'] for r in val])
    forced=~np.array([r['covered'] for r in val])
    s=cfg['scenario'];c=costs(cfg);kw=policy_kwargs(cfg)
    cap=min(1,s['review_capacity_hours_day']*3600/s['review_seconds']/s['daily_volume'])
    capacity_feasible=True
    try:
        best,grid=choose_on_validation(y,score,c,s['base_rate'],force_review=forced,max_review_share=cap,**kw)
    except ValueError as e:
        if str(e)!='No policy satisfies the supplied constraints.':raise
        capacity_feasible=False
        best,grid=choose_on_validation(y,score,c,s['base_rate'],force_review=forced,**kw)
        print('No review policy fits the assumed staffing capacity; report this failure explicitly.',flush=True)
    save_json(ART/'validation-policy-grid.json',grid)
    solo,sweep=choose_model_only(y,score,c,s['base_rate'],**kw)
    save_json(ART/'threshold-sweep.json',sweep)
    save_json(ART/'chosen-policy.json',{'review_policy':best,'model_only_policy':solo,
        'selected_on':'validation only','capacity_share':cap,'capacity_feasible':capacity_feasible,'force_review':'no recognised target box at confidence >= 0.25'})
    ty=np.array([r['y'] for r in test]);ts=np.array([r['score'] for r in test])
    tf=~np.array([r['covered'] for r in test])
    frozen=policy_metrics(ty,ts,best['low'],best['high'],c,s['base_rate'],force_review=tf,**kw)
    alternatives={
        'model_with_review':frozen,
        'model_alone':policy_metrics(ty,ts,solo['low'],solo['high'],c,s['base_rate'],**kw),
        'do_nothing_pass_all':policy_metrics(ty,ts,1.000001,1.000001,c,s['base_rate'],**kw),
        'human_check_everything':policy_metrics(ty,ts,0,1.000001,c,s['base_rate'],**kw)}
    save_json(ART/'test-policy-comparison.json',alternatives)
    det=detection_metrics(model,data,'test',device,imgsz,'parcel-test',names)
    coco=YOLO(cfg['model']['checkpoint'])
    coco_rows=predictions(coco,splits['test'],names,pos,device,imgsz,coco=True)
    save_json(ART/'coco-test-predictions.json',coco_rows)
    # No custom target class is present in COCO's vocabulary; the exact-name zero-shot
    # baseline returns no target detections. Do not relabel people or suitcases as damage.
    mapping=set(coco.names.values())&set(names)
    if mapping:raise ValueError('A COCO class mapping exists; implement its detection evaluation explicitly.')
    missed_counts=[sum(int(row[0])==idx for image in splits['test'] for row in label_rows(image)) for idx in range(len(names))]
    empty_matrix=[[0]*(len(names)+1) for _ in range(len(names)+1)]
    empty_matrix[-1][:-1]=missed_counts
    zero={'confusion_matrix':empty_matrix,'matrix_labels':names+['background'],'AP50':0.,'mAP50_95':0.,'per_class':[{'class':n,'precision_max_f1':0.,'recall_max_f1':0.,'AP50':0.,'mAP50_95':0.} for n in names]}
    baseline={'finetuned':det,'coco_zero_shot':{**zero,'note':'No exact target class names in COCO vocabulary; model was executed and predictions mapped by name.'},
              'predict_nothing':{**zero,'note':'No detections. Detection precision is reported as zero by convention.'}}
    save_json(ART/'detection-results.json',baseline)
    pd.DataFrame(det['per_class']).to_csv(ART/'per-class-results.csv',index=False)
    plt.figure(figsize=(9,5));plt.plot([r['low'] for r in sweep],[r['cost_eur_per_1000'] for r in sweep],color='#167d8d')
    plt.axvline(solo['low'],color='#b4533e',ls='--',label=f"Validation choice {solo['low']:.2f}")
    plt.xlabel('Damage confidence threshold');plt.ylabel('Expected EUR per 1,000 inspection images')
    plt.title('Validation cost sweep — illustrative production scenario');plt.legend();plt.tight_layout()
    plt.savefig(ART/'cost-threshold.png',dpi=150);plt.close()
    sensitivities=[]
    for prevalence in [.01,.05,.1,.2]:
        for fn_cost in [10,20,40,80]:
            cc=Costs(c.false_positive,fn_cost,c.true_positive,c.true_negative,c.review_per_case)
            r=policy_metrics(ty,ts,best['low'],best['high'],cc,prevalence,force_review=tf,**kw)
            sensitivities.append({'prevalence':prevalence,'false_negative_eur':fn_cost,**r})
    save_json(ART/'sensitivity-frozen-policy.json',sensitivities)
    save_json(ART/'run-manifest.json',{'dataset':cfg['dataset'],'names':names,'device':device,'model':cfg['model'],
        'torch':torch.__version__,'inference_imgsz':imgsz,
        'recovered_original_checkpoint':os.environ.get('PARCEL_RECOVER_COMPLETED_TRAINING')=='1',
        'original_completed_epochs':len(pd.read_csv(ROOT/'runs/parcel-training/results.csv')) if os.environ.get('PARCEL_RECOVER_COMPLETED_TRAINING')=='1' else len(pd.read_csv(pretrained.trainer.save_dir/'results.csv')),'weights_sha256':hashlib.sha256((weights/'best.pt').read_bytes()).hexdigest(),
        'test_used_for_selection':False,'new_feedback_used_for_training':False})
    from report_results import main as summarise_results
    summarise_results()
    print('Training and evaluation complete; outputs saved in artifacts/. Nothing submitted.',flush=True)

if __name__=='__main__':main()
