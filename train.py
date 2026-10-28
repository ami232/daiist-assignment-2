"""Compare two image sizes on validation, then evaluate the selected model."""
import os
import platform
import shutil
import time
from pathlib import Path
import pandas as pd
from project import (ROOT, ARTIFACTS, WEIGHTS, EPOCHS, SEED,
                     configure_runtime, prepare_dataset, write_json)


def main():
    configure_runtime()
    data_yaml,names=prepare_dataset()
    from audit import audit_dataset
    audit_dataset(names)
    from ultralytics import YOLO
    from evaluate import cache_predictions,evaluate_models
    from policy import choose_policy
    epochs=int(os.environ.get('TRAIN_EPOCHS',EPOCHS))
    sizes=[320,480]
    candidates=[]
    started=time.time()
    for size in sizes:
        settings=dict(data=str(data_yaml),epochs=epochs,imgsz=size,
                      batch=16,device='cpu',workers=0,seed=SEED,deterministic=True,
                      optimizer='AdamW',lr0=.001,patience=12,cache=True,
                      mosaic=.0,close_mosaic=0,fliplr=.5,
                      project=str(ROOT/'runs'/'training'),
                      name=f'rental-car-damage-{size}',plots=True,verbose=False,save=True)
        # Optional continuation of the already completed first experiment.
        # A normal clean-machine run trains both candidates from the same COCO weights.
        previous=ROOT/'runs'/'training'/'rental-car-damage'
        reuse=(size==320 and os.environ.get('TRAIN_REUSE_320')=='1'
               and (previous/'weights'/'best.pt').exists())
        if reuse:
            checkpoint=previous/'weights'/'best.pt'
            save_dir=previous
            print('Reusing the completed 320-pixel experiment for this continuation.',flush=True)
        else:
            print(f'Finetuning YOLO11n at {size} pixels from the same COCO initialization ...',flush=True)
            model=YOLO('yolo11n.pt')
            model.train(**settings)
            checkpoint=model.trainer.best
            save_dir=model.trainer.save_dir
        candidate=YOLO(checkpoint)
        records=cache_predictions(candidate,'valid',names,size,save=False)
        choice=choose_policy([r['score'] for r in records],[r['positive'] for r in records])
        history=pd.read_csv(save_dir/'results.csv')
        best_idx=history['metrics/mAP50-95(B)'].idxmax()
        row={'image_size':size,'epochs_completed':len(history),
             'best_validation_AP50':float(history.loc[best_idx,'metrics/mAP50(B)']),
             'best_validation_mAP50_95':float(history.loc[best_idx,'metrics/mAP50-95(B)']),
             'validation_triage_cost_per_1000':choice['triage']['cost_per_1000'],
             'validation_lower':choice['triage']['lower'],'validation_upper':choice['triage']['upper'],
             'checkpoint':str(checkpoint),'save_dir':str(save_dir),'settings':settings}
        candidates.append(row)
        write_json(ARTIFACTS/'experiment_progress.json',candidates)
        print('Candidate validation results:',{k:v for k,v in row.items()
              if k not in ['checkpoint','save_dir','settings']},flush=True)
    selected=min(candidates,key=lambda r:(r['validation_triage_cost_per_1000'],r['image_size']))
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy2(selected['checkpoint'],WEIGHTS)
    shutil.copy2(Path(selected['save_dir'])/'results.csv',ARTIFACTS/'training_history.csv')
    pd.DataFrame([{k:v for k,v in c.items() if k not in ['checkpoint','save_dir','settings']}
                  for c in candidates]).to_csv(ARTIFACTS/'experiments.csv',index=False)
    write_json(ARTIFACTS/'training_run.json',{
        'model':'yolo11n.pt','initialization':'COCO pretrained','task':'object detection',
        'settings':selected['settings'],'architecture':platform.machine(),
        'training_seconds_this_invocation':time.time()-started,
        'epochs_completed':selected['epochs_completed'],
        'best_weight_selection':'Highest validation mAP50-95 within each training run',
        'model_selection':'Lowest validation triage cost across image sizes; test is not used',
        'best_weights':'weights/best.pt',
        'development_note':'Initial 320-pixel test results were inspected before deciding to try 480 pixels. The final test is therefore not completely blind.'})
    evaluate_models(data_yaml,names,selected['image_size'])
    print('Training and evaluation finished. Run: uv run python main.py app',flush=True)


if __name__=='__main__':
    main()
