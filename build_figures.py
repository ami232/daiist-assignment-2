"""Save reproducible failure examples and business-assumption comparisons."""
import json
import numpy as np
import pandas as pd
from project import ARTIFACTS, DATA, configure_runtime
from policy import Scenario, evaluate_policy, alternatives


def build_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    choice=json.loads((ARTIFACTS/'operating_point.json').read_text())
    records=json.loads((ARTIFACTS/'test_predictions.json').read_text())
    valid=json.loads((ARTIFACTS/'valid_predictions.json').read_text())
    scores=[r['score'] for r in records]; truth=[r['positive'] for r in records]
    vs=[r['score'] for r in valid]; vy=[r['positive'] for r in valid]
    threshold=choice['single']['lower']
    damage_id=json.loads((ARTIFACTS/'dataset_audit.json').read_text())['classes'].index('Damage')
    rows=[]
    for accuracy in [.9,.95,.99,1.]:
        cfg=Scenario(human_sensitivity=accuracy,human_specificity=accuracy)
        costs=alternatives(scores,truth,choice,cfg)
        rows.append({'assumed_human_sensitivity_and_specificity':accuracy,
                     'human_review_all_cost_per_1000':costs['Human review of every photo'],
                     'fixed_hybrid_cost_per_1000':costs['Model with human review']})
    pd.DataFrame(rows).to_csv(ARTIFACTS/'human_accuracy_sensitivity.csv',index=False)
    # Separate the unconstrained cost minimum from an option that actually uses all three outcomes.
    best=None
    for lower in np.linspace(.01,.99,99):
        for upper in np.linspace(.01,1,100):
            if lower>=upper:
                continue
            result=evaluate_policy(vs,vy,lower,upper)
            if (np.asarray(result['actual_triage_counts']).sum(axis=0)>0).all():
                if best is None or result['cost_per_1000']<best['cost_per_1000']:
                    best=result
    if best:
        result=evaluate_policy(scores,truth,best['lower'],best['upper'])
        (ARTIFACTS/'three_outcome_option.json').write_text(json.dumps({
            'selection_split':'valid','constraint':'All three outcomes used on validation',
            'validation':best,'test':result},indent=2)+'\n')
    errors=[]
    for positive,label in [(False,'False alarm'),(True,'Missed damaged photo')]:
        subset=[r for r in records if r['positive']==positive and (r['score']>=threshold)!=positive]
        subset.sort(key=lambda r:r['score'],reverse=not positive)
        errors.extend([(label,r) for r in subset[:2]])
    if errors:
        fig,axes=plt.subplots(2,2,figsize=(11,9))
        for ax,(label,record) in zip(axes.flat,errors):
            image=Image.open(DATA/'test'/'images'/record['file'])
            width,height=image.size
            ax.imshow(image)
            for c,x,y,w,h in record['truth']:
                ax.add_patch(plt.Rectangle(((x-w/2)*width,(y-h/2)*height),w*width,h*height,
                                           fill=False,color='#2a9d6f',linewidth=2,linestyle='--'))
            damage=sorted((d for d in record['detections'] if d['class']==damage_id and d['confidence']>=threshold),
                          key=lambda d:-d['confidence'])[:3]
            for detection in damage:
                x1,y1,x2,y2=detection['xyxy']
                ax.add_patch(plt.Rectangle((x1*width,y1*height),(x2-x1)*width,(y2-y1)*height,
                                           fill=False,color='#d34d33',linewidth=2))
            ax.set_title(f'{label} | damage score {record["score"]:.3f}'); ax.axis('off')
        for ax in axes.flat[len(errors):]:
            ax.axis('off')
        fig.suptitle('Model-only errors against provided labels: green = original box; red = predicted Damage\nLabels themselves may be imperfect. These are not measured human-review outcomes.')
        fig.tight_layout(); fig.savefig(ARTIFACTS/'failure_examples.png',dpi=140); plt.close(fig)
    summary=json.loads((ARTIFACTS/'test_policy.json').read_text())
    fig,ax=plt.subplots(figsize=(7,3.5))
    counts=np.asarray(summary['triage']['actual_triage_counts'])
    ax.imshow(counts,cmap='Blues',aspect='auto')
    for row in range(2):
        for col in range(3):
            ax.text(col,row,str(counts[row,col]),ha='center',va='center')
    ax.set(xticks=[0,1,2],xticklabels=['Pass','Human review','Fail / hold'],
           yticks=[0,1],yticklabels=['No Damage label','Damage label'],
           title='Observed triage counts on the 84 test photos')
    fig.tight_layout();fig.savefig(ARTIFACTS/'triage_counts.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    configure_runtime()
    build_figures()
