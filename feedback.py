"""Local feedback agents. Verified human observations remain outside training data.

This is an auditable, tool-based feedback workflow, not a claim that an LLM can
discover the true cause of every failure. Diagnoses are hypotheses to investigate.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,uuid,argparse
import numpy as np

ROOT=Path(__file__).resolve().parent
PRIVATE=ROOT/'feedback'/'private'
LOG=PRIVATE/'verified-events.jsonl'

def add_feedback(image,predicted_score,covered,actual_damage,reviewer,low,high):
    if image is None:raise ValueError('Upload an image first.')
    if actual_damage not in ['Damage present','No visible damage']:
        raise ValueError('A verified human label is required.')
    if not str(reviewer).strip():raise ValueError('Record a reviewer identifier.')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    case=str(uuid.uuid4());path=PRIVATE/(case+'.png');image.convert('RGB').save(path)
    a=np.asarray(image.convert('RGB').resize((128,128)),dtype=float)
    grayscale=a.mean(axis=2)
    # Basic acquisition checks suggest possible causes; they do not establish causality.
    contrast=float(grayscale.std());edge=float(np.mean(np.abs(np.diff(grayscale,axis=0))))
    outcome='review' if not covered or low<=predicted_score<high else ('fail' if predicted_score>=high else 'pass')
    row={'id':case,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
         'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'private_image':path.name,
         'reviewer':str(reviewer).strip(),'verified_label':int(actual_damage=='Damage present'),
         'score':float(predicted_score),'covered':bool(covered),'original_action':outcome,
         'weights_sha256':json.loads((ROOT/'artifacts/run-manifest.json').read_text())['weights_sha256'],
         'thresholds':{'low':low,'high':high},'contrast':contrast,'edge_energy':edge,
         'eligible_for_coursework_training':False}
    with LOG.open('a') as f:f.write(json.dumps(row)+'\n')
    diagnose()
    return case

def diagnose():
    rows=[json.loads(s) for s in LOG.read_text().splitlines() if s.strip()] if LOG.exists() else []
    # Observer -> error auditor -> diagnostic agent -> repair planner. No label is inferred.
    errors=[]
    for r in rows:
        wrong=(r['original_action']=='pass' and r['verified_label']==1) or (r['original_action']=='fail' and r['verified_label']==0)
        if not wrong:continue
        hypotheses=[]
        if not r['covered']:hypotheses.append('No recognised box: investigate missed object, framing or domain mismatch.')
        if r['contrast']<20:hypotheses.append('Low contrast: test a brighter controlled capture setup.')
        if r['edge_energy']<3:hypotheses.append('Weak edge detail: check blur, resolution and distance.')
        if not hypotheses:hypotheses.append('Check unfamiliar damage type, occlusion and label consistency against the public class definitions.')
        errors.append({'id':r['id'],'error':'missed_damage' if r['verified_label'] else 'false_hold',
                       'hypotheses_not_proven_causes':hypotheses})
    summary={'verified_feedback_count':len(rows),'observed_wrong_automatic_decisions':len(errors),
             'error_cases':errors,'status':'Local diagnosis only; not a population error-rate estimate.',
             'next_actions':[
                 'Review the original image and class specification with a human; retain disagreement rather than overwrite it.',
                 'Compare error patterns with a representative independently labelled audit sample, including passed cases.',
                 'Re-evaluate thresholds on validation data with the current approved cost scenario.',
                 'For a new model, use an explicitly selected public dataset version and train a separate candidate.',
                 'Require candidate evaluation, capacity/cost checks and recorded human approval before promotion.'],
             'new_labels_added_to_training':0,'champion_weights_overwritten':False}
    folder=ROOT/'feedback';folder.mkdir(exist_ok=True)
    # Do not copy reviewer names or uploaded images into a shareable diagnostic summary.
    (folder/'diagnostic-plan.json').write_text(json.dumps(summary,indent=2))
    return summary

def promotion_gate(candidate_result,champion_result,approval):
    """Gate independently evaluated candidates; never evaluate test data repeatedly to tune."""
    needed={'evaluation_split','independent_audit_id','cost_eur_per_1000','damage_recall','review_share','capacity_share','weights_sha256'}
    if not needed<=set(candidate_result) or not needed<=set(champion_result):
        raise ValueError('Both evaluations need an independent audit record and all gate metrics.')
    if candidate_result['evaluation_split']!='fresh_independent_audit':
        raise ValueError('Promotion requires a fresh independent audit, not the coursework test split.')
    if candidate_result['independent_audit_id']!=champion_result['independent_audit_id']:
        raise ValueError('Compare champion and candidate on the same audit sample.')
    passed=(candidate_result['cost_eur_per_1000']<champion_result['cost_eur_per_1000']
            and candidate_result['damage_recall']>=champion_result['damage_recall']
            and candidate_result['review_share']<=candidate_result['capacity_share'])
    return {'metrics_passed':passed,'approved':bool(str(approval or '').strip()),
            'promote':passed and bool(str(approval or '').strip()),
            'candidate_weights_sha256':candidate_result['weights_sha256'],
            'rollback_weights_sha256':champion_result['weights_sha256']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['diagnose'])
    parser.parse_args();print(json.dumps(diagnose(),indent=2))
