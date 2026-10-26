"""Summarise real saved measurements without inventing student conclusions."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent

def main():
    art=ROOT/'artifacts'
    read=lambda name:json.loads((art/name).read_text())
    health=read('dataset-health.json');det=read('detection-results.json')
    policy=read('chosen-policy.json');comparisons=read('test-policy-comparison.json')
    feedback=read('offline-feedback-cycle.json');manifest=read('run-manifest.json')
    parts=['# Measured parcel inspection results','',
           'Automatically summarised from real saved outputs. Report,  No synthetic values are used as model results.','',
           '## Dataset',f"Public source: {manifest['dataset']['url']}",
           f"Split image counts: {health['split_images']}",
           f"Exact cross-split duplicate findings: {len(health['exact_cross_split_duplicates'])}.",'']
    for row in health['class_split_counts']:
        parts.append(f"- {row['split']} / {row['class']}: {row['instances']} instances; median relative box area {row['median_box_area_fraction']}.")
    parts.extend(['','## Same-test detection comparison',''])
    for name,row in det.items():
        parts.append(f"- {name}: AP50 {row['AP50']:.4f}; mAP50–95 {row['mAP50_95']:.4f}.")
    parts.extend(['','## Fine-tuned per-class measurements',''])
    for row in det['finetuned']['per_class']:parts.append('- '+json.dumps(row))
    parts.extend(['','## Bounded repair cycle',feedback['status'],
                  f"Candidate accepted: {feedback['candidate_accepted']}. Held-out test inspected during repair: {feedback['test_inspected']}.",
                  '', '## Validation-selected policy',json.dumps(policy,indent=2),'',
                  '## Frozen test policy comparisons',''])
    for name,row in comparisons.items():
        parts.append(f"- {name}: €{row['cost_eur_per_1000']:.2f} per 1,000 images; {row['review_share']:.2%} reviewed; {row['review_cases_per_day']:.1f} reviews/day; {row['people_required_fte']:.3f} FTE.")
    parts.extend(['','## Interpretation boundaries','',
        'All business costs and review rates depend on the scenario assumptions. Image-level risk scores are not automatically calibrated probabilities. Cost comparisons are expected scenario calculations rather than observed warehouse profit.',
        'The test set is small and public. Exact duplicates, class imbalance, multiple boxes in one image and a different production camera environment can limit generalisation. The dataset health and per-class results must be considered before any shipping decision.',
        'No operational self-improvement or new-damage recognition is claimed without independently verified feedback and a fresh candidate evaluation. The saved feedback cycle records whether this particular repair attempt succeeded.'])
    (ROOT/'MEASURED_RESULTS.md').write_text('\n'.join(parts)+'\n')

if __name__=='__main__':main()
