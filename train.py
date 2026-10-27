"""Download, prepare, train, evaluate, and export. Run: python main.py train.

On Colab use its GPU-enabled PyTorch and pip-installed dependencies, not uv sync.
Local runs default to CPU if CUDA is unavailable. No input prompts are used.
"""
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.runtime'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.runtime/matplotlib'))
(ROOT/'.runtime').mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from roboflow import Roboflow
from api_key import load_api_key
from dataset_utils import ANOMALY_NAMES, EXTENSIONS, parse_label, prepare_dataset
from decision import comparisons, evaluate, optimize


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def per_class(metrics, names, counts):
    rows = []
    lookup = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    for class_id, name in names.items():
        row = {'Class': name, 'Test instances': counts[class_id],
               'Precision': np.nan, 'Recall': np.nan, 'AP50': np.nan, 'AP50–95': np.nan}
        if counts[class_id] and class_id in lookup:
            i = lookup[class_id]
            row.update({'Precision': float(metrics.box.p[i]), 'Recall': float(metrics.box.r[i]),
                        'AP50': float(metrics.box.ap50[i]), 'AP50–95': float(metrics.box.ap[i])})
        rows.append(row)
    return pd.DataFrame(rows)


def score_images(model, folder, names, device, output, cfg):
    anomaly_ids = {i for i, name in model.names.items() if name in ANOMALY_NAMES}
    true_ids = {i for i, name in names.items() if name in ANOMALY_NAMES}
    images = sorted(p for p in (folder/'images').rglob('*') if p.suffix.lower() in EXTENSIONS)
    rows = []
    for index, image in enumerate(images, 1):
        relative = image.relative_to(folder/'images')
        labels = parse_label((folder/'labels'/relative).with_suffix('.txt'), names)
        result = model.predict(str(image), device=device, imgsz=cfg['imgsz'], batch=1,
                               conf=.001, iou=.7, max_det=300, save=False, verbose=False)[0]
        score = max((float(b.conf.item()) for b in result.boxes if int(b.cls.item()) in anomaly_ids), default=0.0)
        rows.append({'image': str(relative), 'positive': int(any(a[0] in true_ids for a in labels)), 'anomaly_score': score})
        if index % 200 == 0:
            print(f'{folder.name}: scored {index}/{len(images)}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def main():
    cfg = json.loads((ROOT/'pipeline_config.json').read_text())
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    run = ROOT/'runs'/f'pipeline_{timestamp}'
    stage = run/'artifacts'
    stage.mkdir(parents=True)
    device = os.environ.get('TRAIN_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
    device = int(device) if device.isdigit() else device
    print(f'Training device: {device}; epochs: {cfg["epochs"]}', flush=True)
    rf = Roboflow(api_key=load_api_key())
    dataset = rf.workspace(cfg['workspace']).project(cfg['project']).version(cfg['version']).download(
        'yolov8', location=str(ROOT/'datasets'/f'raw_{timestamp}'))
    prepared = ROOT/'datasets'/f'detection_{timestamp}'
    yaml_path, names, duplicates = prepare_dataset(Path(dataset.location), prepared, stage/'dataset_statistics')
    write_json(stage/'dataset_statistics/duplicate_groups.json', duplicates)
    write_json(stage/'provenance.json', {'config': cfg, 'device': str(device), 'timestamp_utc': timestamp,
                                      'torch': torch.__version__, 'source': 'Roboflow public version'})
    model = YOLO(cfg['model'])
    model.train(data=str(yaml_path), epochs=cfg['epochs'], imgsz=cfg['imgsz'], batch=cfg['batch'],
                seed=cfg['seed'], patience=cfg['patience'], device=device, workers=2,
                project=str(run), name='training', save=True, plots=True)
    training_dir = Path(model.trainer.save_dir)
    best = training_dir/'weights/best.pt'
    assert best.is_file()
    (stage/'weights').mkdir()
    shutil.copy2(best, stage/'weights/best.pt')
    (stage/'training').mkdir()
    for path in training_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, stage/'training'/path.name)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trained = YOLO(str(best))
    evaluation = stage/'evaluation'
    metrics = trained.val(data=str(yaml_path), split='test', imgsz=cfg['imgsz'], batch=cfg['batch'],
                          device=device, conf=.001, iou=.7, max_det=300, plots=True,
                          project=str(evaluation), name='finetuned_test')
    write_json(Path(metrics.save_dir)/'overall_metrics.json', {k: float(v) for k, v in metrics.results_dict.items()})
    counts = {i: 0 for i in names}
    for label in (prepared/'test/labels').rglob('*.txt'):
        for annotation in parse_label(label, names):
            counts[annotation[0]] += 1
    fine = per_class(metrics, names, counts)
    fine.to_csv(Path(metrics.save_dir)/'per_class_metrics.csv', index=False)
    trained.val(data=str(yaml_path), split='test', imgsz=cfg['imgsz'], batch=cfg['batch'],
                device=device, conf=.25, iou=.7, max_det=300, plots=True,
                project=str(evaluation), name='finetuned_test_conf025')
    decision_dir = stage/'decision_analysis'
    decision_dir.mkdir()
    validation = score_images(trained, prepared/'valid', names, device, decision_dir/'validation_scores.csv', cfg)
    # Select both hybrid and model-alone thresholds before reading test scores.
    policy = optimize(validation, cfg['assumptions'])
    write_json(decision_dir/'frozen_policy.json', policy)
    write_json(decision_dir/'assumptions.json', cfg['assumptions'])
    test = score_images(trained, prepared/'test', names, device, decision_dir/'test_scores.csv', cfg)
    comparisons(validation, cfg['assumptions'], policy).to_csv(decision_dir/'three_way_policy_comparison.csv', index=False)
    comparisons(test, cfg['assumptions'], policy).to_csv(decision_dir/'test_policy_comparison.csv', index=False)
    sensitivity = []
    for prevalence in [.05, .10, .20]:
        for missed_cost in [1., 5., 20.]:
            for review_seconds in [15, 30, 60]:
                scenario = dict(cfg['assumptions'], production_prevalence=prevalence,
                                missed_anomaly_cost_eur=missed_cost, review_seconds=review_seconds)
                selected_policy = optimize(validation, scenario)
                r = evaluate(validation, selected_policy['lower_threshold'], selected_policy['upper_threshold'], scenario)
                sensitivity.append({'prevalence': prevalence, 'missed_cost_eur': missed_cost,
                                    'review_seconds': review_seconds, 'lower_threshold': selected_policy['lower_threshold'],
                                    'upper_threshold': selected_policy['upper_threshold'],
                                    'cost_per_1000_eur': r['cost_per_1000'], 'review_share': r['review_share']})
    pd.DataFrame(sensitivity).to_csv(decision_dir/'hybrid_sensitivity.csv', index=False)
    # Export a cost curve with the chosen upper threshold held fixed.
    curve_rows = []
    for lower in np.unique(np.r_[np.linspace(0, policy['upper_threshold'], 201), policy['lower_threshold']]):
        r = evaluate(validation, lower, policy['upper_threshold'], cfg['assumptions'])
        curve_rows.append({'lower_threshold': lower, 'cost_per_1000_eur': r['cost_per_1000'], 'review_share': r['review_share']})
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(decision_dir/'three_way_cost_curve.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(curve.lower_threshold, curve.cost_per_1000_eur)
    selected = evaluate(validation, policy['lower_threshold'], policy['upper_threshold'], cfg['assumptions'])
    ax.scatter([policy['lower_threshold']], [selected['cost_per_1000']], color='red')
    ax.set(xlabel='Lower threshold (upper fixed)', ylabel='Cost / 1,000 (€)', title='Validation cost curve')
    fig.tight_layout()
    fig.savefig(decision_dir/'three_way_cost_curve.png', dpi=160)
    plt.close(fig)
    # Exclude test images that exactly match any train image for a supplementary check.
    leaked = {item['image'] for group in duplicates if any(i['split'] == 'train' for i in group)
              for item in group if item['split'] == 'test'}
    test_results = []
    for title, frame in [('original_test', test), ('test_without_known_duplicate', test.loc[~test.image.isin(leaked)])]:
        r = evaluate(frame, policy['lower_threshold'], policy['upper_threshold'], cfg['assumptions'])
        pd.DataFrame(r['counts'], index=['No annotated anomaly', 'Annotated anomaly'],
                     columns=['Pass', 'Human review', 'Automatic incident']).to_csv(decision_dir/f'{title}_action_counts.csv')
        test_results.append({'Evaluation': title, 'Images': len(frame), 'Cost per 1000 (€)': r['cost_per_1000'],
                             'Photo reviews per 1000': 1000*r['review_share'], 'Automatic incidents per 1000': 1000*r['automatic_share'],
                             'Missed positives per 1000': 1000*r['missed_share'], 'Positive cases escalated (%)': 100*r['positive_escalation']})
    pd.DataFrame(test_results).to_csv(decision_dir/'frozen_policy_test_results.csv', index=False)
    # Run the unchanged COCO detector on exactly the same test images.
    coco = YOLO(cfg['model'])
    matches = set(coco.names.values()) & set(names.values())
    if matches:
        raise ValueError('Implement semantic baseline mapping for this changed vocabulary before proceeding.')
    baseline_dir = evaluation/'coco_baseline'
    baseline_dir.mkdir(parents=True)
    with (baseline_dir/'predictions.jsonl').open('w') as output:
        images = sorted(p for p in (prepared/'test/images').rglob('*') if p.suffix.lower() in EXTENSIONS)
        for image in images:
            result = coco.predict(str(image), device=device, imgsz=cfg['imgsz'], conf=.001,
                                  iou=.7, max_det=300, verbose=False)[0]
            records = [{'class_name': coco.names[int(b.cls.item())], 'confidence': float(b.conf.item()),
                        'xyxy': b.xyxy[0].cpu().tolist()} for b in result.boxes]
            output.write(json.dumps({'image': image.name, 'detections': records})+'\n')
    baseline_rows = []
    for title in ['COCO — no target-class outputs', 'Predict nothing']:
        for i, name in names.items():
            baseline_rows.append({'Model': title, 'Class': name, 'Test instances': counts[i],
                                  'Precision': np.nan, 'Recall': 0.0 if counts[i] else np.nan,
                                  'AP50': 0.0 if counts[i] else np.nan, 'AP50–95': 0.0 if counts[i] else np.nan})
    comparison_dir = evaluation/'comparison'
    comparison_dir.mkdir()
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(comparison_dir/'baseline_per_class.csv', index=False)
    fine.insert(0, 'Model', 'Fine-tuned YOLO11n')
    pd.concat([fine, baselines], ignore_index=True).to_csv(comparison_dir/'all_models_per_class.csv', index=False)
    # Publish only after the pipeline succeeds. Preserve the previous artifact set.
    artifacts = ROOT/'artifacts'
    if artifacts.exists():
        artifacts.rename(run/'previous_artifacts')
    shutil.copytree(stage, artifacts)
    (ROOT/'weights').mkdir(exist_ok=True)
    shutil.copy2(best, ROOT/'weights/best.pt')
    print(f'Pipeline complete. Artifacts: {artifacts}; weights: {ROOT/"weights/best.pt"}', flush=True)


if __name__ == '__main__':
    main()
