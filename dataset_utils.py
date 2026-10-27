"""Read mixed YOLO labels and produce a separate detection-format copy."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import yaml

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ANOMALY_NAMES = {'cracked', 'dusty', 'bird_drop'}


def parse_label(path, names):
    annotations = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        raw_id = float(values[0])
        class_id = int(raw_id)
        coords = np.asarray([float(v) for v in values[1:]])
        if raw_id != class_id or class_id not in names or not np.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any():
            raise ValueError(f'Invalid label: {path}:{number}')
        polygon = len(coords) != 4
        if not polygon:
            xc, yc, width, height = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = coords.reshape(-1, 2)
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            xc, yc, width, height = (xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin
        else:
            raise ValueError(f'Unexpected label format: {path}:{number}')
        if width <= 0 or height <= 0:
            raise ValueError(f'Zero-area label: {path}:{number}')
        annotations.append((class_id, float(xc), float(yc), float(width), float(height), polygon))
    return annotations


def prepare_dataset(source, destination, stats_dir):
    source, destination, stats_dir = map(Path, (source, destination, stats_dir))
    if source.resolve() == destination.resolve():
        raise ValueError('The detection copy must differ from the source dataset.')
    if destination.exists():
        raise FileExistsError(f'Use a new destination: {destination}')
    config = yaml.safe_load((source / 'data.yaml').read_text())
    names = config['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if len(names) < 2 or not ANOMALY_NAMES.issubset(set(names.values())):
        raise ValueError('Expected the selected solar dataset classes.')
    anomaly_ids = {i for i, name in names.items() if name in ANOMALY_NAMES}
    summary, coverage, sizes, conversion = [], [], [], []
    hashes = defaultdict(list)
    for split in ['train', 'valid', 'test']:
        images_dir, labels_dir = source/split/'images', source/split/'labels'
        images = sorted(p for p in images_dir.rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f'Empty image split: {split}')
        target = destination/split
        shutil.copytree(images_dir, target/'images')
        (target/'labels').mkdir(parents=True)
        count = Counter()
        present = Counter()
        any_anomaly = only_panel = boxes = polygons = empty = 0
        for image_path in images:
            relative = image_path.relative_to(images_dir)
            label = (labels_dir/relative).with_suffix('.txt')
            if not label.exists():
                raise FileNotFoundError(label)
            annotations = parse_label(label, names)
            classes = {a[0] for a in annotations}
            any_anomaly += bool(classes & anomaly_ids)
            only_panel += bool(classes) and not bool(classes & anomaly_ids)
            empty += not bool(classes)
            for i in classes:
                present[names[i]] += 1
            lines = []
            for i, xc, yc, width, height, polygon in annotations:
                count[names[i]] += 1
                boxes += not polygon
                polygons += polygon
                sizes.append({'Split': split, 'Class': names[i], 'Box area (%)': 100*width*height})
                lines.append(f'{i} {xc:.12g} {yc:.12g} {width:.12g} {height:.12g}')
            output = (target/'labels'/relative).with_suffix('.txt')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('\n'.join(lines) + ('\n' if lines else ''))
            assert len(parse_label(output, names)) == len(annotations)
            with Image.open(image_path) as raw:
                rgb = raw.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            hashes[digest].append({'split': split, 'image': str(relative)})
        summary.append({'Split': split, 'Images': len(images), **{name: count[name] for name in names.values()}})
        coverage.append({'Split': split, 'Images': len(images), 'Missing label file': 0,
                         'Empty label file': empty, 'Any annotated anomaly': any_anomaly,
                         'Panel only': only_panel, **{f'Contains {name}': present[name] for name in sorted(ANOMALY_NAMES)}})
        conversion.append({'Split': split, 'Boxes kept': boxes, 'Polygons converted': polygons,
                           'Total annotations': boxes+polygons})
    destination.mkdir(parents=True, exist_ok=True)
    out_config = {'path': str(destination.resolve()), 'train': 'train/images', 'val': 'valid/images',
                  'test': 'test/images', 'nc': len(names), 'names': names}
    (destination/'data.yaml').write_text(yaml.safe_dump(out_config, sort_keys=False))
    stats_dir.mkdir(parents=True, exist_ok=True)
    for title, rows in [('summary', summary), ('coverage', coverage), ('conversion', conversion)]:
        pd.DataFrame(rows).to_csv(stats_dir/f'{title}.csv', index=False)
    size_frame = pd.DataFrame(sizes)
    size_frame.groupby(['Split', 'Class']).agg(
        Instances=('Box area (%)', 'size'),
        P25_area_pct=('Box area (%)', lambda s: s.quantile(.25)),
        Median_area_pct=('Box area (%)', 'median'),
        P75_area_pct=('Box area (%)', lambda s: s.quantile(.75)),
    ).reset_index().to_csv(stats_dir/'size_summary.csv', index=False)
    duplicates = [group for group in hashes.values() if len(group) > 1]
    return destination/'data.yaml', names, duplicates
