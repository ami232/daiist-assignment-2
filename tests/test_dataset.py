from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import yaml
from dataset_utils import parse_label, prepare_dataset


def test_mixed_annotations_preserve_counts_and_originals(tmp_path):
    source, target, stats = tmp_path/'source', tmp_path/'detection', tmp_path/'stats'
    source.mkdir()
    names = {0: 'bird_drop', 1: 'cracked', 2: 'dusty', 3: 'panel'}
    (source/'data.yaml').write_text(yaml.safe_dump({'names': names}))
    mixed = '3 0.5 0.5 0.8 0.8\n2 0.1 0.2 0.7 0.2 0.7 0.8 0.1 0.8\n'
    for split in ['train', 'valid', 'test']:
        (source/split/'images').mkdir(parents=True)
        (source/split/'labels').mkdir()
        Image.new('RGB', (32, 32), 'white').save(source/split/'images/example.png')
        (source/split/'labels/example.txt').write_text(mixed)
    path, found_names, duplicates = prepare_dataset(source, target, stats)
    assert path.exists() and found_names == names
    assert (source/'train/labels/example.txt').read_text() == mixed
    converted = parse_label(target/'train/labels/example.txt', names)
    assert len(converted) == 2 and not any(row[-1] for row in converted)
    np.testing.assert_allclose(converted[1][1:5], [.4, .5, .6, .6])
    assert len(duplicates) == 1 and len(duplicates[0]) == 3
    assert (stats/'conversion.csv').exists()


def test_invalid_annotations_are_not_silently_dropped(tmp_path):
    path = tmp_path/'bad.txt'
    path.write_text('0 0.2 0.3 0.4\n')
    with pytest.raises(ValueError):
        parse_label(path, {0: 'panel'})
