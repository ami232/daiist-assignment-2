"""Describe the frozen dataset before fitting any model."""
import hashlib
from collections import Counter, defaultdict
import numpy as np
from project import ARTIFACTS, DATA_URL, WORKSPACE, PROJECT, VERSION, image_files, labels_for, write_json


def audit_dataset(names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    summary = {'source': DATA_URL, 'workspace': WORKSPACE, 'project': PROJECT,
               'version': VERSION, 'classes': names, 'splits': {}}
    hashes = defaultdict(list)
    sample = []
    for split in ['train', 'valid', 'test']:
        counts = Counter()
        positives = 0
        areas = defaultdict(list)
        for p in image_files(split):
            labels = labels_for(p)
            positives += any(names[int(row[0])] == 'Damage' for row in labels)
            for cls, x, y, w, h in labels:
                counts[names[int(cls)]] += 1
                areas[names[int(cls)]].append(w*h)
            hashes[hashlib.sha256(p.read_bytes()).hexdigest()].append((split, p.name))
        summary['splits'][split] = {
            'images': len(image_files(split)), 'instances': dict(counts),
            'damage_images': positives, 'no_damage_images': len(image_files(split))-positives,
            'box_area_fraction': {k: {'median': float(np.median(v)),
                                     'p10': float(np.quantile(v,.1)), 'p90':float(np.quantile(v,.9))}
                                  for k,v in areas.items()}}
    duplicates = [v for v in hashes.values() if len({x[0] for x in v}) > 1]
    summary['exact_cross_split_duplicates'] = duplicates
    # A pixel hash also catches identical images saved with different JPEG metadata.
    pixels = defaultdict(list)
    for split in ['train','valid','test']:
        for p in image_files(split):
            with Image.open(p) as img:
                pixels[hashlib.sha256(img.convert('RGB').tobytes()).hexdigest()].append((split,p.name))
    summary['pixel_cross_split_duplicates'] = [v for v in pixels.values() if len({x[0] for x in v})>1]
    write_json(ARTIFACTS / 'dataset_audit.json', summary)
    # Show only training images during the initial visual audit.
    fig, axs = plt.subplots(2, 4, figsize=(12,6))
    for cls in range(len(names)):
        candidates = [p for p in image_files('train') if any(int(r[0])==cls for r in labels_for(p))]
        for j, p in enumerate(candidates[:4]):
            ax=axs[cls,j]
            ax.imshow(Image.open(p)); ax.set_title(names[cls]); ax.axis('off')
            for c,x,y,w,h in labels_for(p):
                with Image.open(p) as img:
                    iw,ih=img.size
                ax.add_patch(plt.Rectangle(((x-w/2)*iw,(y-h/2)*ih),w*iw,h*ih,
                                          fill=False,color='orange',linewidth=2))
    fig.suptitle('Training examples with the original dataset boxes')
    fig.tight_layout(); fig.savefig(ARTIFACTS / 'training_examples.png',dpi=130); plt.close(fig)
    print('Dataset audit:', summary['splits'], flush=True)
    print('Exact cross-split duplicates:',len(duplicates),flush=True)
    return summary


if __name__ == '__main__':
    from project import configure_runtime, prepare_dataset
    configure_runtime()
    _, names = prepare_dataset()
    audit_dataset(names)
