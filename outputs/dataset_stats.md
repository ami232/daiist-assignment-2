# Dataset description: `roboflow-ngkro/package-detection-e1ssd/6`

Computed from the YOLOv11 export on disk by `scripts/describe_dataset.py`.
Positive (compromised) classes: Box_broken, Open_package. Negative (intact) classes: Box, Package.

## Roboflow version metadata

- Version name: 2023-10-16 4:20pm
- Preprocessing: auto-orient=True; resize={'width': 640, 'format': 'Stretch to', 'height': 640}
- Augmentation: none (augmentation field is empty)
- Split: The Roboflow version metadata has no field recording how images were assigned to splits. The split is exactly 70/20/10, which is Roboflow's default ratio when it splits for you, so a Roboflow random split is likely; an author-supplied split would rarely land on exactly those numbers. This is an inference, not a stated fact.
- Licence field in data.yaml: Private

## Images and instances per split

| Split | Images | Instances | Box | Box_broken | Open_package | Package | Boxes/img mean | Boxes/img max |
|---|---|---|---|---|---|---|---|---|
| train | 840 | 1052 | 314 | 206 | 225 | 307 | 1.252 | 11 |
| valid | 240 | 305 | 103 | 64 | 54 | 84 | 1.271 | 5 |
| test | 120 | 136 | 31 | 34 | 29 | 42 | 1.133 | 5 |

## Image-level balance (the one that matters)

| Split | Positive (any Box_broken/Open_package) | of which mixed with negatives | Negative (only Box/Package) | Zero labels | Positive share |
|---|---|---|---|---|---|
| train | 424 | 3 | 416 | 0 | 50.5% |
| valid | 118 | 1 | 122 | 0 | 49.2% |
| test | 62 | 1 | 58 | 0 | 51.7% |

## Box area as a fraction of image area

| Class | Boxes | Median | p10 | p90 | Min | Max |
|---|---|---|---|---|---|---|
| Box | 448 | 25.60% | 8.12% | 52.68% | 0.002% | 94.49% |
| Box_broken | 304 | 29.64% | 17.26% | 77.76% | 0.003% | 99.53% |
| Open_package | 308 | 38.67% | 18.61% | 79.87% | 0.079% | 99.77% |
| Package | 433 | 45.62% | 14.78% | 82.10% | 1.600% | 99.38% |

Histogram: `outputs/box_area_hist.png`

## Images with zero labels

- None in any split.

## Image resolution

| Resolution | Images |
|---|---|
| 640x640 | 1200 |

## Near-duplicates (pHash, Hamming distance <= 4)

- Total pairs: 24 (exact hash matches: 14)
- Within split: train 11, valid 0, test 0
- Across splits: test/train 2, test/valid 4, train/valid 7

### Within-split examples

- d=0: `train/0a08ef4f-291Booooo_jpg.rf.85788dd8a6ec7e68cc6bd60eade12e42.jpg` and `train/b9db7642-261Booooo_jpg.rf.d3bf574688d0bd8c7e267cad7faf42f1.jpg`
- d=0: `train/16aa1d26-193Open_booo_jpg.rf.122640c6969c2e8cd838deef597fd7be.jpg` and `train/299c4183-160Open_booo_jpg.rf.408c7e625f252bef50d83b3616460c45.jpg`
- d=0: `train/17cb342b-252Open_booo_jpg.rf.cd0be55d8dc997bfac2a29118cf96fcf.jpg` and `train/9f366079-254Open_booo_jpg.rf.7e933b973895205028989a3c0698889f.jpg`

### Across-split examples

- d=0: `train/0c660c88-248Open_booo_jpg.rf.21c6c5bc3396f39b95fde4de48421fd0.jpg` and `valid/d6aa445f-253Open_booo_jpg.rf.0ad2006ca7a966b0ba5937b35ea02eae.jpg`
- d=0: `train/1082ae68-33Booooox_jpg.rf.f6818b7faa6bd1f1f0f2bc2fe5cf6b98.jpg` and `valid/ec4c0e01-35Booooox_jpg.rf.1982e4eb3debb8b39b0d10ee03612ccc.jpg`
- d=0: `train/2eff1374-69Booooox_jpg.rf.60487e7bab8048c6fb82df8d75b71470.jpg` and `valid/35ef99d8-66Booooox_jpg.rf.df812c3ce341a97f0cbaded630421e10.jpg`
