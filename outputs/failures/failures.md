# Worst test cases

Lowest-scoring 2 positives and highest-scoring 3 negatives. Predictions at conf 0.001, top 3 boxes drawn; ground truth in white.

| Kind | File | Ground truth | Score | Top-3 predictions |
|---|---|---|---|---|
| missed positive (lowest score) | `5ded88cs-960_jpg.rf.386b08d3ad2096ae1ddfebd676e51ecd.jpg` | Box_broken | 0.0000 | Package 0.792; Package 0.005 |
| missed positive (lowest score) | `670055d7-233Open_boo_jpg.rf.1221d3cd05ea1c8fa8ad629807f7e340.jpg` | Box_broken | 0.0015 | Box 0.721; Box 0.503; Package 0.057 |
| false alarm (highest-scoring negative) | `76c8973b-164Packageeees_jpg.rf.0af5f3106b5a112c852cb7c4fafba70f.jpg` | Package, Package | 0.9338 | Open_package 0.934; Open_package 0.009; Box 0.007 |
| false alarm (highest-scoring negative) | `21926138-50Booooox_jpg.rf.dd6cfd806272833d6bc729968c7a067f.jpg` | Box | 0.8984 | Box_broken 0.898; Box 0.447; Box 0.005 |
| false alarm (highest-scoring negative) | `1packaaaageeeees_jpg.rf.3feee10d7836e2d829a66fc04920728b.jpg` | Package | 0.6410 | Open_package 0.641; Package 0.383; Package 0.011 |
