"""
Remove the round black dermoscope frame from a lesion photo.

Used by both train.py and app.py, so training images and uploaded images get
exactly the same treatment. The frame is found as the dark pixels connected
to the edge of the image, slightly enlarged to cover its blurry border, and
painted with the median colour of the rest of the photo (the skin).
"""

import cv2
import numpy as np
from PIL import Image

DARK = 40  # a pixel is "frame-dark" if all its colour channels are below this


def remove_frame(img: Image.Image) -> Image.Image:
    a = np.asarray(img.convert("RGB")).copy()
    dark = (a.max(axis=2) < DARK).astype(np.uint8)
    _, labels = cv2.connectedComponents(dark)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    frame = np.isin(labels, edge_labels[edge_labels != 0]).astype(np.uint8)
    if frame.sum() == 0:
        return Image.fromarray(a)  # no frame: leave the photo untouched
    frame = cv2.dilate(frame, np.ones((5, 5), np.uint8), iterations=2).astype(bool)
    if frame.all():
        return Image.fromarray(a)
    a[frame] = np.median(a[~frame], axis=0)
    return Image.fromarray(a)
