#!/usr/bin/env python3

import os

import numpy as np
import cv2

from pathlib import Path


"""
The file hierarchy is as follow:
data/
├── labels.txt
├── README.md
└── train
    ├── 01
    │   ├── c01_s00.png
    │   ├── c04_s01.png
    │   ├── c07_s03.png
    │   ├── c30_s03.png
    │   └── c34_s03.png
    ├── 02
    │   ├── c01_s01.png
    │   ├── c09_s00.png
    │   ├── c14_s05.png
    │   ├── c32_s00.png
    │   └── c53_s03.png
    ...
    └── 57
        ├── c24_s00.png
        ├── c25_s01.png
        ├── c31_s03.png
        ├── c33_s07.png
        └── c51_s04.png

Note:
    Be careful class id are 1-indexed
"""

PATH_TO_RESOURCES = "./data/train" # Adapt to your data folder


def get_image_label(path: Path) -> tuple[np.ndarray, int] | None:

    if path.suffix != ".png" and not path.parent.name.isdigit():
        return 

    return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB), int(path.parent.name)

def get_data_set(path: str = PATH_TO_RESOURCES) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns X (features) and Y (labels) as a tupple of np.array.
    The `path` specify where the data is stored.
    """

    img_loc = (Path(dirpath, file) for dirpath, _, files in os.walk(path) for file in files)
    label_image = filter(None, map(get_image_label, img_loc))

    X, Y = zip(*label_image)

    return np.array(X, dtype=object), np.array(Y, dtype=int)
