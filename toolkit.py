#!/usr/bin/env python3

import os

import numpy as np
import cv2


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


def _get_image(directory: str, filename: str) -> np.array:
    """
    Returns the image (as an np.array) from the file located in a
    `directory` and with a `filename`.

    Note:
        The images have variable size.
    """
    img = cv2.imread(os.path.join(PATH_TO_RESOURCES, directory, filename))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _get_class(directory: str, filename: str) -> np.array:
    """
    Returns the class associated with a `directory`.

    Note:
        'filename` is not used here, but still in argument
        for interface convenience
    """
    return int(directory)

def _apply_on_flatten_dir(path: str, *, func=_get_class, type_=object) -> np.array:
    """
    Returns the flatten list of files in each sub-directory at `path`
    and apply `func` to each file.
    """
    return np.array([
        func(directory, filename)
        for directory in os.listdir(PATH_TO_RESOURCES)
        for filename in os.listdir(os.path.join(PATH_TO_RESOURCES, directory))
    ], dtype=type_)

def get_data_set(path: str = PATH_TO_RESOURCES) -> (np.array, np.array):
    """
    Returns X (features) and Y (labels) as a tupple of np.array.
    The `path` specify where the data is stored.
    """
    return (_apply_on_flatten_dir(path, func=_get_image),
            _apply_on_flatten_dir(path, func=_get_class, type_=int))
