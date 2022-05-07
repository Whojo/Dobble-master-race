import cv2
import numpy as np


def _flood_img_border(img, line, mask, start, index):
    seed = [start, 0] if index else [0, start]
    seed[index] = np.argmin(line)

    while True:
        cv2.floodFill(img, mask, seed, (0, 0, 0))
        seed[index] = np.argmin(line)
        if seed[index] == 0:
            break


def mask_with_inner(img_base: np.array, gray_threshold: int) -> np.array:
    img = img_base.copy()
    mask = img.sum(axis=-1) < (gray_threshold * 3)
    # Cf floodFill doc
    pad_mask = np.pad(mask, ((1, 1), (1, 1))).astype(np.uint8)

    # Start floodFill on every pixel of the image borders
    _flood_img_border(img, pad_mask[1, 1:-1], pad_mask, 0, 0)
    _flood_img_border(img, pad_mask[-2, 1:-1], pad_mask, img.shape[0] - 1, 0)
    _flood_img_border(img, pad_mask[1:-1, 1], pad_mask, 0, 1)
    _flood_img_border(img, pad_mask[1:-1, -2], pad_mask, img.shape[1] - 1, 1)

    return ~(pad_mask[1:-1, 1:-1].astype(np.bool8) ^ mask)


def sample_images(
    images: np.ndarray, masks: np.ndarray, nb_samples: int = 50000
) -> np.ndarray:
    nb_sample_per_image = nb_samples // len(images)

    sample_pxls = images[0][masks[0]][
        0
    ]  # Initialize arbitrary to be able to use np.vstack

    for img, mask in zip(images[1:], masks[1:]):
        pxls = img[mask]
        new_sample_pxls_id = np.random.choice(
            pxls.shape[0], size=nb_sample_per_image, replace=False
        )
        sample_pxls = np.vstack((sample_pxls, pxls[new_sample_pxls_id]))

    return sample_pxls


def get_hist(color_map: np.ndarray, nb_colors) -> np.ndarray:
    hist = np.bincount(color_map, minlength=nb_colors)
    norm_hist = hist / hist.max()

    return norm_hist
