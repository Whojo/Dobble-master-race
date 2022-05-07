from typing import Any, Optional
import numpy as np
import cv2 as cv
import pickle

from dobble_master_race.img_utils import mask_with_inner, get_hist, sample_images
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class ColorsHistogram(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        nb_colors: int,
        white_threshold: int = 240,
        shape_inners: bool = True,
        max_samples: int = 50000,
        median_blur: bool = True,
        saved_kmeans: Optional[str] = None,
    ) -> None:
        super().__init__()

        if nb_colors < 2:
            raise ValueError("nb_colors cannot be less than 2")

        self.nb_colors = nb_colors
        self.white_threshold = white_threshold
        self.shape_inners = shape_inners
        self.max_samples = max_samples
        self.median_blur = median_blur

        if saved_kmeans is None:
            self.clustering = KMeans(n_clusters=self.nb_colors, random_state=42)
        else:
            with open(saved_kmeans, "rb") as file:
                self.clustering = pickle.load(file)

    def fit(self, X, y=None):

        if self.median_blur:
            X = np.array([cv.medianBlur(img, 3, None) for img in X], dtype=object)

        if self.shape_inners:
            masks = np.array(
                [mask_with_inner(img, self.white_threshold) for img in X], dtype=object
            )
        else:
            masks = np.array(
                [img.sum(axis=-1) < (self.white_threshold * 3) for img in X],
                dtype=object,
            )

        sample_pxls = sample_images(X, masks, self.max_samples)

        self.clustering.fit(sample_pxls)

        return self

    def transform(self, X, **kwargs):
        masks = np.array([mask_with_inner(img, 240) for img in X], dtype=bool)

        color_maps = np.array(
            [self.clustering.predict(img[mask]) for img, mask in zip(X, masks)],
            dtype=int,
        )

        return np.array(
            [get_hist(color_map, self.nb_colors) for color_map in color_maps],
            dtype=object,
        )


class HuMoments(BaseEstimator, TransformerMixin):
    def __init__(self, nb_moment: int = 6, white_threshold: int = 240):
        super().__init__()
        if nb_moment < 1 or 7 < nb_moment:
            raise ValueError("nb_moment must be between 1 and 7")

        self.nb_moment = nb_moment
        self.white_threshold = white_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kwargs):
        masks = np.array(
            [
                np.array(img.sum(axis=-1) < (self.white_threshold * 3), dtype=np.uint8)
                for img in X
            ],
            dtype=int,
        )

        moments = np.array([cv.moments(mask, binaryImage=True) for mask in masks])

        return np.array(
            [cv.HuMoments(mask_moments)[: self.nb_moment] for mask_moments in moments]
        ).reshape(X.shape[0], -1)


class AspectRatio(BaseEstimator, TransformerMixin):
    def __init__(self, white_threshold: int = 255):
        super().__init__()
        self.white_threshold = white_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kwargs):
        masks = np.array(
            [
                np.array(img.sum(axis=-1) < (self.white_threshold * 3), dtype=np.uint8)
                for img in X
            ],
            dtype=object,
        )

        contours = np.array(
            [
                cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
                for mask in masks
            ],
            dtype=object,
        )
        bounding_rects = np.array(
            [cv.minAreaRect(contour[0]) for contour in contours], dtype=object
        )

        return np.array([max(rect[1]) / min(rect[1]) for rect in bounding_rects])


# Early fusion
class HuMomentsColorsHistogram(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        nb_moment: int,
        nb_colors: int,
        hist_args: dict[str, Any],
        hu_args: dict[str, Any],
    ):
        super().__init__()

        self.hist_args = hist_args
        self.hu_args = hu_args
        self.nb_colors = nb_colors
        self.nb_moment = nb_moment

        self.color_histogram = ColorsHistogram(nb_colors, **hist_args)
        self.hu_moments = HuMoments(nb_moment, **hu_args)

    def fit(self, X, y=None):
        self.color_histogram.fit(X, y=y)
        self.hu_moments.fit(X, y=y)

        return self

    def transform(self, X, y=None, **kwargs):
        hists = self.color_histogram.transform(X, y=y)
        moments = self.hu_moments.transform(X, y=y)

        return np.hstack((hists, moments))
