# features/pipeline.py

import numpy as np
from typing import List
from .base import BaseFeature
from raw_data.iq_container import RawIQData


class FeaturePipeline:
    """
    Combine multiple feature extractors into a single feature matrix.
    """

    def __init__(self, features: List[BaseFeature]):
        self.features = features

    def extract(self, raw: RawIQData) -> np.ndarray:
        feature_blocks = []

        for feat in self.features:
            Xf = feat.extract(raw)
            feature_blocks.append(Xf)

        return np.concatenate(feature_blocks, axis=1)
