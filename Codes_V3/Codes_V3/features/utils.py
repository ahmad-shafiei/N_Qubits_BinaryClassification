# features/utils.py

import numpy as np


def flatten_features(features: np.ndarray) -> np.ndarray:
    """
    Flatten features to (shots, -1)
    """
    return features.reshape(features.shape[0], -1)
