# features/iq_features.py

import numpy as np
from raw_data.iq_container import RawIQData
from .base import FeatureExtractor


class MeanIQFeature(FeatureExtractor):
    """
    Mean I and Q per qubit.
    """

    def __init__(self):
        super().__init__(name="mean_iq")

    def extract(self, raw: RawIQData) -> np.ndarray:
        """
        Returns
        -------
        features : np.ndarray
            Shape: (shots, n_qubits, 2)
        """
        # raw.data shape: (shots, n_qubits, time, 2)
        mean_iq = raw.data.mean(axis=2)
        return mean_iq
