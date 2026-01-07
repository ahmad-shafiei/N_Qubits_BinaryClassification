# features/statistics.py

import numpy as np
from raw_data.iq_container import RawIQData
from .base import FeatureExtractor


class IQVarianceFeature(FeatureExtractor):
    """
    Variance of I and Q per qubit.
    """

    def __init__(self):
        super().__init__(name="iq_variance")

    def extract(self, raw: RawIQData) -> np.ndarray:
        """
        Returns
        -------
        features : np.ndarray
            Shape: (shots, n_qubits, 2)
        """
        var_iq = raw.data.var(axis=2)
        return var_iq
