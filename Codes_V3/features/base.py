# features/base.py

from abc import ABC, abstractmethod
import numpy as np
from raw_data.iq_container import RawIQData


class FeatureExtractor(ABC):
    """
    Base class for all feature extractors.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, raw: RawIQData) -> np.ndarray:
        """
        Parameters
        ----------
        raw : RawIQData

        Returns
        -------
        features : np.ndarray
            Shape: (shots, n_qubits, feature_dim)
            یا
            Shape: (shots, feature_dim)  ← اگر از قبل aggregate شده
        """
        pass
