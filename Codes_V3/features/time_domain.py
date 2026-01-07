import numpy as np
from .base import BaseFeature
from raw_data.iq_container import RawIQData


class MeanIQ(BaseFeature):
    name = "mean_iq"

    def extract(self, raw: RawIQData) -> np.ndarray:
        # mean over time axis
        mean = raw.data.mean(axis=2)  # (states, qubits, 2)
        return mean.reshape(mean.shape[0], -1)


class VarIQ(BaseFeature):
    name = "var_iq"

    def extract(self, raw: RawIQData) -> np.ndarray:
        var = raw.data.var(axis=2)
        return var.reshape(var.shape[0], -1)


class MeanMagnitude(BaseFeature):
    name = "mean_mag"

    def extract(self, raw: RawIQData) -> np.ndarray:
        mag = np.sqrt(
            raw.data[..., 0] ** 2 + raw.data[..., 1] ** 2
        )
        mean_mag = mag.mean(axis=2)  # (states, qubits)
        return mean_mag.reshape(mean_mag.shape[0], -1)
