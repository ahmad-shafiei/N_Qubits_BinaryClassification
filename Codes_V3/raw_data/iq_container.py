# Codes_V3/raw_data/iq_container.py

import numpy as np
from .iq_schema import validate_iq_array


class RawIQData:
    """
    Container for raw multi-qubit IQ traces
    """

    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: float = None,
        metadata: dict = None,
    ):
        validate_iq_array(data)

        self.data = data
        self.sampling_rate = sampling_rate
        self.metadata = metadata or {}

        self.shots = data.shape[0]
        self.n_qubits = data.shape[1]
        self.time_steps = data.shape[2]

    def __repr__(self):
        return (
            f"RawIQData(shots={self.shots}, "
            f"n_qubits={self.n_qubits}, "
            f"time_steps={self.time_steps})"
        )
