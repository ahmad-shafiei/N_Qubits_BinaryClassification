import numpy as np
from raw_data.iq_container import RawIQData

def normalize_per_qubit(raw: RawIQData) -> RawIQData:
    """
    Normalize I/Q per qubit independently
    """
    data = raw.data.copy()  # (shots, qubits, time, 2)

    for q in range(data.shape[1]):
        mean = data[:, q, :, :].mean()
        std = data[:, q, :, :].std() + 1e-12
        data[:, q, :, :] = (data[:, q, :, :] - mean) / std

    return RawIQData(
        data=data,
        sampling_rate=raw.sampling_rate,
        metadata=raw.metadata,
    )
