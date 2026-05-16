import numpy as np
from raw_data.iq_container import RawIQData

def remove_nan(raw: RawIQData) -> RawIQData:
    data = raw.data.copy()
    data[np.isnan(data)] = 0.0
    data[np.isinf(data)] = 0.0

    return RawIQData(
        data=data,
        sampling_rate=raw.sampling_rate,
        metadata=raw.metadata,
    )
