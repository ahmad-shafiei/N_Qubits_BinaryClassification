import numpy as np
from raw_data.iq_container import RawIQData

def downsample(raw: RawIQData, factor: int) -> RawIQData:
    """
    Downsample time axis by an integer factor
    """
    data = raw.data[:, :, ::factor, :]

    return RawIQData(
        data=data,
        sampling_rate=(
            raw.sampling_rate / factor if raw.sampling_rate else None
        ),
        metadata=raw.metadata,
    )
