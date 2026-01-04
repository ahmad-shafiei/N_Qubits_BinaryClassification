from raw_data.iq_container import RawIQData

def copy_raw(raw: RawIQData) -> RawIQData:
    return RawIQData(
        data=raw.data.copy(),
        sampling_rate=raw.sampling_rate,
        metadata=dict(raw.metadata),
    )
