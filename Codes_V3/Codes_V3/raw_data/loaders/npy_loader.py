# Codes_V3/raw_data/loaders/npy_loader.py

import numpy as np
from ..iq_container import RawIQData
from ..data_source import DataSource


class NPYLoader(DataSource):
    def __init__(self, file_path, sampling_rate=None, metadata=None):
        self.file_path = file_path
        self.sampling_rate = sampling_rate
        self.metadata = metadata or {}

    def load(self) -> RawIQData:
        data = np.load(self.file_path)
        return RawIQData(
            data=data,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata,
        )
