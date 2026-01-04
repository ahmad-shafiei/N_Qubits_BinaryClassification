# Codes_V3/raw_data/loaders/hdf5_loader.py

import h5py
from ..iq_container import RawIQData
from ..data_source import DataSource


class HDF5Loader(DataSource):
    def __init__(self, file_path, dataset_key="iq", sampling_rate=None):
        self.file_path = file_path
        self.dataset_key = dataset_key
        self.sampling_rate = sampling_rate

    def load(self) -> RawIQData:
        with h5py.File(self.file_path, "r") as f:
            data = f[self.dataset_key][...]

        return RawIQData(
            data=data,
            sampling_rate=self.sampling_rate,
        )
