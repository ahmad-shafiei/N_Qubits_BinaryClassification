# Codes_V3/raw_data/loaders/simulator_loader.py

import numpy as np
from ..iq_container import RawIQData
from ..data_source import DataSource


class SimulatorLoader(DataSource):
    def __init__(self, shots, n_qubits, time_steps):
        self.shots = shots
        self.n_qubits = n_qubits
        self.time_steps = time_steps

    def load(self) -> RawIQData:
        data = np.random.randn(
            self.shots, self.n_qubits, self.time_steps, 2
        )
        return RawIQData(data=data)
