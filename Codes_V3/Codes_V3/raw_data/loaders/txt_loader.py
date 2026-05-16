import os
import numpy as np
from ..iq_container import RawIQData
from ..data_source import DataSource


def parse_complex_string(s: str) -> complex:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return complex(s)


class TXTMultiStateLoader(DataSource):
    """
    Load multi-qubit IQ data from multiple text files (complex numbers in each line).
    Each file corresponds to one computational basis state.
    """

    def __init__(self, directory, n_qubits, sampling_rate=None):
        self.directory = directory
        self.n_qubits = n_qubits
        self.sampling_rate = sampling_rate

    def bits_from_filename(self, filename: str):
        base = os.path.splitext(os.path.basename(filename))[0]
        bits = [int(b) for b in base[::-1]]  # LSB = qubit1
        return np.array(bits, dtype=int)

    def load(self) -> RawIQData:
        files = sorted([f for f in os.listdir(self.directory) if f.endswith(".txt")],
                       key=lambda x: int(os.path.splitext(x)[0], 2))
        all_shots = []
        labels = []

        for fname in files:
            path = os.path.join(self.directory, fname)
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            if len(lines) != self.n_qubits:
                raise ValueError(f"{fname}: expected {self.n_qubits} lines, got {len(lines)}")

            qubit_arrays = []
            for line in lines:
                qubit_arrays.append(np.array([parse_complex_string(tok) for tok in line.split()]))

            raw = np.stack(qubit_arrays, axis=0)  # shape: (qubits, time)
            num_shots = raw.shape[1]

            # تبدیل به (1, qubits, time, 2) → I/Q
            iq = np.zeros((1, self.n_qubits, num_shots, 2), dtype=float)
            for q in range(self.n_qubits):
                iq[0, q, :, 0] = raw[q, :].real
                iq[0, q, :, 1] = raw[q, :].imag

            all_shots.append(iq)
            label_bits = np.tile(self.bits_from_filename(fname), (num_shots, 1))
            labels.append(label_bits)

        data = np.concatenate(all_shots, axis=0)
        metadata = {"labels": np.vstack(labels), "state_files": files}

        return RawIQData(
            data=data,
            sampling_rate=self.sampling_rate,
            metadata=metadata,
        )
