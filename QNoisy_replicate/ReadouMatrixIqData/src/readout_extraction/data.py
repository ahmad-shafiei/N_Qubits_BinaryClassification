"""Load and parse experimental IQ quadrature files."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from .config import ReadoutDatasetConfig, _list_state_files


def parse_complex_string(value: str) -> complex:
    text = value.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return complex(text)


def bits_from_filename(filename: str, num_qubits: int) -> np.ndarray:
    base = os.path.splitext(os.path.basename(filename))[0]
    if len(base) != num_qubits or set(base) - {"0", "1"}:
        raise ValueError(f"Invalid basis-state filename: {filename}")
    bits = [int(b) for b in base[::-1]]
    return np.array(bits, dtype=int)


def bits_to_int(bits: np.ndarray) -> int:
    return int(sum(int(b) * (2 ** i) for i, b in enumerate(bits)))


def int_to_bits(value: int, num_qubits: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(num_qubits)], dtype=int)


def load_single_file(file_path: str, num_qubits: int) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) != num_qubits:
        raise ValueError(
            f"{file_path}: expected {num_qubits} qubit lines, got {len(lines)}"
        )

    qubit_arrays = []
    shots_per_file = None
    for line in lines:
        tokens = line.split()
        values = [parse_complex_string(token) for token in tokens]
        arr = np.array(values, dtype=np.complex128)
        if shots_per_file is None:
            shots_per_file = arr.size
        elif arr.size != shots_per_file:
            raise ValueError(
                f"{file_path}: inconsistent shot count "
                f"({arr.size} vs {shots_per_file})"
            )
        qubit_arrays.append(arr)
    return np.stack(qubit_arrays, axis=0)


def load_all_data(config: ReadoutDatasetConfig) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load IQ data. Returns (X, y, shots_per_file)."""
    file_paths = _list_state_files(config.data_dir)
    expected = config.num_states
    if len(file_paths) != expected:
        raise ValueError(
            f"{config.dataset_id}: expected {expected} state files, "
            f"found {len(file_paths)}"
        )

    all_x, all_y = [], []
    shots_per_file = None
    for file_path in file_paths:
        print(f"Loading {os.path.basename(file_path)}")
        data = load_single_file(file_path, config.num_qubits)
        if shots_per_file is None:
            shots_per_file = data.shape[1]
        labels = bits_from_filename(file_path, config.num_qubits)
        x_block = data.T
        y_block = np.tile(labels, (shots_per_file, 1))
        all_x.append(x_block)
        all_y.append(y_block)

    x_data = np.vstack(all_x)
    y_data = np.vstack(all_y)
    print("\nFinal dataset shape:")
    print("X:", x_data.shape)
    print("y:", y_data.shape)
    return x_data, y_data, shots_per_file
