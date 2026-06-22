"""Noise model helpers with per-snapshot readout matrix paths."""

import os

import numpy as np

from qiskit_aer.noise import NoiseModel, ReadoutError

from .circuits import (
    build_circuit_independent,
    build_zzfeaturemap_circuit,
)
from .readout_extraction.config import (
    DEFAULT_SNAPSHOT_ID,
    PROJECT_ROOT,
    get_active_snapshot,
    get_data_train_root,
    get_dataset_output_dir,
    get_results_root,
    resolve_snapshot_id,
    set_active_snapshot,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = get_data_train_root()

NUM_QUBITS = 4
CIRCUIT_BUILDERS = {
    "independent": build_circuit_independent,
    "zz_featuremap": build_zzfeaturemap_circuit,
}


def get_active_noise_snapshot() -> str:
    return get_active_snapshot()


def get_noise_matrix_dir(snapshot_id: str | None = None) -> str:
    snapshot_id = resolve_snapshot_id(snapshot_id)
    if snapshot_id.startswith("res_"):
        return os.path.join(get_results_root(), snapshot_id)
    return get_dataset_output_dir(snapshot_id)


NOISE_MATRIX_DIR = get_noise_matrix_dir()
SINGLE_QUBIT_DIR = NOISE_MATRIX_DIR
CORRELATED_MATRIX_PATH = os.path.join(NOISE_MATRIX_DIR, "noise_matrix_16x16.txt")


def create_noise_model(
    mode="synthetic",
    p01=0.05,
    p10=0.05,
    *,
    snapshot_id: str | None = None,
    dataset_id: str | None = None,
):
    noise_model = NoiseModel()
    active_snapshot = resolve_snapshot_id(snapshot_id or dataset_id)
    matrix_dir = get_noise_matrix_dir(active_snapshot)

    if mode == "synthetic":
        readout_error = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        for q in range(NUM_QUBITS):
            noise_model.add_readout_error(readout_error, [q])
    elif mode == "experimental_single":
        for q in range(NUM_QUBITS):
            path = os.path.join(matrix_dir, f"assignment_matrix_q{q + 1}.txt")
            assignment = np.loadtxt(path)
            noise_model.add_readout_error(ReadoutError(assignment), [q])
    elif mode == "experimental_correlated":
        return None
    else:
        raise ValueError(f"Unknown noise mode: {mode}")
    return noise_model


def apply_correlated_readout_noise(
    p_ideal,
    snapshot_id: str | None = None,
    dataset_id: str | None = None,
):
    active_snapshot = resolve_snapshot_id(snapshot_id or dataset_id)
    matrix_path = os.path.join(
        get_noise_matrix_dir(active_snapshot),
        "noise_matrix_16x16.txt",
    )
    assignment = np.loadtxt(matrix_path)
    return p_ideal @ assignment
