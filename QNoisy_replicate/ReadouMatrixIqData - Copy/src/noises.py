import os
import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit import transpile

from .circuits import (
    build_circuit_independent,
    build_zzfeaturemap_circuit
)


# ============================================================
# PROJECT PATHS
# ============================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(SRC_DIR)


DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "DataTrain"
)


NOISE_MATRIX_DIR = os.path.join(
    PROJECT_ROOT,
    "noise_matrix_results"
)


SINGLE_QUBIT_DIR = NOISE_MATRIX_DIR


CORRELATED_MATRIX_PATH = os.path.join(
    NOISE_MATRIX_DIR,
    "noise_matrix_16x16.txt"
)
NUM_QUBITS = 4
CIRCUIT_BUILDERS = {
    "independent":
        build_circuit_independent,
    "zz_featuremap":
        build_zzfeaturemap_circuit,
}

def create_noise_model(
        mode="synthetic",
        p01=0.05,
        p10=0.05):

    noise_model = NoiseModel()
    if mode == "synthetic":
        readout_error = ReadoutError(
            [
                [1-p01, p01],
                [p10, 1-p10]
            ]
        )
        for q in range(NUM_QUBITS):

            noise_model.add_readout_error(
                readout_error,
                [q]
            )
    elif mode == "experimental_single":
        for q in range(NUM_QUBITS):
            path = os.path.join(
                SINGLE_QUBIT_DIR,
                f"assignment_matrix_q{q+1}.txt"
            )
            A = np.loadtxt(path)
            noise_model.add_readout_error(
                ReadoutError(A),
                [q]
            )
    elif mode == "experimental_correlated":
        return None
    else:
        raise ValueError(
            f"Unknown noise mode: {mode}"
        )
    return noise_model
def apply_correlated_readout_noise(
        p_ideal):
    A16 = np.loadtxt(
        CORRELATED_MATRIX_PATH
    )
    return p_ideal @ A16

