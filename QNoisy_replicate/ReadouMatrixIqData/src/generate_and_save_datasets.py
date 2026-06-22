"""Generate, save, and load ML training datasets (noisy vs ideal distributions)."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from .circuits import build_circuit_independent, build_zzfeaturemap_circuit
from .noises import apply_correlated_readout_noise, create_noise_model
from .readout_extraction.config import (
    DEFAULT_SNAPSHOT_ID,
    discover_ml_snapshots,
    get_active_snapshot,
    get_data_train_root,
    get_ml_dataset_dir,
    migrate_datatrain_layout,
    resolve_snapshot_id,
    set_active_snapshot,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_ROOT = get_data_train_root()

NUM_QUBITS = 4
CIRCUIT_BUILDERS = {
    "independent": build_circuit_independent,
    "zz_featuremap": build_zzfeaturemap_circuit,
}


def counts_to_probs_4q(counts):
    probs = np.zeros(16)
    for bitstring, count in counts.items():
        probs[int(bitstring, 2)] = count
    return probs / np.sum(probs)


def get_probs(
    circuit_builder,
    thetas,
    noise_model=None,
    shots=1024,
    **circuit_kwargs,
):
    qc = circuit_builder(thetas, **circuit_kwargs)
    sim = AerSimulator(noise_model=noise_model)
    qc_t = transpile(qc, sim)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    return counts_to_probs_4q(counts)


def generate_dataset(
    num_samples,
    circuit_type="independent",
    noise_mode="synthetic",
    shots=1024,
    noise_kwargs=None,
    circuit_kwargs=None,
    snapshot_id: Optional[str] = None,
):
    """Generate (X_noisy, Y_ideal, thetas) arrays for one circuit/noise pair."""
    if noise_kwargs is None:
        noise_kwargs = {}
    if circuit_kwargs is None:
        circuit_kwargs = {}
    if circuit_type not in CIRCUIT_BUILDERS:
        raise ValueError("Unknown circuit type")

    active_snapshot = resolve_snapshot_id(snapshot_id)
    circuit_builder = CIRCUIT_BUILDERS[circuit_type]
    x_rows, y_rows, thetas_rows = [], [], []

    for _ in range(num_samples):
        thetas = np.random.uniform(0, np.pi, size=NUM_QUBITS)

        if noise_mode == "synthetic":
            p01 = np.random.uniform(
                noise_kwargs.get("p01_min", 0.05),
                noise_kwargs.get("p01_max", 0.15),
            )
            p10 = np.random.uniform(
                noise_kwargs.get("p10_min", 0.05),
                noise_kwargs.get("p10_max", 0.15),
            )
            noise_model = create_noise_model(
                mode="synthetic", p01=p01, p10=p10
            )
        else:
            noise_model = create_noise_model(
                mode=noise_mode,
                snapshot_id=active_snapshot,
            )

        p_ideal = get_probs(
            circuit_builder,
            thetas,
            noise_model=None,
            shots=shots,
            **circuit_kwargs,
        )

        if noise_mode == "experimental_correlated":
            p_noisy = apply_correlated_readout_noise(
                p_ideal,
                snapshot_id=active_snapshot,
            )
        else:
            p_noisy = get_probs(
                circuit_builder,
                thetas,
                noise_model=noise_model,
                shots=shots,
                **circuit_kwargs,
            )

        x_rows.append(p_noisy)
        y_rows.append(p_ideal)
        thetas_rows.append(thetas)

    return np.array(x_rows), np.array(y_rows), np.array(thetas_rows)


def save_dataset(
    X,
    Y,
    THETAS,
    split="train",
    circuit_type="independent",
    noise_mode="synthetic",
    snapshot_id: Optional[str] = None,
):
    """Save to DataTrain/<snapshot>/<circuit_type>/<noise_mode>/."""
    save_dir = get_ml_dataset_dir(
        circuit_type, noise_mode, snapshot_id=snapshot_id
    )
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, f"X_{split}.txt"), X)
    np.savetxt(os.path.join(save_dir, f"Y_{split}.txt"), Y)
    np.savetxt(os.path.join(save_dir, f"theta_{split}.txt"), THETAS)
    print("\nSaved dataset:")
    print(save_dir)


def load_dataset(
    split="train",
    circuit_type="independent",
    noise_mode="synthetic",
    snapshot_id: Optional[str] = None,
):
    """Load from DataTrain/<snapshot>/<circuit_type>/<noise_mode>/."""
    load_dir = get_ml_dataset_dir(
        circuit_type, noise_mode, snapshot_id=snapshot_id
    )
    x_path = os.path.join(load_dir, f"X_{split}.txt")
    if not os.path.isfile(x_path):
        raise FileNotFoundError(
            f"Dataset not found: {load_dir}\n"
            f"Set snapshot via set_active_snapshot('{DEFAULT_SNAPSHOT_ID}') "
            "or pass snapshot_id=..."
        )
    X = np.loadtxt(x_path)
    Y = np.loadtxt(os.path.join(load_dir, f"Y_{split}.txt"))
    THETAS = np.loadtxt(os.path.join(load_dir, f"theta_{split}.txt"))
    return X, Y, THETAS


def build_and_save_dataset(
    num_samples,
    split="train",
    circuit_type="independent",
    noise_mode="synthetic",
    shots=1024,
    noise_kwargs=None,
    circuit_kwargs=None,
    snapshot_id: Optional[str] = None,
):
    active_snapshot = resolve_snapshot_id(snapshot_id)
    print("GENERATING DATASET")
    print("================================================")
    print("Snapshot:", active_snapshot)
    print("Circuit :", circuit_type)
    print("Noise   :", noise_mode)
    print("Samples :", num_samples)

    X, Y, THETAS = generate_dataset(
        num_samples=num_samples,
        circuit_type=circuit_type,
        noise_mode=noise_mode,
        shots=shots,
        noise_kwargs=noise_kwargs,
        circuit_kwargs=circuit_kwargs,
        snapshot_id=active_snapshot,
    )
    save_dataset(
        X,
        Y,
        THETAS,
        split=split,
        circuit_type=circuit_type,
        noise_mode=noise_mode,
        snapshot_id=active_snapshot,
    )
    return X, Y, THETAS
