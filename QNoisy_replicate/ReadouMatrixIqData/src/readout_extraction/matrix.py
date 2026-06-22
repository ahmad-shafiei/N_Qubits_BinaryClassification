"""Classifier training and noise-matrix construction."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ReadoutDatasetConfig
from .data import bits_to_int, int_to_bits


def train_single_qubit_classifier(
    x_data: np.ndarray,
    y_data: np.ndarray,
    qubit_index: int,
    config: ReadoutDatasetConfig,
) -> Tuple[object, StandardScaler, np.ndarray]:
    print("================================================")
    print(f"QUBIT {qubit_index + 1}")

    z = x_data[:, qubit_index]
    features = np.column_stack([z.real, z.imag])
    labels = y_data[:, qubit_index]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    if config.classifier_type == "LDA":
        classifier = LinearDiscriminantAnalysis()
    elif config.classifier_type == "QDA":
        classifier = QuadraticDiscriminantAnalysis(
            reg_param=config.qda_reg_param
        )
    else:
        raise ValueError(f"Unknown classifier: {config.classifier_type}")

    classifier.fit(x_train_s, y_train)
    y_pred = classifier.predict(x_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy = {accuracy:.6f}")

    assignment = confusion_matrix(
        y_test, y_pred, labels=[0, 1], normalize="true"
    )
    print("Single-Qubit Assignment Matrix:")
    print(assignment)

    out_path = os.path.join(
        config.output_dir,
        f"assignment_matrix_q{qubit_index + 1}.txt",
    )
    np.savetxt(out_path, assignment)
    np.save(out_path.replace(".txt", ".npy"), assignment)
    return classifier, scaler, assignment


def train_all_classifiers(
    x_data: np.ndarray,
    y_data: np.ndarray,
    config: ReadoutDatasetConfig,
) -> Tuple[List[object], List[StandardScaler], List[np.ndarray]]:
    classifiers, scalers, assignment_matrices = [], [], []
    for q in range(config.num_qubits):
        clf, scaler, matrix = train_single_qubit_classifier(
            x_data, y_data, q, config
        )
        classifiers.append(clf)
        scalers.append(scaler)
        assignment_matrices.append(matrix)
    return classifiers, scalers, assignment_matrices


def predict_full_dataset(
    x_data: np.ndarray,
    classifiers: Sequence[object],
    scalers: Sequence[StandardScaler],
    num_qubits: int,
) -> np.ndarray:
    n_samples = x_data.shape[0]
    y_pred = np.zeros((n_samples, num_qubits), dtype=int)
    for q in range(num_qubits):
        z = x_data[:, q]
        features = np.column_stack([z.real, z.imag])
        features_s = scalers[q].transform(features)
        y_pred[:, q] = classifiers[q].predict(features_s)
    return y_pred


def build_noise_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    num_states = 2 ** num_qubits
    matrix = np.zeros((num_states, num_states))

    for prep_state in range(num_states):
        prep_bits = int_to_bits(prep_state, num_qubits)
        mask = np.all(y_true == prep_bits, axis=1)
        subset = y_pred[mask]
        total = subset.shape[0]
        if total == 0:
            raise ValueError(f"No samples for prepared state {prep_state}")
        for bits in subset:
            meas_state = bits_to_int(bits)
            matrix[prep_state, meas_state] += 1
        matrix[prep_state, :] /= total
    return matrix


def validate_noise_matrix(matrix: np.ndarray) -> dict:
    print("VALIDATION")
    print("================================================")
    row_sums = matrix.sum(axis=1)
    print("Row sums:")
    print(row_sums)

    diag = np.diag(matrix)
    print("Diagonal elements:")
    print(diag)
    mean_diag = float(np.mean(diag))
    print("Average diagonal fidelity:")
    print(mean_diag)
    return {
        "row_sums": row_sums,
        "diagonal": diag,
        "mean_diagonal_fidelity": mean_diag,
    }


def save_noise_matrix(matrix: np.ndarray, output_dir: str, num_qubits: int) -> None:
    size = 2 ** num_qubits
    base = f"noise_matrix_{size}x{size}"
    txt_path = os.path.join(output_dir, f"{base}.txt")
    np.savetxt(txt_path, matrix, fmt="%.8f")
    np.save(os.path.join(output_dir, f"{base}.npy"), matrix)

    # Backward-compatible alias for 4-qubit workflows.
    if num_qubits == 4:
        np.savetxt(os.path.join(output_dir, "noise_matrix_16x16.txt"), matrix, fmt="%.8f")
        np.save(os.path.join(output_dir, "noise_matrix_16x16.npy"), matrix)


def build_independent_noise_matrix(
    assignment_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    result = assignment_matrices[-1]
    for matrix in reversed(assignment_matrices[:-1]):
        result = np.kron(result, matrix)
    return result


def compare_noise_models(
    matrix_real: np.ndarray,
    matrix_indep: np.ndarray,
) -> dict:
    delta = matrix_real - matrix_indep
    metrics = {
        "frobenius_norm": float(np.linalg.norm(delta)),
        "max_abs_diff": float(np.max(np.abs(delta))),
        "mean_abs_diff": float(np.mean(np.abs(delta))),
        "diag_real": np.diag(matrix_real),
        "diag_indep": np.diag(matrix_indep),
    }
    print("\n================================================")
    print("COMPARISON RESULTS")
    print(f"\nFrobenius norm : {metrics['frobenius_norm']:.6f}")
    print(f"Max abs diff   : {metrics['max_abs_diff']:.6f}")
    print(f"Mean abs diff  : {metrics['mean_abs_diff']:.6f}")
    print("\nDiagonal fidelities (REAL):")
    print(np.round(metrics["diag_real"], 4))
    print("\nDiagonal fidelities (INDEPENDENT):")
    print(np.round(metrics["diag_indep"], 4))
    return metrics


def run_noise_extraction(
    x_data: np.ndarray,
    y_data: np.ndarray,
    config: ReadoutDatasetConfig,
    *,
    classifier_type: str | None = None,
    qda_reg_param: float | None = None,
) -> np.ndarray:
    """Train classifiers, predict, build and validate the noise matrix."""
    run_config = config
    overrides = {}
    if classifier_type is not None:
        overrides["classifier_type"] = classifier_type
    if qda_reg_param is not None:
        overrides["qda_reg_param"] = qda_reg_param
    if overrides:
        run_config = replace(config, **overrides)

    classifiers, scalers, _ = train_all_classifiers(x_data, y_data, run_config)
    y_pred = predict_full_dataset(
        x_data, classifiers, scalers, run_config.num_qubits
    )
    matrix = build_noise_matrix(y_data, y_pred, run_config.num_qubits)
    validate_noise_matrix(matrix)
    return matrix
