"""End-to-end readout noise extraction pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .config import (
    ReadoutDatasetConfig,
    ensure_output_dir,
    run_layout_migrations,
)
from .data import load_all_data
from .matrix import (
    build_independent_noise_matrix,
    build_noise_matrix,
    compare_noise_models,
    predict_full_dataset,
    save_noise_matrix,
    train_all_classifiers,
    validate_noise_matrix,
)
from .plots import plot_matrix_difference, plot_noise_matrix, plot_state_overlay


def run_readout_pipeline(
    config: ReadoutDatasetConfig,
    *,
    save_plots: bool = True,
    compare_independent: bool = True,
    overlay_states: Optional[tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Run the full workflow from readout_matrix.ipynb for one dataset."""
    ensure_output_dir(config)
    print("=" * 60)
    print(f"Campaign : {config.campaign_id}")
    print(f"Snapshot : {config.snapshot_id}")
    print(f"Result   : {config.result_id}")
    print(f"Input    : {config.data_dir}")
    print(f"Output   : {config.output_dir}")
    print(f"Qubits   : {config.num_qubits}")
    print(f"Classifier: {config.classifier_type}")
    print("=" * 60)

    # Step 1 — load IQ data
    x_data, y_data, shots_per_file = load_all_data(config)

    # Step 2 — optional IQ overlay visualization
    if save_plots:
        if overlay_states is None and config.num_qubits == 4:
            overlay_states = ("1010", "0101")
        if overlay_states:
            plot_state_overlay(
                x_data,
                y_data,
                config,
                target_states=overlay_states,
                save_path=os.path.join(config.output_dir, "iq_overlay.png"),
            )

    # Step 3 — train per-qubit classifiers
    classifiers, scalers, assignment_matrices = train_all_classifiers(
        x_data, y_data, config
    )

    # Step 4 — predict all shots
    y_pred = predict_full_dataset(
        x_data, classifiers, scalers, config.num_qubits
    )

    # Step 5 — build full correlated noise matrix
    noise_matrix = build_noise_matrix(y_data, y_pred, config.num_qubits)
    fidelity = float(np.mean(np.diag(noise_matrix)))
    print(f"\nAverage assignment fidelity = {fidelity:.6f}")

    # Step 6 — validate
    validation = validate_noise_matrix(noise_matrix)

    # Step 7 — save matrices
    save_noise_matrix(noise_matrix, config.output_dir, config.num_qubits)

    comparison: Optional[Dict[str, Any]] = None
    matrix_indep = None
    if compare_independent:
        matrix_indep = build_independent_noise_matrix(assignment_matrices)
        comparison = compare_noise_models(noise_matrix, matrix_indep)
        comparison_path = os.path.join(
            config.output_dir, "comparison_independent.txt"
        )
        _write_comparison_report(comparison_path, comparison)

        if save_plots:
            size = config.num_states
            plot_noise_matrix(
                noise_matrix,
                config,
                title=f"Correlated {size}x{size} Readout Matrix",
                save_path=os.path.join(
                    config.output_dir, "noise_matrix_heatmap.png"
                ),
            )
            plot_noise_matrix(
                matrix_indep,
                config,
                title=f"Independent Kronecker {size}x{size} Model",
                save_path=os.path.join(
                    config.output_dir, "noise_matrix_independent_heatmap.png"
                ),
            )
            plot_matrix_difference(
                noise_matrix - matrix_indep,
                config,
                title="Difference: Correlated - Independent",
                save_path=os.path.join(
                    config.output_dir, "noise_matrix_difference_heatmap.png"
                ),
            )
    elif save_plots:
        plot_noise_matrix(
            noise_matrix,
            config,
            save_path=os.path.join(config.output_dir, "noise_matrix_heatmap.png"),
        )

    metadata = {
        "campaign_id": config.campaign_id,
        "snapshot_id": config.snapshot_id,
        "result_id": config.result_id,
        "dataset_id": config.snapshot_id,
        "data_dir": config.data_dir,
        "campaign_dir": config.campaign_dir,
        "output_dir": config.output_dir,
        "num_qubits": config.num_qubits,
        "num_states": config.num_states,
        "shots_per_file": shots_per_file,
        "classifier_type": config.classifier_type,
        "qda_reg_param": config.qda_reg_param,
        "random_seed": config.random_seed,
        "mean_diagonal_fidelity": fidelity,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
    }
    if comparison:
        metadata["comparison_independent"] = {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in comparison.items()
        }

    metadata_path = os.path.join(config.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nSaved results to {config.output_dir}")
    return {
        "noise_matrix": noise_matrix,
        "assignment_matrices": assignment_matrices,
        "independent_matrix": matrix_indep,
        "validation": validation,
        "comparison": comparison,
        "metadata_path": metadata_path,
    }


def run_all_discovered_datasets(
    search_root: Optional[str] = None,
    **pipeline_kwargs,
) -> List[Dict[str, Any]]:
    from .config import discover_datasets

    run_layout_migrations()
    results = []
    for data_dir in discover_datasets(search_root):
        config = ReadoutDatasetConfig.from_data_dir(data_dir)
        results.append(run_readout_pipeline(config, **pipeline_kwargs))
    return results


def _write_comparison_report(path: str, comparison: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("COMPARISON: Correlated vs Independent Kronecker\n")
        handle.write("=" * 48 + "\n")
        handle.write(f"Frobenius norm : {comparison['frobenius_norm']:.6f}\n")
        handle.write(f"Max abs diff   : {comparison['max_abs_diff']:.6f}\n")
        handle.write(f"Mean abs diff  : {comparison['mean_abs_diff']:.6f}\n")
        handle.write("\nDiagonal fidelities (REAL):\n")
        handle.write(f"{np.round(comparison['diag_real'], 6)}\n")
        handle.write("\nDiagonal fidelities (INDEPENDENT):\n")
        handle.write(f"{np.round(comparison['diag_indep'], 6)}\n")
