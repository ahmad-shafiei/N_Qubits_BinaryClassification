"""Plots for readout extraction pipeline."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .config import ReadoutDatasetConfig


def _state_labels(num_qubits: int) -> list[str]:
    width = num_qubits
    num_states = 2 ** num_qubits
    return [format(i, f"0{width}b") for i in range(num_states)]


def plot_qubit_iq_clouds(
    x_data: np.ndarray,
    y_data: np.ndarray,
    qubit_index: int,
    *,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    z = x_data[:, qubit_index]
    mask0 = y_data[:, qubit_index] == 0
    mask1 = y_data[:, qubit_index] == 1
    z0, z1 = z[mask0], z[mask1]

    plt.figure(figsize=(5, 5))
    plt.scatter(z0.real, z0.imag, s=8, alpha=0.4, label="State 0")
    plt.scatter(z1.real, z1.imag, s=8, alpha=0.4, label="State 1")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(f"Qubit {qubit_index + 1} IQ Clouds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_state_overlay(
    x_data: np.ndarray,
    y_data: np.ndarray,
    config: ReadoutDatasetConfig,
    *,
    target_states: Sequence[str] = ("1010", "0101"),
    qubits_to_plot: Optional[Iterable[int]] = None,
    max_points: int = 1000,
    alpha: float = 0.35,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    if qubits_to_plot is None:
        qubits_to_plot = list(range(config.num_qubits))
    else:
        qubits_to_plot = list(qubits_to_plot)

    n_qubits_plot = len(qubits_to_plot)
    fig, axes = plt.subplots(1, n_qubits_plot, figsize=(5 * n_qubits_plot, 4))
    if n_qubits_plot == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    rng = np.random.default_rng(config.random_seed)

    for state_idx, target_state in enumerate(target_states):
        if len(target_state) != config.num_qubits:
            raise ValueError(
                f"State {target_state} length != {config.num_qubits} qubits"
            )
        target_bits = np.array([int(b) for b in target_state[::-1]], dtype=int)
        mask = np.all(y_data == target_bits, axis=1)
        x_sel = x_data[mask]
        if len(x_sel) == 0:
            print(f"No samples for {target_state}")
            continue
        if len(x_sel) > max_points:
            idx = rng.choice(len(x_sel), size=max_points, replace=False)
            x_sel = x_sel[idx]

        for ax_idx, q in enumerate(qubits_to_plot):
            z = x_sel[:, q]
            axes[ax_idx].scatter(
                z.real,
                z.imag,
                s=8,
                alpha=alpha,
                color=colors[state_idx % len(colors)],
                label=target_state,
            )

    for ax_idx, q in enumerate(qubits_to_plot):
        axes[ax_idx].set_title(f"Qubit {q + 1}")
        axes[ax_idx].set_xlabel("I")
        axes[ax_idx].set_ylabel("Q")
        axes[ax_idx].grid(True)
        axes[ax_idx].legend()

    fig.suptitle(
        f"Overlay IQ Clouds ({config.snapshot_id})",
        fontsize=14,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


# Notebook alias (familiar name from readout_matrix.ipynb).
plot_fourqubit_state_overlay = plot_state_overlay


def plot_noise_matrix(
    matrix: np.ndarray,
    config: ReadoutDatasetConfig,
    *,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    cmap=None,
) -> None:
    labels = _state_labels(config.num_qubits)
    num_states = len(labels)

    plt.figure(figsize=(max(5, num_states * 0.35), max(4, num_states * 0.3)))
    plt.imshow(matrix, origin="lower", cmap=cmap)
    plt.colorbar(label="Probability")
    plt.xticks(range(num_states), labels, rotation=90)
    plt.yticks(range(num_states), labels)
    plt.xlabel("Measured State")
    plt.ylabel("Prepared State")
    plt.title(title or f"{num_states}x{num_states} Readout Noise Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_matrix_difference(
    delta: np.ndarray,
    config: ReadoutDatasetConfig,
    *,
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    plot_noise_matrix(
        delta,
        config,
        title=title,
        save_path=save_path,
        show=show,
        cmap="bwr",
    )
