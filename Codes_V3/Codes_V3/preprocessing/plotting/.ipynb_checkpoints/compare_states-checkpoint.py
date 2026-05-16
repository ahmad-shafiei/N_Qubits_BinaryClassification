# Codes_V3/preprocessing/plotting/compare_states.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ..utils import RawIQData  # کلاس داده‌ها

# -------------------------------------------
# Extract samples for a specific qubit and state label
# -------------------------------------------
def get_qubit_samples(raw: RawIQData, state_label: np.ndarray, qubit_index: int):
    """
    raw: RawIQData object
    state_label: np.ndarray, shape=(n_qubits,), e.g., [0,1,0,0]
    qubit_index: int, 0..n_qubits-1
    Returns complex samples of shape (n_shots,)
    """
    mask = np.all(raw.labels == state_label, axis=1)
    return raw.data[mask, qubit_index, :, :].view(np.complex128).reshape(-1)


# -------------------------------------------
# Scatter plot of IQ
# -------------------------------------------
def plot_iq_scatter(samples1, samples2, state1_label, state2_label, qubit_index):
    I1, Q1 = samples1.real, samples1.imag
    I2, Q2 = samples2.real, samples2.imag

    plt.figure(figsize=(5,5))
    plt.scatter(I1, Q1, s=10, alpha=0.5, label=str(state1_label), color="blue")
    plt.scatter(I2, Q2, s=10, alpha=0.5, label=str(state2_label), color="red")
    plt.xlabel("I (Real)")
    plt.ylabel("Q (Imag)")
    plt.title(f"IQ Scatter — Qubit {qubit_index+1}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------
# Histogram + CDF
# -------------------------------------------
def plot_iq_histograms(samples1, samples2, state1_label, state2_label, qubit_index, bins=50):
    I1, Q1 = samples1.real, samples1.imag
    I2, Q2 = samples2.real, samples2.imag

    fig, axes = plt.subplots(2,2, figsize=(6,5))

    axes[0,0].hist(I1, bins=bins, alpha=0.6, color="blue", label=str(state1_label))
    axes[0,0].hist(I2, bins=bins, alpha=0.6, color="red",  label=str(state2_label))
    axes[0,0].set_title("Histogram — I (Real)")
    axes[0,0].legend()

    axes[0,1].hist(Q1, bins=bins, alpha=0.6, color="blue", label=str(state1_label))
    axes[0,1].hist(Q2, bins=bins, alpha=0.6, color="red",  label=str(state2_label))
    axes[0,1].set_title("Histogram — Q (Imag)")
    axes[0,1].legend()

    axes[1,0].plot(np.sort(I1), np.linspace(0,1,len(I1)), color="blue")
    axes[1,0].plot(np.sort(I2), np.linspace(0,1,len(I2)), color="red")
    axes[1,0].set_title("CDF — I (Real)")

    axes[1,1].plot(np.sort(Q1), np.linspace(0,1,len(Q1)), color="blue")
    axes[1,1].plot(np.sort(Q2), np.linspace(0,1,len(Q2)), color="red")
    axes[1,1].set_title("CDF — Q (Imag)")

    plt.suptitle(f"Qubit {qubit_index+1} — {str(state1_label)} vs {str(state2_label)}")
    plt.tight_layout()
    plt.show()


# -------------------------------------------
# Gaussian ellipse
# -------------------------------------------
def compute_gaussian_params(samples):
    I, Q = samples.real, samples.imag
    mean = np.array([I.mean(), Q.mean()])
    cov = np.cov(I, Q)
    return mean, cov

def draw_ellipse(ax, mean, cov, color, label, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2*n_std*np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor='none', lw=2, label=label)
    ax.add_patch(ell)

def plot_gaussian_ellipses(samples1, samples2, state1_label, state2_label, qubit_index):
    mean1, cov1 = compute_gaussian_params(samples1)
    mean2, cov2 = compute_gaussian_params(samples2)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(samples1.real, samples1.imag, s=10, alpha=0.3, color="blue", label=str(state1_label))
    ax.scatter(samples2.real, samples2.imag, s=10, alpha=0.3, color="red", label=str(state2_label))

    draw_ellipse(ax, mean1, cov1, "blue", f"{state1_label} ellipse")
    draw_ellipse(ax, mean2, cov2, "red", f"{state2_label} ellipse")

    ax.scatter(mean1[0], mean1[1], color="blue", marker="x", s=120)
    ax.scatter(mean2[0], mean2[1], color="red", marker="x", s=120)

    ax.set_title(f"Gaussian Ellipses — Qubit {qubit_index+1}")
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
