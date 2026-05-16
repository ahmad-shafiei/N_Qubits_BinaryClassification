import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def compute_gaussian_params(samples):
    """
    Compute mean and covariance of IQ samples
    samples: np.ndarray, shape (time_steps,2)
    Returns: mean (2,), cov (2,2)
    """
    I, Q = samples[:,0], samples[:,1]
    mean = np.array([I.mean(), Q.mean()])
    cov = np.cov(I, Q)
    return mean, cov

def draw_ellipse(ax, mean, cov, color, label, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor="none", lw=2, label=label)
    ax.add_patch(ell)

def plot_gaussian_ellipse(samples1, samples2, label1="State1", label2="State2", qubit_index=0):
    """
    Scatter + Gaussian ellipses for two states
    samples1, samples2: np.ndarray, shape (time_steps, 2)
    """
    mean1, cov1 = compute_gaussian_params(samples1)
    mean2, cov2 = compute_gaussian_params(samples2)

    I1, Q1 = samples1[:,0], samples1[:,1]
    I2, Q2 = samples2[:,0], samples2[:,1]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(I1, Q1, s=10, alpha=0.3, color="blue", label=f"{label1} samples")
    ax.scatter(I2, Q2, s=10, alpha=0.3, color="red", label=f"{label2} samples")

    draw_ellipse(ax, mean1, cov1, "blue", f"{label1} ellipse")
    draw_ellipse(ax, mean2, cov2, "red", f"{label2} ellipse")

    # mark means
    ax.scatter(mean1[0], mean1[1], color="blue", marker="x", s=120)
    ax.scatter(mean2[0], mean2[1], color="red", marker="x", s=120)

    ax.set_title(f"Gaussian Ellipses — Qubit {qubit_index+1}")
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
