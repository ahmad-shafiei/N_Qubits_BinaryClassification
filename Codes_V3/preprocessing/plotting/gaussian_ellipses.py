import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from raw_data.iq_container import RawIQData
from preprocessing.utils import get_complex_iq


def _gaussian_params(samples: np.ndarray):
    I, Q = samples.real, samples.imag
    mean = np.array([I.mean(), Q.mean()])
    cov = np.cov(I, Q)
    return mean, cov


def _draw_ellipse(ax, mean, cov, color, n_std=2.0, label=None):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        lw=2,
        label=label,
    )
    ax.add_patch(ellipse)


def plot_gaussian_ellipse(
    raw: RawIQData,
    shot: int,
    qubit: int,
    max_points: int | None = None,
):
    """
    Scatter + Gaussian covariance ellipse for one qubit and one shot.
    """
    samples = get_complex_iq(raw, shot, qubit, max_points)
    mean, cov = _gaussian_params(samples)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        samples.real,
        samples.imag,
        s=10,
        alpha=0.4,
        label="Samples",
    )

    _draw_ellipse(ax, mean, cov, color="red", label="2σ ellipse")
    ax.scatter(mean[0], mean[1], marker="x", s=100, color="red")

    ax.set_title(f"Gaussian Ellipse — Qubit {qubit+1}, Shot {shot}")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
