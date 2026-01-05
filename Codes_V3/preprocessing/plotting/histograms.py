import numpy as np
import matplotlib.pyplot as plt
from raw_data.iq_container import RawIQData
from preprocessing.utils import get_complex_iq


def plot_iq_histograms(
    raw: RawIQData,
    shot: int = 0,
    qubit: int = 0,
    bins: int = 50,
    max_points: int | None = None,
):
    """
    Plot histograms and CDFs of I and Q components.
    """
    samples = get_complex_iq(raw, shot, qubit, max_points)
    I, Q = samples.real, samples.imag

    fig, axes = plt.subplots(2, 2, figsize=(5, 4))

    axes[0, 0].hist(I, bins=bins, alpha=0.7)
    axes[0, 0].set_title("Histogram — I")

    axes[0, 1].hist(Q, bins=bins, alpha=0.7)
    axes[0, 1].set_title("Histogram — Q")

    axes[1, 0].plot(np.sort(I), np.linspace(0, 1, len(I)))
    axes[1, 0].set_title("CDF — I")

    axes[1, 1].plot(np.sort(Q), np.linspace(0, 1, len(Q)))
    axes[1, 1].set_title("CDF — Q")

    fig.suptitle(f"Qubit {qubit+1}, Shot {shot}")
    plt.tight_layout()
    plt.show()
