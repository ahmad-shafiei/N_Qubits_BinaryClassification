import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Histograms + CDFs
# -----------------------------------------------------------
def plot_histograms_and_cdf(samples1, samples2,
                            state1, state2,
                            qubit_index):

    I1, Q1 = samples1.real, samples1.imag
    I2, Q2 = samples2.real, samples2.imag

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))

    # Histogram I
    axes[0, 0].hist(I1, bins=50, alpha=0.6, label=state1)
    axes[0, 0].hist(I2, bins=50, alpha=0.6, label=state2)
    axes[0, 0].set_title("Histogram — I")
    axes[0, 0].legend()

    # Histogram Q
    axes[0, 1].hist(Q1, bins=50, alpha=0.6, label=state1)
    axes[0, 1].hist(Q2, bins=50, alpha=0.6, label=state2)
    axes[0, 1].set_title("Histogram — Q")
    axes[0, 1].legend()

    # CDF I
    axes[1, 0].plot(np.sort(I1), np.linspace(0, 1, len(I1)))
    axes[1, 0].plot(np.sort(I2), np.linspace(0, 1, len(I2)))
    axes[1, 0].set_title("CDF — I")

    # CDF Q
    axes[1, 1].plot(np.sort(Q1), np.linspace(0, 1, len(Q1)))
    axes[1, 1].plot(np.sort(Q2), np.linspace(0, 1, len(Q2)))
    axes[1, 1].set_title("CDF — Q")

    fig.suptitle(f"Statistics — Qubit {qubit_index + 1}")
    plt.tight_layout()
    plt.show()
