import matplotlib.pyplot as plt

# -----------------------------------------------------------
# IQ scatter comparison
# -----------------------------------------------------------
def plot_iq_scatter(samples1, samples2,
                    state1, state2,
                    qubit_index,
                    ax=None):

    I1, Q1 = samples1.real, samples1.imag
    I2, Q2 = samples2.real, samples2.imag

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(I1, Q1, s=10, alpha=0.4, label=state1)
    ax.scatter(I2, Q2, s=10, alpha=0.4, label=state2)

    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"IQ Scatter — Qubit {qubit_index + 1}")
    ax.grid(True)
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()
