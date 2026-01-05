import matplotlib.pyplot as plt

def plot_iq_scatter(samples1, samples2, label1="State1", label2="State2", qubit_index=0):
    """
    Scatter plot for IQ samples comparison between two states for a given qubit.
    
    samples1, samples2: np.ndarray, shape (time_steps, 2) -> [I, Q]
    qubit_index: int, index of qubit (for title only)
    """
    I1, Q1 = samples1[:,0], samples1[:,1]
    I2, Q2 = samples2[:,0], samples2[:,1]

    plt.figure(figsize=(5,5))
    plt.scatter(I1, Q1, s=10, alpha=0.5, color="blue", label=label1)
    plt.scatter(I2, Q2, s=10, alpha=0.5, color="red",  label=label2)

    plt.xlabel("I (Real)")
    plt.ylabel("Q (Imag)")
    plt.title(f"IQ Scatter — Qubit {qubit_index+1}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_iq_scatter_compare(iq_samples1, iq_samples2, label1="State1", label2="State2", qubit_index=0):
    """
    Plot IQ scatter for two different states for the same qubit.
    iq_samples1, iq_samples2: shape (time_steps, 2)  → [I,Q]
    """
    I1, Q1 = iq_samples1[:,0], iq_samples1[:,1]
    I2, Q2 = iq_samples2[:,0], iq_samples2[:,1]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.scatter(I1, Q1, s=10, alpha=0.5, color="blue", label=label1)
    plt.scatter(I2, Q2, s=10, alpha=0.5, color="red", label=label2)
    plt.xlabel("I (Real)")
    plt.ylabel("Q (Imag)")
    plt.title(f"IQ Scatter — Qubit {qubit_index+1}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
