import matplotlib.pyplot as plt
import numpy as np

def plot_iq_histograms(samples1, samples2, label1="State1", label2="State2", qubit_index=0):
    """
    Plot histograms + CDFs for IQ samples comparison between two states.
    
    samples1, samples2: np.ndarray, shape (time_steps, 2) -> [I,Q]
    """
    I1, Q1 = samples1[:,0], samples1[:,1]
    I2, Q2 = samples2[:,0], samples2[:,1]

    fig, axes = plt.subplots(2,2, figsize=(6,5))

    # Histogram I
    axes[0,0].hist(I1, bins=50, alpha=0.6, color="blue", label=label1)
    axes[0,0].hist(I2, bins=50, alpha=0.6, color="red", label=label2)
    axes[0,0].set_title("Histogram — I")
    axes[0,0].legend()

    # Histogram Q
    axes[0,1].hist(Q1, bins=50, alpha=0.6, color="blue", label=label1)
    axes[0,1].hist(Q2, bins=50, alpha=0.6, color="red", label=label2)
    axes[0,1].set_title("Histogram — Q")
    axes[0,1].legend()

    # CDF I
    axes[1,0].plot(np.sort(I1), np.linspace(0,1,len(I1)), color="blue", label=label1)
    axes[1,0].plot(np.sort(I2), np.linspace(0,1,len(I2)), color="red", label=label2)
    axes[1,0].set_title("CDF — I")

    # CDF Q
    axes[1,1].plot(np.sort(Q1), np.linspace(0,1,len(Q1)), color="blue", label=label1)
    axes[1,1].plot(np.sort(Q2), np.linspace(0,1,len(Q2)), color="red", label=label2)
    axes[1,1].set_title("CDF — Q")

    plt.suptitle(f"Qubit {qubit_index+1}")
    plt.tight_layout()
    plt.show()
