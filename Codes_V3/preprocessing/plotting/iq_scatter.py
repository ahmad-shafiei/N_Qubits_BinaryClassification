import matplotlib.pyplot as plt
from raw_data.iq_container import RawIQData
from preprocessing.utils import get_complex_iq


def plot_iq_scatter(
    raw: RawIQData,
    shot: int = 0,
    qubit: int = 0,
    max_points: int = 500,
):
    """
    Scatter plot of IQ samples for one qubit and one shot.
    """
    samples = get_complex_iq(raw, shot, qubit, max_points)

    plt.figure(figsize=(4, 4))
    plt.scatter(samples.real, samples.imag, s=10, alpha=0.6)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(f"IQ Scatter — Shot {shot}, Qubit {qubit+1}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
