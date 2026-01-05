import matplotlib.pyplot as plt
from raw_data.iq_container import RawIQData

def plot_iq_trace(
    raw: RawIQData,
    shot: int = 0,
    qubit: int = 0,
    max_points: int = 300,
):
    iq = raw.data[shot, qubit, :max_points, :]
    I = iq[:, 0]
    Q = iq[:, 1]

    plt.figure()
    plt.plot(I, label="I")
    plt.plot(Q, label="Q")
    plt.title(f"Shot {shot}, Qubit {qubit}")
    plt.xlabel("Time index")
    plt.legend()
    plt.show()


def plot_iq_plane(
    raw: RawIQData,
    shot: int = 0,
    qubit: int = 0,
    max_points: int = 1000,
):
    iq = raw.data[shot, qubit, :max_points, :]

    plt.figure()
    plt.scatter(iq[:, 0], iq[:, 1], s=5)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(f"IQ Plane | Shot {shot}, Qubit {qubit}")
    plt.axis("equal")
    plt.show()
