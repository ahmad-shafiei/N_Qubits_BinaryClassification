import numpy as np
from raw_data.iq_container import RawIQData


def get_complex_iq(
    raw: RawIQData,
    shot: int,
    qubit: int,
    max_points: int | None = None,
):
    """
    Extract complex IQ samples for a given shot and qubit.

    Returns
    -------
    samples : np.ndarray (complex), shape (time_steps,)
    """
    iq = raw.data[shot, qubit]  # (time_steps, 2)
    if max_points is not None:
        iq = iq[:max_points]

    return iq[:, 0] + 1j * iq[:, 1]


def get_state_mask(raw: RawIQData, state_bits: list[int]):
    """
    Return boolean mask for shots matching a given multi-qubit state.
    state_bits = [q1, q2, ..., qN]
    """
    state = np.array(state_bits, dtype=int)
    return np.all(raw.labels == state, axis=1)
