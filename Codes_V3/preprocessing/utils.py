import numpy as np

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
def load_dataset(X_path, y_path, dtype_complex=complex):
    """
    X: shape (N_samples, N_qubits)
    y: shape (N_samples, N_qubits)
    """
    X = np.loadtxt(X_path, dtype=dtype_complex)
    y = np.loadtxt(y_path, dtype=int)
    return X, y


# -----------------------------------------------------------
# Convert state string → label vector
# state_str = Q_N ... Q_2 Q_1
# label     = [q1, q2, ..., qN]
# -----------------------------------------------------------
def state_to_label(state_str):
    if any(c not in "01" for c in state_str):
        raise ValueError("State must be a binary string like '0101'.")

    bits = np.array([int(b) for b in state_str], dtype=int)
    return bits[::-1]   # reverse: Q_N..Q1 → q1..qN


# -----------------------------------------------------------
# Extract samples for a given qubit and state
# -----------------------------------------------------------
def get_qubit_samples(X, y, state_str, qubit_index):
    target = state_to_label(state_str)

    if y.shape[1] != len(target):
        raise ValueError("State length does not match number of qubits.")

    mask = np.all(y == target, axis=1)
    return X[mask, qubit_index]
