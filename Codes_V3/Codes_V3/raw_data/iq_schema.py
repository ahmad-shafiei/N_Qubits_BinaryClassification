# Codes_V3/raw_data/iq_schema.py

"""
Raw IQ Trace Schema

Expected data shape:
    (shots, qubits, time, 2)

Last dimension:
    0 -> I
    1 -> Q
"""

IQ_DIM = 2


def validate_iq_array(x):
    if x.ndim != 4:
        raise ValueError(
            f"Raw IQ data must be 4D: (shots, qubits, time, 2), got {x.ndim}D"
        )

    if x.shape[-1] != IQ_DIM:
        raise ValueError(
            f"Last dimension must be size {IQ_DIM} (I, Q), got {x.shape[-1]}"
        )
