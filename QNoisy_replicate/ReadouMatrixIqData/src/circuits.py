import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
# ============================================================
# INDEPENDENT CIRCUIT
def build_circuit_independent(thetas,measure=True,**kwargs):
    qc = QuantumCircuit(4, 4)
    for i in range(4):
        qc.ry(thetas[i], i)
    if measure:
        qc.measure([0,1,2,3], [0,1,2,3])
    return qc
# ============================================================
# RZZ CORRELATED CIRCUIT
def build_correlated_circuit(
    thetas,theta_rzz=np.pi/2,measure=True,**kwargs):
    qc = QuantumCircuit(4, 4)
    # amplitude encoding
    for i in range(4):
        qc.ry(thetas[i], i)
    # basis rotation
    for i in range(4):
        qc.h(i)
    # entangling layer
    for j in range(4):
        for k in range(j+1, 4):
            qc.rzz(theta_rzz, j, k)
    # return basis
    for i in range(4):
        qc.h(i)
    if measure:
        qc.measure([0,1,2,3], [0,1,2,3])
    return qc
# ============================================================
# ZZ FEATURE MAP CIRCUIT
def build_zzfeaturemap_circuit(
    thetas,reps=1,entanglement="full", measure=True, **kwargs):
    feature_map = zz_feature_map(feature_dimension=4,
        reps=reps,entanglement=entanglement)
    qc = QuantumCircuit(4, 4)
    qc.compose(feature_map.assign_parameters(thetas),inplace=True )
    if measure:
        qc.measure([0,1,2,3], [0,1,2,3])
    return qc