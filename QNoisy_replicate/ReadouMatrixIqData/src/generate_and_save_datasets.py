import numpy as np 
from qiskit import transpile 
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from .noises import create_noise_model, apply_correlated_readout_noise
from .circuits import build_circuit_independent, build_zzfeaturemap_circuit

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "DataTrain")

NUM_QUBITS = 4
CIRCUIT_BUILDERS = {"independent":
        build_circuit_independent,
        "zz_featuremap":
        build_zzfeaturemap_circuit,}
# ============================================================
def counts_to_probs_4q(counts):
    probs = np.zeros(16)
    for bitstring, count in counts.items():
        probs[int(bitstring, 2)] = count
    return probs / np.sum(probs)
# RUN CIRCUIT
# ============================================================
def get_probs(circuit_builder,thetas,noise_model=None,
    shots=1024,**circuit_kwargs):
    qc = circuit_builder(thetas,**circuit_kwargs)
    sim = AerSimulator(noise_model=noise_model)
    qc_t = transpile(qc, sim)
    result = sim.run(qc_t,shots=shots).result()
    counts = result.get_counts()
    return counts_to_probs_4q(counts)
# GENERATE DATASET
# ============================================================
def generate_dataset(num_samples,circuit_type="independent",
    noise_mode="synthetic",shots=1024,noise_kwargs=None,circuit_kwargs=None):
    """ Generate dataset.
    INPUT:
    circuit_type:
        independent
        zz_featuremap
    noise_mode:
        synthetic
        experimental_single
        experimental_correlated """
    if noise_kwargs is None:
        noise_kwargs = {}
    if circuit_kwargs is None:
        circuit_kwargs = {}
    if circuit_type not in CIRCUIT_BUILDERS:
        raise ValueError("Unknown circuit type")
    circuit_builder = CIRCUIT_BUILDERS[circuit_type]
    X, Y, THETAS  = [], [], []
    for _ in range(num_samples):
        thetas = np.random.uniform(0,np.pi, size=NUM_QUBITS )   ######
        # thetas = np.random.uniform(0,np.pi/4, size=NUM_QUBITS )
        # CREATE NOISE MODEL
        # --------------------------------------------
        if noise_mode == "synthetic":
            p01 = np.random.uniform(noise_kwargs.get("p01_min", 0.05),
                noise_kwargs.get("p01_max", 0.15))
            p10 = np.random.uniform(noise_kwargs.get("p10_min", 0.05),
                noise_kwargs.get("p10_max", 0.15) )
            noise_model = create_noise_model(mode="synthetic", p01=p01,p10=p10)
        else:
            noise_model = create_noise_model(mode=noise_mode)
        # IDEAL
        # --------------------------------------------
        p_ideal = get_probs(circuit_builder,thetas,
            noise_model=None,shots=shots,**circuit_kwargs)      
        # NOISY
        # --------------------------------------------
        if noise_mode == "experimental_correlated":
            p_noisy = apply_correlated_readout_noise(p_ideal )
        else:
            p_noisy = get_probs(circuit_builder,thetas,
             noise_model=noise_model,shots=shots,**circuit_kwargs )
        X.append(p_noisy)
        Y.append(p_ideal)
        THETAS.append(thetas)
    return (np.array(X),np.array(Y), np.array(THETAS))
# SAVE DATASET
# ============================================================
def save_dataset(X,Y,THETAS,split="train",
    circuit_type="independent",noise_mode="synthetic"):
    """ Save dataset to:
     DataTrain/
        circuit_type/
            noise_mode/ """
    save_dir = os.path.join(DATA_ROOT,circuit_type, noise_mode)
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, f"X_{split}.txt"),X)
    np.savetxt(os.path.join(save_dir, f"Y_{split}.txt"),Y )
    np.savetxt(os.path.join(save_dir, f"theta_{split}.txt"),THETAS)
    print("\nSaved dataset:")
    print(save_dir)
# LOAD DATASET
# ============================================================
def load_dataset(split="train",circuit_type="independent",
    noise_mode="synthetic"):
    load_dir = os.path.join(DATA_ROOT,circuit_type,noise_mode)
    X = np.loadtxt(os.path.join(load_dir, f"X_{split}.txt"))
    Y = np.loadtxt(os.path.join(load_dir, f"Y_{split}.txt"))
    THETAS = np.loadtxt(os.path.join(load_dir, f"theta_{split}.txt"))
    return X, Y, THETAS
# BUILD + SAVE DATASET
# ============================================================
def build_and_save_dataset(
    num_samples,split="train",circuit_type="independent",
    noise_mode="synthetic",shots=1024,noise_kwargs=None,
    circuit_kwargs=None):
    print("GENERATING DATASET")
    print("================================================")
    print("Circuit :", circuit_type)
    print("Noise   :", noise_mode)
    print("Samples :", num_samples)
    X, Y, THETAS = generate_dataset(
        num_samples=num_samples,
        circuit_type=circuit_type,
        noise_mode=noise_mode,
        shots=shots,noise_kwargs=noise_kwargs,
        circuit_kwargs=circuit_kwargs)
    save_dataset(X,Y,THETAS,
        split=split,circuit_type=circuit_type,noise_mode=noise_mode)
    return X, Y, THETAS