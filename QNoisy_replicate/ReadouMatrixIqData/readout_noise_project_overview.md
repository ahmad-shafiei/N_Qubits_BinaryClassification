# Quantum Readout Noise Modeling and Dataset Generation Project

## Project Overview

This project focuses on modeling and learning quantum readout noise for 4-qubit quantum systems using both:

1. Real experimental IQ measurement data
2. Simulated noisy quantum circuits

The workflow is divided into two major sections:

- Part 1: Experimental Readout Noise Extraction
- Part 2: Noisy Quantum Circuit Dataset Generation

---

# Part 1 — Experimental Readout Noise Extraction

## Main Goal

The first part of the project extracts:

- Single-qubit readout assignment matrices (2×2)
- Full correlated 4-qubit readout noise matrix (16×16)

from real experimental IQ measurement data.

The generated matrices are later used as realistic noise models for quantum circuit simulation and machine learning dataset generation.

---

# Main Notebook

## File

readout_matrix.ipynb

This notebook is responsible for:

- Loading experimental IQ data
- Parsing complex-valued readout samples
- Training qubit state classifiers
- Constructing single-qubit readout matrices
- Building the correlated 16×16 readout noise matrix
- Visualizing the extracted noise
- Saving all extracted matrices to disk

---

# Experimental Dataset Structure

## Input Directory

./quadrature_data_4qubits

This directory contains:

0000.txt
0001.txt
0010.txt
...
1111.txt

A total of 16 files corresponding to all possible prepared computational basis states of 4 qubits.

---

# Structure of Each Experimental File

Each file contains:

- 4 lines
- Each line corresponds to one qubit
- Each line contains 1000 complex IQ samples

Therefore:

(4 qubits) × (1000 shots)

per prepared basis state.

---

# IQ Data Representation

The measurement data are stored as complex numbers:

I + iQ

where:

- Real part → I quadrature
- Imaginary part → Q quadrature

The parser function:

parse_complex_string()

converts the textual representation into np.complex128 objects.

---

# Important Experimental Pipeline

## Step 1 — Load Experimental Files

Main function:

load_all_data()

Responsibilities:

- Load all 16 prepared-state files
- Preserve ordering of basis states
- Build:
  - X → IQ data
  - Y → prepared labels

Final dataset shapes:

X.shape = (16000, 4)
Y.shape = (16000, 4)

because:

16 states × 1000 shots = 16000 samples

---

## Step 2 — IQ Cloud Visualization

Main function:

plot_qubit_iq_clouds()

Purpose:

- Visualize separation between |0⟩ and |1⟩ measurement clouds for each qubit.

This visualization is critical for:

- Understanding readout quality
- Estimating overlap between states
- Validating classifier performance

---

## Step 3 — Single-Qubit Classifier Training

Main function:

train_single_qubit_classifier()

Classifier used:

LinearDiscriminantAnalysis

Features:

[I, Q]

Labels:

0 or 1

Outputs:

- Classification accuracy
- Confusion matrix
- Single-qubit assignment matrix

---

# Single-Qubit Assignment Matrix

Each qubit produces a 2×2 assignment matrix with Qiskit-compatible convention:

A[i,j] = P(measured=j | prepared=i)

Properties:

- Row-stochastic matrix
- Rows sum to 1
- Diagonal entries represent readout fidelity

Saved files:

noise_matrix_results/
    assignment_matrix_q1.npy
    assignment_matrix_q2.npy
    assignment_matrix_q3.npy
    assignment_matrix_q4.npy

---

# Step 4 — Build Full Correlated 16×16 Noise Matrix

Main function:

build_noise_matrix()

Purpose:

Construct:

M[i,j] = P(measured=j | prepared=i)

for all 16 prepared states.

This matrix captures:

- Correlated readout effects
- Multi-qubit readout correlations
- Crosstalk effects
- Non-factorizable measurement errors

---

# Saved Outputs

## Output Directory

./noise_matrix_results

Generated files include:

- assignment_matrix_q1.npy
- assignment_matrix_q2.npy
- assignment_matrix_q3.npy
- assignment_matrix_q4.npy
- noise_matrix_16x16.npy
- noise_matrix_16x16.txt
- noise_matrix_heatmap.png

---

# Part 2 — Noisy Quantum Circuit Dataset Generation

## Main Goal

The second major section of the project generates:

- Ideal probability distributions
- Noisy probability distributions
- Training datasets
- Test datasets

for quantum machine learning and readout-noise mitigation experiments.

---

# Main Notebook

## File

noisy_4q_zz.ipynb

This notebook contains:

- Quantum circuit definitions
- Noise model construction
- Qiskit simulation
- Correlated noise application
- Dataset generation
- Dataset saving/loading utilities

---

# Quantum Circuits Used in the Project

## Circuit Family 1 — Independent RY Circuit

Main function:

build_circuit_independent()

Circuit structure:

Ry(theta_i)

applied independently to each qubit.

Characteristics:

- No entanglement
- Independent qubit rotations
- Simple probability structure

---
<!-- ## Circuit Family 2 — Correlated RZZ Circuit

Main function:

build_correlated_circuit()

Structure:

1. Initial RY encoding
2. Hadamard basis rotation
3. Full entangling layer using rzz(theta_rzz)
4. Final basis restoration

Characteristics:

- Strong multi-qubit correlations
- Entangled probability distributions -->

---

## Circuit Family 2 — ZZFeatureMap Circuit

Main function:

build_zzfeaturemap_circuit()

Uses Qiskit's zz_feature_map.

Features:

- Standard quantum feature map
- Quantum machine learning oriented
- Supports configurable reps and entanglement

---

# Noise Models

## Noise Mode 1 — Synthetic Noise

Mode name:

synthetic

Implemented using:

ReadoutError

with configurable p01 and p10.

Meaning:

- 0 → 1 flip probability
- 1 → 0 flip probability

---

## Noise Mode 2 — Experimental Single-Qubit Noise

Mode name:

experimental_single

Uses experimentally extracted 2×2 assignment matrices loaded from:

noise_matrix_results/

Each qubit receives its own experimentally calibrated readout error.

---

## Noise Mode 3 — Experimental Correlated Noise

Mode name:

experimental_correlated

Uses the 16×16 correlated readout matrix.

Important limitation:

Qiskit Aer cannot directly apply full correlated readout matrices.

Therefore:

- Correlated noise is applied manually
- Noise is applied directly to probability vectors

Main function:

apply_correlated_readout_noise()

Core equation:

p_noisy = p_ideal @ A16

where:

A16[i,j] = P(measured=j | prepared=i)

---

# Probability Representation

The simulator converts measurement counts into 16-dimensional probability vectors using:

counts_to_probs_4q()

Ordering convention:

0000 → index 0
...
1111 → index 15

This ordering must remain consistent everywhere in the project.

---

# Dataset Generation Pipeline

Main function:

generate_dataset()

For each sample:

1. Random angles are generated
2. Quantum circuit is built
3. Ideal distribution is simulated
4. Noise is applied
5. Input/output pair is stored

Final dataset contents:

- X → noisy probabilities
- Y → ideal probabilities
- THETAS → circuit parameters

---

# Dataset Storage Structure

## Root Directory

./DataTrain

Datasets are organized hierarchically:

DataTrain/
    independent/
        synthetic/
        experimental_single/
        experimental_correlated/

    <!-- zz_rzz/
        synthetic/
        experimental_single/
        experimental_correlated/ -->

    zz_featuremap/
        synthetic/
        experimental_single/
        experimental_correlated/

Each directory contains:

- X_train.npy
- Y_train.npy
- theta_train.npy
- X_test.npy
- Y_test.npy
- theta_test.npy

---

# Important Project Conventions

## 1. Bit Ordering Convention

Critical conversion functions:

- bits_to_int()
- int_to_bits()

The project uses:

q1 + 2*q2 + 4*q3 + 8*q4

This convention must remain consistent everywhere.

---

## 2. Qiskit Noise Matrix Convention

The project follows:

M[i,j] = P(measured=j | prepared=i)

Therefore:

- Rows = prepared states
- Columns = measured states
- Rows sum to 1

---

## 3. Correlated Noise Is Applied Manually

Very important architectural detail:

Qiskit Aer cannot directly implement full correlated 16×16 readout errors.

Therefore experimental_correlated must always use:

apply_correlated_readout_noise()

---

# Critical Files Summary

## Experimental Readout Extraction

readout_matrix.ipynb

Responsible for:

- Reading experimental IQ data
- Training classifiers
- Extracting assignment matrices
- Building correlated 16×16 noise matrix

---

## Noisy Dataset Generation

noisy_4q_zz.ipynb

Responsible for:

- Quantum circuit generation
- Noise model creation
- Applying synthetic/experimental noise
- Dataset generation
- Dataset saving/loading

---

# Recommended Execution Order

## Step 1

Run:

readout_matrix.ipynb

to generate:

noise_matrix_results/

---

## Step 2

Run:

noisy_4q_zz.ipynb

using generated noise matrices.

---

# Final Notes for Cursor Agent

The Cursor agent should treat the following as highly sensitive and important:

1. Bit ordering consistency
2. Probability vector ordering
3. Qiskit row-stochastic convention
4. Correct application of correlated noise
5. Stable directory structure
6. Consistent dataset naming
7. Experimental matrices must be generated before dataset creation
8. Correlated noise is manually applied
9. All probability vectors are 16-dimensional
10. Experimental IQ data are complex-valued
