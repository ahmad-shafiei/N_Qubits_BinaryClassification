# Quantum Readout Noise Modeling and Dataset Generation

## Project Overview

This project models **4-qubit readout noise** from real experimental IQ data and uses the extracted matrices to generate machine-learning datasets from noisy quantum circuit simulations.

The workflow has two main parts:

| Part | Notebook / CLI | Output |
|------|----------------|--------|
| **1 — Readout extraction** | `readout_matrix.ipynb`, `scripts/run_readout_matrix.py` | `noise_matrix_results/res_<snapshot>/` |
| **2 — ML datasets & training** | `noisy_4q_zzfeaturemap.ipynb`, `scripts/build_ml_datasets.py` | `DataTrain/<snapshot>/` |

**Rule:** always complete Part 1 for a snapshot before building datasets or training models on experimental noise.

---

## Directory Layout (current)

```
ReadouMatrixIqData/
├── quadrature_data_4qubits/          # experimental IQ input (per snapshot)
│   └── 1_1_2025/
│       ├── 0000.txt … 1111.txt       # 16 prepared basis states
│
├── noise_matrix_results/             # extracted readout matrices
│   └── res_1_1_2025/
│       ├── assignment_matrix_q1.txt … q4.txt
│       ├── noise_matrix_16x16.txt
│       └── plots / .npy (optional)
│
├── DataTrain/                        # ML datasets (per snapshot)
│   └── 1_1_2025/
│       ├── independent/
│       │   ├── synthetic/
│       │   ├── experimental_single/
│       │   └── experimental_correlated/
│       └── zz_featuremap/
│           └── …
│
├── src/
│   ├── readout_extraction/           # Part 1 pipeline (config, data, matrix, plots)
│   ├── generate_and_save_datasets.py # single-dataset build/load
│   ├── ml_dataset_pipeline.py        # batch build for one snapshot
│   ├── noises.py                     # loads matrices from res_<snapshot>
│   ├── circuits.py, metrics.py, evaluation.py, dashboard.py
│
├── scripts/
│   ├── run_readout_matrix.py         # CLI for Part 1
│   └── build_ml_datasets.py          # CLI for Part 2 dataset generation
│
├── readout_matrix.ipynb              # Part 1 (interactive)
└── noisy_4q_zzfeaturemap.ipynb     # Part 2 (datasets + model experiments)
```

### Snapshot concept

Each **snapshot** (e.g. `1_1_2025`) is one experimental measurement campaign:

- **Input:** `quadrature_data_4qubits/<snapshot>/`
- **Noise matrices:** `noise_matrix_results/res_<snapshot>/`
- **ML data:** `DataTrain/<snapshot>/`

Set the active snapshot in Python:

```python
from src.readout_extraction.config import set_active_snapshot
set_active_snapshot("1_1_2025")
```

Or via environment: `NOISE_SNAPSHOT=1_1_2025`

---

# Part 1 — Experimental Readout Noise Extraction

## Goal

From IQ measurement clouds, extract:

- **Single-qubit assignment matrices** (2×2 per qubit)
- **Correlated 4-qubit readout matrix** (16×16)

Convention (Qiskit-compatible):

`A[i,j] = P(measured = j | prepared = i)` — rows sum to 1.

## Input data

Each file `0000.txt` … `1111.txt` in a snapshot folder contains:

- 4 lines (one per qubit)
- 1000 complex IQ samples per line (`I + iQ`)

Total per snapshot: 16 states × 1000 shots = **16 000 samples**, shape `(16000, 4)`.

## Pipeline steps

1. **Load** — `load_all_data(config)` from `src/readout_extraction/data.py`
2. **Visualize** — IQ clouds, state overlays (`plots.py`)
3. **Train classifiers** — LDA (or QDA) on `[I, Q]` per qubit
4. **Build 16×16 matrix** — from full-dataset predictions
5. **Validate & save** — row sums, diagonal fidelities
6. **Compare** — correlated vs Kronecker-independent approximation

## How to run

**CLI (recommended for batch):**

```bash
python scripts/run_readout_matrix.py --list
python scripts/run_readout_matrix.py --snapshot 1_1_2025
python scripts/run_readout_matrix.py --all
```

**Notebook:** `readout_matrix.ipynb` — set `SNAPSHOT`, run Setup, then step cells or `run_readout_pipeline(config)`.

## Outputs

Under `noise_matrix_results/res_<snapshot>/`:

| File | Role |
|------|------|
| `assignment_matrix_q1.txt` … `q4.txt` | per-qubit readout errors |
| `noise_matrix_16x16.txt` | full correlated readout matrix |
| heatmaps / comparison plots | diagnostics |

These files are consumed by `src/noises.py` in Part 2.

---

# Part 2 — Noisy Circuit Datasets & Model Training

## Goal

Generate datasets where:

- **X** = noisy 16-dim probability vectors
- **Y** = ideal probability vectors
- **THETAS** = circuit parameters

Then train/evaluate mitigation models across noise regimes.

## Circuits

| `circuit_type` | Builder | Notes |
|----------------|---------|-------|
| `independent` | `build_circuit_independent()` | independent RY rotations |
| `zz_featuremap` | `build_zzfeaturemap_circuit()` | Qiskit ZZFeatureMap, typically `reps=2` |

## Noise modes

| `noise_mode` | Source | Application |
|--------------|--------|-------------|
| `synthetic` | fixed p01/p10 | Qiskit `ReadoutError` in Aer |
| `experimental_single` | `assignment_matrix_q*.txt` | per-qubit experimental matrices |
| `experimental_correlated` | `noise_matrix_16x16.txt` | **manual** on probability vectors (`p_noisy = p_ideal @ A16`) |

> Aer cannot apply a full 16×16 correlated readout matrix directly; correlated experimental noise is always applied post-simulation.

## Dataset layout

```
DataTrain/<snapshot>/<circuit_type>/<noise_mode>/
    X_train.npy, Y_train.npy, theta_train.npy
    X_validation.npy, …
    X_test.npy, …
```

Standard job recipe (17 datasets): see `STANDARD_ML_DATASET_JOBS` in `src/ml_dataset_pipeline.py`.

- **train:** 4500 samples
- **validation / test:** 1500 samples

## How to build datasets

**CLI:**

```bash
python scripts/build_ml_datasets.py --list
python scripts/build_ml_datasets.py --snapshot 1_1_2025
python scripts/build_ml_datasets.py --snapshot 1_1_2025 --dry-run
python scripts/build_ml_datasets.py --snapshot 1_1_2025 --skip-existing
python scripts/build_ml_datasets.py --snapshot 1_1_2025 --noise experimental_single --circuit independent
```

**Python:**

```python
from src.ml_dataset_pipeline import build_datasets_for_snapshot
build_datasets_for_snapshot("1_1_2025", skip_existing=True)
```

**Single dataset (manual):**

```python
from src.generate_and_save_datasets import build_and_save_dataset
build_and_save_dataset(
    num_samples=1500,
    split="test",
    circuit_type="independent",
    noise_mode="experimental_single",
    snapshot_id="1_1_2025",
)
```

**Load:**

```python
from src.generate_and_save_datasets import load_dataset
X, Y, TH = load_dataset("test", "independent", "experimental_single")
```

Paths resolve via `get_active_snapshot()` / `DATA_SNAPSHOT`.

## Notebook: `noisy_4q_zzfeaturemap.ipynb`

Sections:

1. **Setup** — imports, `DATA_SNAPSHOT`, seed
2. **Dataset generation** — run before any training phase
3. **Tests** — sanity checks on circuits/noise
4. **Phases 0–3** — train/validate/test experiments per circuit × noise combination
5. **Organize outputs** — `make_test_sets`, summary tables, dashboards (`figs/`)

---

# Recommended Execution Order

```
1. Place IQ files  →  quadrature_data_4qubits/<snapshot>/
2. Extract matrices →  python scripts/run_readout_matrix.py --snapshot <snapshot>
3. Build datasets   →  python scripts/build_ml_datasets.py --snapshot <snapshot>
4. Set snapshot     →  set_active_snapshot("<snapshot>") or DATA_SNAPSHOT in notebook
5. Train / evaluate →  noisy_4q_zzfeaturemap.ipynb (phases)
```

For a new snapshot `2_2_2025`: repeat steps 1–4, then point training cells at the new data.

---

# Critical Conventions

1. **Bit ordering:** `q1 + 2*q2 + 4*q3 + 8*q4` — use `bits_to_int()` / `int_to_bits()` consistently.
2. **Probability index:** `0000 → 0`, …, `1111 → 15` in `counts_to_probs_4q()`.
3. **Matrix convention:** rows = prepared, columns = measured; rows stochastic.
4. **Correlated noise:** always `apply_correlated_readout_noise()` for `experimental_correlated`.
5. **Snapshot alignment:** IQ folder, `res_<snapshot>`, and `DataTrain/<snapshot>` must share the same id.

---

# Evaluation Metrics

Models are evaluated with:

- **FidelityMean** — \((\sum \sqrt{p \cdot y})^2\)
- **L1Mean** — \(\sum |p - y|\)
- **KLMean** — \(\sum y \log(y/p)\)

Dashboard outputs: `figs/` folder; summary tables via `src/dashboard.py`.

---

# Key Source Modules

| Module | Responsibility |
|--------|----------------|
| `src/readout_extraction/` | IQ loading, LDA/QDA, matrix build, plots, `run_readout_pipeline` |
| `src/noises.py` | load experimental matrices; build Aer noise models |
| `src/generate_and_save_datasets.py` | simulate circuits, generate/save/load one dataset |
| `src/ml_dataset_pipeline.py` | orchestrate all standard datasets for a snapshot |
| `src/circuits.py` | independent & ZZFeatureMap builders |
| `src/evaluation.py` | test-set packaging for experiments |
| `src/dashboard.py` | summary tables and heatmap dashboards |

---

# Notes for Developers / AI Agents

**Do not break:**

- snapshot path layout (`quadrature_data_4qubits/<id>`, `res_<id>`, `DataTrain/<id>`)
- bit/probability ordering
- manual correlated-noise path for `experimental_correlated`
- row-stochastic matrix convention

**Safe to extend:**

- new snapshots (add folder + run both CLIs)
- visualization / dashboard metric switching (post-processing only)
- additional circuit types or noise modes (update `STANDARD_ML_DATASET_JOBS` and `noises.py`)

**Legacy:** flat `DataTrain/independent/...` (without snapshot) was migrated to `DataTrain/1_1_2025/...`. Old paths should not be used for new work.
