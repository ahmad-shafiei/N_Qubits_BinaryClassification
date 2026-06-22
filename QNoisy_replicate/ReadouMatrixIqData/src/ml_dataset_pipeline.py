"""Recipe and orchestration for ML dataset generation per IQ snapshot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .readout_extraction.config import (
    discover_snapshots,
    get_campaign_dir,
    get_dataset_output_dir,
    get_snapshot_data_dir,
    resolve_snapshot_id,
    set_active_snapshot,
)
from .generate_and_save_datasets import build_and_save_dataset

ZZ_CIRCUIT_KWARGS = {"reps": 2, "entanglement": "full"}

# Unique (circuit, noise, split) jobs required before model training.
STANDARD_ML_DATASET_JOBS: List[Dict[str, Any]] = [
  # --- independent ---
    {"split": "train", "num_samples": 4500, "circuit_type": "independent", "noise_mode": "synthetic"},
    {"split": "train", "num_samples": 4500, "circuit_type": "independent", "noise_mode": "experimental_single"},
    {"split": "train", "num_samples": 4500, "circuit_type": "independent", "noise_mode": "experimental_correlated"},
    {"split": "validation", "num_samples": 1500, "circuit_type": "independent", "noise_mode": "synthetic"},
    {"split": "validation", "num_samples": 1500, "circuit_type": "independent", "noise_mode": "experimental_correlated"},
    {"split": "test", "num_samples": 1500, "circuit_type": "independent", "noise_mode": "synthetic"},
    {"split": "test", "num_samples": 1500, "circuit_type": "independent", "noise_mode": "experimental_single"},
    {"split": "test", "num_samples": 1500, "circuit_type": "independent", "noise_mode": "experimental_correlated"},
  # --- zz_featuremap (reps=2) ---
    {"split": "train", "num_samples": 4500, "circuit_type": "zz_featuremap", "noise_mode": "synthetic", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "train", "num_samples": 4500, "circuit_type": "zz_featuremap", "noise_mode": "experimental_single", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "train", "num_samples": 4500, "circuit_type": "zz_featuremap", "noise_mode": "experimental_correlated", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "validation", "num_samples": 1500, "circuit_type": "zz_featuremap", "noise_mode": "synthetic", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "validation", "num_samples": 1500, "circuit_type": "zz_featuremap", "noise_mode": "experimental_single", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "validation", "num_samples": 1500, "circuit_type": "zz_featuremap", "noise_mode": "experimental_correlated", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "test", "num_samples": 1500, "circuit_type": "zz_featuremap", "noise_mode": "synthetic", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "test", "num_samples": 1500, "circuit_type": "zz_featuremap", "noise_mode": "experimental_single", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
    {"split": "test", "num_samples": 1500, "circuit_type": "zz_featuremap", "noise_mode": "experimental_correlated", "circuit_kwargs": ZZ_CIRCUIT_KWARGS},
]

EXPERIMENTAL_NOISE_MODES = ("experimental_single", "experimental_correlated")


@dataclass
class SnapshotBuildStatus:
    snapshot_id: str
    iq_data_dir: str
    noise_matrix_dir: str
    has_iq_data: bool
    has_single_qubit_matrices: bool
    has_correlated_matrix: bool
    ready_for_experimental: bool

    def missing_noise_files(self) -> List[str]:
        missing = []
        matrix_dir = self.noise_matrix_dir
        if not self.has_single_qubit_matrices:
            for q in range(1, 5):
                missing.append(os.path.join(matrix_dir, f"assignment_matrix_q{q}.txt"))
        if not self.has_correlated_matrix:
            missing.append(os.path.join(matrix_dir, "noise_matrix_16x16.txt"))
        return missing


def get_snapshot_build_status(
    snapshot_id: str,
    *,
    campaign_id: str = "quadrature_data_4qubits",
) -> SnapshotBuildStatus:
    snapshot_id = resolve_snapshot_id(snapshot_id)
    iq_dir = get_snapshot_data_dir(snapshot_id, campaign_id=campaign_id)
    matrix_dir = get_dataset_output_dir(snapshot_id)

    has_single = all(
        os.path.isfile(os.path.join(matrix_dir, f"assignment_matrix_q{q}.txt"))
        for q in range(1, 5)
    )
    has_corr = os.path.isfile(os.path.join(matrix_dir, "noise_matrix_16x16.txt"))

    return SnapshotBuildStatus(
        snapshot_id=snapshot_id,
        iq_data_dir=iq_dir,
        noise_matrix_dir=matrix_dir,
        has_iq_data=os.path.isdir(iq_dir),
        has_single_qubit_matrices=has_single,
        has_correlated_matrix=has_corr,
        ready_for_experimental=has_single and has_corr,
    )


def list_snapshots_with_noise_matrices() -> List[str]:
    """Snapshots whose res_<id> folder contains experimental noise matrices."""
    from .readout_extraction.config import get_results_root

    root = get_results_root()
    if not os.path.isdir(root):
        return []

    snapshots = []
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("res_"):
            continue
        snapshot_id = entry[len("res_") :]
        status = get_snapshot_build_status(snapshot_id)
        if status.ready_for_experimental:
            snapshots.append(snapshot_id)
    return snapshots


def list_snapshots_with_iq_data(
    campaign_id: str = "quadrature_data_4qubits",
) -> List[str]:
    campaign_dir = get_campaign_dir(campaign_id)
    paths = discover_snapshots(campaign_dir)
    return [os.path.basename(p) for p in paths if p != os.path.abspath(campaign_dir)]


def list_buildable_snapshots(
    campaign_id: str = "quadrature_data_4qubits",
) -> List[str]:
    """Snapshots with IQ data AND extracted noise matrices."""
    iq = set(list_snapshots_with_iq_data(campaign_id))
    noise = set(list_snapshots_with_noise_matrices())
    return sorted(iq & noise)


def _job_needs_experimental(noise_mode: str) -> bool:
    return noise_mode in EXPERIMENTAL_NOISE_MODES


def _job_can_run(job: Dict[str, Any], status: SnapshotBuildStatus) -> bool:
    noise_mode = job["noise_mode"]
    if noise_mode == "synthetic":
        return True
    if noise_mode == "experimental_single":
        return status.has_single_qubit_matrices
    if noise_mode == "experimental_correlated":
        return status.has_correlated_matrix
    return False


def filter_jobs_for_snapshot(
    jobs: Sequence[Dict[str, Any]],
    status: SnapshotBuildStatus,
    *,
    noise_modes: Optional[Iterable[str]] = None,
    circuits: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    allowed_noise = set(noise_modes) if noise_modes else None
    allowed_circuits = set(circuits) if circuits else None
    selected = []
    for job in jobs:
        if allowed_noise and job["noise_mode"] not in allowed_noise:
            continue
        if allowed_circuits and job["circuit_type"] not in allowed_circuits:
            continue
        if _job_can_run(job, status):
            selected.append(dict(job))
    return selected


def build_datasets_for_snapshot(
    snapshot_id: Optional[str] = None,
    *,
    jobs: Optional[Sequence[Dict[str, Any]]] = None,
    noise_modes: Optional[Iterable[str]] = None,
    circuits: Optional[Iterable[str]] = None,
    shots: int = 1024,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Build all standard ML datasets for one experimental snapshot.

    Select snapshot (e.g. ``1_1_2025``), verify noise matrices in
    ``noise_matrix_results/res_<snapshot>/``, then write files under
    ``DataTrain/<snapshot>/``.
    """
    snapshot_id = set_active_snapshot(resolve_snapshot_id(snapshot_id))
    status = get_snapshot_build_status(snapshot_id)
    job_list = list(jobs or STANDARD_ML_DATASET_JOBS)
    runnable = filter_jobs_for_snapshot(
        job_list, status, noise_modes=noise_modes, circuits=circuits
    )

    print("=" * 60)
    print(f"Snapshot          : {snapshot_id}")
    print(f"IQ data           : {status.iq_data_dir}")
    print(f"Noise matrices    : {status.noise_matrix_dir}")
    print(f"Experimental ready: {status.ready_for_experimental}")
    print(f"Jobs to run       : {len(runnable)} / {len(job_list)}")
    print("=" * 60)

    if not status.ready_for_experimental:
        print("WARNING: experimental noise matrices incomplete:")
        for path in status.missing_noise_files():
            print(f"  missing: {path}")
        print("Synthetic jobs will still run; experimental jobs are skipped.")

    results = {"built": [], "skipped": [], "dry_run": dry_run}

    for job in runnable:
        from .readout_extraction.config import get_ml_dataset_dir

        out_dir = get_ml_dataset_dir(
            job["circuit_type"], job["noise_mode"], snapshot_id
        )
        x_path = os.path.join(out_dir, f"X_{job['split']}.txt")
        label = (
            f"{job['circuit_type']} | {job['noise_mode']} | "
            f"{job['split']} ({job['num_samples']})"
        )

        if skip_existing and os.path.isfile(x_path):
            print(f"SKIP (exists): {label}")
            results["skipped"].append(label)
            continue

        if dry_run:
            print(f"DRY-RUN: {label} -> {out_dir}")
            results["built"].append(label)
            continue

        print(f"\n>>> {label}")
        build_and_save_dataset(
            num_samples=job["num_samples"],
            split=job["split"],
            circuit_type=job["circuit_type"],
            noise_mode=job["noise_mode"],
            shots=shots,
            circuit_kwargs=job.get("circuit_kwargs"),
            snapshot_id=snapshot_id,
        )
        results["built"].append(label)

    return results


def build_all_buildable_snapshots(**kwargs) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for snapshot_id in list_buildable_snapshots():
        summary[snapshot_id] = build_datasets_for_snapshot(snapshot_id, **kwargs)
    return summary
