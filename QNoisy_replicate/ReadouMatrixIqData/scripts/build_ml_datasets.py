#!/usr/bin/env python3
"""Build ML training datasets for a chosen experimental IQ snapshot.

Examples
--------
List snapshots that have both IQ data and noise matrices::

    python scripts/build_ml_datasets.py --list

Build all standard datasets for one snapshot::

    python scripts/build_ml_datasets.py --snapshot 1_1_2025

Only experimental_single datasets for independent circuit::

    python scripts/build_ml_datasets.py --snapshot 1_1_2025 \\
        --noise experimental_single --circuit independent

Preview without generating::

    python scripts/build_ml_datasets.py --snapshot 1_1_2025 --dry-run

Build for every ready snapshot::

    python scripts/build_ml_datasets.py --all
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.readout_extraction.config import DEFAULT_SNAPSHOT_ID
from src.ml_dataset_pipeline import (  # noqa: E402
    build_all_buildable_snapshots,
    build_datasets_for_snapshot,
    get_snapshot_build_status,
    list_buildable_snapshots,
    list_snapshots_with_iq_data,
    list_snapshots_with_noise_matrices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DataTrain/<snapshot>/ datasets using "
            "noise_matrix_results/res_<snapshot>/ matrices."
        ),
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help=f"IQ/noise snapshot id (default: {DEFAULT_SNAPSHOT_ID})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build for every snapshot with IQ data + noise matrices.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available snapshots and readiness.",
    )
    parser.add_argument(
        "--noise",
        nargs="+",
        choices=("synthetic", "experimental_single", "experimental_correlated"),
        help="Restrict to specific noise modes.",
    )
    parser.add_argument(
        "--circuit",
        nargs="+",
        choices=("independent", "zz_featuremap"),
        help="Restrict to specific circuit types.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Shots per circuit simulation (default: 1024).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose X_<split>.txt already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without generating data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("IQ snapshots:", list_snapshots_with_iq_data())
        print("Noise-ready :", list_snapshots_with_noise_matrices())
        print("Buildable   :", list_buildable_snapshots())
        for snapshot_id in list_buildable_snapshots():
            status = get_snapshot_build_status(snapshot_id)
            print(
                f"\n{snapshot_id}:"
                f"\n  IQ      : {status.has_iq_data}"
                f"\n  single  : {status.has_single_qubit_matrices}"
                f"\n  corr    : {status.has_correlated_matrix}"
                f"\n  matrix  : {status.noise_matrix_dir}"
            )
        return 0

    kwargs = {
        "noise_modes": args.noise,
        "circuits": args.circuit,
        "shots": args.shots,
        "skip_existing": args.skip_existing,
        "dry_run": args.dry_run,
    }

    if args.all:
        build_all_buildable_snapshots(**kwargs)
        return 0

    snapshot = args.snapshot or DEFAULT_SNAPSHOT_ID
    build_datasets_for_snapshot(snapshot, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
