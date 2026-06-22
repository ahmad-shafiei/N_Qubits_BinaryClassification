#!/usr/bin/env python3
"""Run the readout_matrix.ipynb workflow from the command line.

Layout
------
    quadrature_data_4qubits/1_1_2025/*.txt   ->  noise_matrix_results/res_1_1_2025/

Examples
--------
Default campaign + snapshot::

    python scripts/run_readout_matrix.py

Explicit snapshot::

    python scripts/run_readout_matrix.py --snapshot 1_1_2025
    python scripts/run_readout_matrix.py --campaign quadrature_data_4qubits --snapshot 2_2_2025

Direct path to a snapshot folder::

    python scripts/run_readout_matrix.py --data-dir ./quadrature_data_4qubits/1_1_2025

All snapshots in all campaigns::

    python scripts/run_readout_matrix.py --all

Migrate old flat layout to snapshot folders::

    python scripts/run_readout_matrix.py --migrate-layout
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.readout_extraction.config import (  # noqa: E402
    DEFAULT_CAMPAIGN_ID,
    DEFAULT_SNAPSHOT_ID,
    ReadoutDatasetConfig,
    discover_datasets,
    get_campaign_dir,
    get_results_root,
    get_snapshot_data_dir,
    run_layout_migrations,
)
from src.readout_extraction.pipeline import (  # noqa: E402
    run_all_discovered_datasets,
    run_readout_pipeline,
)


def _resolve_data_dir(
    campaign: str | None,
    snapshot: str | None,
    data_dir: str | None,
) -> str:
    if data_dir:
        return os.path.abspath(data_dir)

    campaign_id = campaign or DEFAULT_CAMPAIGN_ID
    snapshot_id = snapshot or DEFAULT_SNAPSHOT_ID
    path = get_snapshot_data_dir(
        snapshot_id,
        campaign_id=campaign_id,
        project_root=PROJECT_ROOT,
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Snapshot folder not found: {path}\n"
            f"Expected layout: {campaign_id}/{snapshot_id}/"
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract readout noise matrices from IQ quadrature snapshots.",
    )
    parser.add_argument(
        "--campaign",
        default=DEFAULT_CAMPAIGN_ID,
        help=(
            "Quadrature campaign folder under project root "
            f"(default: {DEFAULT_CAMPAIGN_ID})."
        ),
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help=(
            "Snapshot subfolder name inside the campaign "
            f"(default: {DEFAULT_SNAPSHOT_ID})."
        ),
    )
    parser.add_argument(
        "--data-dir",
        help="Path to a snapshot folder that directly contains state .txt files.",
    )
    parser.add_argument(
        "--results-root",
        default=get_results_root(),
        help="Root directory for res_<snapshot> noise-matrix outputs.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every snapshot in every quadrature_data_* campaign.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discoverable snapshot folders and exit.",
    )
    parser.add_argument(
        "--classifier",
        choices=("LDA", "QDA"),
        default="LDA",
        help="Per-qubit IQ classifier (default: LDA).",
    )
    parser.add_argument(
        "--qda-reg",
        type=float,
        default=0.3,
        help="Regularization for QDA (ignored for LDA).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=None,
        help="Override automatic qubit-count inference.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plot images.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip correlated vs independent Kronecker comparison.",
    )
    parser.add_argument(
        "--migrate-layout",
        action="store_true",
        help=(
            "Move legacy flat layouts into snapshot subfolders "
            "(quadrature, noise_matrix_results, DataTrain)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        datasets = discover_datasets(PROJECT_ROOT)
        if not datasets:
            print("No snapshot folders found.")
            return 0
        for path in datasets:
            cfg = ReadoutDatasetConfig.from_data_dir(
                path, results_root=args.results_root
            )
            print(
                f"{cfg.snapshot_id}\t{cfg.result_id}\t{path}\t"
                f"{cfg.num_qubits} qubits"
            )
        return 0

    if args.migrate_layout:
        summary = run_layout_migrations(
            campaign_id=args.campaign,
            snapshot_id=args.snapshot or DEFAULT_SNAPSHOT_ID,
        )
        for key, value in summary.items():
            print(f"{key}: {value or 'no changes'}")
        return 0

    pipeline_kwargs = {
        "save_plots": not args.no_plots,
        "compare_independent": not args.no_compare,
    }
    classifier_kwargs = {
        "classifier_type": args.classifier,
        "qda_reg_param": args.qda_reg,
        "random_seed": args.seed,
    }

    if args.all:
        run_layout_migrations()
        run_all_discovered_datasets(PROJECT_ROOT, **pipeline_kwargs)
        return 0

    data_dir = _resolve_data_dir(args.campaign, args.snapshot, args.data_dir)
    config = ReadoutDatasetConfig.from_data_dir(
        data_dir,
        results_root=args.results_root,
        num_qubits=args.num_qubits,
        **classifier_kwargs,
    )
    run_readout_pipeline(config, **pipeline_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
