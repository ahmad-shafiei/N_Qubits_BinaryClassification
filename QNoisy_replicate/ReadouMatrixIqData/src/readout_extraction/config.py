"""Path layout and dataset discovery for readout matrix extraction.

Layout
------
Input (per measurement campaign)::

    quadrature_data_4qubits/
        1_1_2025/          # snapshot: 16 state .txt files
        2_2_2025/          # another snapshot
        ...

Output (one folder per snapshot)::

    noise_matrix_results/
        res_1_1_2025/
        res_2_2_2025/
        ...
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import List, Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))

QUADRATURE_PREFIX = "quadrature_data_"
RESULTS_ROOT_NAME = "noise_matrix_results"
RESULT_PREFIX = "res_"

DEFAULT_CAMPAIGN_ID = "quadrature_data_4qubits"
DEFAULT_SNAPSHOT_ID = "1_1_2025"
# Snapshot label used by downstream noise helpers (maps to res_<snapshot>).
DATA_TRAIN_ROOT_NAME = "DataTrain"
ML_CIRCUIT_DIRS = ("independent", "zz_featuremap")

# Runtime override (notebook / scripts can call set_active_snapshot).
_runtime_snapshot: Optional[str] = None


def get_active_snapshot() -> str:
    """Active IQ / noise / ML dataset snapshot, e.g. 1_1_2025."""
    if _runtime_snapshot is not None:
        return _runtime_snapshot
    return os.environ.get(
        "NOISE_SNAPSHOT",
        os.environ.get("NOISE_DATASET", DEFAULT_SNAPSHOT_ID),
    )


def set_active_snapshot(snapshot_id: str) -> str:
    """Set snapshot for this Python session (also updates NOISE_SNAPSHOT)."""
    global _runtime_snapshot
    _runtime_snapshot = snapshot_id
    os.environ["NOISE_SNAPSHOT"] = snapshot_id
    return snapshot_id


def resolve_snapshot_id(snapshot_id: Optional[str] = None) -> str:
    return snapshot_id if snapshot_id is not None else get_active_snapshot()


@dataclass
class ReadoutDatasetConfig:
    """Configuration for one IQ snapshot inside a quadrature campaign."""

    data_dir: str
    output_dir: str
    snapshot_id: str
    result_id: str
    campaign_id: str
    campaign_dir: str
    num_qubits: int
    random_seed: int = 42
    classifier_type: str = "LDA"
    qda_reg_param: float = 0.3
    test_size: float = 0.2

    @property
    def dataset_id(self) -> str:
        """Alias kept for logging and metadata compatibility."""
        return self.snapshot_id

    @property
    def num_states(self) -> int:
        return 2 ** self.num_qubits

    @classmethod
    def from_snapshot(
        cls,
        campaign_dir: str,
        snapshot_id: str,
        *,
        results_root: Optional[str] = None,
        num_qubits: Optional[int] = None,
        **kwargs,
    ) -> "ReadoutDatasetConfig":
        campaign_dir = os.path.abspath(campaign_dir)
        campaign_id = os.path.basename(os.path.normpath(campaign_dir))
        data_dir = os.path.join(campaign_dir, snapshot_id)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Snapshot folder not found: {data_dir}")
        return cls.from_data_dir(
            data_dir,
            results_root=results_root,
            num_qubits=num_qubits,
            campaign_dir=campaign_dir,
            campaign_id=campaign_id,
            snapshot_id=snapshot_id,
            **kwargs,
        )

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str,
        *,
        results_root: Optional[str] = None,
        num_qubits: Optional[int] = None,
        campaign_dir: Optional[str] = None,
        campaign_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        **kwargs,
    ) -> "ReadoutDatasetConfig":
        data_dir = os.path.abspath(data_dir)

        if not _list_state_files(data_dir):
            raise FileNotFoundError(
                f"No computational-basis .txt files in {data_dir}. "
                "Expected a snapshot folder such as quadrature_data_4qubits/1_1_2025/"
            )

        if snapshot_id is None:
            snapshot_id = os.path.basename(os.path.normpath(data_dir))
        if campaign_dir is None:
            campaign_dir = os.path.dirname(data_dir)
        if campaign_id is None:
            campaign_id = os.path.basename(os.path.normpath(campaign_dir))

        if results_root is None:
            results_root = get_results_root()
        result_id = snapshot_to_result_id(snapshot_id)
        output_dir = os.path.join(results_root, result_id)

        if num_qubits is None:
            num_qubits = infer_num_qubits(data_dir, campaign_dir=campaign_dir)

        return cls(
            data_dir=data_dir,
            output_dir=output_dir,
            snapshot_id=snapshot_id,
            result_id=result_id,
            campaign_id=campaign_id,
            campaign_dir=campaign_dir,
            num_qubits=num_qubits,
            **kwargs,
        )


def snapshot_to_result_id(snapshot_id: str) -> str:
    if snapshot_id.startswith(RESULT_PREFIX):
        return snapshot_id
    return f"{RESULT_PREFIX}{snapshot_id}"


def result_id_to_snapshot(result_id: str) -> str:
    if result_id.startswith(RESULT_PREFIX):
        return result_id[len(RESULT_PREFIX) :]
    return result_id


def infer_num_qubits(
    data_dir: str,
    *,
    campaign_dir: Optional[str] = None,
) -> int:
    """Infer qubit count from campaign folder name, file count, or first file."""
    search_names = []
    if campaign_dir:
        search_names.append(os.path.basename(os.path.normpath(campaign_dir)))
    search_names.append(os.path.basename(os.path.normpath(data_dir)))

    for name in search_names:
        match = re.search(r"(\d+)\s*qubits?", name, re.IGNORECASE)
        if match:
            return int(match.group(1))

    txt_files = _list_state_files(data_dir)
    if not txt_files:
        raise FileNotFoundError(f"No state .txt files found in {data_dir}")

    n_states = len(txt_files)
    if n_states & (n_states - 1) != 0:
        raise ValueError(
            f"Expected 2^n state files in {data_dir}, found {n_states}"
        )
    num_qubits = n_states.bit_length() - 1

    with open(txt_files[0], "r", encoding="utf-8") as handle:
        line_count = sum(1 for line in handle if line.strip())
    if line_count != num_qubits:
        raise ValueError(
            f"Folder implies {num_qubits} qubits but "
            f"{os.path.basename(txt_files[0])} has {line_count} lines"
        )
    return num_qubits


def discover_campaign_roots(
    search_root: Optional[str] = None,
    *,
    pattern: str = f"{QUADRATURE_PREFIX}*",
) -> List[str]:
    if search_root is None:
        search_root = PROJECT_ROOT
    import glob

    paths = glob.glob(os.path.join(search_root, pattern))
    return sorted(p for p in paths if os.path.isdir(p))


def discover_snapshots(campaign_dir: str) -> List[str]:
    """Return snapshot data directories that directly contain state .txt files."""
    snapshots: List[str] = []

    if _list_state_files(campaign_dir):
        snapshots.append(os.path.abspath(campaign_dir))

    for entry in sorted(os.listdir(campaign_dir)):
        path = os.path.join(campaign_dir, entry)
        if os.path.isdir(path) and _list_state_files(path):
            snapshots.append(os.path.abspath(path))

    return snapshots


def discover_datasets(
    search_root: Optional[str] = None,
    *,
    campaign: Optional[str] = None,
) -> List[str]:
    """Return absolute paths to all snapshot folders with IQ state files."""
    if campaign is not None:
        campaign_dir = campaign
        if not os.path.isabs(campaign_dir):
            campaign_dir = os.path.join(search_root or PROJECT_ROOT, campaign)
        return discover_snapshots(campaign_dir)

    datasets: List[str] = []
    for campaign_dir in discover_campaign_roots(search_root):
        datasets.extend(discover_snapshots(campaign_dir))
    return datasets


def get_results_root(project_root: Optional[str] = None) -> str:
    root = os.path.normpath(project_root or PROJECT_ROOT)
    if os.path.basename(root) == RESULTS_ROOT_NAME:
        return root
    return os.path.join(root, RESULTS_ROOT_NAME)


def get_campaign_dir(
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    project_root: Optional[str] = None,
) -> str:
    return os.path.join(project_root or PROJECT_ROOT, campaign_id)


def get_snapshot_data_dir(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    *,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    project_root: Optional[str] = None,
) -> str:
    return os.path.join(get_campaign_dir(campaign_id, project_root), snapshot_id)


def get_dataset_output_dir(
    snapshot_id: str,
    results_root: Optional[str] = None,
) -> str:
    return os.path.join(
        get_results_root(results_root),
        snapshot_to_result_id(snapshot_id),
    )


def ensure_output_dir(config: ReadoutDatasetConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    return config.output_dir


def get_data_train_root(project_root: Optional[str] = None) -> str:
    return os.path.join(project_root or PROJECT_ROOT, DATA_TRAIN_ROOT_NAME)


def get_ml_dataset_dir(
    circuit_type: str,
    noise_mode: str,
    snapshot_id: Optional[str] = None,
    *,
    data_root: Optional[str] = None,
) -> str:
    """Path: DataTrain/<snapshot>/<circuit_type>/<noise_mode>/"""
    snapshot_id = resolve_snapshot_id(snapshot_id)
    return os.path.join(
        get_data_train_root(data_root),
        snapshot_id,
        circuit_type,
        noise_mode,
    )


def discover_ml_snapshots(data_root: Optional[str] = None) -> List[str]:
    """List snapshot folders under DataTrain that contain circuit subdirs."""
    root = get_data_train_root(data_root)
    if not os.path.isdir(root):
        return []
    snapshots = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        if any(
            os.path.isdir(os.path.join(path, circuit))
            for circuit in ML_CIRCUIT_DIRS
        ):
            snapshots.append(entry)
    return snapshots


def migrate_datatrain_layout(
    data_root: Optional[str] = None,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
) -> Optional[str]:
    """Move legacy DataTrain/<circuit>/<noise>/ into DataTrain/<snapshot>/."""
    root = get_data_train_root(data_root)
    if not os.path.isdir(root):
        return None

    has_legacy = any(
        os.path.isdir(os.path.join(root, circuit))
        for circuit in ML_CIRCUIT_DIRS
    )
    if not has_legacy:
        return None

    target = os.path.join(root, snapshot_id)
    os.makedirs(target, exist_ok=True)
    for circuit in ML_CIRCUIT_DIRS:
        src = os.path.join(root, circuit)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(target, circuit)
        if os.path.exists(dst):
            for entry in os.listdir(src):
                item_src = os.path.join(src, entry)
                item_dst = os.path.join(dst, entry)
                if os.path.isdir(item_src):
                    if os.path.exists(item_dst):
                        shutil.rmtree(item_dst)
                    shutil.move(item_src, item_dst)
                elif os.path.isfile(item_src):
                    if os.path.exists(item_dst):
                        os.remove(item_src)
                    else:
                        shutil.move(item_src, item_dst)
            if not os.listdir(src):
                os.rmdir(src)
        else:
            shutil.move(src, dst)
    return target


def migrate_quadrature_layout(
    campaign_dir: Optional[str] = None,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
) -> Optional[str]:
    """Move flat .txt files in a campaign root into a snapshot subfolder."""
    campaign_dir = campaign_dir or get_campaign_dir()
    if not os.path.isdir(campaign_dir):
        return None

    state_files = _list_state_files(campaign_dir)
    if not state_files:
        return None

    target = os.path.join(campaign_dir, snapshot_id)
    os.makedirs(target, exist_ok=True)
    for path in state_files:
        dst = os.path.join(target, os.path.basename(path))
        if os.path.exists(dst):
            os.remove(path)
        else:
            shutil.move(path, dst)
    return target


def migrate_results_layout(
    results_root: Optional[str] = None,
    *,
    legacy_folder: str = DEFAULT_CAMPAIGN_ID,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
) -> Optional[str]:
    """Rename legacy result folder named after campaign to res_<snapshot>."""
    results_root = get_results_root(results_root)
    legacy_path = os.path.join(results_root, legacy_folder)
    target = get_dataset_output_dir(snapshot_id, results_root)

    if not os.path.isdir(legacy_path):
        return None
    if os.path.abspath(legacy_path) == os.path.abspath(target):
        return target

    if os.path.isdir(target):
        for entry in os.listdir(legacy_path):
            src = os.path.join(legacy_path, entry)
            dst = os.path.join(target, entry)
            if os.path.exists(dst):
                if os.path.isfile(src):
                    os.remove(src)
                continue
            shutil.move(src, dst)
        if not os.listdir(legacy_path):
            os.rmdir(legacy_path)
    else:
        shutil.move(legacy_path, target)
    return target


def migrate_legacy_flat_results(
    results_root: Optional[str] = None,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    *,
    remove_duplicates: bool = True,
) -> Optional[str]:
    """Move or deduplicate flat files from noise_matrix_results/."""
    results_root = get_results_root(results_root)
    if not os.path.isdir(results_root):
        return None

    legacy_markers = (
        "noise_matrix_16x16.txt",
        "assignment_matrix_q1.txt",
    )
    has_legacy = any(
        os.path.isfile(os.path.join(results_root, name))
        for name in legacy_markers
    )
    if not has_legacy:
        return None

    target = get_dataset_output_dir(snapshot_id, results_root)
    os.makedirs(target, exist_ok=True)
    moved = []
    for entry in os.listdir(results_root):
        src = os.path.join(results_root, entry)
        if not os.path.isfile(src):
            continue
        if entry in ("README.md",):
            continue
        dst = os.path.join(target, entry)
        if os.path.exists(dst):
            if remove_duplicates:
                os.remove(src)
                moved.append(entry)
            continue
        shutil.move(src, dst)
        moved.append(entry)
    return target if moved else None


def run_layout_migrations(
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
) -> dict:
    """Apply input + output layout migrations for the default campaign."""
    campaign_dir = get_campaign_dir(campaign_id)
    return {
        "quadrature_snapshot": migrate_quadrature_layout(campaign_dir, snapshot_id),
        "results_renamed": migrate_results_layout(
            legacy_folder=campaign_id,
            snapshot_id=snapshot_id,
        ),
        "results_flat": migrate_legacy_flat_results(snapshot_id=snapshot_id),
        "datatrain_snapshot": migrate_datatrain_layout(snapshot_id=snapshot_id),
    }


def _list_state_files(data_dir: str) -> List[str]:
    import glob

    paths = glob.glob(os.path.join(data_dir, "[01]*.txt"))
    valid = []
    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]
        if base and set(base) <= {"0", "1"}:
            valid.append(path)
    return sorted(
        valid,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0], 2),
    )
