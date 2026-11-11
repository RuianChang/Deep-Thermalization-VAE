"""Cluster-aware loading for hierarchical deep-thermalization shadows."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ClusterRecord:
    """All A-side shadows associated with one outer bath draw."""

    outer_id: int
    z_b: np.ndarray
    pauli_basis: np.ndarray
    outcome_class: np.ndarray
    q_b: int

    @property
    def z_b_string(self) -> str:
        return "".join(str(int(bit)) for bit in self.z_b)


@dataclass(frozen=True)
class DataContract:
    """Column and categorical-value metadata inferred from the CSV."""

    bath_columns: tuple[str, ...]
    pauli_columns: tuple[str, ...]
    outcome_columns: tuple[str, ...]
    pauli_value_to_index: dict[int, int]
    n_bath: int
    n_subsystem: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bath_columns": list(self.bath_columns),
            "pauli_columns": list(self.pauli_columns),
            "outcome_columns": list(self.outcome_columns),
            "pauli_value_to_index": {
                str(key): value for key, value in self.pauli_value_to_index.items()
            },
            "n_bath": self.n_bath,
            "n_subsystem": self.n_subsystem,
        }


def _numbered_columns(columns: Iterable[str], prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matched: list[tuple[int, str]] = []
    for column in columns:
        match = pattern.match(str(column))
        if match:
            matched.append((int(match.group(1)), str(column)))
    return [name for _, name in sorted(matched)]


def _integer_frame(df: pd.DataFrame, columns: Sequence[str], name: str) -> np.ndarray:
    numeric = df.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} must contain integer-valued entries.")
    return rounded.astype(np.int64, copy=False)


def load_cluster_records(csv_path: str | Path) -> tuple[list[ClusterRecord], DataContract]:
    """Load the per-qubit CSV and group every outer_id into one sample."""

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "outer_id" not in df.columns:
        raise ValueError("The CSV must contain an outer_id column.")

    bath_columns = _numbered_columns(df.columns, "z")
    pauli_columns = _numbered_columns(df.columns, "P")
    outcome_columns = _numbered_columns(df.columns, "b")
    if not bath_columns:
        raise ValueError("No bath columns z1, z2, ... were found.")
    if not pauli_columns or pauli_columns != [
        f"P{i}" for i in range(1, len(pauli_columns) + 1)
    ]:
        raise ValueError("Pauli columns must be consecutively named P1, P2, ...")
    if not outcome_columns or outcome_columns != [
        f"b{i}" for i in range(1, len(outcome_columns) + 1)
    ]:
        raise ValueError("Outcome columns must be consecutively named b1, b2, ...")
    if len(pauli_columns) != len(outcome_columns):
        raise ValueError("The numbers of Pauli and outcome columns must match.")
    if len(outcome_columns) != 2:
        raise ValueError(
            "This first implementation uses the N_A=2 four-class decoder and "
            "therefore requires exactly b1, b2 and P1, P2."
        )

    z_all = _integer_frame(df, bath_columns, "Bath bits")
    p_all = _integer_frame(df, pauli_columns, "Pauli labels")
    b_all = _integer_frame(df, outcome_columns, "A-side outcomes")
    if not np.isin(z_all, [0, 1]).all():
        raise ValueError("Bath columns must contain only 0 or 1.")
    if not np.isin(b_all, [0, 1]).all():
        raise ValueError("Outcome columns must contain only 0 or 1.")

    raw_pauli_values = sorted(int(value) for value in np.unique(p_all))
    if len(raw_pauli_values) != 3:
        raise ValueError(
            "Expected exactly three local Pauli labels (X, Y, Z), but found "
            f"{raw_pauli_values}."
        )
    pauli_value_to_index = {
        raw_value: index for index, raw_value in enumerate(raw_pauli_values)
    }
    p_mapped = np.vectorize(pauli_value_to_index.__getitem__, otypes=[np.int64])(
        p_all
    )

    working = df.copy()
    working["_row_index"] = np.arange(len(working), dtype=np.int64)
    records: list[ClusterRecord] = []
    for outer_id, group in working.groupby("outer_id", sort=False):
        row_indices = group["_row_index"].to_numpy(dtype=np.int64)
        z_group = z_all[row_indices]
        if not np.all(z_group == z_group[0]):
            raise ValueError(
                f"outer_id={outer_id!r} contains more than one exact bath record."
            )
        try:
            outer_id_int = int(outer_id)
        except (TypeError, ValueError) as error:
            raise ValueError("outer_id values must be integer-like.") from error
        z_b = z_group[0].copy()
        outcomes = b_all[row_indices]
        outcome_class = 2 * outcomes[:, 0] + outcomes[:, 1]
        records.append(
            ClusterRecord(
                outer_id=outer_id_int,
                z_b=z_b,
                pauli_basis=p_mapped[row_indices].copy(),
                outcome_class=outcome_class.astype(np.int64, copy=False),
                q_b=int(z_b.sum()),
            )
        )

    if not records:
        raise ValueError("The CSV contains no clusters.")
    if len({record.outer_id for record in records}) != len(records):
        raise ValueError("outer_id must uniquely identify a cluster.")

    contract = DataContract(
        bath_columns=tuple(bath_columns),
        pauli_columns=tuple(pauli_columns),
        outcome_columns=tuple(outcome_columns),
        pauli_value_to_index=pauli_value_to_index,
        n_bath=len(bath_columns),
        n_subsystem=len(outcome_columns),
    )
    return records, contract


class ClusterDataset(Dataset[ClusterRecord]):
    """A dataset whose unit of sampling is one complete bath cluster."""

    def __init__(self, records: Sequence[ClusterRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ClusterRecord:
        return self.records[index]


def collate_clusters(records: Sequence[ClusterRecord]) -> dict[str, Any]:
    """Pad variable-R_A clusters while retaining a reconstruction mask."""

    if not records:
        raise ValueError("Cannot collate an empty batch.")
    batch_size = len(records)
    n_bath = records[0].z_b.shape[0]
    n_subsystem = records[0].pauli_basis.shape[1]
    max_shadows = max(record.pauli_basis.shape[0] for record in records)

    z_b = torch.empty((batch_size, n_bath), dtype=torch.long)
    pauli_basis = torch.zeros(
        (batch_size, max_shadows, n_subsystem), dtype=torch.long
    )
    outcome_class = torch.zeros((batch_size, max_shadows), dtype=torch.long)
    shadow_mask = torch.zeros((batch_size, max_shadows), dtype=torch.bool)
    q_b = torch.empty(batch_size, dtype=torch.long)
    outer_id = torch.empty(batch_size, dtype=torch.long)
    z_b_string: list[str] = []

    for index, record in enumerate(records):
        n_shadows = record.pauli_basis.shape[0]
        z_b[index] = torch.from_numpy(record.z_b)
        pauli_basis[index, :n_shadows] = torch.from_numpy(record.pauli_basis)
        outcome_class[index, :n_shadows] = torch.from_numpy(record.outcome_class)
        shadow_mask[index, :n_shadows] = True
        q_b[index] = record.q_b
        outer_id[index] = record.outer_id
        z_b_string.append(record.z_b_string)

    return {
        "z_b": z_b,
        "pauli_basis": pauli_basis,
        "outcome_class": outcome_class,
        "shadow_mask": shadow_mask,
        "q_b": q_b,
        "outer_id": outer_id,
        "z_b_string": z_b_string,
    }


def _split_counts(
    n_groups: int, val_fraction: float, test_fraction: float
) -> tuple[int, int, int]:
    if n_groups < 3:
        raise ValueError(
            "Each charge sector needs at least three split groups for "
            "stratified train/validation/test splitting."
        )
    n_test = max(1, int(round(n_groups * test_fraction)))
    n_val = max(1, int(round(n_groups * val_fraction)))
    while n_test + n_val >= n_groups:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Could not leave at least one training group.")
    return n_groups - n_val - n_test, n_val, n_test


def split_records(
    records: Sequence[ClusterRecord],
    *,
    seed: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_by: str = "exact_zb",
) -> dict[str, list[ClusterRecord]]:
    """Create charge-stratified splits without breaking the requested groups."""

    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation plus test fraction must be less than 1.")
    if split_by not in {"exact_zb", "outer_id"}:
        raise ValueError("split_by must be either 'exact_zb' or 'outer_id'.")

    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key: Any = record.z_b_string if split_by == "exact_zb" else record.outer_id
        grouped_indices[key].append(index)

    groups_by_charge: dict[int, list[Any]] = defaultdict(list)
    for key, indices in grouped_indices.items():
        charges = {records[index].q_b for index in indices}
        if len(charges) != 1:
            raise ValueError(f"Split group {key!r} contains multiple bath charges.")
        groups_by_charge[next(iter(charges))].append(key)

    rng = np.random.default_rng(seed)
    group_assignment: dict[Any, str] = {}
    for charge, keys in sorted(groups_by_charge.items()):
        keys = list(keys)
        rng.shuffle(keys)
        n_train, n_val, _ = _split_counts(
            len(keys), val_fraction=val_fraction, test_fraction=test_fraction
        )
        for key in keys[:n_train]:
            group_assignment[key] = "train"
        for key in keys[n_train : n_train + n_val]:
            group_assignment[key] = "val"
        for key in keys[n_train + n_val :]:
            group_assignment[key] = "test"

    result: dict[str, list[ClusterRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for record in records:
        key = record.z_b_string if split_by == "exact_zb" else record.outer_id
        result[group_assignment[key]].append(record)

    for split_name, split_records_ in result.items():
        if not split_records_:
            raise ValueError(f"The {split_name} split is empty.")
    return result


def records_summary(records: Sequence[ClusterRecord]) -> dict[str, Any]:
    shadows_per_cluster = Counter(
        int(record.pauli_basis.shape[0]) for record in records
    )
    charge_counts = Counter(int(record.q_b) for record in records)
    exact_bath_counts = Counter(record.z_b_string for record in records)
    return {
        "n_clusters": len(records),
        "n_rows": int(sum(record.pauli_basis.shape[0] for record in records)),
        "n_unique_exact_zb": len(exact_bath_counts),
        "max_outer_draws_per_exact_zb": max(exact_bath_counts.values()),
        "shadows_per_cluster": {
            str(key): value for key, value in sorted(shadows_per_cluster.items())
        },
        "charge_counts": {
            str(key): value for key, value in sorted(charge_counts.items())
        },
    }


def split_manifest(splits: dict[str, Sequence[ClusterRecord]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for split_name, records in splits.items():
        for record in records:
            manifest.append(
                {
                    "split": split_name,
                    "outer_id": record.outer_id,
                    "z_b": record.z_b_string,
                    "q_b": record.q_b,
                    "n_shadows": int(record.pauli_basis.shape[0]),
                }
            )
    return manifest


def make_loader(
    records: Sequence[ClusterRecord],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ClusterDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_clusters,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
