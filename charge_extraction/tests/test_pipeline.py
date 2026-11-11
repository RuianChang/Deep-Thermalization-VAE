"""Fast tests for data grouping, split isolation, and the grouped objective."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from charge_extraction.data import (
    collate_clusters,
    load_cluster_records,
    split_records,
)
from charge_extraction.model import ChargeVIB, ChargeVIBConfig


def synthetic_csv(path: Path) -> None:
    rng = np.random.default_rng(7)
    rows: list[dict[str, int]] = []
    outer_id = 0
    for charge in (4, 5, 6):
        seen: set[tuple[int, ...]] = set()
        while len(seen) < 9:
            one_positions = rng.choice(10, size=charge, replace=False)
            z_b = np.zeros(10, dtype=np.int64)
            z_b[one_positions] = 1
            key = tuple(int(value) for value in z_b)
            if key in seen:
                continue
            seen.add(key)
            repetitions = 2 if len(seen) == 1 else 1
            for _ in range(repetitions):
                for _ in range(4):
                    row = {"outer_id": outer_id}
                    row.update({f"z{i + 1}": int(z_b[i]) for i in range(10)})
                    row.update(
                        {
                            "P1": int(rng.choice([2, 3, 4])),
                            "b1": int(rng.integers(2)),
                            "P2": int(rng.choice([2, 3, 4])),
                            "b2": int(rng.integers(2)),
                        }
                    )
                    rows.append(row)
                outer_id += 1
    pd.DataFrame(rows).to_csv(path, index=False)


class PipelineTest(unittest.TestCase):
    def test_grouping_split_and_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "data.csv"
            synthetic_csv(csv_path)
            records, contract = load_cluster_records(csv_path)
        self.assertEqual(contract.n_bath, 10)
        self.assertTrue(all(record.pauli_basis.shape == (4, 2) for record in records))

        splits = split_records(records, seed=11, split_by="exact_zb")
        exact = {
            name: {record.z_b_string for record in records_}
            for name, records_ in splits.items()
        }
        self.assertTrue(exact["train"].isdisjoint(exact["val"]))
        self.assertTrue(exact["train"].isdisjoint(exact["test"]))
        self.assertTrue(exact["val"].isdisjoint(exact["test"]))

        batch = collate_clusters(splits["train"][:5])
        config = ChargeVIBConfig(
            n_bath=10,
            latent_dim=3,
            embed_dim=16,
            n_heads=2,
            n_encoder_layers=1,
            encoder_ff_dim=32,
            basis_embed_dim=4,
            decoder_hidden_dim=16,
            dropout=0.0,
        )
        model = ChargeVIB(config)
        metrics = model.objective(batch, beta=1e-3, k_h=2)
        self.assertEqual(set(metrics), {
            "loss",
            "expected_nll",
            "marginal_nll",
            "kl_sum",
            "kl_mean",
            "accuracy",
        })
        self.assertTrue(torch.isfinite(metrics["loss"]))
        metrics["loss"].backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
