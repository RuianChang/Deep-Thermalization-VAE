"""Run the provisional beta and latent-dimension grid sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd


DEFAULT_BETAS = "0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1"
DEFAULT_LATENT_DIMS = "1,2,4,8,16"


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--betas", default=DEFAULT_BETAS)
    parser.add_argument("--latent-dims", default=DEFAULT_LATENT_DIMS)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, train_args = parser.parse_known_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for seed in _ints(args.seeds):
        for latent_dim in _ints(args.latent_dims):
            for beta in _floats(args.betas):
                run_name = f"beta_{beta:.8g}_d_{latent_dim}_seed_{seed}"
                run_dir = output_root / run_name
                summary_path = run_dir / "summary.json"
                if summary_path.is_file() and not args.overwrite:
                    print(f"Skipping completed run: {run_name}")
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "charge_extraction.train",
                        "--csv",
                        str(args.csv),
                        "--output-dir",
                        str(run_dir),
                        "--beta",
                        str(beta),
                        "--latent-dim",
                        str(latent_dim),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        *train_args,
                    ]
                    print("Running:", " ".join(command))
                    subprocess.run(command, check=True)
                with summary_path.open() as handle:
                    summaries.append(json.load(handle))
                pd.DataFrame(summaries).to_csv(
                    output_root / "sweep_summary.csv", index=False
                )


if __name__ == "__main__":
    main()
