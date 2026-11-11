"""Train one charge-extraction VIB model and run frozen-latent diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd
import torch

from .data import (
    load_cluster_records,
    make_loader,
    records_summary,
    split_manifest,
    split_records,
)
from .diagnostics import (
    collect_latents,
    fit_charge_probe,
    fit_microscopic_bit_probe,
    latent_geometry,
    save_pca_plot,
    write_latent_csv,
)
from .model import ChargeVIB, ChargeVIBConfig
from .training import evaluate, fit_model, resolve_device, save_json, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "U1_dataset_LA=2_LB=10_Neel.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a cluster-aware variational information bottleneck and test "
            "whether its frozen latent representation extracts Q_B."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-by", choices=("exact_zb", "outer_id"), default="exact_zb")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)

    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-ff-dim", type=int, default=256)
    parser.add_argument("--basis-embed-dim", type=int, default=16)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--variance-mode",
        choices=("input_dependent", "global"),
        default="input_dependent",
    )
    parser.add_argument("--kl-reduction", choices=("sum", "mean"), default="sum")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--k-train", type=int, default=1)
    parser.add_argument("--k-eval", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 autocast on CUDA. It is disabled automatically on CPU.",
    )
    parser.add_argument("--probe-epochs", type=int, default=500)
    return parser


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return metadata


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.beta < 0:
        raise ValueError("beta must be non-negative.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    set_seed(args.seed, deterministic=args.deterministic)

    records, contract = load_cluster_records(args.csv)
    splits = split_records(
        records,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_by=args.split_by,
    )
    split_summaries = {
        name: records_summary(split_records_) for name, split_records_ in splits.items()
    }
    if args.split_by == "exact_zb":
        exact_sets = {
            name: {record.z_b_string for record in split_records_}
            for name, split_records_ in splits.items()
        }
        assert exact_sets["train"].isdisjoint(exact_sets["val"])
        assert exact_sets["train"].isdisjoint(exact_sets["test"])
        assert exact_sets["val"].isdisjoint(exact_sets["test"])

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        splits["train"],
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        splits["val"],
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        splits["test"],
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 2,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = ChargeVIBConfig(
        n_bath=contract.n_bath,
        n_subsystem=contract.n_subsystem,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        encoder_ff_dim=args.encoder_ff_dim,
        basis_embed_dim=args.basis_embed_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout,
        variance_mode=args.variance_mode,
        kl_reduction=args.kl_reduction,
    )
    model = ChargeVIB(model_config).to(device)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())

    run_config = {
        "arguments": vars(args),
        "model": model_config.to_dict(),
        "data_contract": contract.to_dict(),
        "full_dataset": records_summary(records),
        "splits": split_summaries,
        "environment": environment_metadata(device),
        "n_trainable_parameters": n_parameters,
        "amp_active": amp,
        "scientific_guardrail": (
            "Q_B is not used in the VIB objective. It is used only by post-hoc "
            "probes after the encoder has been frozen."
        ),
    }
    save_json(output_dir / "run_config.json", run_config)
    pd.DataFrame(split_manifest(splits)).to_csv(
        output_dir / "split_manifest.csv", index=False
    )

    print(f"device={device} amp={amp} parameters={n_parameters:,}")
    print(
        "clusters="
        + ", ".join(
            f"{name}:{summary['n_clusters']}"
            for name, summary in split_summaries.items()
        )
    )
    fit_model(
        model,
        train_loader,
        val_loader,
        device=device,
        output_dir=output_dir,
        beta=args.beta,
        k_train=args.k_train,
        k_eval=args.k_eval,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        amp=amp,
        eval_seed=args.seed + 10_000,
        checkpoint_metadata=run_config,
    )

    val_metrics = evaluate(
        model,
        val_loader,
        device=device,
        beta=args.beta,
        k_h=args.k_eval,
        amp=amp,
        eval_seed=args.seed + 20_000,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        beta=args.beta,
        k_h=args.k_eval,
        amp=amp,
        eval_seed=args.seed + 30_000,
    )
    predictive_metrics = {"val": val_metrics, "test": test_metrics}
    save_json(output_dir / "predictive_metrics.json", predictive_metrics)

    diagnostic_loaders = {
        name: make_loader(
            split_records_,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 100 + index,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        for index, (name, split_records_) in enumerate(splits.items())
    }
    latent_by_split = {
        name: collect_latents(model, loader, device)
        for name, loader in diagnostic_loaders.items()
    }
    write_latent_csv(output_dir / "latent_embeddings.csv", latent_by_split)
    charge_probe = fit_charge_probe(
        latent_by_split,
        device=device,
        seed=args.seed,
        epochs=args.probe_epochs,
    )
    microscopic_probe = fit_microscopic_bit_probe(
        latent_by_split,
        device=device,
        seed=args.seed,
        epochs=args.probe_epochs,
    )
    geometry = latent_geometry(latent_by_split)
    pca_metadata = save_pca_plot(output_dir / "latent_pca.png", latent_by_split)
    diagnostics = {
        "charge_probe": charge_probe,
        "microscopic_bit_probe": microscopic_probe,
        "latent_geometry": geometry,
        "pca": pca_metadata,
    }
    save_json(output_dir / "latent_diagnostics.json", diagnostics)

    charge_test = charge_probe["test"]
    microscopic_test = microscopic_probe["test"]
    summary = {
        "output_dir": output_dir,
        "beta": args.beta,
        "latent_dim": args.latent_dim,
        "split_by": args.split_by,
        "test_marginal_nll": test_metrics["marginal_nll"],
        "test_outcome_accuracy": test_metrics["accuracy"],
        "test_kl_sum_nats": test_metrics["kl_sum"],
        "test_charge_probe_accuracy": charge_test["accuracy"],
        "test_charge_probe_macro_f1": charge_test["macro_f1"],
        "test_charge_majority_baseline": charge_test[
            "majority_baseline_accuracy"
        ],
        "test_microscopic_bit_accuracy_above_charge_only": microscopic_test[
            "accuracy_above_charge_only"
        ],
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in summary.items()}, indent=2))
    return summary


if __name__ == "__main__":
    main()
