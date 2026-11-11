"""Training and evaluation utilities for the charge VIB."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import ChargeVIB


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return device


def move_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def run_epoch(
    model: ChargeVIB,
    loader: DataLoader,
    *,
    device: torch.device,
    beta: float,
    k_h: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n_clusters = 0

    context = nullcontext() if training else torch.inference_mode()
    with context:
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            batch_size = int(batch["z_b"].shape[0])
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, amp):
                metrics = model.objective(batch, beta=beta, k_h=k_h)
            if training:
                assert scaler is not None
                scaler.scale(metrics["loss"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            n_clusters += batch_size

    return {name: value / n_clusters for name, value in totals.items()}


def evaluate(
    model: ChargeVIB,
    loader: DataLoader,
    *,
    device: torch.device,
    beta: float,
    k_h: int,
    amp: bool,
    eval_seed: int,
) -> dict[str, float]:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(eval_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(eval_seed)
        return run_epoch(
            model,
            loader,
            device=device,
            beta=beta,
            k_h=k_h,
            optimizer=None,
            scaler=None,
            amp=amp,
            grad_clip=math.inf,
        )


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value

    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True) + "\n")


def fit_model(
    model: ChargeVIB,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    output_dir: Path,
    beta: float,
    k_train: int,
    k_eval: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    grad_clip: float,
    amp: bool,
    eval_seed: int,
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            beta=beta,
            k_h=k_train,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            beta=beta,
            k_h=k_eval,
            amp=amp,
            eval_seed=eval_seed,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_marginal_nll={val_metrics['marginal_nll']:.6f} "
            f"val_kl_sum={val_metrics['kl_sum']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss - 1e-8:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "metadata": checkpoint_metadata,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        save_json(output_dir / "training_history.json", history)
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    # The checkpoint is written by this process and includes run metadata in
    # addition to tensors, so PyTorch 2.6+ must not use weights_only=True.
    checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return history
