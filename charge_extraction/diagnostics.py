"""Post-hoc probes and latent visualizations for charge extraction."""

from __future__ import annotations

from collections import Counter
import copy
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import ChargeVIB
from .training import move_batch


def collect_latents(
    model: ChargeVIB, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    model.eval()
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    bath_bits: list[np.ndarray] = []
    charges: list[np.ndarray] = []
    outer_ids: list[np.ndarray] = []
    z_b_strings: list[str] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            mu, logvar = model.encode(batch["z_b"])
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            bath_bits.append(batch["z_b"].cpu().numpy())
            charges.append(batch["q_b"].cpu().numpy())
            outer_ids.append(batch["outer_id"].cpu().numpy())
            z_b_strings.extend(raw_batch["z_b_string"])
    return {
        "mu": np.concatenate(mus),
        "logvar": np.concatenate(logvars),
        "z_b": np.concatenate(bath_bits),
        "q_b": np.concatenate(charges),
        "outer_id": np.concatenate(outer_ids),
        "z_b_string": z_b_strings,
    }


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std


def _confusion_matrix(
    target: np.ndarray, prediction: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true_label, predicted_label in zip(target, prediction):
        matrix[
            class_to_index[int(true_label)], class_to_index[int(predicted_label)]
        ] += 1
    return matrix


def _classification_metrics(
    target: np.ndarray, prediction: np.ndarray, classes: np.ndarray
) -> dict[str, Any]:
    matrix = _confusion_matrix(target, prediction, classes)
    f1_values: list[float] = []
    per_class_accuracy: dict[str, float] = {}
    for index, label in enumerate(classes):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        row_total = matrix[index, :].sum()
        per_class_accuracy[str(int(label))] = (
            0.0 if row_total == 0 else float(true_positive / row_total)
        )
    return {
        "accuracy": float((target == prediction).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_charge_accuracy": per_class_accuracy,
        "confusion_matrix": matrix.tolist(),
    }


def fit_charge_probe(
    latent_by_split: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float = 0.03,
    patience: int = 60,
) -> dict[str, Any]:
    """Train a linear Q_B probe while keeping the VIB encoder frozen."""

    torch.manual_seed(seed)
    x_train = latent_by_split["train"]["mu"].astype(np.float32)
    q_train = latent_by_split["train"]["q_b"].astype(np.int64)
    classes = np.unique(q_train)
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    mean, std = _standardize_fit(x_train)

    def tensors(split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        split = latent_by_split[split_name]
        x = ((split["mu"] - mean) / std).astype(np.float32)
        unknown = set(int(value) for value in np.unique(split["q_b"])) - set(
            class_to_index
        )
        if unknown:
            raise ValueError(f"Probe split contains unseen charge labels: {unknown}")
        y = np.array(
            [class_to_index[int(value)] for value in split["q_b"]],
            dtype=np.int64,
        )
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    x_train_t, y_train_t = tensors("train")
    x_val_t, y_val_t = tensors("val")
    probe = nn.Linear(x_train.shape[1], len(classes)).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    counts = np.bincount(y_train_t.cpu().numpy(), minlength=len(classes))
    class_weights = counts.sum() / np.maximum(counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
    best_state = copy.deepcopy(probe.state_dict())
    best_val_loss = float("inf")
    wait = 0
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            probe(x_train_t), y_train_t, weight=class_weights_t
        )
        loss.backward()
        optimizer.step()
        probe.eval()
        with torch.inference_mode():
            val_loss = float(
                nn.functional.cross_entropy(probe(x_val_t), y_val_t).cpu()
            )
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = copy.deepcopy(probe.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    probe.load_state_dict(best_state)
    probe.eval()

    majority_charge = Counter(int(value) for value in q_train).most_common(1)[0][0]
    result: dict[str, Any] = {
        "classes": classes.astype(int).tolist(),
        "probe_type": "linear_softmax_on_frozen_latent_mean",
        "standardization": "fit_on_train_only",
        "best_val_cross_entropy": best_val_loss,
    }
    for split_name in ("train", "val", "test"):
        x_t, _ = tensors(split_name)
        with torch.inference_mode():
            predicted_index = probe(x_t).argmax(dim=-1).cpu().numpy()
        prediction = classes[predicted_index]
        target = latent_by_split[split_name]["q_b"]
        split_metrics = _classification_metrics(target, prediction, classes)
        split_metrics["majority_baseline_accuracy"] = float(
            (target == majority_charge).mean()
        )
        result[split_name] = split_metrics
    return result


def _sector_centers(x: np.ndarray, q_b: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(charge): x[q_b == charge].mean(axis=0, keepdims=True)
        for charge in np.unique(q_b)
    }


def fit_microscopic_bit_probe(
    latent_by_split: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float = 0.03,
    patience: int = 60,
) -> dict[str, Any]:
    """Probe z_B positions after removing charge-sector latent centroids."""

    torch.manual_seed(seed + 1)
    train = latent_by_split["train"]
    centers = _sector_centers(train["mu"], train["q_b"])

    def residual_features(split_name: str) -> np.ndarray:
        split = latent_by_split[split_name]
        residual = np.empty_like(split["mu"], dtype=np.float32)
        for index, (mu, charge) in enumerate(zip(split["mu"], split["q_b"])):
            residual[index] = mu - centers[int(charge)][0]
        return residual

    train_residual = residual_features("train")
    mean, std = _standardize_fit(train_residual)

    def tensors(split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        x = ((residual_features(split_name) - mean) / std).astype(np.float32)
        y = latent_by_split[split_name]["z_b"].astype(np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    x_train, y_train = tensors("train")
    x_val, y_val = tensors("val")
    n_bath = y_train.shape[1]
    probe = nn.Linear(x_train.shape[1], n_bath).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    positive = y_train.sum(dim=0)
    negative = y_train.shape[0] - positive
    pos_weight = negative / positive.clamp_min(1.0)
    best_state = copy.deepcopy(probe.state_dict())
    best_val_loss = float("inf")
    wait = 0
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            probe(x_train), y_train, pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        probe.eval()
        with torch.inference_mode():
            val_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    probe(x_val), y_val
                ).cpu()
            )
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = copy.deepcopy(probe.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    probe.load_state_dict(best_state)
    probe.eval()

    sector_bit_probability = {
        charge: train["z_b"][train["q_b"] == charge].mean(axis=0)
        for charge in np.unique(train["q_b"])
    }
    result: dict[str, Any] = {
        "probe_type": "linear_bit_probe_on_charge_centered_frozen_latent_mean",
        "best_val_binary_cross_entropy": best_val_loss,
    }
    for split_name in ("train", "val", "test"):
        x, y = tensors(split_name)
        with torch.inference_mode():
            prediction = (probe(x).sigmoid() >= 0.5).cpu().numpy()
        target = y.cpu().numpy().astype(bool)
        charge = latent_by_split[split_name]["q_b"]
        baseline = np.stack(
            [sector_bit_probability[int(value)] >= 0.5 for value in charge]
        )
        result[split_name] = {
            "bit_accuracy": float((prediction == target).mean()),
            "charge_only_baseline_bit_accuracy": float(
                (baseline == target).mean()
            ),
            "accuracy_above_charge_only": float(
                (prediction == target).mean() - (baseline == target).mean()
            ),
            "per_bit_accuracy": (prediction == target).mean(axis=0).tolist(),
        }
    return result


def latent_geometry(
    latent_by_split: dict[str, dict[str, Any]]
) -> dict[str, float]:
    all_mu = np.concatenate(
        [latent_by_split[name]["mu"] for name in ("train", "val", "test")]
    )
    all_q = np.concatenate(
        [latent_by_split[name]["q_b"] for name in ("train", "val", "test")]
    )
    global_center = all_mu.mean(axis=0, keepdims=True)
    between = 0.0
    within = 0.0
    for charge in np.unique(all_q):
        sector = all_mu[all_q == charge]
        center = sector.mean(axis=0, keepdims=True)
        between += len(sector) * float(np.square(center - global_center).sum())
        within += float(np.square(sector - center).sum())
    denominator = max(len(all_mu) - 1, 1)
    return {
        "between_charge_variance": between / denominator,
        "within_charge_variance": within / denominator,
        "between_to_within_ratio": between / max(within, 1e-12),
    }


def write_latent_csv(
    path: str | Path, latent_by_split: dict[str, dict[str, Any]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latent_dim = latent_by_split["train"]["mu"].shape[1]
    fieldnames = [
        "split",
        "outer_id",
        "z_b",
        "q_b",
        *[f"mu_{index}" for index in range(latent_dim)],
        *[f"logvar_{index}" for index in range(latent_dim)],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            split = latent_by_split[split_name]
            for row in range(len(split["q_b"])):
                record: dict[str, Any] = {
                    "split": split_name,
                    "outer_id": int(split["outer_id"][row]),
                    "z_b": split["z_b_string"][row],
                    "q_b": int(split["q_b"][row]),
                }
                record.update(
                    {
                        f"mu_{index}": float(split["mu"][row, index])
                        for index in range(latent_dim)
                    }
                )
                record.update(
                    {
                        f"logvar_{index}": float(split["logvar"][row, index])
                        for index in range(latent_dim)
                    }
                )
                writer.writerow(record)


def save_pca_plot(
    path: str | Path, latent_by_split: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Save a dependency-light PCA projection colored by Q_B."""

    path = Path(path)
    matplotlib_cache = path.parent / ".matplotlib_cache"
    font_cache = path.parent / ".cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_mu = np.concatenate(
        [latent_by_split[name]["mu"] for name in ("train", "val", "test")]
    )
    all_q = np.concatenate(
        [latent_by_split[name]["q_b"] for name in ("train", "val", "test")]
    )
    centered = all_mu - all_mu.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, right_vectors.shape[0])
    projected = centered @ right_vectors[:n_components].T
    if n_components == 1:
        projected = np.column_stack((projected[:, 0], np.zeros(len(projected))))
    variance = np.square(singular_values)
    explained = variance / max(variance.sum(), 1e-12)

    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    colors = {4: "#1f77b4", 5: "#ff7f0e", 6: "#2ca02c"}
    for charge in np.unique(all_q):
        mask = all_q == charge
        axis.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=20,
            alpha=0.65,
            label=rf"$Q_B={int(charge)}$",
            color=colors.get(int(charge)),
        )
    axis.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
    pc2_variance = 100 * explained[1] if len(explained) > 1 else 0.0
    axis.set_ylabel(f"PC2 ({pc2_variance:.1f}% variance)")
    axis.set_title("Latent means colored by bath charge")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "explained_variance_ratio": explained[:n_components].tolist(),
        "n_components": n_components,
    }
