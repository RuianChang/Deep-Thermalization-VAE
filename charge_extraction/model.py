"""Variational information-bottleneck model for conditional A-side shadows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ChargeVIBConfig:
    n_bath: int = 10
    n_subsystem: int = 2
    latent_dim: int = 2
    embed_dim: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 2
    encoder_ff_dim: int = 256
    basis_embed_dim: int = 16
    decoder_hidden_dim: int = 128
    dropout: float = 0.1
    variance_mode: str = "input_dependent"
    kl_reduction: str = "sum"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BathEncoder(nn.Module):
    """Encode the exact bath bitstring into one global Gaussian latent."""

    def __init__(self, config: ChargeVIBConfig):
        super().__init__()
        if config.embed_dim % config.n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads.")
        if config.variance_mode not in {"input_dependent", "global"}:
            raise ValueError("variance_mode must be input_dependent or global.")

        self.config = config
        self.bit_embedding = nn.Embedding(2, config.embed_dim)
        self.position_embedding = nn.Embedding(config.n_bath + 1, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.n_encoder_layers,
            norm=nn.LayerNorm(config.embed_dim),
        )
        self.mu_head = nn.Linear(config.embed_dim, config.latent_dim)
        if config.variance_mode == "input_dependent":
            self.logvar_head: nn.Module | None = nn.Linear(
                config.embed_dim, config.latent_dim
            )
            self.global_logvar = None
        else:
            self.logvar_head = None
            self.global_logvar = nn.Parameter(torch.zeros(config.latent_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, z_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z_b.ndim != 2 or z_b.shape[1] != self.config.n_bath:
            raise ValueError(
                f"z_b must have shape [batch, {self.config.n_bath}], "
                f"received {tuple(z_b.shape)}."
            )
        batch_size = z_b.shape[0]
        token_embeddings = self.bit_embedding(z_b)
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls, token_embeddings), dim=1)
        positions = torch.arange(
            self.config.n_bath + 1, device=z_b.device
        ).unsqueeze(0)
        sequence = sequence + self.position_embedding(positions)
        pooled = self.transformer(sequence)[:, 0]
        mu = self.mu_head(pooled)
        if self.logvar_head is not None:
            logvar = self.logvar_head(pooled)
        else:
            assert self.global_logvar is not None
            logvar = self.global_logvar.unsqueeze(0).expand_as(mu)
        return mu, logvar.clamp(min=-12.0, max=8.0)


class MeasurementDecoder(nn.Module):
    """Predict the joint two-qubit outcome for a chosen local Pauli basis."""

    def __init__(self, config: ChargeVIBConfig):
        super().__init__()
        self.config = config
        self.basis_embedding = nn.Embedding(3, config.basis_embed_dim)
        self.position_embedding = nn.Embedding(
            config.n_subsystem, config.basis_embed_dim
        )
        decoder_input_dim = (
            config.latent_dim + config.n_subsystem * config.basis_embed_dim
        )
        self.network = nn.Sequential(
            nn.LayerNorm(decoder_input_dim),
            nn.Linear(decoder_input_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, 2**config.n_subsystem),
        )

    def forward(self, h: torch.Tensor, pauli_basis: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("h must have shape [batch, K_h, latent_dim].")
        if pauli_basis.ndim != 3:
            raise ValueError(
                "pauli_basis must have shape [batch, R_A, n_subsystem]."
            )
        batch_size, k_h, latent_dim = h.shape
        basis_batch, n_shadows, n_subsystem = pauli_basis.shape
        if batch_size != basis_batch:
            raise ValueError("h and pauli_basis batch dimensions do not match.")
        if latent_dim != self.config.latent_dim:
            raise ValueError("Unexpected latent dimension.")
        if n_subsystem != self.config.n_subsystem:
            raise ValueError("Unexpected subsystem size.")

        positions = torch.arange(n_subsystem, device=pauli_basis.device)
        basis = self.basis_embedding(pauli_basis)
        basis = basis + self.position_embedding(positions)[None, None, :, :]
        basis = basis.flatten(start_dim=-2)
        basis = basis[:, None, :, :].expand(-1, k_h, -1, -1)
        latent = h[:, :, None, :].expand(-1, -1, n_shadows, -1)
        return self.network(torch.cat((latent, basis), dim=-1))


class ChargeVIB(nn.Module):
    """Cluster-aware beta-VAE / variational information bottleneck."""

    def __init__(self, config: ChargeVIBConfig):
        super().__init__()
        self.config = config
        self.encoder = BathEncoder(config)
        self.decoder = MeasurementDecoder(config)

    def encode(self, z_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(z_b)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor, k_h: int
    ) -> torch.Tensor:
        if k_h < 1:
            raise ValueError("k_h must be positive.")
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn(
            (mu.shape[0], k_h, mu.shape[1]),
            dtype=mu.dtype,
            device=mu.device,
        )
        return mu[:, None, :] + std[:, None, :] * epsilon

    @staticmethod
    def kl_by_dimension(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * (mu.square() + logvar.exp() - logvar - 1.0)

    def forward(
        self,
        z_b: torch.Tensor,
        pauli_basis: torch.Tensor,
        *,
        k_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(z_b)
        h = self.reparameterize(mu, logvar, k_h=k_h)
        logits = self.decoder(h, pauli_basis)
        return logits, mu, logvar

    def objective(
        self,
        batch: dict[str, torch.Tensor],
        *,
        beta: float,
        k_h: int,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the grouped objective and both predictive NLL conventions."""

        logits, mu, logvar = self(
            batch["z_b"], batch["pauli_basis"], k_h=k_h
        )
        targets = batch["outcome_class"][:, None, :, None]
        targets = targets.expand(-1, k_h, -1, 1)
        # Keep likelihood arithmetic in float32 even under CUDA autocast.
        log_probabilities = logits.float().log_softmax(dim=-1)
        selected_log_probability = (
            log_probabilities.gather(-1, targets).squeeze(-1)
        )
        mask = batch["shadow_mask"][:, None, :].to(logits.dtype)
        shadows_per_cluster = mask.sum(dim=-1).clamp_min(1.0)

        nll_by_k_cluster = -(
            selected_log_probability * mask
        ).sum(dim=-1) / shadows_per_cluster
        expected_nll_by_cluster = nll_by_k_cluster.mean(dim=1)

        selected_probability = selected_log_probability.exp()
        marginal_probability = selected_probability.mean(dim=1).clamp_min(1e-12)
        row_mask = batch["shadow_mask"].to(logits.dtype)
        marginal_nll_by_cluster = -(
            marginal_probability.log() * row_mask
        ).sum(dim=-1) / row_mask.sum(dim=-1).clamp_min(1.0)

        kl_dim = self.kl_by_dimension(mu.float(), logvar.float())
        kl_sum_by_cluster = kl_dim.sum(dim=-1)
        kl_mean_by_cluster = kl_dim.mean(dim=-1)
        if self.config.kl_reduction == "sum":
            penalty_by_cluster = kl_sum_by_cluster
        elif self.config.kl_reduction == "mean":
            penalty_by_cluster = kl_mean_by_cluster
        else:
            raise ValueError("kl_reduction must be sum or mean.")
        total_by_cluster = expected_nll_by_cluster + beta * penalty_by_cluster

        marginal_prediction = (
            log_probabilities.exp().mean(dim=1).argmax(dim=-1)
        )
        correct = (
            (marginal_prediction == batch["outcome_class"]) * batch["shadow_mask"]
        ).sum()
        n_valid_rows = batch["shadow_mask"].sum().clamp_min(1)

        return {
            "loss": total_by_cluster.mean(),
            "expected_nll": expected_nll_by_cluster.mean(),
            "marginal_nll": marginal_nll_by_cluster.mean(),
            "kl_sum": kl_sum_by_cluster.mean(),
            "kl_mean": kl_mean_by_cluster.mean(),
            "accuracy": correct.to(logits.dtype) / n_valid_rows,
        }
