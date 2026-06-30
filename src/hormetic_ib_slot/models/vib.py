"""
Variational Information Bottleneck head for per-slot latent compression.

Applied after slot attention: maps each slot vector h_k into a stochastic
latent z_k ~ N(mu_k, sigma_k^2).

The VIB objective (Alemi et al., 2017):
    L = E[log p(y|z)] - beta * KL[q(z|x) || p(z)]

where KL has closed form for diagonal Gaussians:
    KL = 0.5 * sum_d ( exp(log_var) + mu^2 - 1 - log_var )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VIBHead(nn.Module):
    """
    Variational IB head applied per slot.

    Maps slot vectors (B, num_slots, slot_dim) to stochastic latent codes
    (B, num_slots, latent_dim) via diagonal Gaussian reparameterisation.
    """

    def __init__(
        self,
        slot_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def encode(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: (B, num_slots, slot_dim)

        Returns:
            mu:      (B, num_slots, latent_dim)
            log_var: (B, num_slots, latent_dim)  — clamped for stability
        """
        out = self.encoder(slots)                          # (B, num_slots, 2*latent_dim)
        mu, log_var = out.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -10.0, 10.0)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Standard reparameterisation trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # deterministic at eval time

    def forward(
        self, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: (B, num_slots, slot_dim)

        Returns:
            z:        (B, num_slots, latent_dim)
            mu:       (B, num_slots, latent_dim)
            log_var:  (B, num_slots, latent_dim)
            kl_loss:  scalar — unweighted KL; multiply by beta in the loss function
        """
        mu, log_var = self.encode(slots)
        z = self.reparameterize(mu, log_var)

        # Closed-form KL: mean over (B, num_slots), sum over latent_dim
        kl_per_slot = 0.5 * (log_var.exp() + mu.pow(2) - 1.0 - log_var)
        kl_loss = kl_per_slot.sum(dim=-1).mean()          # scalar

        return z, mu, log_var, kl_loss


class VIBPredictionHead(nn.Module):
    """
    Projection head for contrastive slot-identity loss.

    Maps latent z → L2-normalised prediction vector used to encourage
    the same physical object to map to consistent latent codes across frames,
    even through occlusion.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, num_slots, latent_dim)

        Returns:
            proj: (B, num_slots, latent_dim) — L2-normalised
        """
        proj = self.mlp(z)
        return F.normalize(proj, p=2, dim=-1)
