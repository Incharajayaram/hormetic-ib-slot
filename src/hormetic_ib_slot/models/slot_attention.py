"""
SAVi-style Slot Attention with temporal slot propagation.

Based on:
  Locatello et al. "Object-Centric Learning with Slot Attention." NeurIPS 2020.
  Kipf et al.      "Conditional Object-Centric Learning from Video."  ICLR 2022.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotInitializer(nn.Module):
    """
    Learnable slot initialiser.

    Maintains (mu, log_sigma) parameters and samples slot vectors via the
    reparameterisation trick, then projects them to slot_dim.
    """

    def __init__(self, num_slots: int, slot_dim: int) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        mu = self.mu.expand(batch_size, -1, -1)
        sigma = self.log_sigma.exp().expand(batch_size, -1, -1)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # (B, num_slots, slot_dim)


class SlotAttention(nn.Module):
    """
    Slot Attention module (single-frame).

    Implements the iterative cross-attention update loop:
      1. LayerNorm inputs and slots
      2. Compute attention weights (slots attend to inputs)
      3. Update slots via GRU
      4. Refine slots via residual MLP
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        hidden_dim: int,
        num_iters: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        # Input projections
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.proj_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.proj_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.proj_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.norm_pre_ff = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(
        self, inputs: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, N, slot_dim)  — flattened spatial features
            slots:  (B, num_slots, slot_dim) — initial slot vectors

        Returns:
            slots: (B, num_slots, slot_dim) — updated slots
        """
        B, N, D = inputs.shape
        inputs_norm = self.norm_inputs(inputs)  # (B, N, D)
        k = self.proj_k(inputs_norm)            # (B, N, D)
        v = self.proj_v(inputs_norm)            # (B, N, D)

        for _ in range(self.num_iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)  # (B, num_slots, D)
            q = self.proj_q(slots_norm)          # (B, num_slots, D)

            # Attention: (B, num_slots, N)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn = F.softmax(attn_logits, dim=1)   # softmax over slots dim

            # Normalise attention weights over input positions (weighted mean)
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            # Aggregate: (B, num_slots, D)
            updates = torch.bmm(attn_weights, v)

            # GRU update (requires reshaping to (B*num_slots, D))
            slots_flat = slots_prev.reshape(B * self.num_slots, D)
            updates_flat = updates.reshape(B * self.num_slots, D)
            slots = self.gru(updates_flat, slots_flat).reshape(B, self.num_slots, D)

            # Residual MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class TemporalSlotAttention(nn.Module):
    """
    Video slot attention with temporal slot propagation (SAVi-style).

    Processes a sequence of frame features. Slot state from frame t−1
    initialises the attention at frame t, enabling identity tracking across
    time and occlusions.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        hidden_dim: int,
        num_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.initializer = SlotInitializer(num_slots, slot_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_iters=num_iters,
        )

    def forward(
        self,
        features_seq: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features_seq: (B, T, H*W, slot_dim) — projected frame features
            prev_slots:   (B, num_slots, slot_dim) or None

        Returns:
            slots_seq: (B, T, num_slots, slot_dim)
            final_slots: (B, num_slots, slot_dim) — last-frame slots for chaining
        """
        B, T, N, D = features_seq.shape
        device = features_seq.device

        if prev_slots is None:
            slots = self.initializer(B, device)
        else:
            slots = prev_slots

        all_slots = []
        for t in range(T):
            frame_features = features_seq[:, t]          # (B, N, D)
            slots = self.slot_attention(frame_features, slots)
            all_slots.append(slots.unsqueeze(1))          # (B, 1, num_slots, D)

        slots_seq = torch.cat(all_slots, dim=1)           # (B, T, num_slots, D)
        return slots_seq, slots
