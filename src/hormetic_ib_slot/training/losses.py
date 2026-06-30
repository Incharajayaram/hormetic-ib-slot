import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target, reduction='mean')


def vib_loss(recon: torch.Tensor, target: torch.Tensor, kl_loss: torch.Tensor, beta: float):
    recon_l = reconstruction_loss(recon, target)
    kl_l = kl_loss
    total = recon_l + beta * kl_l
    return total, recon_l, kl_l


def hungarian_match(pred_slots: torch.Tensor, gt_slots: torch.Tensor) -> torch.Tensor:
    """
    pred_slots: (B, num_slots, latent_dim)
    gt_slots:   (B, num_slots, latent_dim)
    Returns indices (B, num_slots) — permutation of pred that best matches gt (L2).
    """
    B, num_slots, _ = pred_slots.shape
    indices = torch.zeros(B, num_slots, dtype=torch.long, device=pred_slots.device)

    pred_np = pred_slots.detach().cpu().numpy()
    gt_np = gt_slots.detach().cpu().numpy()

    for b in range(B):
        # cost[i, j] = L2 distance between pred_slots[b, i] and gt_slots[b, j]
        diff = pred_np[b][:, None, :] - gt_np[b][None, :, :]  # (S, S, D)
        cost = np.linalg.norm(diff, axis=-1)  # (S, S)
        row_ind, col_ind = linear_sum_assignment(cost)
        # row_ind[k] = pred slot index, col_ind[k] = gt slot index it's matched to
        # We want: for each pred slot position, which gt slot it maps to
        perm = np.zeros(num_slots, dtype=np.int64)
        perm[row_ind] = col_ind
        indices[b] = torch.from_numpy(perm)

    return indices


def slot_identity_loss(
    z_t: torch.Tensor,
    z_t_future: torch.Tensor,
    match_indices: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE-style identity loss between slots at time t and t+k.
    z_t, z_t_future: (B, num_slots, latent_dim)
    match_indices: (B, num_slots) from hungarian_match — if None, computed internally.
    Pulls matched pairs together, pushes all other pairs apart.
    """
    B, num_slots, latent_dim = z_t.shape

    # Normalize
    z_t_norm = F.normalize(z_t, dim=-1)          # (B, S, D)
    z_f_norm = F.normalize(z_t_future, dim=-1)   # (B, S, D)

    if match_indices is None:
        match_indices = hungarian_match(z_t, z_t_future)

    total_loss = torch.tensor(0.0, device=z_t.device)

    for b in range(B):
        # Permute z_t_future according to matching
        perm = match_indices[b]  # (S,)
        z_f_perm = z_f_norm[b][perm]  # (S, D) — reordered future slots

        # Similarity matrix: query = z_t, keys = all future slots
        # sim[i, j] = cosine similarity between z_t[i] and z_f_perm[j]
        sim = torch.matmul(z_t_norm[b], z_f_perm.T) / temperature  # (S, S)

        # Positive pairs are on the diagonal (slot i matches slot i after permutation)
        labels = torch.arange(num_slots, device=z_t.device)
        loss_b = F.cross_entropy(sim, labels)
        total_loss = total_loss + loss_b

    return total_loss / B


def total_training_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    kl_loss: torch.Tensor,
    beta: float,
    z_t: Optional[torch.Tensor] = None,
    z_t_future: Optional[torch.Tensor] = None,
    lambda_identity: float = 0.1,
) -> dict:
    total, recon_l, kl_l = vib_loss(recon, target, kl_loss, beta)

    identity_l = torch.tensor(0.0, device=recon.device)
    if z_t is not None and z_t_future is not None:
        identity_l = slot_identity_loss(z_t, z_t_future)
        total = total + lambda_identity * identity_l

    return {
        'total': total,
        'recon': recon_l,
        'kl': kl_l,
        'identity': identity_l,
    }
