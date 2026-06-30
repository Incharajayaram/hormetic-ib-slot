import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional


def _hungarian_cosine_match(slots_a: np.ndarray, slots_b: np.ndarray) -> np.ndarray:
    """
    slots_a, slots_b: (num_slots, slot_dim) numpy arrays (already L2-normalized)
    Returns permutation perm such that slots_b[perm[i]] is matched to slots_a[i].
    """
    num_slots = slots_a.shape[0]
    # Cosine similarity matrix: maximize <=> minimize negative
    sim = slots_a @ slots_b.T   # (S, S)
    cost = -sim
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros(num_slots, dtype=np.int64)
    perm[row_ind] = col_ind
    return perm


def compute_slot_cosine_stability(slots_seq: torch.Tensor) -> float:
    """
    Measures how stable slot representations are across consecutive frames.

    slots_seq: (B, T, num_slots, slot_dim)
    Returns mean cosine similarity of Hungarian-matched slots across consecutive frames.
    Higher = more temporally stable.
    """
    B, T, num_slots, slot_dim = slots_seq.shape
    if T < 2:
        return 1.0

    # Normalize once
    slots_norm = F.normalize(slots_seq, dim=-1).detach().cpu().numpy()  # (B, T, S, D)

    total_sim = 0.0
    n_pairs = 0

    for b in range(B):
        for t in range(T - 1):
            a = slots_norm[b, t]      # (S, D)
            b_next = slots_norm[b, t + 1]  # (S, D)
            perm = _hungarian_cosine_match(a, b_next)
            # Cosine similarity of matched pairs
            sims = (a * b_next[perm]).sum(axis=-1)  # (S,)
            total_sim += sims.mean()
            n_pairs += 1

    return float(total_sim / n_pairs) if n_pairs > 0 else 1.0


def compute_slot_collapse_rate(slots_seq: torch.Tensor, threshold: float = 0.95) -> float:
    """
    Fraction of (batch, timestep) pairs where any two slots have cosine similarity > threshold.
    High rate indicates representation collapse (slots not differentiating objects).

    slots_seq: (B, T, num_slots, slot_dim)
    """
    B, T, num_slots, slot_dim = slots_seq.shape
    slots_norm = F.normalize(slots_seq, dim=-1)  # (B, T, S, D)

    # Pairwise cosine similarity between slots at each timestep
    # (B, T, S, S)
    sim = torch.bmm(
        slots_norm.view(B * T, num_slots, slot_dim),
        slots_norm.view(B * T, num_slots, slot_dim).transpose(1, 2),
    ).view(B, T, num_slots, num_slots)

    # Mask diagonal (self-similarity = 1)
    eye = torch.eye(num_slots, device=slots_seq.device).bool()
    sim = sim.masked_fill(eye.unsqueeze(0).unsqueeze(0), 0.0)

    # Any off-diagonal pair exceeds threshold
    collapsed = (sim > threshold).any(dim=-1).any(dim=-1)  # (B, T)
    return float(collapsed.float().mean().item())


def _compute_ari_manual(pred_flat: np.ndarray, true_flat: np.ndarray) -> float:
    """Manual ARI implementation for when sklearn is unavailable."""
    pred_labels = np.unique(pred_flat)
    true_labels = np.unique(true_flat)

    # Contingency table
    contingency = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    true_to_idx = {v: i for i, v in enumerate(true_labels)}
    pred_to_idx = {v: i for i, v in enumerate(pred_labels)}
    for t, p in zip(true_flat, pred_flat):
        contingency[true_to_idx[t], pred_to_idx[p]] += 1

    # ARI formula
    n = len(pred_flat)
    sum_comb_c = np.sum([np.math.comb(int(n_ij), 2) for n_ij in contingency.flatten()])
    sum_comb_a = np.sum([np.math.comb(int(a_i), 2) for a_i in contingency.sum(axis=1)])
    sum_comb_b = np.sum([np.math.comb(int(b_j), 2) for b_j in contingency.sum(axis=0)])
    comb_n = np.math.comb(n, 2)

    expected_index = sum_comb_a * sum_comb_b / (comb_n + 1e-10)
    max_index = (sum_comb_a + sum_comb_b) / 2.0
    ari = (sum_comb_c - expected_index) / (max_index - expected_index + 1e-10)
    return float(ari)


def compute_ari(pred_masks: torch.Tensor, true_masks: torch.Tensor) -> float:
    """
    Adjusted Rand Index between slot segmentation and ground-truth.

    pred_masks: (B, H, W) int — argmax of slot attention masks
    true_masks: (B, H, W) int — ground-truth object labels
    Returns mean ARI over batch.
    """
    pred_np = pred_masks.detach().cpu().numpy()
    true_np = true_masks.detach().cpu().numpy()
    B = pred_np.shape[0]
    ari_scores = []

    try:
        from sklearn.metrics import adjusted_rand_score
        for b in range(B):
            score = adjusted_rand_score(true_np[b].flatten(), pred_np[b].flatten())
            ari_scores.append(score)
    except ImportError:
        for b in range(B):
            score = _compute_ari_manual(pred_np[b].flatten(), true_np[b].flatten())
            ari_scores.append(score)

    return float(np.mean(ari_scores))
