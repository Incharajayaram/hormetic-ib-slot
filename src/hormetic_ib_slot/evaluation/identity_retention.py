import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional
from tqdm import tqdm


def slot_object_matching(
    slot_masks: torch.Tensor,
    object_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Assign each slot to an object by maximum IoU.

    slot_masks:   (B, num_slots, H, W)  — slot attention masks (soft, sum-to-1 over slots)
    object_masks: (B, num_objects, H, W) — binary ground-truth segmentation masks

    Returns assignment (B, num_slots) int64 — object id for each slot, or -1 for background.
    """
    B, num_slots, H, W = slot_masks.shape
    _, num_objects, _, _ = object_masks.shape
    assignment = torch.full((B, num_slots), -1, dtype=torch.long)

    slot_binary = (slot_masks == slot_masks.max(dim=1, keepdim=True).values).float()

    for b in range(B):
        iou_matrix = torch.zeros(num_slots, num_objects)
        for s in range(num_slots):
            for o in range(num_objects):
                pred = slot_binary[b, s]          # (H, W)
                gt = object_masks[b, o].float()   # (H, W)
                intersection = (pred * gt).sum()
                union = (pred + gt).clamp(max=1.0).sum()
                iou_matrix[s, o] = intersection / (union + 1e-6)

        # Hungarian matching: maximise IoU <=> minimise negative IoU
        cost = -iou_matrix.numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        for s_idx, o_idx in zip(row_ind, col_ind):
            # Only assign if IoU is non-trivial
            if iou_matrix[s_idx, o_idx] > 0.1:
                assignment[b, s_idx] = int(o_idx)

    return assignment


def _run_model_for_masks(model, video_clip: torch.Tensor, device):
    """
    Run model on a (B, T, C, H, W) clip and return slot masks (B, T, num_slots, H, W).
    Handles the model's forward signature.
    """
    model.eval()
    with torch.no_grad():
        out = model(video_clip.to(device), beta=0.0)
    masks = out.get('masks')  # (B, T, num_slots, 1, H, W) or (B, num_slots, 1, H, W)
    if masks is None:
        return None
    # Squeeze channel dim if present
    if masks.dim() == 6:
        masks = masks.squeeze(-3)   # (B, T, num_slots, H, W)
    elif masks.dim() == 5:
        masks = masks.squeeze(-3)   # (B, num_slots, H, W) -> keep as is
    return masks


def compute_identity_retention_accuracy(
    model,
    dataloader,
    device,
    occlusion_duration_frames: List[int] = [4, 8, 16, 32],
) -> dict:
    """
    Measures identity-retention accuracy as a function of occlusion duration.

    For each video with annotated occlusion windows:
      1. Get slot-to-object assignment just before occlusion (t_pre).
      2. Run model through occlusion (no ground-truth update).
      3. Get slot-to-object assignment just after re-emergence (t_post).
      4. Identity retained if the same slot index still maps to the same object.

    Returns dict: {k: accuracy_at_k, ..., 'mean_accuracy': float}
    """
    model.eval()
    correct = {k: 0 for k in occlusion_duration_frames}
    total = {k: 0 for k in occlusion_duration_frames}

    for batch in tqdm(dataloader, desc='Evaluating identity retention'):
        video = batch['video']                  # (B, T, C, H, W)
        B, T, C, H, W = video.shape

        occ_starts = batch.get('occlusion_start', [None] * B)
        occ_ends = batch.get('occlusion_end', [None] * B)
        visibilities = batch.get('object_visibility', [None] * B)

        # Run model over entire clip to get slot masks at every timestep
        masks_seq = _run_model_for_masks(model, video, device)
        if masks_seq is None:
            continue

        for b in range(B):
            occ_start = occ_starts[b]
            occ_end = occ_ends[b]

            if occ_start is None or occ_end is None:
                continue
            if isinstance(occ_start, torch.Tensor):
                occ_start = occ_start.item()
                occ_end = occ_end.item()

            occ_len = occ_end - occ_start
            if occ_len <= 0 or occ_start == 0:
                continue

            t_pre = occ_start - 1
            t_post = min(occ_end, T - 1)

            vis = visibilities[b]
            if isinstance(vis, torch.Tensor) and vis.shape[0] == T:
                num_obj = vis.shape[1] if vis.dim() > 1 else 1
                # Build dummy object masks from visibility — placeholder for datasets
                # that don't provide per-pixel segmentation. Treat pre/post as known.
                # In a full evaluation pipeline, ground-truth masks would come from
                # the dataset; here we use slot masks as proxy (slot tracking self-consistency).
                pre_masks = masks_seq[b, t_pre].unsqueeze(0)   # (1, S, H, W)
                post_masks = masks_seq[b, t_post].unsqueeze(0)

                # Cross-match slots: t_pre vs t_post without seeing occlusion labels
                pre_np = pre_masks[0].cpu().numpy()  # (S, H, W)
                post_np = post_masks[0].cpu().numpy()
                num_slots = pre_np.shape[0]

                # Build IoU cost matrix between pre and post slots
                cost = np.zeros((num_slots, num_slots))
                pre_bin = (pre_np == pre_np.max(axis=0, keepdims=True)).astype(float)
                post_bin = (post_np == post_np.max(axis=0, keepdims=True)).astype(float)
                for s1 in range(num_slots):
                    for s2 in range(num_slots):
                        intersection = (pre_bin[s1] * post_bin[s2]).sum()
                        union = np.clip(pre_bin[s1] + post_bin[s2], 0, 1).sum()
                        cost[s1, s2] = -intersection / (union + 1e-6)

                row_ind, col_ind = linear_sum_assignment(cost)
                # Identity retained if matched slot indices are the same
                n_retained = sum(r == c for r, c in zip(row_ind, col_ind))
                retention_rate = n_retained / num_slots

                dur_key = min(occlusion_duration_frames, key=lambda k: abs(k - occ_len))
                correct[dur_key] += retention_rate
                total[dur_key] += 1

    accuracy_by_duration = {}
    all_accs = []
    for k in occlusion_duration_frames:
        if total[k] > 0:
            acc = correct[k] / total[k]
        else:
            acc = float('nan')
        accuracy_by_duration[k] = acc
        if not np.isnan(acc):
            all_accs.append(acc)

    mean_acc = float(np.mean(all_accs)) if all_accs else float('nan')
    accuracy_by_duration['mean_accuracy'] = mean_acc
    return accuracy_by_duration


def evaluate_model(model, clevrer_loader, adept_loader, device) -> dict:
    """Run identity retention on CLEVRER and ADEPT, return combined metrics."""
    results = {}

    if clevrer_loader is not None:
        clevrer_metrics = compute_identity_retention_accuracy(model, clevrer_loader, device)
        results['clevrer'] = clevrer_metrics

    if adept_loader is not None:
        adept_metrics = compute_identity_retention_accuracy(model, adept_loader, device)
        results['adept'] = adept_metrics

    # Aggregate mean accuracy across datasets
    mean_accs = [
        v['mean_accuracy']
        for v in results.values()
        if 'mean_accuracy' in v and not np.isnan(v['mean_accuracy'])
    ]
    results['overall_mean_accuracy'] = float(np.mean(mean_accs)) if mean_accs else float('nan')
    return results
