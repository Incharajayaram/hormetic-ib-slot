import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path
from typing import Optional


def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)


def visualize_slots(
    video_frame: torch.Tensor,
    recon: torch.Tensor,
    per_slot_recon: torch.Tensor,
    masks: torch.Tensor,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot:  original | reconstruction | slot_1 | slot_2 | ... | mask_1 | mask_2 | ...

    video_frame:   (3, H, W) or (H, W, 3)
    recon:         (3, H, W)
    per_slot_recon:(num_slots, 3, H, W)
    masks:         (num_slots, 1, H, W) or (num_slots, H, W)
    """
    def hwc(t):
        t = _to_np(t)
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.transpose(1, 2, 0)
        return np.clip(t, 0, 1)

    frame = hwc(video_frame)
    recon_np = hwc(recon)
    num_slots = per_slot_recon.shape[0]
    slot_recons = [hwc(per_slot_recon[s]) for s in range(num_slots)]

    if masks.dim() == 4:
        masks = masks.squeeze(1)  # (S, H, W)
    masks_np = [_to_np(masks[s]) for s in range(num_slots)]

    ncols = 2 + 2 * num_slots
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3))
    axes[0].imshow(frame)
    axes[0].set_title('Original')
    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstruction')
    for s in range(num_slots):
        axes[2 + s].imshow(slot_recons[s])
        axes[2 + s].set_title(f'Slot {s+1}')
        axes[2 + num_slots + s].imshow(masks_np[s], cmap='viridis', vmin=0, vmax=1)
        axes[2 + num_slots + s].set_title(f'Mask {s+1}')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_beta_schedule(
    schedule,
    total_steps: int = 10000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot β vs training step for a single schedule."""
    steps = np.arange(total_steps)
    betas = [schedule.get_beta(int(s), total_steps) for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, betas, linewidth=2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('β')
    ax.set_title(f'β schedule: {schedule.__class__.__name__}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_all_schedules(
    schedules: dict,
    total_steps: int = 10000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot all β schedules on a single axes.
    schedules: dict mapping name -> BetaSchedule instance
    """
    steps = np.arange(total_steps)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, sched in schedules.items():
        betas = [sched.get_beta(int(s), total_steps) for s in steps]
        ax.plot(steps, betas, linewidth=2, label=name)
    ax.set_xlabel('Training step')
    ax.set_ylabel('β')
    ax.set_title('β Schedules Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_training_curves(
    log_csv_path: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training loss curves from a logs.csv file.
    Expected columns: step, epoch, beta, total_loss, recon_loss, kl_loss, identity_loss
    """
    import csv

    rows = []
    with open(log_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if v not in ('', 'nan')})

    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    steps = [r['step'] for r in rows]
    keys_to_plot = ['total_loss', 'recon_loss', 'kl_loss', 'beta']
    n = sum(1 for k in keys_to_plot if k in rows[0])
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    ax_idx = 0
    for key in keys_to_plot:
        if key not in rows[0]:
            continue
        vals = [r[key] for r in rows]
        axes[ax_idx].plot(steps, vals, linewidth=1.5)
        axes[ax_idx].set_xlabel('Step')
        axes[ax_idx].set_ylabel(key)
        axes[ax_idx].set_title(key.replace('_', ' ').title())
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig
