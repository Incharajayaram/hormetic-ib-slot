import torch
from pathlib import Path
from typing import Tuple, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
):
    """Save model + optimizer state with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device,
) -> Tuple[int, dict]:
    """
    Load checkpoint from path.
    Returns (epoch, metrics).
    optimizer may be None if loading for inference only.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt.get('epoch', 0)
    metrics = ckpt.get('metrics', {})
    return epoch, metrics
