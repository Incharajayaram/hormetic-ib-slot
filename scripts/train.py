"""
Main training entry point for Hormetic IB Slot experiments.

Usage:
    python scripts/train.py --config configs/base.yaml \
                            --experiment configs/experiments/hormetic_sigmoid.yaml \
                            --seed 0 \
                            --device cuda:0

    # Override specific values:
    python scripts/train.py --config configs/base.yaml \
                            --experiment configs/experiments/fixed_beta.yaml \
                            --schedule.beta_max 0.5 \
                            --seed 1

    # Quick debug run:
    python scripts/train.py --config configs/base.yaml \
                            --experiment configs/experiments/hormetic_sigmoid.yaml \
                            --training.num_epochs 2 \
                            --data.max_videos 50 \
                            --debug
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hormetic_ib_slot.models.model import HormeticIBSlot
from hormetic_ib_slot.schedules.beta_schedules import make_schedule
from hormetic_ib_slot.training.trainer import Trainer
from hormetic_ib_slot.data.clevrer import get_clevrer_loader
from hormetic_ib_slot.utils.logging import setup_experiment_dir


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def apply_cli_overrides(config: dict, overrides: list) -> dict:
    """Apply dot-notation CLI overrides like --schedule.beta_max 0.5"""
    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.lstrip("-").split(".")
        # Parse value type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.lower() == "null":
                    value = None
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Hormetic IB Slot model")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment-specific config to merge over base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true",
                        help="Quick debug run: 2 epochs, 20 videos")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args, unknown = parser.parse_known_args()

    # Load and merge configs
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.experiment) as f:
        exp_config = yaml.safe_load(f)
    config = deep_merge(config, exp_config)

    # Apply CLI overrides (--key.subkey=value format)
    if unknown:
        config = apply_cli_overrides(config, unknown)

    if args.debug:
        config["training"]["num_epochs"] = 2
        config["data"]["max_videos"] = 20
        config["training"]["log_interval"] = 5
        config["training"]["save_interval"] = 1

    set_seed(args.seed)
    device = torch.device(args.device)

    # Setup experiment directory
    schedule_name = config["schedule"]["name"]
    exp_dir = setup_experiment_dir(config["paths"]["results_dir"], schedule_name, args.seed)
    print(f"Experiment dir: {exp_dir}")

    # Save merged config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Build model
    model_cfg = config["model"]
    model = HormeticIBSlot(
        num_slots=model_cfg["num_slots"],
        slot_dim=model_cfg["slot_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_iters=model_cfg["num_iters"],
        resolution=tuple(model_cfg["resolution"]),
        backbone=model_cfg["backbone"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Build schedule
    sched_cfg = config["schedule"]
    schedule = make_schedule(
        sched_cfg["name"],
        beta_min=sched_cfg.get("beta_min", 0.0),
        beta_max=sched_cfg.get("beta_max", 1.0),
        steepness=sched_cfg.get("sigmoid_steepness", 10.0),
        midpoint=sched_cfg.get("sigmoid_midpoint", 0.5),
        seed=sched_cfg.get("random_seed", 42),
    )

    # Optimizer
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["num_epochs"]
    )

    # Data loaders
    data_cfg = config["data"]
    train_loader = get_clevrer_loader(
        root_dir=config["paths"]["clevrer_root"],
        split="train",
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        num_frames=data_cfg["num_frames"],
        frame_stride=data_cfg["frame_stride"],
        resolution=tuple(model_cfg["resolution"]),
        max_videos=data_cfg.get("max_videos"),
    )
    val_loader = get_clevrer_loader(
        root_dir=config["paths"]["clevrer_root"],
        split="val",
        batch_size=config["evaluation"]["batch_size"],
        num_workers=data_cfg["num_workers"],
        num_frames=data_cfg["num_frames"],
        frame_stride=data_cfg["frame_stride"],
        resolution=tuple(model_cfg["resolution"]),
        max_videos=data_cfg.get("max_videos"),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        schedule=schedule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=train_cfg,
        device=device,
        save_dir=exp_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from: {args.resume}")

    # Train
    print(f"\nStarting training: {schedule_name} | seed={args.seed} | "
          f"epochs={train_cfg['num_epochs']} | device={device}")
    history = trainer.train(train_loader, val_loader, train_cfg["num_epochs"])

    print(f"\nTraining complete. Results in: {exp_dir}")
    return history


if __name__ == "__main__":
    main()
