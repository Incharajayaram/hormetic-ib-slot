"""
Evaluate a trained Hormetic IB Slot model for identity-retention accuracy.

Usage:
    python scripts/evaluate.py \
        --checkpoint results/hormetic_sigmoid/seed_0/checkpoint_best.pt \
        --config results/hormetic_sigmoid/seed_0/config.yaml \
        --device cuda:0 \
        --clevrer /data/clevrer \
        --adept /data/adept \
        --output results/eval_hormetic_sigmoid_seed0.json

    # Evaluate all experiments in a results dir:
    python scripts/evaluate.py --results_dir results/ --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hormetic_ib_slot.models.model import HormeticIBSlot
from hormetic_ib_slot.data.clevrer import get_clevrer_loader
from hormetic_ib_slot.data.adept import get_adept_loader
from hormetic_ib_slot.evaluation.identity_retention import evaluate_model
from hormetic_ib_slot.evaluation.slot_stability import compute_slot_cosine_stability, compute_slot_collapse_rate
from hormetic_ib_slot.utils.checkpoint import load_checkpoint


def evaluate_single(checkpoint_path: str, config_path: str, clevrer_root: str,
                    adept_root: str, device: torch.device) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

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

    epoch, train_metrics = load_checkpoint(model, None, checkpoint_path, device)
    model.eval()

    eval_cfg = config.get("evaluation", {})
    occlusion_durations = eval_cfg.get("occlusion_durations", [4, 8, 16, 32])
    batch_size = eval_cfg.get("batch_size", 16)
    data_cfg = config["data"]

    clevrer_loader = get_clevrer_loader(
        root_dir=clevrer_root,
        split="val",
        batch_size=batch_size,
        num_workers=2,
        num_frames=data_cfg["num_frames"],
        frame_stride=data_cfg["frame_stride"],
        resolution=tuple(model_cfg["resolution"]),
        max_videos=200,  # Use subset for eval speed
    )

    adept_loader = get_adept_loader(
        root_dir=adept_root,
        split="val",
        batch_size=batch_size,
        num_workers=2,
        num_frames=data_cfg["num_frames"],
        resolution=tuple(model_cfg["resolution"]),
    )

    metrics = evaluate_model(
        model=model,
        clevrer_loader=clevrer_loader,
        adept_loader=adept_loader,
        device=device,
        occlusion_duration_frames=occlusion_durations,
    )

    metrics["checkpoint"] = checkpoint_path
    metrics["schedule"] = config["schedule"]["name"]
    metrics["epoch"] = epoch
    if train_metrics:
        metrics["final_train_metrics"] = train_metrics

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Evaluate all experiments under this directory")
    parser.add_argument("--clevrer", type=str, default="/data/clevrer")
    parser.add_argument("--adept", type=str, default="/data/adept")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    all_results = []

    if args.results_dir:
        # Find all checkpoint directories
        results_dir = Path(args.results_dir)
        for exp_dir in sorted(results_dir.glob("*/seed_*")):
            ckpt = exp_dir / "checkpoint_best.pt"
            cfg = exp_dir / "config.yaml"
            if ckpt.exists() and cfg.exists():
                print(f"Evaluating: {exp_dir.name}...")
                try:
                    metrics = evaluate_single(str(ckpt), str(cfg), args.clevrer, args.adept, device)
                    all_results.append(metrics)
                    print(f"  Mean identity retention: {metrics.get('mean_accuracy', 'N/A'):.3f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
    elif args.checkpoint and args.config:
        metrics = evaluate_single(args.checkpoint, args.config, args.clevrer, args.adept, device)
        all_results.append(metrics)
        print(json.dumps(metrics, indent=2))
    else:
        parser.error("Provide either --checkpoint + --config, or --results_dir")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return all_results


if __name__ == "__main__":
    main()
