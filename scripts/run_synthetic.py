"""
End-to-end synthetic ablation for all 6 beta schedules.

Fixes vs. original:
  - T=6 (matches base.yaml num_frames; allows k in {2,4,6} measurements)
  - eval condition: T >= k (was T > k, causing k=4 to silently return 0.0 with T=4)
  - 3 seeds per condition; reports mean +/- std
  - 300 steps per run (was 100)
  - per-step collapse tracking to show collapse trajectory, not just endpoint
  - slot diversity auxiliary loss (cosine-similarity penalty) to combat collapse

Usage:
    python scripts/run_synthetic.py
    python scripts/run_synthetic.py --steps 300 --seeds 3
    python scripts/run_synthetic.py --steps 50 --seeds 1 --quick
"""

import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from hormetic_ib_slot.models.model import HormeticIBSlot
from hormetic_ib_slot.schedules.beta_schedules import make_schedule, SCHEDULE_REGISTRY, plot_schedules
from hormetic_ib_slot.training.losses import total_training_loss
from hormetic_ib_slot.evaluation.slot_stability import (
    compute_slot_cosine_stability, compute_slot_collapse_rate
)
from hormetic_ib_slot.utils.checkpoint import save_checkpoint, load_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────

SCHEDULES = [
    "hormetic_sigmoid",
    "hormetic_cosine",
    "linear",
    "reverse",
    "random_permutation",
    "fixed_beta",
]

MODEL_CFG = dict(
    num_slots=4,
    slot_dim=32,
    latent_dim=16,
    hidden_dim=64,
    num_iters=2,
    resolution=(32, 32),
    backbone="small",
    vib_hidden=64,
    dec_hidden=32,
)

TRAIN_CFG = dict(
    lr=4e-4,
    batch_size=2,
    T=6,           # matches base.yaml num_frames; allows k in {2, 4, 6}
    beta_max=1.0,
    beta_min=0.0,
    clip_grad_norm=1.0,
    lambda_identity=0.3,   # increased from 0.1 to provide stronger slot separation
    lambda_diversity=0.05, # NEW: slot diversity penalty (pushes slots apart)
)

OCCLUSION_K = [2, 4, 6]  # occlusion durations to measure


# ── Slot diversity loss ───────────────────────────────────────────────────────

def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    """
    Penalise high cosine similarity between distinct slots.
    slots: (B, num_slots, slot_dim)
    Returns mean off-diagonal cosine similarity (lower is better; we minimise this).
    """
    B, S, D = slots.shape
    slots_norm = F.normalize(slots, dim=-1)           # (B, S, D)
    sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # (B, S, S)
    eye = torch.eye(S, device=slots.device).unsqueeze(0)     # (1, S, S)
    off_diag = sim * (1.0 - eye)
    # Mean over off-diagonal elements
    n_pairs = S * (S - 1)
    return off_diag.sum(dim=(1, 2)).mean() / n_pairs


# ── Synthetic data ────────────────────────────────────────────────────────────

class SyntheticVideoDataset:
    """
    Random (B, T, 3, H, W) video clips with coloured blobs and occlusion events.
    """

    def __init__(self, num_videos=60, T=6, H=32, W=32, num_objects=3,
                 batch_size=2, seed=0):
        rng = np.random.default_rng(seed)
        self.T = T
        self.H = H
        self.W = W
        self.num_objects = num_objects
        self.batch_size = batch_size
        self.num_videos = num_videos

        videos, object_tracks, visibilities = [], [], []

        for _ in range(num_videos):
            pos = rng.uniform(0.15, 0.85, size=(num_objects, 2))
            vel = rng.uniform(-0.04, 0.04, size=(num_objects, 2))
            colors = rng.uniform(0.4, 1.0, size=(num_objects, 3))

            frames = np.zeros((T, 3, H, W), dtype=np.float32)
            tracks_t = np.zeros((T, num_objects, 2))
            vis_t = np.ones((T, num_objects), dtype=bool)

            # Occlude one object for a mid-clip span (leave first and last frames visible)
            occ_obj = rng.integers(0, num_objects)
            occ_start = rng.integers(1, max(2, T - 2))
            occ_len = rng.integers(1, max(2, T // 2))
            occ_end = min(occ_start + occ_len, T - 1)  # keep last frame visible

            for t in range(T):
                pos = np.clip(pos + vel, 0.05, 0.95)
                tracks_t[t] = pos.copy()
                frame = np.zeros((3, H, W), dtype=np.float32)

                for k in range(num_objects):
                    visible = not (occ_obj == k and occ_start <= t < occ_end)
                    vis_t[t, k] = visible
                    if visible:
                        cx = int(pos[k, 0] * W)
                        cy = int(pos[k, 1] * H)
                        r = max(2, min(H, W) // 7)
                        y_lo = max(0, cy - r); y_hi = min(H, cy + r)
                        x_lo = max(0, cx - r); x_hi = min(W, cx + r)
                        for c in range(3):
                            frame[c, y_lo:y_hi, x_lo:x_hi] = colors[k, c]

                frame += rng.normal(0, 0.015, frame.shape).astype(np.float32)
                frame = np.clip(frame, 0, 1)
                frames[t] = frame

            videos.append(frames)
            object_tracks.append(tracks_t)
            visibilities.append(vis_t)

        self.videos = np.stack(videos)
        self.tracks = np.stack(object_tracks)
        self.visibilities = np.stack(visibilities)

    def get_batch(self, step: int) -> dict:
        N = len(self.videos)
        idx = np.arange(step * self.batch_size % N,
                        (step * self.batch_size % N) + self.batch_size) % N
        return {
            "video": torch.from_numpy(self.videos[idx]),
            "visibility": torch.from_numpy(self.visibilities[idx]),
        }

    def __len__(self):
        return self.num_videos // self.batch_size


# ── Training loop ─────────────────────────────────────────────────────────────

def run_schedule(schedule_name, num_steps, dataset, cfg, device, seed):
    torch.manual_seed(seed)

    model = HormeticIBSlot(**MODEL_CFG).to(device)
    schedule = make_schedule(schedule_name, beta_max=cfg["beta_max"],
                             beta_min=cfg["beta_min"], seed=42)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    history = []
    model.train()

    for step in range(num_steps):
        batch = dataset.get_batch(step)
        video = batch["video"].to(device)
        B, T, C, H, W = video.shape

        beta = schedule.get_beta(step, num_steps)
        out = model(video, beta=beta)

        recon = out["recon"]
        kl_loss = out["kl_loss"]
        z_slots = out["z_slots"]
        slots_seq = out["slots"]   # (B, T, S, slot_dim)

        losses = total_training_loss(
            recon=recon.reshape(B * T, C, H, W),
            target=video.reshape(B * T, C, H, W),
            kl_loss=kl_loss,
            beta=beta,
            z_t=z_slots[:, 0],
            z_t_future=z_slots[:, -1],
            lambda_identity=cfg["lambda_identity"],
        )

        # Slot diversity loss (averaged over time)
        div_loss = slot_diversity_loss(slots_seq.reshape(B * T, slots_seq.shape[2], slots_seq.shape[3]))
        total = losses["total"] + cfg["lambda_diversity"] * div_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])
        optimizer.step()

        # Track collapse every 25 steps
        if step % 25 == 0 or step == num_steps - 1:
            with torch.no_grad():
                collapse = compute_slot_collapse_rate(slots_seq.detach())
            history.append({
                "step": step,
                "beta": beta,
                "total": total.item(),
                "recon": losses["recon"].item(),
                "kl": losses["kl"].item(),
                "collapse": collapse,
            })

    model.eval()
    eval_results = _evaluate_model(model, dataset, device)

    return history, eval_results


def _evaluate_model(model, dataset, device):
    model.eval()
    stability_scores, collapse_rates = [], []
    identity_acc = {k: [] for k in OCCLUSION_K}

    with torch.no_grad():
        for step in range(min(15, len(dataset))):
            batch = dataset.get_batch(step)
            video = batch["video"].to(device)
            B, T, C, H, W = video.shape
            vis = batch["visibility"]

            out = model(video, beta=1.0)
            slots_seq = out["slots"]
            masks_seq = out["masks"]

            stab = compute_slot_cosine_stability(slots_seq)
            collapse = compute_slot_collapse_rate(slots_seq)
            stability_scores.append(stab)
            collapse_rates.append(collapse)

            for k in OCCLUSION_K:
                # BUG FIX: was `T > k`, must be `T >= k`
                if T >= k:
                    acc = _synthetic_identity_retention(
                        masks_seq, vis, k, H, W, dataset.num_objects
                    )
                    identity_acc[k].append(acc)

    return {
        "mean_slot_stability": float(np.mean(stability_scores)),
        "mean_collapse_rate": float(np.mean(collapse_rates)),
        "identity_retention": {
            k: float(np.mean(v)) if v else None
            for k, v in identity_acc.items()
        },
        "mean_identity_retention": float(np.mean([
            np.mean(v) for v in identity_acc.values() if v
        ])),
    }


def _synthetic_identity_retention(masks_seq, vis, k_frames, H, W, num_objects):
    from scipy.optimize import linear_sum_assignment

    B, T, num_slots, _, h, w = masks_seq.shape
    masks_np = masks_seq[:, :, :, 0].detach().cpu().numpy()
    vis_np = vis.numpy()

    total_correct = 0
    total_objects = 0

    for b in range(B):
        def get_gt_masks(t):
            gt = np.zeros((num_objects, h, w), dtype=np.float32)
            quad_size = h // 2
            for obj in range(num_objects):
                if vis_np[b, t, obj]:
                    qr = (obj // 2) * quad_size
                    qc = (obj % 2) * quad_size
                    gt[obj, qr:qr + quad_size, qc:qc + quad_size] = 1.0
            return gt

        t_end = min(k_frames, T - 1)
        gt0 = get_gt_masks(0)
        gt_k = get_gt_masks(t_end)
        pred0 = masks_np[b, 0]
        pred_k = masks_np[b, t_end]

        def iou_matrix(pred, gt):
            cost = np.zeros((len(pred), len(gt)))
            for i, p in enumerate(pred):
                for j, g in enumerate(gt):
                    inter = (p * g).sum()
                    union = p.sum() + g.sum() - inter + 1e-8
                    cost[i, j] = inter / union
            return cost

        iou0 = iou_matrix(pred0, gt0)
        iou_k = iou_matrix(pred_k, gt_k)

        row0, col0 = linear_sum_assignment(-iou0)
        slot_to_obj0 = {row0[i]: col0[i] for i in range(len(row0))}
        row_k, col_k = linear_sum_assignment(-iou_k)
        slot_to_obj_k = {row_k[i]: col_k[i] for i in range(len(row_k))}

        for slot, obj0 in slot_to_obj0.items():
            if vis_np[b, 0, obj0] and vis_np[b, t_end, obj0]:
                total_correct += int(slot_to_obj_k.get(slot, -1) == obj0)
                total_objects += 1

    return total_correct / max(total_objects, 1)


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_all_seeds(num_steps, num_seeds, device):
    results_dir = ROOT / "results" / "synthetic"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Shared dataset (fixed seed so all conditions see same videos)
    dataset = SyntheticVideoDataset(
        num_videos=60,
        T=TRAIN_CFG["T"],
        H=MODEL_CFG["resolution"][0],
        W=MODEL_CFG["resolution"][1],
        num_objects=3,
        batch_size=TRAIN_CFG["batch_size"],
        seed=42,
    )

    print(f"Synthetic dataset: {len(dataset.videos)} videos, "
          f"T={dataset.T}, {dataset.num_objects} objects/video")
    print(f"Conditions: {len(SCHEDULES)}, Seeds: {num_seeds}, Steps: {num_steps}\n")

    plot_beta_trajectories(num_steps, results_dir)

    all_results = {}   # schedule_name -> list of per-seed dicts
    total_t0 = time.time()

    for schedule_name in SCHEDULES:
        print(f"\n{'='*60}")
        print(f"Schedule: {schedule_name}")
        seed_results = []

        for seed in range(num_seeds):
            t0 = time.time()
            history, eval_res = run_schedule(
                schedule_name=schedule_name,
                num_steps=num_steps,
                dataset=dataset,
                cfg=TRAIN_CFG,
                device=device,
                seed=seed,
            )
            elapsed = time.time() - t0

            ir_str = "  ".join(
                f"k={k}: {eval_res['identity_retention'].get(k, 0.0):.3f}"
                for k in OCCLUSION_K
            )
            final_collapse = history[-1]["collapse"] if history else 1.0
            print(f"  seed={seed}  {elapsed:.1f}s  "
                  f"collapse={final_collapse:.2f}  {ir_str}")

            seed_results.append({
                "seed": seed,
                "history": history,
                "eval": eval_res,
                "final_loss": history[-1]["total"] if history else None,
            })

        all_results[schedule_name] = seed_results

    total_elapsed = time.time() - total_t0
    print(f"\nTotal wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    return all_results, dataset, results_dir


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(all_results):
    """Compute mean +/- std across seeds for each schedule."""
    agg = {}
    for name, seed_results in all_results.items():
        stabilities = [r["eval"]["mean_slot_stability"] for r in seed_results]
        collapses = [r["eval"]["mean_collapse_rate"] for r in seed_results]
        ir_by_k = {k: [] for k in OCCLUSION_K}
        for r in seed_results:
            for k in OCCLUSION_K:
                v = r["eval"]["identity_retention"].get(k)
                if v is not None:
                    ir_by_k[k].append(v)

        agg[name] = {
            "stability_mean": float(np.mean(stabilities)),
            "stability_std": float(np.std(stabilities)),
            "collapse_mean": float(np.mean(collapses)),
            "collapse_std": float(np.std(collapses)),
            "ir": {
                k: {
                    "mean": float(np.mean(v)) if v else None,
                    "std": float(np.std(v)) if v else None,
                }
                for k, v in ir_by_k.items()
            },
        }
    return agg


def print_summary(agg):
    print("\n" + "=" * 90)
    print("MULTI-SEED ABLATION SUMMARY (synthetic, mean +/- std)")
    print("=" * 90)

    k_cols = "  ".join(f"{'IR@k='+str(k):>12s}" for k in OCCLUSION_K)
    print(f"{'Schedule':22s}  {'Collapse':>12s}  {'Stability':>10s}  {k_cols}")
    print("-" * 90)

    for name, a in agg.items():
        collapse_s = f"{a['collapse_mean']:.3f}+/-{a['collapse_std']:.3f}"
        stab_s = f"{a['stability_mean']:.3f}"
        ir_parts = []
        for k in OCCLUSION_K:
            m = a["ir"][k]["mean"]
            s = a["ir"][k]["std"]
            if m is not None:
                ir_parts.append(f"{m:.3f}+/-{s:.3f}")
            else:
                ir_parts.append("        N/A")
        ir_s = "  ".join(f"{p:>12s}" for p in ir_parts)
        print(f"{name:22s}  {collapse_s:>12s}  {stab_s:>10s}  {ir_s}")

    print("=" * 90)
    print("Note: IR = identity retention accuracy. Higher is better.")
    print("      Collapse fraction: fraction of frames where any two slots have cos-sim > 0.95.")


def save_results(all_results, agg, path):
    out = {
        "aggregated": agg,
        "per_seed": {
            name: [
                {
                    "seed": r["seed"],
                    "eval": r["eval"],
                    "final_loss": r["final_loss"],
                    "loss_trajectory": r["history"][::5],
                }
                for r in seed_results
            ]
            for name, seed_results in all_results.items()
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

COLORS = {
    "hormetic_sigmoid":   "#e41a1c",
    "hormetic_cosine":    "#377eb8",
    "linear":             "#4daf4a",
    "reverse":            "#ff7f00",
    "random_permutation": "#984ea3",
    "fixed_beta":         "#a65628",
}


def plot_beta_trajectories(num_steps, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plot_schedules(total_steps=num_steps,
                         save_path=str(results_dir / "beta_trajectories.png"))
    plt.close(fig)


def plot_loss_curves(all_results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = ["total", "recon", "kl"]
    titles = ["Total Loss", "Reconstruction Loss", "KL Loss (unweighted)"]

    for ax, key, title in zip(axes, keys, titles):
        for name, seed_results in all_results.items():
            # Average histories across seeds
            max_len = max(len(r["history"]) for r in seed_results)
            vals_per_seed = []
            for r in seed_results:
                h = r["history"]
                steps = [e["step"] for e in h]
                vals = [e[key] for e in h]
                vals_per_seed.append((steps, vals))

            # Use first seed for steps reference
            ref_steps = vals_per_seed[0][0]
            all_vals = np.array([v for _, v in vals_per_seed
                                 if len(v) == len(ref_steps)])
            if len(all_vals) == 0:
                continue
            mean_v = all_vals.mean(axis=0)
            std_v = all_vals.std(axis=0)
            color = COLORS.get(name, "black")
            ax.plot(ref_steps, mean_v, label=name.replace("_", " "),
                    color=color, linewidth=1.5)
            ax.fill_between(ref_steps, mean_v - std_v, mean_v + std_v,
                            alpha=0.15, color=color)

        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Loss Curves by beta Schedule (synthetic, mean +/- std)", fontsize=12)
    fig.tight_layout()
    path = results_dir / "synthetic_loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved to {path}")


def plot_collapse_trajectories(all_results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, seed_results in all_results.items():
        all_steps, all_vals = [], []
        for r in seed_results:
            steps = [e["step"] for e in r["history"]]
            vals = [e["collapse"] for e in r["history"]]
            all_steps = steps
            all_vals.append(vals)
        if not all_vals:
            continue
        min_len = min(len(v) for v in all_vals)
        all_vals = np.array([v[:min_len] for v in all_vals])
        steps = all_steps[:min_len]
        mean_v = all_vals.mean(axis=0)
        std_v = all_vals.std(axis=0)
        color = COLORS.get(name, "black")
        ax.plot(steps, mean_v, label=name.replace("_", " "),
                color=color, linewidth=1.8)
        ax.fill_between(steps, mean_v - std_v, mean_v + std_v,
                        alpha=0.15, color=color)

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Slot collapse rate", fontsize=11)
    ax.set_title("Slot Collapse Rate over Training by beta Schedule", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = results_dir / "collapse_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Collapse trajectories saved to {path}")


def plot_eval_comparison(agg, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agg.keys())
    fig, axes = plt.subplots(1, len(OCCLUSION_K) + 1, figsize=(5 * (len(OCCLUSION_K) + 1), 4))

    # IR plots
    for ax, k in zip(axes[:-1], OCCLUSION_K):
        means = [agg[n]["ir"][k]["mean"] or 0 for n in names]
        stds = [agg[n]["ir"][k]["std"] or 0 for n in names]
        colors = [COLORS.get(n, "gray") for n in names]
        bars = ax.bar(range(len(names)), means, yerr=stds, color=colors,
                      edgecolor="black", linewidth=0.5, capsize=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_title(f"Identity Retention @ k={k}", fontsize=10)
        ax.set_ylim(0, max(max(means) * 1.3, 0.15))
        ax.grid(True, axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    # Collapse rate
    ax = axes[-1]
    collapse_means = [agg[n]["collapse_mean"] for n in names]
    collapse_stds = [agg[n]["collapse_std"] for n in names]
    colors = [COLORS.get(n, "gray") for n in names]
    ax.bar(range(len(names)), collapse_means, yerr=collapse_stds, color=colors,
           edgecolor="black", linewidth=0.5, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_title("Slot Collapse Rate", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Evaluation by beta Schedule (synthetic, mean +/- std)", fontsize=11)
    fig.tight_layout()
    path = results_dir / "synthetic_eval_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Eval comparison saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="50 steps, 1 seed, 20 videos")
    args = parser.parse_args()

    if args.quick:
        args.steps = 50
        args.seeds = 1

    device = torch.device("cpu")
    print(f"=== Hormetic IB Slot — Synthetic Ablation ===")
    print(f"Steps: {args.steps}  Seeds: {args.seeds}  T={TRAIN_CFG['T']}")
    print(f"Model: {MODEL_CFG['resolution'][0]}x{MODEL_CFG['resolution'][1]}, "
          f"{MODEL_CFG['num_slots']} slots, {MODEL_CFG['latent_dim']}D latent")
    print(f"lambda_identity={TRAIN_CFG['lambda_identity']}  "
          f"lambda_diversity={TRAIN_CFG['lambda_diversity']}\n")

    all_results, dataset, results_dir = run_all_seeds(
        num_steps=args.steps,
        num_seeds=args.seeds,
        device=device,
    )

    agg = aggregate(all_results)
    print_summary(agg)
    save_results(all_results, agg, results_dir / "synthetic_results.json")
    plot_loss_curves(all_results, results_dir)
    plot_collapse_trajectories(all_results, results_dir)
    plot_eval_comparison(agg, results_dir)

    print("\nDone. All outputs in:", results_dir)


if __name__ == "__main__":
    main()
