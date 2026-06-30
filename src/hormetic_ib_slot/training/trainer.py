import os
import csv
import time
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .losses import total_training_loss


class Trainer:
    def __init__(self, model, schedule, optimizer, config: dict, device, save_dir,
                 lr_scheduler=None):
        """
        model:        HormeticIBSlot instance
        schedule:     BetaSchedule instance
        optimizer:    torch optimizer
        config:       dict with keys: lr, weight_decay, beta_max, beta_min,
                      clip_grad_norm, save_interval, log_interval,
                      lambda_identity, T_context, T_occlude
        device:       torch.device
        save_dir:     str or Path — experiment output directory
        lr_scheduler: optional torch LR scheduler (stepped once per epoch)
        """
        self.model = model
        self.schedule = schedule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.save_dir / 'logs.csv'
        self._csv_initialized = False
        self.global_step = 0
        self.history = {
            'step': [], 'epoch': [], 'beta': [],
            'total_loss': [], 'recon_loss': [], 'kl_loss': [], 'identity_loss': [],
        }

    def _get_beta(self, total_steps: int) -> float:
        return self.schedule.get_beta(self.global_step, total_steps)

    def _write_log_row(self, row: dict):
        file_exists = self.log_path.exists()
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    def train_epoch(self, dataloader, epoch: int, total_steps: int) -> dict:
        self.model.train()
        epoch_metrics = {k: 0.0 for k in ['total_loss', 'recon_loss', 'kl_loss', 'identity_loss']}
        n_batches = 0
        lambda_identity = self.config.get('lambda_identity', 0.1)
        clip_norm = self.config.get('clip_grad_norm', 1.0)
        log_interval = self.config.get('log_interval', 100)

        for batch in tqdm(dataloader, desc=f'Epoch {epoch}', leave=False):
            video = batch['video'].to(self.device)  # (B, T, C, H, W)
            B, T, C, H, W = video.shape
            beta = self._get_beta(total_steps)

            # Use first T_context frames for context, predict future for identity loss
            T_ctx = min(self.config.get('T_context', T), T)
            video_ctx = video[:, :T_ctx]

            out = self.model(video_ctx, beta=beta)
            recon = out['recon']          # (B, T_ctx, C, H, W)
            kl_loss = out['kl_loss']
            z_slots = out['z_slots']     # (B, T_ctx, num_slots, latent_dim)

            # Identity loss: compare first and last frame slots
            z_t = z_slots[:, 0]           # (B, num_slots, latent_dim)
            z_t_future = z_slots[:, -1]   # (B, num_slots, latent_dim)

            losses = total_training_loss(
                recon=recon.reshape(B * T_ctx, C, H, W),
                target=video_ctx.reshape(B * T_ctx, C, H, W),
                kl_loss=kl_loss,
                beta=beta,
                z_t=z_t,
                z_t_future=z_t_future,
                lambda_identity=lambda_identity,
            )

            self.optimizer.zero_grad()
            losses['total'].backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.optimizer.step()

            for k in epoch_metrics:
                val = losses.get(k.replace('_loss', ''), losses.get(k, torch.tensor(0.0)))
                epoch_metrics[k] += val.item() if hasattr(val, 'item') else float(val)
            n_batches += 1

            if self.global_step % log_interval == 0:
                row = {
                    'step': self.global_step,
                    'epoch': epoch,
                    'beta': beta,
                    'total_loss': losses['total'].item(),
                    'recon_loss': losses['recon'].item(),
                    'kl_loss': losses['kl'].item(),
                    'identity_loss': losses['identity'].item()
                    if hasattr(losses['identity'], 'item') else float(losses['identity']),
                }
                self._write_log_row(row)
                for k, v in row.items():
                    if k in self.history:
                        self.history[k].append(v)

            self.global_step += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

    def validate(self, dataloader) -> dict:
        self.model.eval()
        val_metrics = {k: 0.0 for k in ['total_loss', 'recon_loss', 'kl_loss']}
        n_batches = 0
        beta = self.config.get('beta_max', 1.0)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation', leave=False):
                video = batch['video'].to(self.device)
                B, T, C, H, W = video.shape
                T_ctx = min(self.config.get('T_context', T), T)
                video_ctx = video[:, :T_ctx]

                out = self.model(video_ctx, beta=beta)
                recon = out['recon']
                kl_loss = out['kl_loss']

                losses = total_training_loss(
                    recon=recon.reshape(B * T_ctx, C, H, W),
                    target=video_ctx.reshape(B * T_ctx, C, H, W),
                    kl_loss=kl_loss,
                    beta=beta,
                )
                for k in val_metrics:
                    key = k.replace('_loss', '')
                    val = losses.get(key, losses.get(k, torch.tensor(0.0)))
                    val_metrics[k] += val.item() if hasattr(val, 'item') else float(val)
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in val_metrics.items()}

    def train(self, train_loader, val_loader, num_epochs: int) -> dict:
        total_steps = num_epochs * len(train_loader)
        save_interval = self.config.get('save_interval', 10)
        best_val_loss = float('inf')

        print(f"Starting training: {num_epochs} epochs, ~{total_steps} steps")
        print(f"Schedule: {self.schedule.__class__.__name__}, beta_max={self.config.get('beta_max', 1.0)}")
        print(f"Logs: {self.log_path}")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch, total_steps)
            val_metrics = self.validate(val_loader)

            elapsed = time.time() - t0
            current_beta = self._get_beta(total_steps)
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"β={current_beta:.4f} | "
                f"train_loss={train_metrics['total_loss']:.4f} | "
                f"val_loss={val_metrics['total_loss']:.4f} | "
                f"recon={val_metrics['recon_loss']:.4f} | "
                f"kl={val_metrics['kl_loss']:.4f} | "
                f"time={elapsed:.1f}s"
            )

            if epoch % save_interval == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, {**train_metrics, **{'val_' + k: v for k, v in val_metrics.items()}})

            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(epoch, val_metrics, name='checkpoint_best')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.save_checkpoint(num_epochs, {}, name='checkpoint_latest')
        return self.history

    def save_checkpoint(self, epoch: int, metrics: dict, name: Optional[str] = None):
        fname = f'checkpoint_epoch{epoch}.pt' if name is None else f'{name}.pt'
        path = self.save_dir / fname
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'schedule': self.schedule.__class__.__name__,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.global_step = ckpt.get('global_step', 0)
        return ckpt.get('epoch', 0), ckpt.get('metrics', {})
