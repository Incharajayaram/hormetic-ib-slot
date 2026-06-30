"""
Full HormeticIBSlot model.

Combines:
  backbone (CNN feature extractor + positional embedding)
  → TemporalSlotAttention
  → VIBHead (per-slot stochastic latent code)
  → SpatialBroadcastDecoder (per-frame reconstruction)

The only variable across experimental conditions is the β schedule fed to
VIBHead at each training step.  Architecture and parameter count are identical
across all six conditions.
"""

from typing import Optional

import torch
import torch.nn as nn

from .backbone import SmallCNN, ResNetBackbone, PositionEmbedding
from .slot_attention import TemporalSlotAttention
from .vib import VIBHead, VIBPredictionHead
from .decoder import SpatialBroadcastDecoder


class HormeticIBSlot(nn.Module):
    """
    Hormetic Information-Bottleneck Slot model.

    Args:
        num_slots:    Number of slot vectors (one per candidate object).
        slot_dim:     Dimensionality of each slot vector.
        latent_dim:   Dimensionality of the VIB latent code per slot.
        hidden_dim:   MLP/GRU hidden size inside slot attention.
        num_iters:    Number of slot attention iterations per frame.
        resolution:   (H, W) of input frames.
        backbone:     'small' (SmallCNN) or 'resnet' (ResNetBackbone).
        vib_hidden:   Hidden size for the VIB encoder MLP.
        dec_hidden:   Hidden channels for the spatial broadcast decoder.
    """

    def __init__(
        self,
        num_slots: int = 7,
        slot_dim: int = 64,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_iters: int = 3,
        resolution: tuple = (64, 64),
        backbone: str = "small",
        vib_hidden: int = 128,
        dec_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim
        self.resolution = resolution

        # --- Feature extractor ---
        if backbone == "small":
            self.backbone = SmallCNN(in_channels=3, out_channels=64)
        elif backbone == "resnet":
            self.backbone = ResNetBackbone(in_channels=3, out_channels=64)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Use 'small' or 'resnet'.")
        feature_channels = 64

        self.pos_emb = PositionEmbedding(channels=feature_channels)

        # Project CNN features to slot_dim for slot attention
        self.input_proj = nn.Linear(feature_channels, slot_dim)

        # --- Slot attention ---
        self.temporal_slots = TemporalSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_iters=num_iters,
        )

        # --- VIB head ---
        self.vib = VIBHead(
            slot_dim=slot_dim,
            latent_dim=latent_dim,
            hidden_dim=vib_hidden,
        )
        self.pred_head = VIBPredictionHead(
            latent_dim=latent_dim,
            hidden_dim=vib_hidden,
        )

        # --- Decoder ---
        self.decoder = SpatialBroadcastDecoder(
            latent_dim=latent_dim,
            output_channels=3,
            resolution=resolution,
            hidden_channels=dec_hidden,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from all frames simultaneously.

        Args:
            video: (B, T, 3, H, W)

        Returns:
            features_seq: (B, T, H*W, slot_dim)
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        feats = self.backbone(frames)                   # (B*T, 64, H, W)
        feats = self.pos_emb(feats)                     # (B*T, 64, H, W) + pos
        feats = feats.flatten(2).permute(0, 2, 1)       # (B*T, H*W, 64)
        feats = self.input_proj(feats)                  # (B*T, H*W, slot_dim)
        feats = feats.reshape(B, T, H * W, self.slot_dim)
        return feats

    def _decode_sequence(self, z_seq: torch.Tensor) -> tuple:
        """
        Decode latent slot sequences frame by frame.

        Args:
            z_seq: (B, T, num_slots, latent_dim)

        Returns:
            recon_seq:  (B, T, 3, H, W)
            masks_seq:  (B, T, num_slots, 1, H, W)
            sr_seq:     (B, T, num_slots, 3, H, W) — per-slot RGB
        """
        B, T, K, D = z_seq.shape
        recons, masks_list, sr_list = [], [], []
        for t in range(T):
            recon, masks, sr = self.decoder(z_seq[:, t])
            recons.append(recon.unsqueeze(1))
            masks_list.append(masks.unsqueeze(1))
            sr_list.append(sr.unsqueeze(1))
        recon_seq = torch.cat(recons, dim=1)
        masks_seq = torch.cat(masks_list, dim=1)
        sr_seq = torch.cat(sr_list, dim=1)
        return recon_seq, masks_seq, sr_seq

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        video: torch.Tensor,
        beta: float = 1.0,
        prev_slots: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            video:      (B, T, 3, H, W) — normalised to [-0.5, 0.5] or [0, 1]
            beta:       Current β value from the schedule (used by caller for loss).
            prev_slots: (B, num_slots, slot_dim) or None — slots from prior clip.

        Returns:
            dict with keys:
              'slots':      (B, T, num_slots, slot_dim)
              'z_slots':    (B, T, num_slots, latent_dim)
              'mu':         (B, T, num_slots, latent_dim)
              'log_var':    (B, T, num_slots, latent_dim)
              'recon':      (B, T, 3, H, W)
              'masks':      (B, T, num_slots, 1, H, W)
              'per_slot_recon': (B, T, num_slots, 3, H, W)
              'predictions':    (B, T, num_slots, latent_dim) — L2-norm projected z
              'kl_loss':    scalar tensor — sum of per-frame KL losses (mean over B, slots)
              'prev_slots': (B, num_slots, slot_dim) — last-frame slots for chaining
        """
        B, T, C, H, W = video.shape

        # 1. Extract CNN features for all frames
        features_seq = self._extract_features(video)    # (B, T, H*W, slot_dim)

        # 2. Temporal slot attention
        slots_seq, final_slots = self.temporal_slots(features_seq, prev_slots)
        # slots_seq: (B, T, num_slots, slot_dim)

        # 3. VIB head for each frame
        z_list, mu_list, lv_list, kl_total = [], [], [], 0.0
        for t in range(T):
            z_t, mu_t, lv_t, kl_t = self.vib(slots_seq[:, t])
            z_list.append(z_t.unsqueeze(1))
            mu_list.append(mu_t.unsqueeze(1))
            lv_list.append(lv_t.unsqueeze(1))
            kl_total = kl_total + kl_t

        z_seq = torch.cat(z_list, dim=1)      # (B, T, num_slots, latent_dim)
        mu_seq = torch.cat(mu_list, dim=1)
        lv_seq = torch.cat(lv_list, dim=1)
        kl_loss = kl_total / T               # average over time

        # 4. Contrastive prediction head
        pred_seq = self.pred_head(z_seq.reshape(B * T, self.num_slots, self.latent_dim))
        pred_seq = pred_seq.reshape(B, T, self.num_slots, self.latent_dim)

        # 5. Decode
        recon_seq, masks_seq, sr_seq = self._decode_sequence(z_seq)

        return {
            "slots": slots_seq,
            "z_slots": z_seq,
            "mu": mu_seq,
            "log_var": lv_seq,
            "recon": recon_seq,
            "masks": masks_seq,
            "per_slot_recon": sr_seq,
            "predictions": pred_seq,
            "kl_loss": kl_loss,
            "prev_slots": final_slots,
        }
