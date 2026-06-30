"""
Spatial Broadcast Decoder (Watters et al., 2019).

Each slot's latent vector is tiled over the output spatial grid, positional
coordinates are concatenated, and a small CNN produces per-slot RGB + alpha.
Alpha softmax across slots gives an alpha-composite reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBroadcastDecoder(nn.Module):
    """
    Decodes a set of slot latent codes into an image via spatial broadcast.

    Args:
        latent_dim:       Dimensionality of each slot's latent code.
        output_channels:  Number of output image channels (3 for RGB).
        resolution:       (H, W) of the output image.
        hidden_channels:  Number of channels in the internal CNN.
    """

    def __init__(
        self,
        latent_dim: int,
        output_channels: int = 3,
        resolution: tuple = (64, 64),
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.H, self.W = resolution
        self.hidden_channels = hidden_channels

        # Input to decoder CNN: latent_dim + 2 (x,y position channels)
        in_c = latent_dim + 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_c, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels + 1, 3, padding=1),
            # Last channel is alpha (unnormalised logit)
        )

        # Fixed 2D positional grid, registered as a buffer so it moves with
        # the module when .to(device) is called.
        self._register_pos_grid()

    def _register_pos_grid(self) -> None:
        y = torch.linspace(-1, 1, self.H)
        x = torch.linspace(-1, 1, self.W)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")  # (H, W)
        # (1, 2, H, W) — broadcastable over batch and slots
        pos_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        self.register_buffer("pos_grid", pos_grid)

    def forward(
        self, z_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_slots: (B, num_slots, latent_dim)

        Returns:
            recon:          (B, C, H, W)           — alpha-composited image
            masks:          (B, num_slots, 1, H, W) — per-slot soft masks
            per_slot_recon: (B, num_slots, C, H, W) — per-slot RGB before mixing
        """
        B, num_slots, D = z_slots.shape
        H, W = self.H, self.W

        # --- Spatial broadcast ---
        # Tile z over (H, W): (B*num_slots, D, H, W)
        z_flat = z_slots.reshape(B * num_slots, D)
        z_tiled = z_flat[:, :, None, None].expand(-1, -1, H, W)

        # Tile positional grid over (B*num_slots): (B*num_slots, 2, H, W)
        pos = self.pos_grid.expand(B * num_slots, -1, -1, -1)

        # Concatenate latent + position
        inp = torch.cat([z_tiled, pos], dim=1)   # (B*num_slots, D+2, H, W)

        # --- CNN decode ---
        out = self.cnn(inp)                       # (B*num_slots, C+1, H, W)
        slot_rgb = out[:, :self.output_channels]  # (B*num_slots, C, H, W)
        slot_alpha_logit = out[:, self.output_channels:]  # (B*num_slots, 1, H, W)

        # Reshape back to (B, num_slots, ...)
        slot_rgb = slot_rgb.reshape(B, num_slots, self.output_channels, H, W)
        slot_alpha_logit = slot_alpha_logit.reshape(B, num_slots, 1, H, W)

        # Softmax alpha across slots dimension
        masks = torch.softmax(slot_alpha_logit, dim=1)   # (B, num_slots, 1, H, W)

        # Alpha-composite reconstruction
        recon = (masks * slot_rgb).sum(dim=1)            # (B, C, H, W)

        return recon, masks, slot_rgb
