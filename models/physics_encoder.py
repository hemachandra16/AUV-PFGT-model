from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_physics_priors(x: torch.Tensor, window: int = 15) -> torch.Tensor:
    """Closed-form underwater degradation priors, computed in closed form from RGB.

    The previous encoder was a plain ``Conv3x3 -> GELU -> 2x ResBlock -> Conv3x3`` stack whose
    only input was the same RGB the wavelet branch already sees. It was named "physics
    encoder" but contained no physics: it could only re-derive what the rest of the network
    could compute itself, and — because every layer is a 3x3 convolution with no global
    pooling — it could not represent image-wide per-channel statistics *at any depth*.

    That matters because those statistics are what predict the answer. Regressing the eight
    priors below against the per-channel gain the UIEB reference actually applies gives
    R^2 = 0.24 / 0.49 / 0.62 for R / G / B (``tools/_physicsprobe.py``).

    Channels returned (8):
      0  dark channel prior      — min over colour channels, local min-pooled. Underwater the
                                   dark channel is dominated by backscatter, so it tracks
                                   transmission/depth.
      1  bright channel prior    — max over channels, local max-pooled. Tracks the veiling
                                   light / background illuminant.
      2  local contrast (std)    — scattering suppresses local contrast, so this is a proxy
                                   for how much haze sits in front of a region.
      3-5 per-channel image mean — broadcast. The direct colour-cast measurement. Red is
                                   attenuated fastest with depth, blue/green least.
      6  R/G ratio               — broadcast. Depth/attenuation cue, invariant to exposure.
      7  R/B ratio               — broadcast.

    Channels 3-7 are image-global by construction and are exactly what a stack of 3x3 convs
    cannot produce.
    """
    pad = window // 2
    min_c = x.min(dim=1, keepdim=True).values
    max_c = x.max(dim=1, keepdim=True).values

    dark = -F.max_pool2d(-min_c, kernel_size=window, stride=1, padding=pad)
    bright = F.max_pool2d(max_c, kernel_size=window, stride=1, padding=pad)

    mean_local = F.avg_pool2d(x, kernel_size=window, stride=1, padding=pad)
    sq_local = F.avg_pool2d(x * x, kernel_size=window, stride=1, padding=pad)
    local_std = (sq_local - mean_local * mean_local).clamp_min(0).sqrt().mean(dim=1, keepdim=True)

    mu = x.mean(dim=(2, 3), keepdim=True)                      # (B, 3, 1, 1)
    ones = torch.ones_like(dark)
    mu_maps = mu * ones                                        # broadcast to (B, 3, H, W)
    ratio_rg = (mu[:, 0:1] / (mu[:, 1:2] + 1e-6)).clamp(0, 4) * ones
    ratio_rb = (mu[:, 0:1] / (mu[:, 2:3] + 1e-6)).clamp(0, 4) * ones

    return torch.cat([dark, bright, local_std, mu_maps, ratio_rg, ratio_rb], dim=1)


NUM_PRIORS = 8


class ResidualBlock(nn.Module):
    """A lightweight two-convolution residual block with GELU activation."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        return residual + out


class PhysicsPriorEncoder(nn.Module):
    """Encoder producing a physics feature map AND an image-global physics context vector.

    Two things changed relative to the original, both for measured reasons:

    1. **It is now fed actual physics.** ``compute_physics_priors`` supplies eight closed-form
       degradation cues alongside the RGB input (see that function for why each one).

    2. **It can now see the whole image.** A global-average-pool branch produces a context
       vector which is (a) used to modulate the feature map, SE-style, and (b) returned so the
       decoder can apply a per-image colour correction. This is the fix for the dominant
       error mode: measured on the held-out set, an oracle per-image per-channel offset is
       worth **+3.20 dB** and a per-image affine **+5.46 dB**, but no pathway existed to emit
       one — the refinement head has a 9x9 receptive field, no pooling, and its input is
       InstanceNorm'd to exactly zero per-image per-channel mean.

    ``forward`` returns ``(features, context)``. ``features`` keeps its original
    ``(B, hidden_channels, H, W)`` shape so the attention and fusion modules are unchanged.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64,
                 context_dim: int = 64, use_priors: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.use_priors = use_priors

        stem_in = in_channels + (NUM_PRIORS if use_priors else 0)
        self.initial_conv = nn.Conv2d(stem_in, hidden_channels, kernel_size=3, padding=1)
        self.initial_act = nn.GELU()

        self.res_block1 = ResidualBlock(hidden_channels)
        self.res_block2 = ResidualBlock(hidden_channels)

        self.final_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # Global context: pooled features + the raw global priors, which stay exact rather
        # than being smeared by convolution.
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_channels + NUM_PRIORS, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        # SE-style channel modulation so the global signal actually reaches the feature map.
        self.se = nn.Sequential(
            nn.Linear(context_dim, hidden_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image into (physics feature map, global physics context).

        Args:
            x: Input tensor of shape (B, 3, H, W), values in [0, 1].

        Returns:
            features: (B, hidden_channels, H, W)
            context:  (B, context_dim) — image-global degradation descriptor.
        """
        self._validate_input(x)

        if self.use_priors:
            priors = compute_physics_priors(x)
            stem_in = torch.cat([x, priors], dim=1)
            prior_global = priors.mean(dim=(2, 3))                  # (B, NUM_PRIORS)
        else:
            stem_in = x
            prior_global = x.new_zeros(x.shape[0], NUM_PRIORS)

        features = self.initial_conv(stem_in)
        features = self.initial_act(features)
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.final_conv(features)

        pooled = features.mean(dim=(2, 3))                          # (B, hidden_channels)
        context = self.context_mlp(torch.cat([pooled, prior_global], dim=1))

        # Re-inject the global signal into the spatial map.
        gate = self.se(context).unsqueeze(-1).unsqueeze(-1)         # (B, C, 1, 1)
        features = features * gate

        return features, context

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a torch.Tensor, got {type(x).__name__}.")
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor of shape (B, C, H, W), got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
