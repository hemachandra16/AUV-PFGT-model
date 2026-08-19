from __future__ import annotations

import torch
import torch.nn as nn


class ResidualFusionBlock(nn.Module):
    """A lightweight residual block used after channel projection."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        return residual + out


class FeatureFusion(nn.Module):
    """Fuse low-frequency, high-frequency, and physics feature maps.

    The module expects three feature tensors with identical spatial dimensions and
    128 channels each. These tensors are concatenated, projected back to 128
    channels via a 1x1 convolution, refined with a residual block, and then
    processed by a 3x3 convolution to preserve spatial detail.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")

        self.channels = channels
        self.projection = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.residual_block = ResidualFusionBlock(channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(self, low_features: torch.Tensor, high_features: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        """Fuse three feature maps into a single 128-channel representation.

        Args:
            low_features: Tensor of shape (B, 128, H, W).
            high_features: Tensor of shape (B, 128, H, W).
            physics_features: Tensor of shape (B, 128, H, W).

        Returns:
            Fused feature tensor of shape (B, 128, H, W).
        """
        self._validate_inputs(low_features, high_features, physics_features)

        fused = torch.cat([low_features, high_features, physics_features], dim=1)
        fused = self.projection(fused)
        fused = self.residual_block(fused)
        fused = self.final_conv(fused)
        return fused

    def _validate_inputs(
        self,
        low_features: torch.Tensor,
        high_features: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> None:
        for name, tensor in (
            ("low_features", low_features),
            ("high_features", high_features),
            ("physics_features", physics_features),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(tensor).__name__}.")
            if tensor.dim() != 4:
                raise ValueError(f"Expected {name} to have shape (B, C, H, W), got {tuple(tensor.shape)}.")
            if tensor.shape[1] != self.channels:
                raise ValueError(
                    f"Expected {name} to have {self.channels} channels, got {tensor.shape[1]}."
                )

        if low_features.shape != high_features.shape or low_features.shape != physics_features.shape:
            raise ValueError(
                "All input feature maps must share the same shape. "
                f"Got low_features={tuple(low_features.shape)}, high_features={tuple(high_features.shape)}, "
                f"physics_features={tuple(physics_features.shape)}."
            )
