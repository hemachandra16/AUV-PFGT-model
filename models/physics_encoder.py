from __future__ import annotations

import torch
import torch.nn as nn


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
    """Lightweight encoder that produces a physics-aware feature map from RGB input.

    The encoder transforms a raw RGB image into a 64-channel physics feature map,
    capturing degradation-related cues such as attenuation, scattering, and color
    distortion that can later guide transformer attention.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.initial_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.initial_act = nn.GELU()

        self.res_block1 = ResidualBlock(hidden_channels)
        self.res_block2 = ResidualBlock(hidden_channels)

        self.final_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image tensor into a physics feature map.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            A physics feature map of shape (B, 64, H, W).
        """
        self._validate_input(x)

        features = self.initial_conv(x)
        features = self.initial_act(features)

        features = self.res_block1(features)
        features = self.res_block2(features)

        features = self.final_conv(features)
        return features

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a torch.Tensor, got {type(x).__name__}.")
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor of shape (B, C, H, W), got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
