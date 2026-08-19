from __future__ import annotations

import torch
import torch.nn as nn


class ResidualRefinementBlock(nn.Module):
    """A lightweight residual block used in the refinement head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return residual + out


class ImageRefinementHead(nn.Module):
    """Convert reconstructed feature maps into an enhanced RGB image.

    The head progressively reduces the channel dimension while preserving the
    spatial resolution, then applies a sigmoid to constrain the output to the
    expected [0, 1] image range.
    """

    def __init__(self, in_channels: int = 128) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")

        self.in_channels = in_channels

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.residual_block = ResidualRefinementBlock(64)

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a reconstructed feature map into an RGB image.

        Args:
            x: Input tensor of shape (B, 128, H, W).

        Returns:
            Output tensor of shape (B, 3, H, W).
        """
        self._validate_input(x)

        features = self.stage1(x)
        features = self.residual_block(features)
        features = self.stage2(features)
        output = self.final_conv(features)
        return output

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}.")
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor of shape (B, C, H, W), got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
