"""Per-image global colour correction, conditioned on the physics context.

This module exists because of the single largest measured finding of the design review.

On the held-out 89 images, with the converged model at 24.9015 dB:

    + oracle per-image per-channel OFFSET  -> 28.1038 dB   (+3.202)
    + oracle per-image per-channel AFFINE  -> 30.3572 dB   (+5.456)

i.e. 43% of the model's remaining error energy is a single per-image per-channel constant.
Yet nothing in the network could emit one:

  * ``models/model.py`` applies ``InstanceNorm2d(affine=False)`` to the refinement head's
    input, which sets its per-image per-channel mean to exactly zero (measured GAP magnitude
    7.0e-09), and does the same to the fusion output one line after it is produced.
  * The refinement head's measured receptive field is 9x9, with no global-pooling path.
  * Every convolution in the head is ``bias=False``.
  * The physics encoder was all 3x3 convolutions, so it could not compute image-wide
    statistics either.

So the last stage able to apply a colour correction was handed a tensor with that exact
information deleted, and could only look through a 9x9 window. This module restores the
pathway: it predicts a per-image, per-channel affine (gain and offset) from the physics
context vector and applies it to the decoded image.

It is initialised to an **exact identity** (final layer zero-weight, zero-bias), so at step 0
the model computes precisely what it did before. Any change it produces is therefore learned,
and attributable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GlobalColorCorrection(nn.Module):
    """Predict a per-image per-channel affine correction from a physics context vector.

    Args:
        context_dim: Width of the physics context vector.
        hidden: Width of the predictor MLP.
        out_channels: Image channels (3 for RGB).
        max_gain_delta: Bound on how far the gain may move from 1.0.
        max_shift: Bound on the additive shift.

    The bounds keep the correction in a physically sensible range — underwater colour
    correction is a moderate per-channel rebalancing, not an arbitrary remap — and stop a
    single mispredicted image from producing a wildly saturated output early in training.
    """

    def __init__(
        self,
        context_dim: int = 64,
        hidden: int = 64,
        out_channels: int = 3,
        max_gain_delta: float = 0.5,
        max_shift: float = 0.25,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.max_gain_delta = max_gain_delta
        self.max_shift = max_shift

        self.predictor = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels * 2),
        )
        # Exact identity at initialisation.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply the predicted per-image per-channel affine.

        Args:
            image: (B, 3, H, W) decoded image, expected in [0, 1].
            context: (B, context_dim) physics context.

        Returns:
            Corrected image, clamped to [0, 1].
        """
        if image.shape[1] != self.out_channels:
            raise ValueError(
                f"Expected {self.out_channels} image channels, got {image.shape[1]}."
            )
        if context.shape[0] != image.shape[0]:
            raise ValueError(
                f"Batch mismatch: image {image.shape[0]} vs context {context.shape[0]}."
            )

        params = self.predictor(context)                       # (B, 6)
        gain_raw, shift_raw = params.chunk(2, dim=1)           # (B, 3) each

        # tanh keeps both bounded and keeps the zero-init exactly at gain=1, shift=0.
        gain = 1.0 + self.max_gain_delta * torch.tanh(gain_raw)
        shift = self.max_shift * torch.tanh(shift_raw)

        gain = gain.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        return (image * gain + shift).clamp(0.0, 1.0)

    def predicted_params(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (gain, shift) as (B, 3) tensors — for inspection and verification."""
        params = self.predictor(context)
        gain_raw, shift_raw = params.chunk(2, dim=1)
        return (1.0 + self.max_gain_delta * torch.tanh(gain_raw),
                self.max_shift * torch.tanh(shift_raw))
