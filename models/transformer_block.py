from __future__ import annotations

import torch
import torch.nn as nn

from models.attention.physics_attention import PhysicsGuidedAttention


class TransformerBlock(nn.Module):
    """A lightweight Pre-LayerNorm transformer block with physics-guided attention.

    The block processes a sequence of tokens of shape (B, N, C) and injects
    physics-aware guidance via the existing PhysicsGuidedAttention module.
    It follows the standard Pre-LN residual structure:

    1. LayerNorm
    2. Physics-guided attention
    3. Residual add
    4. LayerNorm
    5. Feed-forward network
    6. Residual add
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Pre-LN normalization for the attention branch.
        self.norm1 = nn.LayerNorm(embed_dim)

        # Physics-guided attention module reused from the dedicated attention file.
        self.attn = PhysicsGuidedAttention(embed_dim=embed_dim, num_heads=num_heads)

        # Pre-LN normalization for the feed-forward branch.
        self.norm2 = nn.LayerNorm(embed_dim)

        # Lightweight MLP used in the transformer block.
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        """Apply one transformer block to a token sequence.

        Args:
            x: Input tensor of shape (B, N, C).
            physics_features: Physics feature tensor of shape (B, 64, H, W).

        Returns:
            Output tensor of shape (B, N, C).
        """
        self._validate_inputs(x, physics_features)

        # First normalization and attention branch.
        attn_input = self.norm1(x)
        attn_output = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            physics_features=physics_features,
        )
        x = x + attn_output

        # Second normalization and feed-forward branch.
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        return x

    def _validate_inputs(self, x: torch.Tensor, physics_features: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}.")
        if not isinstance(physics_features, torch.Tensor):
            raise TypeError(
                f"Expected physics_features to be a torch.Tensor, got {type(physics_features).__name__}."
            )

        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B, N, C), got {tuple(x.shape)}.")
        if physics_features.dim() != 4:
            raise ValueError(
                f"Expected physics_features to have shape (B, 64, H, W), got {tuple(physics_features.shape)}."
            )

        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected x to have embedding dimension {self.embed_dim}, got {embed_dim}."
            )
        if physics_features.shape[0] != batch_size:
            raise ValueError(
                "The batch size of x must match the batch size of physics_features. "
                f"Got {batch_size} and {physics_features.shape[0]}."
            )
        if physics_features.shape[1] != 64:
            raise ValueError(
                f"Expected physics_features to have 64 channels, got {physics_features.shape[1]}."
            )
        if seq_len <= 0:
            raise ValueError(f"The sequence length must be positive, got {seq_len}.")
