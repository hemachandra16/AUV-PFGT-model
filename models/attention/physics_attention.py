from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedAttention(nn.Module):
    """Physics-guided scaled dot-product attention.

    This module follows the project specification by computing standard
    attention scores from query and key tensors and then adding a learnable
    bias derived from physics features before applying softmax. The bias is
    intended to inject degradation-aware guidance into the attention process.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # A lightweight 1x1 convolution projects the physics feature map to a
        # compact representation that can be turned into physics tokens.
        self.physics_projection = nn.Conv2d(64, 1, kernel_size=1, bias=True)

        # Learnable scalar controlling the strength of the physics bias.
        self.physics_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply physics-guided attention to the provided tensors.

        Args:
            query: Query tensor of shape (B, N, C).
            key: Key tensor of shape (B, N, C).
            value: Value tensor of shape (B, N, C).
            physics_features: Physics feature tensor of shape (B, C, H, W).

        Returns:
            Attended output tensor of shape (B, N, C).
        """
        self._validate_inputs(query, key, value, physics_features)

        batch_size, seq_len, _ = query.shape

        # Standard scaled dot-product attention logits.
        # We use the last dimension of the embeddings as the attention width.
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Convert the physics feature map into a bias tensor aligned with the
        # attention logits. The projected feature map is reshaped from (B, 1, H, W)
        # to (B, N, N) so that it can be added directly to the attention scores.
        bias = self._build_physics_bias(physics_features, batch_size, seq_len)

        # Inject the physics bias before softmax.
        guided_scores = scores + self.physics_scale * bias

        # Apply softmax over the last dimension so each query attends to the
        # corresponding keys.
        attention_weights = F.softmax(guided_scores, dim=-1)

        # Compute the attended output.
        attended = torch.matmul(attention_weights, value)
        return attended

    def _build_physics_bias(
        self,
        physics_features: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Project the physics feature map into a token-based attention bias.

        The physics feature map is projected to a compact one-channel map and
        converted into a sequence of physics tokens whose length matches the
        transformer sequence length. The bias matrix is then formed from the
        pairwise similarity between these tokens, allowing the attention logits
        to receive a structured physics prior instead of relying on arbitrary
        flattening.
        """
        if physics_features.dim() != 4:
            raise ValueError(
                f"Expected physics_features to have shape (B, C, H, W), got {tuple(physics_features.shape)}."
            )
        if physics_features.shape[1] != self.physics_projection.in_channels:
            raise ValueError(
                "Expected physics_features to have the same channel count as the "
                f"projection layer. Got {physics_features.shape[1]} channels, "
                f"expected {self.physics_projection.in_channels}."
            )
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")

        # Project the physics feature map to a compact one-channel map.
        projected = self.physics_projection(physics_features)  # (B, 1, H, W)

        # Calculate target spatial grid height/width matching seq_len
        grid_dim = int(math.isqrt(seq_len))
        if grid_dim * grid_dim == seq_len:
            pooled = F.adaptive_avg_pool2d(projected, output_size=(grid_dim, grid_dim))
            tokens = pooled.flatten(2).squeeze(1)  # (B, seq_len)
        else:
            pooled = F.adaptive_avg_pool2d(projected, output_size=(seq_len, 1))
            tokens = pooled.squeeze(1).squeeze(-1)  # (B, seq_len)

        # Standardize each sample's physics tokens so the bias remains stable.
        tokens = tokens - tokens.mean(dim=1, keepdim=True)
        tokens = tokens / (tokens.std(dim=1, keepdim=True) + 1e-6)

        # Form a physics bias matrix from pairwise token similarity.
        # Each entry measures how strongly two physics tokens align, and that
        # alignment is injected into the attention logits before softmax.
        tokens = tokens.unsqueeze(-1)  # (B, seq_len, 1)
        bias = torch.bmm(tokens, tokens.transpose(1, 2))
        return bias

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> None:
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(tensor).__name__}.")
            if tensor.dim() != 3:
                raise ValueError(f"Expected {name} to have shape (B, N, C), got {tuple(tensor.shape)}.")

        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "query, key, and value must have identical shapes. "
                f"Got query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}."
            )

        if physics_features.dim() != 4:
            raise ValueError(
                f"Expected physics_features to have shape (B, C, H, W), got {tuple(physics_features.shape)}."
            )

        if query.shape[0] != physics_features.shape[0]:
            raise ValueError(
                "The batch size of query/key/value must match the batch size of physics_features. "
                f"Got query batch={query.shape[0]} and physics batch={physics_features.shape[0]}."
            )

        if query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected the embedding dimension to be {self.embed_dim}, got {query.shape[-1]}."
            )
