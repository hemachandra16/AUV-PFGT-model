from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedAttention(nn.Module):
    """Physics-guided multi-head scaled dot-product attention.

    Implements the core novelty described in ``docs/math.md`` section 5:

        Attention(Q, K, V) = Softmax(Q Kᵀ / sqrt(d) + lambda * P) V

    where ``P`` is the *projected physics feature map* and ``lambda`` (the learnable
    ``physics_scale``) controls how strongly physics guidance influences attention.

    Two things are essential for this to actually work, and both are implemented here:

    1. **Q, K and V must be distinct learned projections of the input tokens.**
       If Q = K = V (as in the previous revision of this file), then
       ``Softmax(Q Kᵀ / sqrt(d)) V`` is a row-stochastic mixing matrix applied to the
       tokens themselves. Its output is therefore always a convex combination of
       existing tokens: every output channel is bounded by the min/max of that channel
       across the sequence. Such an operator can smooth and re-weight, but it can never
       move colour or content outside the convex hull of its own input — which is
       exactly what underwater colour correction requires. That is why the enhanced
       output used to look like a near-copy of the hazy input.

    2. **``P`` must be a genuine projection of the physics feature map**, not a scalar
       collapse. The previous revision squeezed the 64-channel physics map down to one
       channel, standardised it to a vector ``t``, and formed the rank-1 outer product
       ``t tᵀ``. That carries a single scalar of physics information per token and is
       shared by every head. Here the physics map is instead projected to ``num_heads``
       channels and pooled onto the token grid, producing a per-head, per-position
       additive logit bias of shape ``(B, num_heads, 1, N)``. Position ``j`` receives its
       own physics-derived offset, and each head learns to read a different physics cue.

    The per-key formulation is both the faithful reading of "P is the projected physics
    feature map" (a feature map has one value per spatial position) and the memory-cheap
    one: a full ``(B, H, N, N)`` bias at N = 4096 tokens would cost over a gigabyte per
    branch and does not fit this project's 8 GB VRAM budget. Because the bias broadcasts
    over the query axis, ``scaled_dot_product_attention`` can use its memory-efficient
    kernels instead of materialising the N x N score matrix at all.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        physics_channels: int = 64,
        attn_dropout: float = 0.0,
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
        if physics_channels <= 0:
            raise ValueError(f"physics_channels must be positive, got {physics_channels}.")
        if not 0.0 <= attn_dropout < 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1), got {attn_dropout}.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.physics_channels = physics_channels
        self.attn_dropout = attn_dropout

        # Distinct learned projections for query, key and value. These are what allow
        # the attention output to leave the convex hull of the input tokens.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Standard multi-head output projection, mixing information across heads.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Projects the physics feature map to one bias channel per attention head.
        self.physics_projection = nn.Conv2d(physics_channels, num_heads, kernel_size=1, bias=True)

        # Learnable scalar controlling the strength of the physics bias (lambda).
        self.physics_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise projections so attention starts near-identity in scale."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        # Start the physics bias small so training is not destabilised before the
        # attention itself has learned anything useful.
        nn.init.zeros_(self.physics_projection.bias)
        nn.init.normal_(self.physics_projection.weight, std=0.02)

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
            physics_features: Physics feature tensor of shape (B, physics_channels, H, W).

        Returns:
            Attended output tensor of shape (B, N, C).
        """
        self._validate_inputs(query, key, value, physics_features)

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V and split into heads -> (B, num_heads, N, head_dim).
        q = self._split_heads(self.q_proj(query), batch_size, seq_len)
        k = self._split_heads(self.k_proj(key), batch_size, seq_len)
        v = self._split_heads(self.v_proj(value), batch_size, seq_len)

        # Physics bias P, shape (B, num_heads, 1, N), broadcast over the query axis
        # and added to the attention logits *before* softmax.
        bias = self._build_physics_bias(physics_features, batch_size, seq_len)
        attn_bias = (self.physics_scale * bias).to(q.dtype)

        # Softmax((Q Kᵀ)/sqrt(head_dim) + lambda*P) V.
        # scaled_dot_product_attention applies the 1/sqrt(head_dim) scale internally
        # and adds `attn_mask` to the logits before the softmax, which is exactly the
        # equation above.
        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        # Merge heads back -> (B, N, C) and apply the output projection.
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attended)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape (B, N, C) into (B, num_heads, N, head_dim)."""
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _build_physics_bias(
        self,
        physics_features: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Project the physics feature map into a per-head, per-position attention bias.

        The 64-channel physics map is projected to ``num_heads`` channels by a learned
        1x1 convolution, pooled onto the transformer's token grid, and standardised per
        head so the logit offsets stay in a stable range regardless of input statistics.

        Returns:
            Bias tensor of shape (B, num_heads, 1, N), broadcastable over queries.
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

        # Project to one bias map per attention head -> (B, num_heads, H, W).
        projected = self.physics_projection(physics_features)

        # Pool onto the token grid. The token grid is square whenever seq_len is a
        # perfect square (the usual case, since the model tokenises a HxW feature map);
        # otherwise fall back to a 1-D pooling that still yields seq_len positions.
        grid_dim = int(math.isqrt(seq_len))
        if grid_dim * grid_dim == seq_len:
            pooled = F.adaptive_avg_pool2d(projected, output_size=(grid_dim, grid_dim))
        else:
            pooled = F.adaptive_avg_pool2d(projected, output_size=(seq_len, 1))
        tokens = pooled.flatten(2)  # (B, num_heads, seq_len)

        # Standardise each head's bias vector so the added logits stay well-scaled.
        tokens = tokens.float()
        tokens = tokens - tokens.mean(dim=-1, keepdim=True)
        tokens = tokens / (tokens.std(dim=-1, keepdim=True) + 1e-6)

        # (B, num_heads, 1, seq_len) — broadcasts across the query dimension.
        return tokens.unsqueeze(2)

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
