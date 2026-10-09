"""Reference multi-head causal self-attention with rotary positions."""

import math

import torch
from torch import nn

from .config import ModelConfig
from .rope import RoPE


class CausalSelfAttention(nn.Module):
    """Map [batch, sequence, d_model] to the same shape without attending forward."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RoPE(config.head_dim, config.rope_theta)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        return x.reshape(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [batch, sequence, d_model]")
        batch, sequence, _ = x.shape
        if not 0 < sequence <= self.max_seq_len:
            raise ValueError("sequence length must be between 1 and max_seq_len")

        # q, k, v: [batch, heads, sequence, head_dim]. Rotate queries and keys only.
        q = self.rope(self._split_heads(self.q_proj(x)))
        k = self.rope(self._split_heads(self.k_proj(x)))
        v = self._split_heads(self.v_proj(x))

        # scores: [batch, heads, query_position, key_position].
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        future = torch.ones(sequence, sequence, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        # Restore [batch, sequence, d_model] before the output projection.
        context = context.transpose(1, 2).contiguous().reshape(batch, sequence, self.d_model)
        return self.out_proj(context)
