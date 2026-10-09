"""Rotary position embeddings using adjacent coordinate pairs."""

import math

import torch
from torch import nn


class RoPE(nn.Module):
    """Rotate the final dimension of [batch, heads, sequence, head_dim]."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
            or head_dim % 2
        ):
            raise ValueError("head_dim must be a positive even integer")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("theta must be finite and positive")
        self.head_dim = head_dim
        self.theta = theta
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        self.register_buffer("inv_freq", theta**-exponent, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.head_dim:
            raise ValueError("x must have shape [batch, heads, sequence, head_dim]")
        seq_len = x.shape[-2]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        elif (
            positions.ndim != 1
            or positions.numel() != seq_len
            or positions.dtype not in (torch.int32, torch.int64)
            or bool(torch.any(positions < 0))
        ):
            raise ValueError("positions must be a nonnegative integer vector of sequence length")
        angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * self.inv_freq.to(
            x.device
        )
        cosine = angles.cos().to(x.dtype)[None, None, :, :]
        sine = angles.sin().to(x.dtype)[None, None, :, :]
        pairs = x.reshape(*x.shape[:-1], self.head_dim // 2, 2)
        left, right = pairs.unbind(dim=-1)
        return torch.stack(
            (left * cosine - right * sine, left * sine + right * cosine), dim=-1
        ).flatten(-2)
