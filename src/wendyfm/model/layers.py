"""Normalization and feed-forward layers without model-library wrappers."""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Scale the last dimension by its root mean square and a learned weight."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not 0 < eps < float("inf"):
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.numel():
            raise ValueError("last dimension must equal d_model")
        rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return x * rms.to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    """Compute down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, intermediate_size: int) -> None:
        super().__init__()
        for name, value in (("d_model", d_model), ("intermediate_size", intermediate_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.gate = nn.Linear(d_model, intermediate_size, bias=False)
        self.up = nn.Linear(d_model, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
