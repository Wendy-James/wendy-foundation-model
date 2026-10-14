"""Minimal decoder-only Transformer; intentionally explicit attention for inspection."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    ffn_multiplier: int = 4

    def __post_init__(self):
        if self.vocab_size < 2 or min(self.d_model, self.n_heads, self.n_layers, self.max_seq_len) <= 0:
            raise ValueError("vocab_size >= 2 and dimensions must be positive")
        if self.d_model % self.n_heads != 0 or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer for RoPE")
        if self.ffn_multiplier < 1:
            raise ValueError("ffn_multiplier must be positive")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability, then restore original dtype.
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, T, head_dim] with even head_dim."""
    _, _, seq_len, dim = x.shape
    freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = torch.arange(seq_len, device=x.device).float()[:, None] * freq[None, :]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    pairs = x.float().reshape(*x.shape[:-1], dim // 2, 2)
    x0, x1 = pairs.unbind(-1)
    result = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return result.flatten(-2).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q), apply_rope(k)
        scores = q @ k.transpose(-2, -1) * self.head_dim ** -0.5
        future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(diagonal=1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.ffn_multiplier * cfg.d_model
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # tied embedding/output weights

    def forward(self, token_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be int64 with shape [batch, time]")
        if token_ids.shape[1] == 0 or token_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("sequence length out of configured bounds")
        if token_ids.numel() and (int(token_ids.min()) < 0 or int(token_ids.max()) >= self.cfg.vocab_size):
            raise ValueError("token id out of vocabulary range")
        if labels is not None and (labels.shape != token_ids.shape or labels.dtype != torch.long):
            raise ValueError("labels must match token_ids shape and dtype")
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), ignore_index=-100) if labels is not None else None
        return logits, loss
