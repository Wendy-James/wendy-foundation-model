"""Validated dimensions and numerical settings for a decoder-only model."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model dimensions.

    For BPETokenizer, use tokenizer.spec.config.vocab_size: reserved special
    IDs may exceed tokenizer.vocab_size when training learns fewer merges.
    """

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    intermediate_size: int
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "n_heads",
            "n_layers",
            "max_seq_len",
            "intermediate_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
