"""Pre-norm decoder-only Transformer for next-token language modeling."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CausalSelfAttention
from .config import ModelConfig
from .layers import RMSNorm, SwiGLU


class DecoderBlock(nn.Module):
    """Apply causal attention and SwiGLU, each to a normalized residual stream."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class DecoderOnlyTransformer(nn.Module):
    """Produce [batch, sequence, vocab] logits from integer token IDs.

    Position information enters through RoPE in each attention block. The final
    logit at position t predicts the token following input position t.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights reproducibly from PyTorch's current random seed."""
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        # Two residual branches per block: control their initial output scale.
        residual_scale = 1 / math.sqrt(2 * self.config.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.out_proj.weight.mul_(residual_scale)
                block.ffn.down.weight.mul_(residual_scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must have shape [batch, sequence] and integer dtype")
        batch, sequence = input_ids.shape
        if batch < 1 or not 0 < sequence <= self.config.max_seq_len:
            raise ValueError("batch must be nonempty and sequence length within max_seq_len")
        if bool(torch.any(input_ids < 0)) or bool(torch.any(input_ids >= self.config.vocab_size)):
            raise ValueError("input_ids must be within the configured vocabulary")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def next_token_loss(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mean next-token cross entropy, with exactly one shift when targets are omitted.

        With no targets, input position t predicts input position t+1 and the
        final logit is unused. Explicit targets must already be aligned with
        input positions; all logits are scored without another shift.
        """
        if target_ids is None and input_ids.ndim == 2 and input_ids.shape[1] < 2:
            raise ValueError("next-token loss requires at least two input tokens")
        logits = self(input_ids)
        if target_ids is None:
            predictions = logits[:, :-1]
            targets = input_ids[:, 1:]
        else:
            if target_ids.shape != input_ids.shape or target_ids.device != input_ids.device:
                raise ValueError("target_ids must match input_ids shape and device")
            if target_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("target_ids must have integer dtype")
            if bool(torch.any(target_ids < 0)) or bool(
                torch.any(target_ids >= self.config.vocab_size)
            ):
                raise ValueError("target_ids must be within the configured vocabulary")
            predictions = logits
            targets = target_ids
        return F.cross_entropy(
            predictions.reshape(-1, self.config.vocab_size), targets.reshape(-1).long()
        )
