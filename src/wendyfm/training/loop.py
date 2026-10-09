"""Minimal CPU pretraining with explicit, already-aligned next-token targets."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from wendyfm.model import DecoderOnlyTransformer

BatchProvider = Callable[[], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class TrainConfig:
    """Schedule parameters refer to global, completed optimizer steps."""

    max_steps: int
    learning_rate: float = 3e-4
    min_learning_rate: float = 0.0
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or not 0 <= self.warmup_steps <= self.max_steps
        ):
            raise ValueError("warmup_steps must be between zero and max_steps")
        for name in ("learning_rate", "min_learning_rate", "weight_decay", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")


@dataclass(frozen=True)
class TrainResult:
    """Losses are measured before the first and last updates of this call."""

    step: int
    initial_loss: float
    final_loss: float
    mean_step_seconds: float
    tokens_per_second: float
    optimizer: torch.optim.AdamW


def learning_rate_for_step(step: int, config: TrainConfig) -> float:
    """LR for one-based update `step`; cosine decays after warmup when possible."""
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= config.max_steps:
        raise ValueError("step must be in [1, max_steps]")
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_steps = config.max_steps - config.warmup_steps
    if decay_steps == 1:
        return config.learning_rate
    progress = (step - config.warmup_steps - 1) / (decay_steps - 1)
    return config.min_learning_rate + 0.5 * (
        config.learning_rate - config.min_learning_rate
    ) * (1 + math.cos(math.pi * progress))


def train(
    model: DecoderOnlyTransformer,
    batch_provider: BatchProvider,
    config: TrainConfig,
    *,
    optimizer: torch.optim.AdamW | None = None,
    start_step: int = 0,
    end_step: int | None = None,
) -> TrainResult:
    """Train through end_step (default max_steps), using global schedule steps.

    The provider returns CPU long tensors [B, T]. Targets are shifted by the
    provider exactly once and are passed unchanged to model.next_token_loss.
    Save `result.optimizer` and `result.step` in a checkpoint to resume.
    """
    if (
        isinstance(start_step, bool)
        or not isinstance(start_step, int)
        or not 0 <= start_step < config.max_steps
    ):
        raise ValueError("start_step must be in [0, max_steps)")
    if end_step is None:
        end_step = config.max_steps
    if (
        isinstance(end_step, bool)
        or not isinstance(end_step, int)
        or not start_step < end_step <= config.max_steps
    ):
        raise ValueError("end_step must be in (start_step, max_steps]")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("train requires a CPU model")
    if optimizer is None:
        if start_step:
            raise ValueError("resuming requires the restored optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("optimizer must be AdamW")
    model.train()
    initial_loss = 0.0
    final_loss = 0.0
    tokens = 0
    started = time.perf_counter()
    for step in range(start_step + 1, end_step + 1):
        input_ids, target_ids = batch_provider()
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(target_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape != target_ids.shape
            or not all(input_ids.shape)
            or input_ids.dtype != torch.long
            or target_ids.dtype != torch.long
            or input_ids.device.type != "cpu"
            or target_ids.device.type != "cpu"
        ):
            raise ValueError("batch provider must return matching nonempty CPU long [B, T] tensors")
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_for_step(step, config)
        optimizer.zero_grad(set_to_none=True)
        loss = model.next_token_loss(input_ids, target_ids)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"nonfinite gradient at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if not bool(torch.isfinite(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"nonfinite gradient norm at step {step}")
        optimizer.step()
        final_loss = float(loss.detach())
        if step == start_step + 1:
            initial_loss = final_loss
        tokens += target_ids.numel()
    elapsed = time.perf_counter() - started
    completed = end_step - start_step
    return TrainResult(
        step=end_step,
        initial_loss=initial_loss,
        final_loss=final_loss,
        mean_step_seconds=elapsed / completed,
        tokens_per_second=tokens / elapsed,
        optimizer=optimizer,
    )
