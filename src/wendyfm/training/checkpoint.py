"""Atomic CPU checkpoints for exact model, optimizer, and RNG continuation."""

import os
import pickle
import random
import tempfile
from pathlib import Path

import torch
from torch import nn

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int
) -> None:
    """Atomically replace path after writing a complete checkpoint in its directory."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    destination = Path(path)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as file:
            temporary = file.name
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Restore state and return the number of completed optimizer updates."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("invalid checkpoint file") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "format_version", "model", "optimizer", "step", "torch_rng_state", "python_rng_state"
    }:
        raise ValueError("invalid checkpoint contents")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("invalid checkpoint step")
    if not isinstance(payload["model"], dict) or not isinstance(payload["optimizer"], dict):
        raise ValueError("invalid checkpoint state")
    rng = payload["torch_rng_state"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("invalid checkpoint torch RNG state")
    try:
        # Check the Python RNG payload before changing any live state.
        probe = random.Random()
        probe.setstate(payload["python_rng_state"])
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(rng)
        random.setstate(payload["python_rng_state"])
    except (TypeError, ValueError, RuntimeError, KeyError) as exc:
        raise ValueError("invalid checkpoint state") from exc
    return step
