"""Atomic checkpoints for exact model, optimizer, and RNG continuation.

Format v1 contains no sampler; v2 contains sampler state. Both remain loadable.
Format v3 also fingerprints the training and model configurations and batch
size. Its scheduler is defined by the saved global step and TrainConfig.
Format v4 adds the CUDA device index and its RNG state; CPU saves remain v3.
"""

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .data import NextTokenBatchSampler
from .loop import TrainConfig, learning_rate_for_step

FORMAT_VERSION = 3
CUDA_FORMAT_VERSION = 4
_LEGACY_KEYS = {
    "format_version", "model", "optimizer", "step", "torch_rng_state", "python_rng_state"
}


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _exact_metadata(model: nn.Module, config: TrainConfig, batch_size: int) -> dict:
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not hasattr(model, "config"):
        raise TypeError("exact checkpoints require a model with a dataclass config")
    try:
        model_config = asdict(model.config)
    except TypeError as exc:
        raise TypeError("exact checkpoints require a model with a dataclass config") from exc
    return {
        "train_config_sha256": _fingerprint(asdict(config)),
        "model_config_sha256": _fingerprint(model_config),
        "batch_size": batch_size,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    sampler: NextTokenBatchSampler | None = None,
    config: TrainConfig | None = None,
    batch_size: int | None = None,
) -> None:
    """Atomically write v1, fingerprinted CPU v3, or CUDA v4."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if sampler is not None and not isinstance(sampler, NextTokenBatchSampler):
        raise TypeError("sampler must be a NextTokenBatchSampler")
    if (sampler is None) != (config is None) or (config is None) != (batch_size is None):
        raise ValueError("exact checkpoints require sampler, config, and batch_size")
    metadata = _exact_metadata(model, config, batch_size) if config is not None else None
    if config is not None and step > config.max_steps:
        raise ValueError("step exceeds config max_steps")
    if config is not None:
        expected_lr = config.learning_rate if step == 0 else learning_rate_for_step(step, config)
        if any(group["lr"] != expected_lr for group in optimizer.param_groups):
            raise ValueError("optimizer learning rate does not match scheduler step")
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise ValueError("model parameters must share one device")
    device = devices.pop()
    if device.type == "cuda" and metadata is None:
        raise ValueError("CUDA checkpoints require sampler, config, and batch_size")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("unsupported checkpoint device")
    destination = Path(path)
    payload = {
        "format_version": (
            CUDA_FORMAT_VERSION if device.type == "cuda" else FORMAT_VERSION if metadata else 1
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if sampler is not None:
        payload["sampler"] = sampler.state_dict()
    if metadata is not None:
        payload["exact_metadata"] = metadata
    if device.type == "cuda":
        payload["cuda_device_index"] = device.index
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
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


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    sampler: NextTokenBatchSampler | None = None,
    config: TrainConfig | None = None,
    batch_size: int | None = None,
) -> int:
    """Restore completed updates and RNG states; validate exact fingerprints."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid checkpoint file") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid checkpoint contents")
    version = payload.get("format_version")
    if type(version) is not int or version not in (1, 2, FORMAT_VERSION, CUDA_FORMAT_VERSION):
        raise ValueError("unsupported checkpoint format version")
    required = _LEGACY_KEYS | ({"sampler"} if version >= 2 else set())
    if version >= FORMAT_VERSION:
        required |= {"exact_metadata"}
    if version == CUDA_FORMAT_VERSION:
        required |= {"cuda_device_index", "cuda_rng_state"}
    if set(payload) != required:
        raise ValueError("invalid checkpoint contents")
    if (version == 1 and sampler is not None) or (version >= 2 and sampler is None):
        raise ValueError("checkpoint sampler state does not match restore request")
    if sampler is not None and not isinstance(sampler, NextTokenBatchSampler):
        raise TypeError("sampler must be a NextTokenBatchSampler")
    if version >= FORMAT_VERSION:
        if config is None or batch_size is None:
            raise ValueError("checkpoint requires config and batch_size")
        expected = _exact_metadata(model, config, batch_size)
        saved = payload["exact_metadata"]
        if (
            not isinstance(saved, dict)
            or set(saved) != set(expected)
            or any(type(saved[key]) is not type(value) or saved[key] != value
                   for key, value in expected.items())
        ):
            raise ValueError("incompatible checkpoint config or batch_size fingerprint")
    elif config is not None or batch_size is not None:
        raise ValueError("checkpoint has no config fingerprint")
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("invalid checkpoint step")
    if config is not None and step > config.max_steps:
        raise ValueError("checkpoint step exceeds config max_steps")
    model_devices = {parameter.device for parameter in model.parameters()}
    if len(model_devices) != 1:
        raise ValueError("model parameters must share one device")
    device = model_devices.pop()
    if version == CUDA_FORMAT_VERSION:
        if (
            device.type != "cuda"
            or type(payload["cuda_device_index"]) is not int
            or payload["cuda_device_index"] != device.index
        ):
            raise ValueError("checkpoint CUDA device does not match model")
        cuda_rng = payload["cuda_rng_state"]
        if (
            not isinstance(cuda_rng, torch.Tensor)
            or cuda_rng.device.type != "cpu"
            or cuda_rng.dtype != torch.uint8
            or cuda_rng.ndim != 1
        ):
            raise ValueError("invalid checkpoint CUDA RNG state")
    elif device.type == "cuda":
        raise ValueError("checkpoint has no CUDA RNG state")
    if not isinstance(payload["model"], dict) or not isinstance(payload["optimizer"], dict):
        raise ValueError("invalid checkpoint state")
    rng = payload["torch_rng_state"]
    if (
        not isinstance(rng, torch.Tensor)
        or rng.device.type != "cpu"
        or rng.dtype != torch.uint8
        or rng.ndim != 1
    ):
        raise ValueError("invalid checkpoint torch RNG state")
    try:
        # Check the Python RNG payload before changing any live state.
        probe = random.Random()
        probe.setstate(payload["python_rng_state"])
        torch.Generator(device="cpu").set_state(rng)
        if config is not None:
            expected_lr = (
                config.learning_rate if step == 0 else learning_rate_for_step(step, config)
            )
            groups = payload["optimizer"]["param_groups"]
            if not groups or any(group["lr"] != expected_lr for group in groups):
                raise ValueError("checkpoint scheduler state does not match step")
        if sampler is not None:
            sampler.load_state_dict(payload["sampler"])
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(rng)
        random.setstate(payload["python_rng_state"])
        if version == CUDA_FORMAT_VERSION:
            torch.cuda.set_rng_state(cuda_rng, device=device)
    except Exception as exc:
        raise ValueError(f"invalid checkpoint state: {exc}") from exc
    return step
