"""Bounded FP32 single-device pretraining entry point for a GPU pilot."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.checkpoint import load_checkpoint, save_checkpoint
from wendyfm.training.data import NextTokenBatchSampler, prepare_token_ids
from wendyfm.training.loop import TrainConfig, train


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _documents(path: Path) -> list[str]:
    documents = path.read_text(encoding="utf-8").splitlines()
    if not documents or any(not document.strip() for document in documents):
        raise ValueError(f"expected nonempty, one-document-per-line text: {path}")
    return documents


@torch.no_grad()
def validation_loss(
    model: DecoderOnlyTransformer, ids: torch.Tensor, seq_len: int, batch_size: int
) -> float:
    """Score every full held-out window once, without changing parameters."""
    if ids.numel() < seq_len + 1:
        raise ValueError("validation corpus needs at least seq_len + 1 tokens")
    model.eval()
    windows = ids.unfold(0, seq_len + 1, 1)
    device = next(model.parameters()).device
    weighted_loss = 0.0
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device)
        loss = model.next_token_loss(batch[:, :-1], batch[:, 1:])
        weighted_loss += float(loss) * batch.shape[0]
    return weighted_loss / windows.shape[0]


def run(
    config_path: Path, output_dir: Path, *, device_name: str = "cuda",
    end_step: int | None = None, resume: bool = False,
) -> dict:
    """Train a bounded segment and save a factual manifest plus exact checkpoint."""
    started = time.perf_counter()
    if device_name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device_name == "cuda":
        # Set before CUDA initialization for deterministic cuBLAS operations.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    config_bytes = config_path.read_bytes()
    settings = json.loads(config_bytes)
    model_config = ModelConfig(**settings["model"])
    train_config = TrainConfig(**settings["train"])
    batch_size = settings["batch_size"]
    seed = settings["seed"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    torch.set_num_threads(settings["cpu_threads"])
    random.seed(seed)
    torch.manual_seed(seed)
    train_path = (config_path.parent / settings["train_text"]).resolve()
    validation_path = (config_path.parent / settings["validation_text"]).resolve()
    if train_path == validation_path:
        raise ValueError("training and validation paths must differ")
    train_documents = _documents(train_path)
    validation_documents = _documents(validation_path)
    if set(train_documents) & set(validation_documents):
        raise ValueError("training and validation documents must be disjoint")
    tokenizer = BPETokenizer.train(
        train_documents, vocab_size=model_config.vocab_size,
        special_tokens=(settings["end_of_document"],),
    )
    tokenizer_bytes = (
        json.dumps(tokenizer.spec.serialization_dict(), ensure_ascii=False,
                   sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    train_ids = prepare_token_ids(
        train_documents, tokenizer, model_vocab_size=model_config.vocab_size,
        end_of_document=settings["end_of_document"],
    )
    validation_ids = prepare_token_ids(
        validation_documents, tokenizer, model_vocab_size=model_config.vocab_size,
        end_of_document=settings["end_of_document"],
    )
    sampler = NextTokenBatchSampler(
        train_ids, seq_len=model_config.max_seq_len,
        model_vocab_size=model_config.vocab_size, seed=seed,
    )
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    tokenizer_path = output_dir / "tokenizer.json"
    manifest_path = output_dir / "manifest.json"
    fingerprints = {
        "config_sha256": _digest(config_bytes),
        "train_text_sha256": _digest(train_path.read_bytes()),
        "validation_text_sha256": _digest(validation_path.read_bytes()),
        "tokenizer_sha256": _digest(tokenizer_bytes),
    }
    start_step = 0
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in fingerprints.items()):
            raise ValueError("resume inputs differ from the saved run")
        if previous.get("device") != str(device):
            raise ValueError("resume device differs from the saved run")
        if _digest(tokenizer_path.read_bytes()) != fingerprints["tokenizer_sha256"]:
            raise ValueError("saved tokenizer differs from the training corpus")
        start_step = load_checkpoint(
            checkpoint_path, model, optimizer, sampler=sampler,
            config=train_config, batch_size=batch_size,
        )
    elif any(path.exists() for path in (checkpoint_path, manifest_path, tokenizer_path)):
        raise ValueError("output already contains a run; use --resume or a new directory")
    if end_step is None:
        end_step = train_config.max_steps
    if type(end_step) is not int or not start_step < end_step <= train_config.max_steps:
        raise ValueError("end_step must be after the checkpoint step and at most max_steps")

    initial_validation_loss = validation_loss(
        model, validation_ids, model_config.max_seq_len, batch_size
    )
    result = train(
        model, lambda: sampler.sample(batch_size), train_config,
        optimizer=optimizer, start_step=start_step, end_step=end_step,
    )
    final_validation_loss = validation_loss(
        model, validation_ids, model_config.max_seq_len, batch_size
    )
    save_checkpoint(
        checkpoint_path, model, result.optimizer, result.step,
        sampler=sampler, config=train_config, batch_size=batch_size,
    )
    tokenizer_path.write_bytes(tokenizer_bytes)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        **fingerprints,
        "source_commit_sha": source_sha,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device_name == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "precision": "fp32",
        "seed": seed,
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "train_tokens": train_ids.numel(),
        "validation_tokens": validation_ids.numel(),
        "validation_windows": validation_ids.numel() - model_config.max_seq_len,
        "start_step": start_step,
        "optimizer_steps": result.step,
        "segment_steps": result.step - start_step,
        "timed_target_tokens": (result.step - start_step) * batch_size * model_config.max_seq_len,
        "train_loss_first_batch": result.initial_loss,
        "train_loss_last_batch": result.final_loss,
        "initial_validation_loss": initial_validation_loss,
        "validation_loss": final_validation_loss,
        "tokens_per_second": result.tokens_per_second,
        "training_wall_time_seconds": result.mean_step_seconds * (result.step - start_step),
        "peak_cuda_memory_bytes": result.peak_cuda_memory_bytes,
        "wall_time_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
