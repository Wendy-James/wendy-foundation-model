"""Offline-prepared, single-device M6 pretraining pilot."""

from __future__ import annotations

import json
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
from wendyfm.training.real_data import (
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    sha256,
    split_for_id,
)


@torch.no_grad()
def validation_loss(
    model: DecoderOnlyTransformer, ids: torch.Tensor, seq_len: int, batch_size: int,
    precision: str = "fp32",
) -> float:
    """Score disjoint held-out windows once; avoid overlapping-window inflation."""
    if ids.numel() < seq_len + 1:
        raise ValueError("validation corpus needs at least one full window")
    model.eval()
    windows = ids.unfold(0, seq_len + 1, seq_len)
    device = next(model.parameters()).device
    total = 0.0
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=precision == "fp16"):
            loss = model.next_token_loss(batch[:, :-1], batch[:, 1:])
        total += float(loss) * len(batch)
    return total / len(windows)


def _source_documents(data_dir: Path, manifest: dict) -> tuple[list[str], list[str]]:
    expected = manifest["documents"]
    if len({item["id"] for item in expected}) != len(expected):
        raise ValueError("duplicate document IDs in source manifest")
    grouped = {}
    for split in ("train", "validation"):
        path = data_dir / f"{split}.jsonl"
        if sha256(path.read_bytes()) != manifest[f"{split}_artifact_sha256"]:
            raise ValueError(f"{split} artifact hash differs from source manifest")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        actual = [item for item in expected if item["split"] == split]
        if len(rows) != len(actual):
            raise ValueError("source document count mismatch")
        for row, record in zip(rows, actual, strict=True):
            if (row["id"] != record["id"] or split_for_id(row["id"]) != split or
                    sha256(row["text"].encode()) != record["text_sha256"]):
                raise ValueError("source document ID, split, or content differs")
        grouped[split] = [row["text"] for row in rows]
    return grouped["train"], grouped["validation"]


def run(config_path: Path, data_dir: Path, source_manifest_path: Path, output_dir: Path, *,
        device_name: str = "cpu", precision: str = "fp32", end_step: int | None = None,
        resume: bool = False) -> dict:
    """Run a bounded segment; a real pilot requires prepared source artifacts."""
    started = time.perf_counter()
    if device_name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if precision not in ("fp32", "fp16") or (precision == "fp16" and device_name != "cuda"):
        raise ValueError("fp16 requires CUDA; otherwise use fp32")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    config_bytes = config_path.read_bytes()
    settings = json.loads(config_bytes)
    model_config = ModelConfig(**settings["model"])
    train_config = TrainConfig(**settings["train"])
    batch_size = settings["microbatch_size"]
    accum = settings["gradient_accumulation_steps"]
    warmup = settings["metric_warmup_steps"]
    seed = settings["seed"]
    if any(type(value) is not int or value < 1 for value in (batch_size, accum)):
        raise ValueError("microbatch and accumulation must be positive integers")
    if type(warmup) is not int or warmup < 0 or warmup >= train_config.max_steps:
        raise ValueError("invalid metric warmup")
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid seed")
    torch.set_num_threads(settings["cpu_threads"])
    random.seed(seed)
    torch.manual_seed(seed)
    source_bytes = source_manifest_path.read_bytes()
    source = json.loads(source_bytes)
    if (source.get("dataset_repo"), source.get("dataset_config"),
            source.get("dataset_revision")) != (DATASET_REPO, DATASET_CONFIG, DATASET_REVISION):
        raise ValueError("source manifest is not the pinned FineWeb-Edu revision/config")
    train_documents, validation_documents = _source_documents(data_dir, source)
    tokenizer_path = data_dir / "tokenizer.json"
    if sha256(tokenizer_path.read_bytes()) != source["tokenizer_sha256"]:
        raise ValueError("tokenizer hash differs from source manifest")
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.spec.config.vocab_size != model_config.vocab_size:
        raise ValueError("model vocabulary must equal frozen tokenizer capacity")
    train_ids = prepare_token_ids(train_documents, tokenizer,
                                  model_vocab_size=model_config.vocab_size,
                                  end_of_document="<|eod|>")
    validation_ids = prepare_token_ids(validation_documents, tokenizer,
                                       model_vocab_size=model_config.vocab_size,
                                       end_of_document="<|eod|>")
    sampler = NextTokenBatchSampler(train_ids, seq_len=model_config.max_seq_len,
                                    model_vocab_size=model_config.vocab_size, seed=seed)
    model = DecoderOnlyTransformer(model_config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                  weight_decay=train_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    result_path = output_dir / "result.json"
    fingerprints = {
        "config_sha256": sha256(config_bytes),
        "source_manifest_sha256": sha256(source_bytes),
        "tokenizer_sha256": source["tokenizer_sha256"],
        "train_artifact_sha256": source["train_artifact_sha256"],
        "validation_artifact_sha256": source["validation_artifact_sha256"],
    }
    start_step = 0
    if resume:
        previous = json.loads(result_path.read_text())
        if any(previous.get(key) != value for key, value in fingerprints.items()):
            raise ValueError("resume provenance differs")
        if previous["precision"] != precision or previous["device"] != str(device):
            raise ValueError("resume precision or device differs")
        start_step = load_checkpoint(checkpoint_path, model, optimizer, sampler=sampler,
                                     config=train_config, batch_size=batch_size, scaler=scaler)
    elif checkpoint_path.exists() or result_path.exists():
        raise ValueError("output already contains a run")
    if end_step is None:
        end_step = train_config.max_steps
    if type(end_step) is not int or not start_step < end_step <= train_config.max_steps:
        raise ValueError("end_step must advance within max_steps")
    metric_warmup = min(warmup, end_step - start_step - 1)
    initial_validation = validation_loss(model, validation_ids, model_config.max_seq_len,
                                         batch_size, precision)
    result = train(model, lambda: sampler.sample(batch_size), train_config,
                   optimizer=optimizer, start_step=start_step, end_step=end_step,
                   grad_accum_steps=accum, precision=precision, scaler=scaler,
                   metric_warmup_steps=metric_warmup)
    final_validation = validation_loss(model, validation_ids, model_config.max_seq_len,
                                       batch_size, precision)
    save_checkpoint(checkpoint_path, model, result.optimizer, result.step, sampler=sampler,
                    config=train_config, batch_size=batch_size, scaler=scaler)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    record = {
        **fingerprints,
        "source_commit_sha": source_commit,
        "device": str(device), "precision": precision,
        "attention_backend_requested": model_config.attention_backend,
        "sdpa_kernel_observed": None,
        "trainable_parameters": parameter_count,
        "microbatch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": batch_size * accum,
        "sequence_length": model_config.max_seq_len,
        "start_step": start_step, "optimizer_steps": result.step,
        "measured_steps": result.measured_steps,
        "timed_target_tokens": result.timed_target_tokens,
        "mean_step_seconds": result.timed_seconds / result.measured_steps,
        "tokens_per_second": result.tokens_per_second,
        "train_loss_first_batch": result.initial_loss,
        "train_loss_last_batch": result.final_loss,
        "initial_validation_loss": initial_validation,
        "validation_loss": final_validation,
        "peak_cuda_allocated_bytes": result.peak_cuda_memory_bytes,
        "peak_cuda_reserved_bytes": result.peak_cuda_reserved_bytes,
        "training_wall_seconds": result.mean_step_seconds * (result.step - start_step),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_path.name,
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record
