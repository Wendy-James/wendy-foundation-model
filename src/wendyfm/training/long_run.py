"""Long-lived, segmented real-data training with durable interval evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.checkpoint import load_checkpoint, save_checkpoint
from wendyfm.training.data import NextTokenBatchSampler, prepare_token_ids
from wendyfm.training.long_data import M6_MANIFEST, ROOT, SPLIT_RULE, file_sha256
from wendyfm.training.loop import TrainConfig, learning_rate_for_step, train
from wendyfm.training.real_data import (
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SHARD,
)
from wendyfm.training.real_pilot import (
    _source_documents,
    parameter_update_witness,
    validation_loss,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as file:
            temporary = file.name
            json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_time(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
        not math.isfinite(value) or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint_{step:08d}.pt"


def _event_path(output_dir: Path, step: int) -> Path:
    return output_dir / "events" / f"step_{step:08d}.json"


def _code_sha(expected: str | None) -> str:
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if expected is not None:
        if (len(expected) != 40 or any(character not in "0123456789abcdef" for character in
                                        expected) or actual != expected):
            raise ValueError("executable commit SHA differs from pinned SHA")
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                               capture_output=True, text=True).stdout
        if dirty:
            raise ValueError("pinned executable checkout has uncommitted changes")
    return actual


def _check_memory(device: torch.device, limit: int) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    if reserved >= limit:
        raise MemoryError("CUDA reserved peak reached the run limit")
    return allocated, reserved


def _runtime_source(data_dir: Path, manifest_bytes: bytes,
                    expected_tokenizer_sha256: str | None) -> dict:
    """Check a previously verified corpus without a second full tokenization."""
    source = json.loads(manifest_bytes)
    if (source.get("schema_version"), source.get("dataset_repo"),
            source.get("dataset_config"), source.get("dataset_revision"),
            source.get("dataset_split"), source.get("dataset_shard"),
            source.get("dataset_license"), source.get("split_rule")) != (
                1, DATASET_REPO, DATASET_CONFIG, DATASET_REVISION, "train",
                DATASET_SHARD, "odc-by", SPLIT_RULE):
        raise ValueError("source provenance differs from the pinned M7 contract")
    frozen_tokenizer = json.loads(M6_MANIFEST.read_text())["tokenizer_sha256"]
    expected = expected_tokenizer_sha256 or frozen_tokenizer
    if (source.get("tokenizer_sha256") != expected or
            file_sha256(data_dir / "tokenizer.json") != expected):
        raise ValueError("tokenizer hash differs from the frozen source")
    documents = source.get("documents")
    if not isinstance(documents, list) or not all(isinstance(item, dict) for item in documents):
        raise ValueError("invalid source documents")
    text_hashes = [item.get("text_sha256") for item in documents]
    if len(set(text_hashes)) != len(text_hashes):
        raise ValueError("duplicate document content in source manifest")
    for split in ("train", "validation"):
        tokens = source.get(f"{split}_tokens_with_eod")
        floor = source.get(f"min_{split}_tokens")
        actual = sum(item.get("token_count_with_eod", 0) for item in documents
                     if item.get("split") == split)
        if (type(tokens) is not int or type(floor) is not int or
                tokens < floor or actual != tokens):
            raise ValueError(f"{split} token accounting differs")
    if expected_tokenizer_sha256 is None and (
        source["train_tokens_with_eod"] < 20_000_000 or
        source["validation_tokens_with_eod"] < 200_000
    ):
        raise ValueError("M7 corpus is below the required real-data token floors")
    return source


def _latest_event(output_dir: Path, plan_sha256: str, tokens_per_step: int) -> dict:
    paths = sorted((output_dir / "events").glob("step_*.json"))
    if not paths:
        raise ValueError("resume lacks a completed checkpoint event")
    latest = paths[-1]
    event = json.loads(latest.read_text(encoding="utf-8"))
    step = event.get("step")
    if (type(step) is not int or step < 0 or
            latest != _event_path(output_dir, step) or
            event.get("plan_sha256") != plan_sha256 or
            event.get("cumulative_target_tokens") != step * tokens_per_step or
            not _checkpoint_path(output_dir, step).is_file()):
        raise ValueError("latest event or checkpoint is incomplete or incompatible")
    _positive_time(event.get("work_wall_seconds"), "event work_wall_seconds")
    return event


def _run_long(
    config_path: Path, data_dir: Path, output_dir: Path, *,
    expected_manifest_sha256: str, device_name: str = "cpu", resume: bool = False,
    stop_after_step: int | None = None, expected_tokenizer_sha256: str | None = None,
    expected_commit_sha: str | None = None,
    progress: dict[str, str],
) -> dict:
    """Run all intervals, or stop cleanly at an interval for exact later resume.

    ``expected_tokenizer_sha256`` is a synthetic-test override. The CLI never
    exposes it, so real runs require the M6-frozen tokenizer from M7-A.
    """
    process_started = time.perf_counter()
    config_path, data_dir, output_dir = map(Path, (config_path, data_dir, output_dir))
    if device_name not in ("cpu", "cuda") or (device_name == "cuda" and
                                           not torch.cuda.is_available()):
        raise ValueError("requested device is unavailable")
    resolved = output_dir.resolve()
    if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "outputs"):
        raise ValueError("repository-local training output must be under ignored outputs")
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    config_bytes = config_path.read_bytes()
    source_commit_sha = _code_sha(expected_commit_sha)
    settings = json.loads(config_bytes)
    model_config = ModelConfig(**settings["model"])
    train_config = TrainConfig(**settings["train"])
    long_settings = settings["long_run"]
    if settings.get("precision") != "fp32" or model_config.attention_backend != "sdpa":
        raise ValueError("M7-B protocol requires SDPA FP32")
    batch_size = _positive_int(settings["microbatch_size"], "microbatch_size")
    accum = _positive_int(settings["gradient_accumulation_steps"],
                          "gradient_accumulation_steps")
    warmup = _nonnegative_int(settings["metric_warmup_steps"], "metric_warmup_steps")
    interval = _positive_int(long_settings["validation_interval_steps"],
                             "validation_interval_steps")
    target_tokens = _positive_int(long_settings["target_tokens"], "target_tokens")
    memory_limit = _positive_int(long_settings["max_cuda_reserved_bytes"],
                                 "max_cuda_reserved_bytes")
    wall_limit = _positive_time(long_settings["max_wall_seconds"], "max_wall_seconds")
    seed = settings["seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    if type(settings["cpu_threads"]) is not int or settings["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be positive")
    tokens_per_step = batch_size * accum * model_config.max_seq_len
    if (train_config.max_steps != math.ceil(target_tokens / tokens_per_step) or
            warmup >= train_config.max_steps or interval > train_config.max_steps):
        raise ValueError("training steps, warmup, or interval violate the token budget")
    if stop_after_step is not None and (
        type(stop_after_step) is not int or not 0 < stop_after_step <= train_config.max_steps or
        (stop_after_step % interval and stop_after_step != train_config.max_steps)
    ):
        raise ValueError("stop_after_step must be an interval boundary")
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha256
    ):
        raise ValueError("expected manifest SHA256 is invalid")
    manifest_path = data_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if _digest(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("source manifest SHA256 mismatch")
    source = _runtime_source(data_dir, manifest_bytes, expected_tokenizer_sha256)
    progress["stage"] = "initialization"
    train_documents, validation_documents = _source_documents(data_dir, source)
    tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")
    if tokenizer.spec.config.vocab_size != model_config.vocab_size:
        raise ValueError("model vocabulary differs from frozen tokenizer")
    train_ids = prepare_token_ids(train_documents, tokenizer,
                                  model_vocab_size=model_config.vocab_size,
                                  end_of_document="<|eod|>")
    validation_ids = prepare_token_ids(validation_documents, tokenizer,
                                       model_vocab_size=model_config.vocab_size,
                                       end_of_document="<|eod|>")
    if (train_ids.numel() != source["train_tokens_with_eod"] or
            validation_ids.numel() != source["validation_tokens_with_eod"]):
        raise ValueError("tokenized corpus totals differ from frozen manifest")
    torch.set_num_threads(settings["cpu_threads"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    gpu_name = None
    if device.type == "cuda":
        if torch.cuda.device_count() != 1:
            raise ValueError("M7-B requires exactly one visible CUDA device")
        gpu_name = torch.cuda.get_device_name(device)
        if "T4" not in gpu_name:
            raise ValueError("M7-B protocol requires a T4")
    sampler = NextTokenBatchSampler(train_ids, seq_len=model_config.max_seq_len,
                                    model_vocab_size=model_config.vocab_size, seed=seed)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                  weight_decay=train_config.weight_decay)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    plan = {
        "schema_version": 1, "config_sha256": _digest(config_bytes),
        "source_commit_sha": source_commit_sha,
        "source_manifest_sha256": expected_manifest_sha256,
        "train_artifact_sha256": source["train_artifact_sha256"],
        "validation_artifact_sha256": source["validation_artifact_sha256"],
        "tokenizer_sha256": source["tokenizer_sha256"],
        "dataset_revision": source["dataset_revision"],
        "device": str(device), "precision": "fp32",
        "gpu_name": gpu_name, "cuda_device_count": torch.cuda.device_count()
        if device.type == "cuda" else 0,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "attention_backend": model_config.attention_backend,
        "trainable_parameters": parameter_count,
        "target_tokens": target_tokens, "tokens_per_step": tokens_per_step,
        "max_steps": train_config.max_steps, "validation_interval_steps": interval,
        "max_cuda_reserved_bytes": memory_limit, "max_wall_seconds": wall_limit,
    }
    plan_sha256 = _digest(json.dumps(plan, sort_keys=True, allow_nan=False).encode())
    if resume:
        progress["stage"] = "resume"
        saved_plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
        if saved_plan != plan:
            raise ValueError("frozen run plan differs on resume")
        previous = _latest_event(output_dir, plan_sha256, tokens_per_step)
        step = previous["step"]
        loaded_step = load_checkpoint(_checkpoint_path(output_dir, step), model, optimizer,
                                      sampler=sampler, config=train_config, batch_size=batch_size)
        if loaded_step != step:
            raise ValueError("checkpoint and event step differ")
        if step == train_config.max_steps:
            return previous
    else:
        if output_dir.exists():
            raise ValueError("fresh output directory must not exist")
        (output_dir / "events").mkdir(parents=True)
        _atomic_json(output_dir / "plan.json", plan)
        step = 0
        previous = None
        startup_seconds = time.perf_counter() - process_started
        progress["stage"] = "initial_validation"
        validation_started = time.perf_counter()
        initial_validation = validation_loss(model, validation_ids, model_config.max_seq_len,
                                             batch_size)
        validation_seconds = time.perf_counter() - validation_started
        allocated, reserved = _check_memory(device, memory_limit)
        progress["stage"] = "checkpoint"
        checkpoint_started = time.perf_counter()
        save_checkpoint(_checkpoint_path(output_dir, 0), model, optimizer, 0,
                        sampler=sampler, config=train_config, batch_size=batch_size)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        work_seconds = startup_seconds + validation_seconds + checkpoint_seconds
        previous = {
            "event": "initial", "step": 0, "plan_sha256": plan_sha256,
            "cumulative_target_tokens": 0, "validation_loss": initial_validation,
            "train_loss_curve": [], "training_wall_seconds": 0.0,
            "timed_seconds": 0.0, "timed_target_tokens": 0,
            "timed_target_tokens_per_second": None,
            "full_run_target_tokens_per_second": 0.0,
            "startup_seconds": startup_seconds,
            "validation_wall_seconds": validation_seconds,
            "checkpoint_wall_seconds": checkpoint_seconds,
            "work_wall_seconds": work_seconds,
            "peak_cuda_allocated_bytes": allocated,
            "peak_cuda_reserved_bytes": reserved,
            "learning_rate": train_config.learning_rate,
        }
        _atomic_json(_event_path(output_dir, 0), previous)
    startup_pending = time.perf_counter() - process_started if resume else 0.0
    end_limit = stop_after_step or train_config.max_steps
    if step >= end_limit:
        return previous
    while step < end_limit:
        end_step = min(step + interval, train_config.max_steps)
        if end_step > end_limit:
            raise ValueError("stop_after_step must align with an interval")
        startup_seconds = startup_pending
        startup_pending = 0.0
        prior_work = previous["work_wall_seconds"]
        if prior_work + startup_seconds >= wall_limit:
            raise TimeoutError("run wall-time limit reached before training")
        before = model.final_norm.weight.detach().cpu().clone()
        curve: list[dict] = []
        segment_started = time.perf_counter()
        progress["stage"] = "training"

        def observe(global_step: int, loss: float, grad_norm: float) -> None:
            if prior_work + startup_seconds + time.perf_counter() - segment_started >= wall_limit:
                raise TimeoutError("run wall-time limit reached during training")
            curve.append({"step": global_step, "train_loss": loss, "gradient_norm": grad_norm})

        result = train(model, lambda: sampler.sample(batch_size), train_config,
                       optimizer=optimizer, start_step=step, end_step=end_step,
                       step_observer=observe, grad_accum_steps=accum,
                       metric_warmup_steps=min(warmup, end_step - step - 1) if step == 0 else 0,
                       max_cuda_reserved_bytes=memory_limit if device.type == "cuda" else None)
        training_seconds = time.perf_counter() - segment_started
        if parameter_update_witness(before, model.final_norm.weight)["changed_elements"] == 0:
            raise RuntimeError("optimizer did not update final_norm.weight")
        if len(curve) != end_step - step:
            raise RuntimeError("missing per-step training observations")
        progress["stage"] = "validation"
        validation_started = time.perf_counter()
        final_validation = validation_loss(model, validation_ids, model_config.max_seq_len,
                                           batch_size)
        validation_seconds = time.perf_counter() - validation_started
        allocated, reserved = _check_memory(device, memory_limit)
        if prior_work + startup_seconds + training_seconds + validation_seconds >= wall_limit:
            raise TimeoutError("run wall-time limit reached before checkpoint")
        progress["stage"] = "checkpoint"
        checkpoint_started = time.perf_counter()
        save_checkpoint(_checkpoint_path(output_dir, end_step), model, result.optimizer,
                        end_step, sampler=sampler, config=train_config, batch_size=batch_size)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        work_seconds = (prior_work + startup_seconds + training_seconds +
                        validation_seconds + checkpoint_seconds)
        if work_seconds >= wall_limit:
            raise TimeoutError("run wall-time limit reached after checkpoint")
        tokens = end_step * tokens_per_step
        event = {
            "event": "complete" if end_step == train_config.max_steps else "interval",
            "step": end_step, "plan_sha256": plan_sha256,
            "cumulative_target_tokens": tokens,
            "validation_loss": final_validation,
            "train_loss_first_batch": result.initial_loss,
            "train_loss_last_batch": result.final_loss,
            "train_loss_curve": curve,
            "training_wall_seconds": training_seconds,
            "timed_seconds": result.timed_seconds,
            "timed_target_tokens": result.timed_target_tokens,
            "timed_target_tokens_per_second": result.tokens_per_second,
            "full_run_target_tokens_per_second": tokens / work_seconds,
            "startup_seconds": startup_seconds,
            "validation_wall_seconds": validation_seconds,
            "checkpoint_wall_seconds": checkpoint_seconds,
            "work_wall_seconds": work_seconds,
            "peak_cuda_allocated_bytes": allocated,
            "peak_cuda_reserved_bytes": reserved,
            "learning_rate": learning_rate_for_step(end_step, train_config),
        }
        progress["stage"] = "evidence"
        _atomic_json(_event_path(output_dir, end_step), event)
        old_checkpoint = _checkpoint_path(output_dir, step)
        old_checkpoint.unlink(missing_ok=True)
        step, previous = end_step, event
    progress["stage"] = "complete"
    return previous


def run_long(
    config_path: Path, data_dir: Path, output_dir: Path, *,
    expected_manifest_sha256: str, device_name: str = "cpu", resume: bool = False,
    stop_after_step: int | None = None, expected_tokenizer_sha256: str | None = None,
    expected_commit_sha: str | None = None,
) -> dict:
    """Run one job or exact continuation; record concise private failure evidence."""
    progress = {"stage": "preflight"}
    try:
        return _run_long(
            config_path, data_dir, output_dir,
            expected_manifest_sha256=expected_manifest_sha256,
            device_name=device_name, resume=resume,
            stop_after_step=stop_after_step,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            expected_commit_sha=expected_commit_sha, progress=progress,
        )
    except Exception as error:
        output = Path(output_dir)
        if (output / "plan.json").is_file():
            message = str(error)
            if (not message or len(message) > 200 or
                    any(character in message for character in ("/", "\\", "\n", "\r"))):
                message = "see private local traceback"
            try:
                _atomic_json(output / "failure.json", {
                    "status": "FAIL", "stage": progress["stage"],
                    "error_type": type(error).__name__, "error_message": message,
                })
            except OSError:
                pass
        raise
