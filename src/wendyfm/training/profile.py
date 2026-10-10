"""Fixed, single-CUDA-device Phase A FP32 training baseline."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.data import NextTokenBatchSampler, prepare_token_ids
from wendyfm.training.gpu import _documents
from wendyfm.training.loop import TrainConfig, train

ROOT = Path(__file__).resolve().parents[3]
BASELINE = {"batch_size": 4, "warmup_steps": 5, "measured_steps": 30, "repeats": 3,
            "seed": 20261009, "cpu_threads": 1, "end_of_document": "<|eod|>"}
MODEL = {"d_model": 32, "n_heads": 4, "n_layers": 2, "intermediate_size": 64,
         "max_seq_len": 16, "vocab_size": 272}
TRAIN = {"max_steps": 35, "learning_rate": 0.003, "min_learning_rate": 0.003,
         "warmup_steps": 0, "weight_decay": 0.01}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: Path) -> dict:
    settings = json.loads(path.read_text(encoding="utf-8"))
    if set(settings) != set(BASELINE) | {"model", "train", "train_text"}:
        raise ValueError("Phase A config keys differ from the frozen baseline")
    if any(type(settings[key]) is not type(value) or settings[key] != value
           for key, value in BASELINE.items()):
        raise ValueError("Phase A baseline bounds differ")
    if settings["model"] != MODEL or settings["train"] != TRAIN:
        raise ValueError("Phase A model or optimizer differs from the frozen baseline")
    if settings["train_text"] != "../data/fixtures/cpu_pilot_train.txt":
        raise ValueError("Phase A corpus differs from the M4 fixture")
    ModelConfig(**settings["model"])
    TrainConfig(**settings["train"])
    return settings


def aggregate(repeats: list[dict]) -> dict:
    if len(repeats) != 3 or any(item.get("status") != "PASS" for item in repeats):
        raise ValueError("three completed repeats are required")
    rates = sorted(item["target_tokens_per_second"] for item in repeats)
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("repeat rates must be finite and positive")
    median = rates[1]
    spread = (rates[-1] - rates[0]) / median
    return {"median_target_tokens_per_second": median, "throughput_spread": spread,
            "unstable": spread > 0.10}


def _source_sha(expected_sha: str) -> str:
    if len(expected_sha) != 40 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("expected SHA must be 40 lowercase hexadecimal characters")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != expected_sha:
        raise ValueError("source SHA differs from expected commit")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("source checkout has uncommitted files")
    return actual


def _gpu_platform(device_name: str) -> dict:
    completed = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=name,driver_version,power.limit,pci.bus_id",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    rows = completed.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("expected one selected nvidia-smi GPU row")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4 or fields[0] != device_name or not all(fields):
        raise ValueError("selected physical GPU does not match logical cuda:0")
    ordering = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    devices = [[part.strip() for part in line.split(",")]
               for line in ordering.stdout.strip().splitlines()]
    if (not devices or any(len(row) != 2 for row in devices)
            or [row[1] for row in devices if row[0] == "0"] != [fields[3]]
            or fields[3] != min(row[1] for row in devices)):
        raise ValueError("selected physical GPU does not match PCI CUDA ordering")
    power_limit = float(fields[2])
    if not math.isfinite(power_limit) or power_limit <= 0:
        raise ValueError("invalid GPU power limit")
    return {"driver_version": fields[1], "gpu_power_limit_watts": power_limit}


def _memory_on_failure() -> dict:
    try:
        return {"allocated_bytes": torch.cuda.memory_allocated(0),
                "reserved_bytes": torch.cuda.memory_reserved(0)}
    except RuntimeError:
        return {}


def _preflight(model: DecoderOnlyTransformer, sampler: NextTokenBatchSampler,
               batch_size: int, config: TrainConfig) -> None:
    inputs, targets = sampler.sample(batch_size)
    if not torch.equal(inputs[:, 1:], targets[:, :-1]):
        raise ValueError("sampler targets are not aligned exactly once")
    device = next(model.parameters()).device
    with torch.no_grad():
        x, y = inputs.to(device), targets.to(device)
        explicit = F.cross_entropy(model(x).reshape(-1, model.config.vocab_size),
                                   y.reshape(-1))
        actual = model.next_token_loss(x, y)
        torch.testing.assert_close(actual, explicit)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = train(model, lambda: (inputs, targets), config, end_step=1)
    if result.step != 1 or not any(not torch.equal(old, new) for old, new
                                   in zip(before, model.parameters(), strict=True)):
        raise ValueError("M4 reference step did not update parameters")
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("nonfinite parameter after M4 reference step")


def _check_fp32_model(model: DecoderOnlyTransformer) -> None:
    if any(parameter.device != torch.device("cuda:0") or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError("profile model must have only FP32 parameters on cuda:0")


def _repeat(index: int, ids: torch.Tensor, settings: dict) -> dict:
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    config = TrainConfig(**settings["train"])
    model = DecoderOnlyTransformer(ModelConfig(**settings["model"])).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    sampler = NextTokenBatchSampler(ids, seq_len=settings["model"]["max_seq_len"],
                                    model_vocab_size=settings["model"]["vocab_size"],
                                    seed=settings["seed"])
    observations: list[dict] = []
    phase = "preflight"
    try:
        _check_fp32_model(model)
        if index == 1:
            # The reference check uses the actual M4 train() path, outside timing.
            _preflight(model, sampler, settings["batch_size"], config)
            del model, optimizer, sampler
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(settings["seed"])
            model = DecoderOnlyTransformer(ModelConfig(**settings["model"])).to("cuda:0")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
            sampler = NextTokenBatchSampler(ids, seq_len=settings["model"]["max_seq_len"],
                                            model_vocab_size=settings["model"]["vocab_size"],
                                            seed=settings["seed"])
            _check_fp32_model(model)
        phase = "warmup"
        train(model, lambda: sampler.sample(settings["batch_size"]), config,
              optimizer=optimizer, end_step=settings["warmup_steps"])
        prior_parameters = [parameter.detach().cpu().clone() for parameter in model.parameters()]
        device = torch.device("cuda:0")
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        def observe(step: int, loss: float, grad_norm: float) -> None:
            observations.append({"step": step, "loss": loss, "gradient_norm": grad_norm})

        phase = "measurement"
        result = train(model, lambda: sampler.sample(settings["batch_size"]), config,
                       optimizer=optimizer, start_step=settings["warmup_steps"],
                       step_observer=observe)
        changed = any(
            not torch.equal(old, new.detach().cpu()) for old, new
            in zip(prior_parameters, model.parameters(), strict=True)
        )
        if not changed:
            raise ValueError("measured optimizer steps did not update parameters")
        elapsed = result.mean_step_seconds * settings["measured_steps"]
        tokens = (settings["measured_steps"] * settings["batch_size"]
                  * settings["model"]["max_seq_len"])
        return {"status": "PASS", "repeat": index, "warmup_steps": 5,
                "measured_steps": len(observations), "elapsed_seconds": elapsed,
                "mean_step_seconds": result.mean_step_seconds, "target_tokens": tokens,
                "target_tokens_per_second": tokens / elapsed,
                "allocated_baseline_bytes": allocated, "reserved_baseline_bytes": reserved,
                "peak_allocated_bytes": result.peak_cuda_memory_bytes,
                "peak_reserved_bytes": result.peak_cuda_reserved_bytes,
                "measured_run_changed_parameter": changed,
                "loss_min": min(item["loss"] for item in observations),
                "loss_max": max(item["loss"] for item in observations),
                "steps": observations}
    except Exception as error:
        return {"status": "FAIL", "repeat": index, "phase": phase,
                "error_type": type(error).__name__, "completed_measured_steps": len(observations),
                **_memory_on_failure()}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def run(config_path: Path, output_path: Path, *, expected_sha: str) -> dict:
    """Run one bounded cell; always write a factual result, including failures."""
    settings = load_config(config_path)
    sha = _source_sha(expected_sha)
    if output_path.exists():
        raise ValueError("output already exists")
    config_bytes = config_path.read_bytes()
    train_path = (config_path.parent / settings["train_text"]).resolve()
    train_bytes = train_path.read_bytes()
    result: dict = {"status": "FAIL", "source_commit_sha": sha,
                    "config_sha256": digest(config_bytes), "train_text_sha256": digest(train_bytes),
                    "seed": settings["seed"], "precision": "fp32", "device": "cuda:0",
                    "batch_size": 4, "sequence_length": 16, "warmup_steps": 5,
                    "measured_steps": 30, "requested_repeats": 3, "retries": 0,
                    "repeats": [], "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device is required")
        result["cuda_device_count"] = 1
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(settings["cpu_threads"])
        device = torch.cuda.get_device_properties(0)
        if "T4" not in device.name:
            raise RuntimeError("Phase A requires one Tesla T4 GPU")
        result.update({"gpu_name": device.name, "gpu_total_memory_bytes": device.total_memory,
                       "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                       "parameter_count": sum(p.numel() for p in DecoderOnlyTransformer(
                           ModelConfig(**settings["model"])).parameters())})
        result.update(_gpu_platform(device.name))
        documents = _documents(train_path)
        tokenizer = BPETokenizer.train(documents, vocab_size=settings["model"]["vocab_size"],
                                       special_tokens=(settings["end_of_document"],))
        tokenizer_bytes = (json.dumps(tokenizer.spec.serialization_dict(), sort_keys=True,
                                      ensure_ascii=False) + "\n").encode("utf-8")
        result["tokenizer_sha256"] = digest(tokenizer_bytes)
        ids = prepare_token_ids(documents, tokenizer,
                                model_vocab_size=settings["model"]["vocab_size"],
                                end_of_document=settings["end_of_document"])
        result["train_tokens"] = ids.numel()
        for index in range(1, 4):
            repeat = _repeat(index, ids, settings)
            result["repeats"].append(repeat)
            gc.collect()
            torch.cuda.empty_cache()
            if repeat["status"] != "PASS":
                result["failure_phase"] = repeat["phase"]
                result["error_type"] = repeat["error_type"]
                break
        else:
            result.update(aggregate(result["repeats"]))
            result["status"] = "PASS"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["failure_phase"] = "setup"
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return result
