"""Package frozen M6 artifacts and strictly verify the private Kaggle result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path(__file__).with_name("remote_m6_pilot.py")
MANIFEST = ROOT / "data/manifests/fineweb_edu_m6.json"
CONFIG = ROOT / "configs/m6_real_pilot.json"
MARKER = "__WENDYFM_COMMIT_SHA__"
SLUG = "wendyfm-m6-b-realdata-pilot"
ARMS = (("A", "reference", "fp32"), ("B", "sdpa", "fp32"),
        ("C", "sdpa", "fp16"))
MEMORY_LIMIT_BYTES = 12 * 1024**3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha(value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("commit SHA must be 40 lowercase hexadecimal characters")


def verify_artifacts(data_dir: Path) -> dict:
    source = json.loads(MANIFEST.read_text())
    for name, key in (("train.jsonl", "train_artifact_sha256"),
                      ("validation.jsonl", "validation_artifact_sha256"),
                      ("tokenizer.json", "tokenizer_sha256")):
        if digest(data_dir / name) != source[key]:
            raise ValueError(f"{name} hash differs from frozen manifest")
    return source


def prepare_job(job_dir: Path, data_dir: Path, user: str, sha: str) -> None:
    valid_sha(sha)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", user):
        raise ValueError("invalid Kaggle username")
    if job_dir.resolve().is_relative_to(ROOT):
        raise ValueError("job directory must be outside repository")
    if job_dir.exists() and any(job_dir.iterdir()):
        raise ValueError("job directory must be empty")
    verify_artifacts(data_dir)
    source = REMOTE.read_text()
    if source.count(MARKER) != 1:
        raise ValueError("remote script must contain one SHA marker")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / REMOTE.name).write_text(source.replace(MARKER, sha))
    (job_dir / "data").mkdir()
    for name in ("train.jsonl", "validation.jsonl", "tokenizer.json"):
        shutil.copyfile(data_dir / name, job_dir / "data" / name)
    verify_artifacts(job_dir / "data")
    metadata = {"id": f"{user}/{SLUG}", "title": SLUG,
                "code_file": REMOTE.name, "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
    (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def positive(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise ValueError(f"invalid {name}")
    return float(value)


def verify_result(path: Path, sha: str) -> dict:
    valid_sha(sha)
    result = json.loads(path.read_text())
    expected = {"status": "PASS", "test": "m6_b_realdata_pilot", "retries": 0,
                "expected_commit_sha": sha, "tested_commit_sha": sha,
                "failed_arm": None}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("wrapper status, retry count, or SHA mismatch")
    arms = result.get("arms")
    if not isinstance(arms, list) or len(arms) != 3:
        raise ValueError("expected three completed sequential arms")
    source = json.loads(MANIFEST.read_text())
    settings = json.loads(CONFIG.read_text())
    expected_common = {
        "source_commit_sha": sha, "device": "cuda:0", "cuda_device_count": 1,
        "config_sha256": digest(CONFIG), "source_manifest_sha256": digest(MANIFEST),
        "train_artifact_sha256": source["train_artifact_sha256"],
        "validation_artifact_sha256": source["validation_artifact_sha256"],
        "tokenizer_sha256": source["tokenizer_sha256"],
        "trainable_parameters": 21_502_464, "microbatch_size": 2,
        "gradient_accumulation_steps": 4, "effective_batch_size": 8,
        "sequence_length": 256, "start_step": 0, "optimizer_steps": 20,
        "measured_steps": 18, "timed_target_tokens": 18 * 2048,
        "sdpa_kernel_observed": None, "checkpoint": "checkpoint.pt",
    }
    if settings["train"]["max_steps"] != 20 or settings["seed"] != 20261012:
        raise ValueError("frozen training budget changed")
    for item, (label, backend, precision) in zip(arms, ARMS, strict=True):
        if not isinstance(item, dict) or any(item.get(key) != value for key, value in {
                **expected_common, "arm": label, "attention_backend_requested": backend,
                "precision": precision}.items()):
            raise ValueError(f"arm {label} provenance, mode, or budget mismatch")
        if any(not isinstance(item.get(key), str) or not item[key] for key in
               ("torch_version", "cuda_version", "gpu_name")) or "T4" not in item["gpu_name"]:
            raise ValueError(f"arm {label} lacks T4 or runtime identity")
        for key in ("train_loss_first_batch", "train_loss_last_batch",
                    "initial_validation_loss", "validation_loss", "tokens_per_second",
                    "mean_step_seconds", "timed_seconds", "wall_seconds",
                    "training_wall_seconds"):
            positive(item.get(key), key)
        allocated = positive(item.get("peak_cuda_allocated_bytes"), "peak CUDA allocated")
        reserved = positive(item.get("peak_cuda_reserved_bytes"), "peak CUDA reserved")
        if allocated > reserved or reserved >= MEMORY_LIMIT_BYTES:
            raise ValueError("CUDA memory gate violated")
        if not math.isclose(item["tokens_per_second"],
                            item["timed_target_tokens"] / item["timed_seconds"],
                            rel_tol=1e-6):
            raise ValueError("throughput arithmetic mismatch")
        if not math.isclose(item["mean_step_seconds"],
                            item["timed_seconds"] / item["measured_steps"],
                            rel_tol=1e-6):
            raise ValueError("step timing arithmetic mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--job-dir", type=Path, required=True)
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--user", required=True)
    prepare.add_argument("--sha", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--sha", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_job(args.job_dir, args.data_dir, args.user, args.sha)
            print(f"Prepared private M6-B package: {args.job_dir}")
        else:
            verify_result(args.result, args.sha)
            print("PASS: M6-B result verified")
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
