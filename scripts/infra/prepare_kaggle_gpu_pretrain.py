"""Prepare and verify a private, SHA-pinned Kaggle FP32 pretraining job."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_SCRIPT = Path(__file__).with_name("remote_gpu_pretrain.py")
MARKER = "__WENDYFM_COMMIT_SHA__"
SLUG = "wendyfm-fp32-pretrain"


def valid_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("commit SHA must be exactly 40 lowercase hexadecimal characters")
    return value


def prepare_job(job_dir: Path, user: str, sha: str) -> None:
    valid_sha(sha)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", user):
        raise ValueError("invalid Kaggle username")
    if job_dir.resolve().is_relative_to(ROOT):
        raise ValueError("job directory must be outside the repository")
    if job_dir.exists() and any(job_dir.iterdir()):
        raise ValueError("job directory must be empty")
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")
    if source.count(MARKER) != 1:
        raise ValueError("remote script must contain exactly one SHA marker")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "remote_gpu_pretrain.py").write_text(source.replace(MARKER, sha), encoding="utf-8")
    metadata = {
        "id": f"{user}/{SLUG}",
        "title": SLUG,
        "code_file": "remote_gpu_pretrain.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (job_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def verify_result(path: Path, sha: str) -> None:
    valid_sha(sha)
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or any(
        result.get(key) != expected
        for key, expected in {
            "status": "PASS",
            "expected_commit_sha": sha,
            "tested_commit_sha": sha,
            "test": "fp32_single_gpu_pretrain",
            "checkpoint_resume_exact": True,
        }.items()
    ):
        raise ValueError("result failed status, SHA, or exact resume checks")
    manifest = result.get("manifest")
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value
        for key, value in {
            "source_commit_sha": sha, "device": "cuda:0", "precision": "fp32",
            "optimizer_steps": 8, "segment_steps": 8, "timed_target_tokens": 512,
        }.items()
    ):
        raise ValueError("result has an incompatible training manifest")
    for key in (
        "train_loss_first_batch", "train_loss_last_batch", "initial_validation_loss",
        "validation_loss", "tokens_per_second", "training_wall_time_seconds",
        "peak_cuda_memory_bytes",
    ):
        value = manifest.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value < float("inf")
        ):
            raise ValueError(f"result has invalid {key}")
    if not isinstance(manifest.get("cuda_device_name"), str) or not manifest["cuda_device_name"]:
        raise ValueError("result has no CUDA device name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--job-dir", type=Path, required=True)
    prepare.add_argument("--user", required=True)
    prepare.add_argument("--sha", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--sha", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_job(args.job_dir, args.user, args.sha)
            print(f"Prepared private FP32 GPU job: {args.job_dir}")
        else:
            verify_result(args.result, args.sha)
            print("PASS: FP32 GPU pretraining result and exact resume verified")
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
