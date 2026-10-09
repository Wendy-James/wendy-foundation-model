"""Prepare a private Kaggle T4 model smoke job and validate its result."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_SCRIPT = Path(__file__).with_name("remote_model_smoke.py")
MARKER = "__WENDYFM_COMMIT_SHA__"
SLUG = "wendyfm-model-smoke"


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
    (job_dir / "remote_model_smoke.py").write_text(source.replace(MARKER, sha), encoding="utf-8")
    metadata = {
        "id": f"{user}/{SLUG}",
        "title": SLUG,
        "code_file": "remote_model_smoke.py",
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
            "test": "model_cuda_adamw",
        }.items()
    ):
        raise ValueError("result failed status, SHA, or test identity checks")
    for key in ("loss", "step_time_s", "peak_gpu_memory_bytes"):
        value = result.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value < float("inf")
        ):
            raise ValueError(f"result has invalid {key}")
    if not isinstance(result.get("cuda_device"), str) or not result["cuda_device"]:
        raise ValueError("result has no CUDA device")


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
            print(f"Prepared private GPU job: {args.job_dir}")
        else:
            verify_result(args.result, args.sha)
            print("PASS: model CUDA smoke result and commit SHA verified")
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
