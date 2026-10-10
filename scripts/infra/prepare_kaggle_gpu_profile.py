"""Package and verify one private, SHA-pinned M5 Kaggle profiling job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path(__file__).with_name("remote_gpu_profile.py")
CONFIG = ROOT / "configs/m5_phase_a_fp32.json"
MARKER = "__WENDYFM_COMMIT_SHA__"
SLUG = "wendyfm-m5-fp32-profile"


def valid_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("commit SHA must be 40 lowercase hexadecimal characters")
    return value


def prepare_job(job_dir: Path, user: str, sha: str) -> None:
    valid_sha(sha)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", user):
        raise ValueError("invalid Kaggle username")
    if job_dir.resolve().is_relative_to(ROOT):
        raise ValueError("job directory must be outside repository")
    if job_dir.exists() and any(job_dir.iterdir()):
        raise ValueError("job directory must be empty")
    source = REMOTE.read_text(encoding="utf-8")
    if source.count(MARKER) != 1:
        raise ValueError("remote script must contain one SHA marker")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / REMOTE.name).write_text(source.replace(MARKER, sha), encoding="utf-8")
    metadata = {"id": f"{user}/{SLUG}", "title": SLUG, "code_file": REMOTE.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
    (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                  encoding="utf-8")


def verify_result(path: Path, sha: str) -> dict:
    valid_sha(sha)
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if any(wrapper.get(key) != value for key, value in {
        "status": "PASS", "expected_commit_sha": sha, "tested_commit_sha": sha,
        "test": "m5_phase_a_fp32"}.items()):
        raise ValueError("wrapper status or SHA mismatch")
    result = wrapper.get("manifest")
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    train_text = (CONFIG.parent / settings["train_text"]).resolve()
    if not isinstance(result, dict) or any(result.get(key) != value for key, value in {
        "status": "PASS", "source_commit_sha": sha, "precision": "fp32", "device": "cuda:0",
        "cuda_device_count": 1,
        "batch_size": 4, "sequence_length": 16, "warmup_steps": 5,
        "measured_steps": 30, "requested_repeats": 3, "retries": 0,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "train_text_sha256": hashlib.sha256(train_text.read_bytes()).hexdigest()}.items()):
        raise ValueError("manifest provenance or baseline mismatch")
    if type(result["cuda_device_count"]) is not int:
        raise ValueError("invalid CUDA device count")
    if any(not isinstance(result.get(key), str) or not result[key] for key in
           ("torch_version", "cuda_version", "driver_version")):
        raise ValueError("missing CUDA or PyTorch identity")
    if not isinstance(result.get("gpu_name"), str) or "T4" not in result["gpu_name"]:
        raise ValueError("Phase A requires a Tesla T4 GPU")
    for key in ("gpu_total_memory_bytes", "gpu_power_limit_watts", "parameter_count"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (
            not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"invalid {key}")
    if any(type(result[key]) is not int for key in ("gpu_total_memory_bytes", "parameter_count")):
        raise ValueError("invalid GPU memory or parameter count type")
    repeats = result.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != 3:
        raise ValueError("expected three independent repeats")
    for index, repeat in enumerate(repeats, 1):
        if any(repeat.get(key) != value for key, value in {
            "status": "PASS", "repeat": index, "warmup_steps": 5,
            "measured_steps": 30, "target_tokens": 1920,
            "measured_run_changed_parameter": True}.items()):
            raise ValueError("incomplete repeat")
        for key in ("elapsed_seconds", "mean_step_seconds", "target_tokens_per_second",
                    "allocated_baseline_bytes", "reserved_baseline_bytes",
                    "peak_allocated_bytes", "peak_reserved_bytes"):
            value = repeat.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                not math.isfinite(value) or value < 0 or (
                    key not in ("allocated_baseline_bytes", "reserved_baseline_bytes")
                    and value == 0
                )
            ):
                raise ValueError(f"invalid {key}")
        if (repeat["peak_allocated_bytes"] < repeat["allocated_baseline_bytes"]
                or repeat["peak_reserved_bytes"] < repeat["reserved_baseline_bytes"]):
            raise ValueError("peak memory below warmup baseline")
        if not math.isclose(repeat["mean_step_seconds"], repeat["elapsed_seconds"] / 30,
                            rel_tol=1e-9):
            raise ValueError("mean step timing mismatch")
        if not math.isclose(repeat["target_tokens_per_second"], 1920 / repeat["elapsed_seconds"],
                            rel_tol=1e-9):
            raise ValueError("target-token accounting mismatch")
        if not isinstance(repeat.get("steps"), list) or len(repeat["steps"]) != 30:
            raise ValueError("missing measured step observations")
        for step_number, observation in enumerate(repeat["steps"], 6):
            if observation.get("step") != step_number or any(
                isinstance(observation.get(key), bool)
                or not isinstance(observation.get(key), (int, float))
                or not math.isfinite(observation[key])
                or observation[key] < 0
                for key in ("loss", "gradient_norm")
            ):
                raise ValueError("invalid measured step observation")
        for key, expected in (("loss_min", min(step["loss"] for step in repeat["steps"])),
                              ("loss_max", max(step["loss"] for step in repeat["steps"]))):
            if repeat.get(key) != expected:
                raise ValueError(f"invalid {key}")
    rates = sorted(repeat["target_tokens_per_second"] for repeat in repeats)
    median = rates[1]
    spread = (rates[2] - rates[0]) / median
    reported_median = result.get("median_target_tokens_per_second")
    reported_spread = result.get("throughput_spread")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in (reported_median, reported_spread))
            or not math.isclose(reported_median, median, rel_tol=1e-9)
            or not math.isclose(reported_spread, spread, rel_tol=1e-9)
            or result.get("unstable") is not (spread > 0.10)):
        raise ValueError("repeat aggregation mismatch")
    return result


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
            print(f"Prepared private M5 profile job: {args.job_dir}")
        else:
            verify_result(args.result, args.sha)
            print("PASS: M5 Phase A result verified")
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
