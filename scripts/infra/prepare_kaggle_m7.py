"""Prepare a private M7 Kaggle job without copying corpus bytes into its source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from wendyfm.training.long_data import file_sha256

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path(__file__).with_name("remote_m7.py")
CONFIG = ROOT / "configs/m7_t4_1m.json"
CONFIG_SHA256 = "71c39117cd56105f892470cd5a20d823c78c64ccdaf88b86721963d9cc523ad4"
TOKENIZER_SHA256 = "c518f4ee754fb30aed2497fd366e4a795f780572e0e1a463ea8f4c01b41a952f"
ARTIFACTS = {"train.jsonl": "train_artifact_sha256",
             "validation.jsonl": "validation_artifact_sha256",
             "tokenizer.json": "tokenizer_sha256"}
FILES = frozenset(("manifest.json", *ARTIFACTS))
MARKERS = {"__WENDYFM_COMMIT_SHA__": "sha", "__WENDYFM_MANIFEST_SHA256__": "manifest",
           "__WENDYFM_DATASET_SLUG__": "slug"}


def valid_hash(value: str, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def finite_number(value: object, *, positive: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and (value > 0 if positive else value >= 0))


def matches_expected(actual: object, expected: object) -> bool:
    """Compare frozen integer fields without accepting bool as int."""
    return (type(actual) is int if type(expected) is int else True) and actual == expected


def verify_attachment(data_dir: Path, manifest_sha256: str,
                      tokenizer_sha256: str | None = None) -> dict:
    """Verify only the four allowed files and their byte hashes before packaging."""
    if not valid_hash(manifest_sha256, 64):
        raise ValueError("invalid manifest SHA256")
    if not data_dir.is_dir() or {p.name for p in data_dir.iterdir()} != FILES or any(
        not (data_dir / name).is_file() or (data_dir / name).is_symlink() for name in FILES
    ):
        raise ValueError("attachment must contain exactly four regular files")
    if file_sha256(data_dir / "manifest.json") != manifest_sha256:
        raise ValueError("manifest SHA256 mismatch")
    source = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("invalid manifest")
    for name, key in ARTIFACTS.items():
        if not valid_hash(source.get(key), 64) or file_sha256(data_dir / name) != source[key]:
            raise ValueError(f"{name} SHA256 mismatch")
    if source["tokenizer_sha256"] != (tokenizer_sha256 or TOKENIZER_SHA256):
        raise ValueError("tokenizer differs from frozen M6 artifact")
    if (type(source.get("train_tokens_with_eod")) is not int or
            type(source.get("validation_tokens_with_eod")) is not int or
            source["train_tokens_with_eod"] < 20_000_000 or
            source["validation_tokens_with_eod"] < 200_000):
        raise ValueError("M7 corpus below required token floors")
    return source


def prepare_job(job_dir: Path, data_dir: Path, user: str, dataset_slug: str,
                sha: str, manifest_sha256: str) -> None:
    if not valid_hash(sha, 40):
        raise ValueError("commit SHA must be 40 lowercase hexadecimal characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", user) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]*", dataset_slug
    ):
        raise ValueError("invalid Kaggle username or dataset slug")
    if job_dir.resolve().is_relative_to(ROOT):
        raise ValueError("job directory must be outside repository")
    if job_dir.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("job directory must be outside private corpus")
    if job_dir.exists() and any(job_dir.iterdir()):
        raise ValueError("job directory must be empty")
    verify_attachment(data_dir, manifest_sha256)
    if file_sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("frozen M7 config SHA256 mismatch")
    template = REMOTE.read_text(encoding="utf-8")
    if any(template.count(marker) != 1 for marker in MARKERS):
        raise ValueError("remote wrapper markers differ")
    script = template.replace("__WENDYFM_COMMIT_SHA__", sha).replace(
        "__WENDYFM_MANIFEST_SHA256__", manifest_sha256).replace(
        "__WENDYFM_DATASET_SLUG__", dataset_slug)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / REMOTE.name).write_text(script, encoding="utf-8")
    metadata = {"id": f"{user}/wendyfm-m7-1m-private", "title": "WendyFM M7 1M private",
                "code_file": REMOTE.name, "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [f"{user}/{dataset_slug}"],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def verify_result(path: Path, data_dir: Path, sha: str, manifest_sha256: str) -> dict:
    """Reject partial runs and provenance or metric arithmetic mismatches."""
    if not valid_hash(sha, 40) or not valid_hash(manifest_sha256, 64):
        raise ValueError("invalid expected SHA")
    source = verify_attachment(data_dir, manifest_sha256)
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {"status": "PASS", "test": "m7_1m_private_t4", "retries": 0,
                "expected_commit_sha": sha, "tested_commit_sha": sha,
                "expected_manifest_sha256": manifest_sha256}
    if any(not matches_expected(result.get(key), value) for key, value in expected.items()):
        raise ValueError("result status, retry count, or provenance mismatch")
    plan = result.get("plan")
    if not isinstance(plan, dict) or any(not matches_expected(plan.get(key), value) for key, value in {
        "source_commit_sha": sha, "source_manifest_sha256": manifest_sha256,
        "config_sha256": CONFIG_SHA256, "tokenizer_sha256": TOKENIZER_SHA256,
        "device": "cuda:0", "cuda_device_count": 1, "precision": "fp32",
        "attention_backend": "sdpa", "max_steps": 489, "tokens_per_step": 2048,
        "target_tokens": 1_000_000, "max_wall_seconds": 3600.0,
        "max_cuda_reserved_bytes": 12 * 1024**3,
        "train_artifact_sha256": source["train_artifact_sha256"],
        "validation_artifact_sha256": source["validation_artifact_sha256"],
    }.items()) or "T4" not in str(plan.get("gpu_name", "")):
        raise ValueError("run plan provenance or T4 protocol mismatch")
    events = result.get("events")
    if (not isinstance(events, list) or len(events) != 2 or
            any(not isinstance(e, dict) for e in events) or
            any(type(e.get("step")) is not int for e in events) or
            [e["step"] for e in events] != [0, 489]):
        raise ValueError("missing step-zero or completion event")
    if events[0].get("event") != "initial" or events[1].get("event") != "complete":
        raise ValueError("event sequence mismatch")
    plan_sha256 = hashlib.sha256(json.dumps(plan, sort_keys=True,
                                          allow_nan=False).encode()).hexdigest()
    for event in events:
        if (type(event.get("cumulative_target_tokens")) is not int or
                event["cumulative_target_tokens"] != event["step"] * 2048 or
                not finite_number(event.get("validation_loss"), positive=True)):
            raise ValueError("event token accounting or validation loss mismatch")
        allocated = event.get("peak_cuda_allocated_bytes")
        reserved = event.get("peak_cuda_reserved_bytes")
        if (type(allocated) is not int or type(reserved) is not int or
                not 0 < allocated <= reserved < 12 * 1024**3):
            raise ValueError("CUDA memory gate violated")
        if event.get("plan_sha256") != plan_sha256:
            raise ValueError("event plan hash mismatch")
        for key in ("startup_seconds", "validation_wall_seconds",
                    "checkpoint_wall_seconds", "training_wall_seconds"):
            if not finite_number(event.get(key), positive=key != "startup_seconds" and
                                 (key != "training_wall_seconds" or event["step"] > 0)):
                raise ValueError("event timing missing or nonfinite")
        expected_work = sum(event[key] for key in (
            "startup_seconds", "validation_wall_seconds", "checkpoint_wall_seconds",
            "training_wall_seconds"))
        if event["step"]:
            expected_work += events[0]["work_wall_seconds"]
        if (not finite_number(event.get("work_wall_seconds"), positive=True) or
                not math.isclose(event["work_wall_seconds"], expected_work, rel_tol=1e-6)):
            raise ValueError("event wall-time accounting mismatch")
    final = events[1]
    curve = final.get("train_loss_curve")
    if not isinstance(curve, list) or len(curve) != 489 or any(
        not isinstance(item, dict) or type(item.get("step")) is not int or
        item["step"] != index or any(
            not finite_number(item.get(key), positive=key == "train_loss")
            for key in ("train_loss", "gradient_norm")
        ) for index, item in enumerate(curve, 1)
    ):
        raise ValueError("incomplete or nonfinite training loss curve")
    if (final.get("train_loss_first_batch") != curve[0]["train_loss"] or
            final.get("train_loss_last_batch") != curve[-1]["train_loss"]):
        raise ValueError("training loss endpoints differ from curve")
    timed_tokens = final.get("timed_target_tokens")
    timed_seconds = final.get("timed_seconds")
    throughput = final.get("timed_target_tokens_per_second")
    wall = final.get("work_wall_seconds")
    if (type(timed_tokens) is not int or timed_tokens != (489 - 2) * 2048 or
            not finite_number(timed_seconds, positive=True) or
            timed_seconds > final["training_wall_seconds"] or
            not finite_number(throughput, positive=True) or
            not math.isclose(throughput, timed_tokens / timed_seconds, rel_tol=1e-6) or
            not finite_number(wall, positive=True) or
            not 0 < wall < 3600 or
            not finite_number(final.get("full_run_target_tokens_per_second"), positive=True) or
            not math.isclose(final["full_run_target_tokens_per_second"],
                             1_001_472 / wall, rel_tol=1e-6)):
        raise ValueError("incomplete or inconsistent measured result")
    checkpoint = result.get("checkpoint")
    if (not isinstance(checkpoint, dict) or type(checkpoint.get("step")) is not int or
            checkpoint["step"] != 489 or not valid_hash(
        checkpoint.get("sha256"), 64
    ) or type(checkpoint.get("bytes")) is not int or checkpoint["bytes"] <= 0):
        raise ValueError("checkpoint metadata missing")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for flag in ("job-dir", "data-dir"):
        prepare.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("user", "dataset-slug", "sha", "manifest-sha256"):
        prepare.add_argument(f"--{flag}", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--sha", required=True)
    verify.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_job(args.job_dir, args.data_dir, args.user, args.dataset_slug,
                        args.sha, args.manifest_sha256)
            print(f"Prepared private M7 package: {args.job_dir}")
        else:
            verify_result(args.result, args.data_dir, args.sha, args.manifest_sha256)
            print("PASS: M7 result verified")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
