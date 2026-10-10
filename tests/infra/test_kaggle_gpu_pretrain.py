"""Offline checks for the private, SHA-pinned FP32 pretraining job."""

import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_gpu_pretrain.py"))
REMOTE = runpy.run_path(str(ROOT / "scripts/infra/remote_gpu_pretrain.py"))


def test_private_bounded_job_uses_pinned_source(tmp_path) -> None:
    sha = "a" * 40
    job = tmp_path / "job"
    LOCAL["prepare_job"](job, "wendy_user", sha)
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    script = (job / "remote_gpu_pretrain.py").read_text()
    assert metadata["id"] == "wendy_user/wendyfm-fp32-pretrain"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["dataset_sources"] == []
    assert metadata["code_file"] == "remote_gpu_pretrain.py"
    assert f'EXPECTED_COMMIT_SHA = "{sha}"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    assert '"--end-step", "4"' in script
    compile(script, str(job / "remote_gpu_pretrain.py"), "exec")
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, "wendy_user", sha)


def test_invalid_pin_and_mismatched_checkout_rejected(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="40 lowercase"):
        LOCAL["prepare_job"](tmp_path / "job", "wendy_user", "bad")
    command = Mock(return_value=Mock(stdout="b" * 40 + "\n"))
    monkeypatch.setattr(REMOTE["subprocess"], "run", command)
    assert REMOTE["verify_checkout"](tmp_path, "b" * 40) == "b" * 40
    with pytest.raises(ValueError, match="differs"):
        REMOTE["verify_checkout"](tmp_path, "c" * 40)


def test_result_verifier_requires_measured_fp32_resume_evidence(tmp_path) -> None:
    sha = "d" * 40
    path = tmp_path / "result.json"
    manifest = {
        "source_commit_sha": sha, "device": "cuda:0", "precision": "fp32",
        "optimizer_steps": 8, "segment_steps": 8, "timed_target_tokens": 512,
        "train_loss_first_batch": 5.6, "train_loss_last_batch": 5.1,
        "initial_validation_loss": 5.6, "validation_loss": 5.2,
        "tokens_per_second": 100.0, "training_wall_time_seconds": 5.12,
        "peak_cuda_memory_bytes": 123456, "cuda_device_name": "Tesla T4",
    }
    result = {
        "status": "PASS", "expected_commit_sha": sha, "tested_commit_sha": sha,
        "test": "fp32_single_gpu_pretrain", "checkpoint_resume_exact": True,
        "manifest": manifest,
    }
    path.write_text(json.dumps(result), encoding="utf-8")
    LOCAL["verify_result"](path, sha)
    for invalid in (
        {**result, "tested_commit_sha": "e" * 40},
        {**result, "checkpoint_resume_exact": False},
        {**result, "manifest": {**manifest, "tokens_per_second": float("nan")}},
        {**result, "manifest": {**manifest, "peak_cuda_memory_bytes": 0}},
        {**result, "manifest": {**manifest, "source_commit_sha": "e" * 40}},
    ):
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError):
            LOCAL["verify_result"](path, sha)
