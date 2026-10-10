"""Offline verification of private, SHA-pinned M5 Kaggle packaging."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_gpu_profile.py"))


def test_private_single_gpu_package(tmp_path) -> None:
    sha = "a" * 40
    job = tmp_path / "job"
    LOCAL["prepare_job"](job, "wendy_user", sha)
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    script = (job / "remote_gpu_profile.py").read_text()
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["dataset_sources"] == []
    assert metadata["code_file"] == "remote_gpu_profile.py"
    assert f'EXPECTED_COMMIT_SHA = "{sha}"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    assert "timeout=300" in script
    compile(script, "remote_gpu_profile.py", "exec")
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, "wendy_user", sha)
    with pytest.raises(ValueError, match="40 lowercase"):
        LOCAL["prepare_job"](tmp_path / "other", "wendy_user", "bad")


def test_result_rejects_incomplete_or_mismatched_repeats(tmp_path) -> None:
    sha = "b" * 40
    config_hash = LOCAL["hashlib"].sha256(LOCAL["CONFIG"].read_bytes()).hexdigest()
    repeats = [{"status": "PASS", "repeat": i, "warmup_steps": 5,
                "measured_steps": 30, "target_tokens": 1920,
                "last_update_changed_parameter": True, "elapsed_seconds": 2.0,
                "mean_step_seconds": 2 / 30, "target_tokens_per_second": 960.0,
                "allocated_baseline_bytes": 50, "reserved_baseline_bytes": 100,
                "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                "steps": [{"step": step, "loss": 2.0, "gradient_norm": 1.0}
                          for step in range(6, 36)]} for i in range(1, 4)]
    manifest = {"status": "PASS", "source_commit_sha": sha, "precision": "fp32",
                "device": "cuda:0", "batch_size": 4, "sequence_length": 16,
                "warmup_steps": 5, "measured_steps": 30, "requested_repeats": 3,
                "retries": 0, "config_sha256": config_hash, "gpu_name": "Tesla T4",
                "repeats": repeats}
    wrapper = {"status": "PASS", "expected_commit_sha": sha,
               "tested_commit_sha": sha, "test": "m5_phase_a_fp32", "manifest": manifest}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    assert LOCAL["verify_result"](path, sha) == manifest
    wrapper["manifest"]["repeats"][2]["measured_steps"] = 29
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        LOCAL["verify_result"](path, sha)
