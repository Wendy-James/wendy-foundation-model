"""Offline verification of private, SHA-pinned M5 Kaggle packaging."""

import copy
import json
import runpy
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

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
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == []
    assert metadata["code_file"] == "remote_gpu_profile.py"
    assert f'EXPECTED_COMMIT_SHA = "{sha}"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    assert "TOTAL_TIMEOUT_SECONDS = 300" in script
    assert "remaining_seconds(started, TOTAL_TIMEOUT_SECONDS)" in script
    compile(script, "remote_gpu_profile.py", "exec")
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, "wendy_user", sha)
    with pytest.raises(ValueError, match="40 lowercase"):
        LOCAL["prepare_job"](tmp_path / "other", "wendy_user", "bad")


def test_remote_wrapper_deadline_is_total_and_bounded() -> None:
    remote = runpy.run_path(str(LOCAL["REMOTE"]))
    remaining = remote["remaining_seconds"]
    assert 0 < remaining(time.monotonic() - 1, 60) <= 60
    with pytest.raises(TimeoutError, match="runtime budget"):
        remaining(time.monotonic() - 301, 10)


def test_remote_masks_profiler_child_before_start(monkeypatch, tmp_path) -> None:
    remote = runpy.run_path(str(LOCAL["REMOTE"]))
    globals_ = remote["main"].__globals__
    globals_["EXPECTED_COMMIT_SHA"] = "a" * 40
    globals_["RESULT_PATH"] = tmp_path / "result.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    calls = []

    def fake_run(command, **kwargs):
        if "profile_gpu.py" in str(command):
            calls.append(kwargs["env"])
            raise subprocess.CalledProcessError(1, command)
        if "rev-parse" in command:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(remote["subprocess"], "run", fake_run)
    with pytest.raises(SystemExit) as error:
        remote["main"]()
    assert error.value.code == 1
    assert len(calls) == 1
    assert calls[0]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[0]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert json.loads(globals_["RESULT_PATH"].read_text())["status"] == "FAIL"


def test_result_rejects_incomplete_or_mismatched_repeats(tmp_path) -> None:
    sha = "b" * 40
    config_hash = LOCAL["hashlib"].sha256(LOCAL["CONFIG"].read_bytes()).hexdigest()
    settings = json.loads(LOCAL["CONFIG"].read_text())
    train_text = (LOCAL["CONFIG"].parent / settings["train_text"]).resolve()
    train_hash = LOCAL["hashlib"].sha256(train_text.read_bytes()).hexdigest()
    repeats = [{"status": "PASS", "repeat": i, "warmup_steps": 5,
                "measured_steps": 30, "target_tokens": 1920,
                "measured_run_changed_parameter": True, "elapsed_seconds": 2.0,
                "mean_step_seconds": 2 / 30, "target_tokens_per_second": 960.0,
                "allocated_baseline_bytes": 50, "reserved_baseline_bytes": 100,
                "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                "loss_min": 2.0, "loss_max": 2.0,
                "steps": [{"step": step, "loss": 2.0, "gradient_norm": 1.0}
                          for step in range(6, 36)]} for i in range(1, 4)]
    manifest = {"status": "PASS", "source_commit_sha": sha, "precision": "fp32",
                "device": "cuda:0", "cuda_device_count": 1,
                "batch_size": 4, "sequence_length": 16,
                "warmup_steps": 5, "measured_steps": 30, "requested_repeats": 3,
                "retries": 0, "config_sha256": config_hash, "gpu_name": "Tesla T4",
                "train_text_sha256": train_hash, "torch_version": "2.5.1",
                "cuda_version": "12.1", "driver_version": "550.0",
                "gpu_total_memory_bytes": 16_000_000_000,
                "gpu_power_limit_watts": 70.0, "parameter_count": 100_000,
                "median_target_tokens_per_second": 960.0,
                "throughput_spread": 0.0, "unstable": False,
                "repeats": repeats}
    wrapper = {"status": "PASS", "expected_commit_sha": sha,
               "tested_commit_sha": sha, "test": "m5_phase_a_fp32", "manifest": manifest}
    valid_wrapper = copy.deepcopy(wrapper)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    assert LOCAL["verify_result"](path, sha) == manifest
    wrapper["manifest"]["repeats"][2]["measured_steps"] = 29
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        LOCAL["verify_result"](path, sha)
    for mutate, message in (
        (lambda data: data["manifest"].update(train_text_sha256="bad"), "provenance"),
        (lambda data: data["manifest"].update(gpu_name="NVIDIA L4"), "Tesla T4"),
        (lambda data: data["manifest"].update(cuda_device_count=2), "provenance"),
        (lambda data: data["manifest"].update(cuda_version=""), "identity"),
        (lambda data: data["manifest"]["repeats"][0].update(mean_step_seconds=1.0),
         "timing"),
        (lambda data: data["manifest"]["repeats"][0]["steps"][0].update(gradient_norm=-1),
         "observation"),
        (lambda data: data["manifest"].update(throughput_spread=0.2), "aggregation"),
    ):
        changed = copy.deepcopy(valid_wrapper)
        mutate(changed)
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            LOCAL["verify_result"](path, sha)
