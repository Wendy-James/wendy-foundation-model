"""Offline package, bounds, hardware mask, and verifier checks for C1."""

import hashlib
import json
import runpy
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_model_scaling.py"))
REMOTE = runpy.run_path(str(LOCAL["REMOTE"]))


def test_private_bounded_single_visible_gpu_package(tmp_path) -> None:
    job = tmp_path / "job"
    sha = "a" * 40
    LOCAL["prepare_job"](job, "wendy_user", sha)
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    script = (job / "remote_model_scaling.py").read_text()
    assert metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True and metadata["enable_tpu"] is False
    assert metadata["dataset_sources"] == metadata["kernel_sources"] == []
    assert f'EXPECTED_COMMIT_SHA = "{sha}"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    assert "CELL_TIMEOUT_SECONDS = 240" in script
    assert "TOTAL_TIMEOUT_SECONDS = 900" in script
    compile(script, "remote_model_scaling.py", "exec")
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, "wendy_user", sha)
    with pytest.raises(ValueError, match="40 lowercase"):
        LOCAL["prepare_job"](tmp_path / "bad", "wendy_user", "bad")


def test_total_deadline_and_failure_stops_before_retry(monkeypatch, tmp_path) -> None:
    assert 0 < REMOTE["remaining_seconds"](time.monotonic() - 1, 240) <= 240
    with pytest.raises(TimeoutError, match="runtime budget"):
        REMOTE["remaining_seconds"](time.monotonic() - 901, 240)
    globals_ = REMOTE["main"].__globals__
    globals_["EXPECTED_COMMIT_SHA"] = "a" * 40
    globals_["RESULT_PATH"] = tmp_path / "result.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    calls = []

    def fake_run(command, **kwargs):
        if "profile_scaling.py" in str(command):
            calls.append((command, kwargs["env"]))
            raise subprocess.TimeoutExpired(command, 240)
        if "rev-parse" in command:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(REMOTE["subprocess"], "run", fake_run)
    with pytest.raises(SystemExit) as error:
        REMOTE["main"]()
    assert error.value.code == 1
    assert len(calls) == 1
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[0][1]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    record = json.loads(globals_["RESULT_PATH"].read_text())
    assert record["failed_cell"] == 1
    assert record["error_type"] == "TimeoutExpired"
    assert record["retries"] == 0


def test_verifier_rejects_bad_wrapper_and_wrong_cell(tmp_path) -> None:
    sha = "a" * 40
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "FAIL", "cells": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="wrapper status"):
        LOCAL["verify_result"](path, sha)
    wrapper = {"status": "PASS", "test": "m5_phase_c1_fp32",
               "expected_commit_sha": sha, "tested_commit_sha": sha, "retries": 0,
               "cells": [{"cell": 2, "model": LOCAL["MODELS"][0],
                          "parameter_count": LOCAL["parameter_counts"]()[0]}] * 3}
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="cell model"):
        LOCAL["verify_result"](path, sha)


def test_verifier_checks_three_cells_metrics_and_provenance(tmp_path) -> None:
    sha = "b" * 40
    config = LOCAL["CONFIG"]
    settings = LOCAL["load_sweep_config"](config)
    train_path = (config.parent / settings["train_text"]).resolve()
    tokenizer = LOCAL["BPETokenizer"].train(LOCAL["_documents"](train_path), vocab_size=272,
                                             special_tokens=(settings["end_of_document"],))
    tokenizer_bytes = (json.dumps(tokenizer.spec.serialization_dict(), sort_keys=True,
                                  ensure_ascii=False) + "\n").encode()
    cells = []
    for index, (model, count) in enumerate(zip(LOCAL["MODELS"],
                                                LOCAL["parameter_counts"](), strict=True), 1):
        repeats = []
        for repeat in (1, 2, 3):
            elapsed = 2.0 + repeat * 0.02
            repeats.append({"status": "PASS", "repeat": repeat, "warmup_steps": 5,
                            "measured_steps": 30, "target_tokens": 1920,
                            "measured_run_changed_parameter": True,
                            "elapsed_seconds": elapsed, "mean_step_seconds": elapsed / 30,
                            "target_tokens_per_second": 1920 / elapsed,
                            "allocated_baseline_bytes": 50, "reserved_baseline_bytes": 100,
                            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                            "loss_min": 2.0, "loss_max": 2.0,
                            "steps": [{"step": step, "loss": 2.0, "gradient_norm": 1.0}
                                      for step in range(6, 36)]})
        rates = sorted(item["target_tokens_per_second"] for item in repeats)
        cells.append({"status": "PASS", "cell": index, "model": model,
                      "source_commit_sha": sha, "precision": "fp32", "device": "cuda:0",
                      "cuda_device_count": 1, "batch_size": 4, "sequence_length": 16,
                      "warmup_steps": 5, "measured_steps": 30, "requested_repeats": 3,
                      "retries": 0,
                      "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                      "train_text_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                      "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
                      "gpu_name": "Tesla T4", "torch_version": "2.5.1",
                      "cuda_version": "12.1", "driver_version": "550.0",
                      "gpu_total_memory_bytes": 16_000_000_000,
                      "gpu_power_limit_watts": 70.0, "parameter_count": count,
                      "median_target_tokens_per_second": rates[1],
                      "throughput_spread": (rates[2] - rates[0]) / rates[1],
                      "unstable": False, "repeats": repeats})
    wrapper = {"status": "PASS", "test": "m5_phase_c1_fp32",
               "expected_commit_sha": sha, "tested_commit_sha": sha,
               "retries": 0, "cells": cells}
    path = tmp_path / "valid.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    assert LOCAL["verify_result"](path, sha) == wrapper
    wrapper["cells"][1]["repeats"][0]["target_tokens_per_second"] = -1
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid target_tokens_per_second"):
        LOCAL["verify_result"](path, sha)
    wrapper["cells"][1]["repeats"][0]["target_tokens_per_second"] = 1920 / 2.02
    wrapper["cells"][2]["tokenizer_sha256"] = "0" * 64
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer hash"):
        LOCAL["verify_result"](path, sha)
