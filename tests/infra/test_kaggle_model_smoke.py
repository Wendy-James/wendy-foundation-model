"""Offline packaging and validation checks for the Kaggle model job."""

import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_model_smoke.py"))
REMOTE = runpy.run_path(str(ROOT / "scripts/infra/remote_model_smoke.py"))


def test_prepare_private_gpu_job_with_pinned_source(tmp_path):
    sha = "a" * 40
    job = tmp_path / "job"
    LOCAL["prepare_job"](job, "wendy_user", sha)
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    script = (job / "remote_model_smoke.py").read_text()
    assert metadata["id"] == "wendy_user/wendyfm-model-smoke"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == []
    assert metadata["code_file"] == "remote_model_smoke.py"
    assert f'EXPECTED_COMMIT_SHA = "{sha}"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    compile(script, str(job / "remote_model_smoke.py"), "exec")
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, "wendy_user", sha)


@pytest.mark.parametrize("sha", ["a" * 39, "A" * 40, "a" * 41, "x" * 40])
def test_invalid_sha_rejected_before_packaging(tmp_path, sha):
    with pytest.raises(ValueError, match="40 lowercase"):
        LOCAL["prepare_job"](tmp_path / "job", "wendy_user", sha)
    assert not (tmp_path / "job").exists()


def test_checkout_verification_before_model_import(monkeypatch, tmp_path):
    sha = "b" * 40
    command = Mock(return_value=Mock(stdout=sha + "\n"))
    monkeypatch.setattr(REMOTE["subprocess"], "run", command)
    assert REMOTE["verify_checkout"](tmp_path, sha) == sha
    command.return_value.stdout = "c" * 40 + "\n"
    with pytest.raises(ValueError, match="differs"):
        REMOTE["verify_checkout"](tmp_path, sha)
    with pytest.raises(ValueError, match="invalid"):
        REMOTE["verify_checkout"](tmp_path, "bad")


def test_result_validation(tmp_path):
    sha = "d" * 40
    path = tmp_path / "result.json"
    result = {
        "status": "PASS", "expected_commit_sha": sha, "tested_commit_sha": sha,
        "test": "model_cuda_adamw", "loss": 3.2, "cuda_device": "Tesla T4",
        "step_time_s": 0.1, "peak_gpu_memory_bytes": 123456,
    }
    path.write_text(json.dumps(result))
    LOCAL["verify_result"](path, sha)
    for key, bad in [("tested_commit_sha", "e" * 40), ("loss", float("nan")),
                     ("cuda_device", ""), ("step_time_s", 0),
                     ("peak_gpu_memory_bytes", -1)]:
        invalid = {**result, key: bad}
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            LOCAL["verify_result"](path, sha)
