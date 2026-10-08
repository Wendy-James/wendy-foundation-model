"""Local checks for job preparation, result validation, and tokenizer smoke logic."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/kaggle_smoke.py"))
REMOTE = runpy.run_path(str(ROOT / "scripts/infra/remote_tokenizer_smoke.py"))


def test_prepared_job_is_private_cpu_only_and_pinned(tmp_path):
    sha = "a" * 40
    job_dir = tmp_path / "job"
    LOCAL["prepare_job"](job_dir, "wendy_user", sha)
    metadata = json.loads((job_dir / "kernel-metadata.json").read_text())
    script = (job_dir / "remote_tokenizer_smoke.py").read_text()
    assert metadata["id"] == "wendy_user/wendyfm-tokenizer-smoke"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["code_file"] == "remote_tokenizer_smoke.py"
    assert f'EXPECTED_COMMIT_SHA = "{sha}"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    compile(script, str(job_dir / "remote_tokenizer_smoke.py"), "exec")


def test_local_tokenizer_smoke_logic():
    result = REMOTE["run_tokenizer_check"](ROOT)
    assert result["test"] == "tokenizer_round_trip"
    assert result["token_count"] > 0
    assert result["round_trip"] is True
    assert result["deterministic"] is True


def test_verifier_rejects_failure_or_wrong_commit(tmp_path):
    sha = "b" * 40
    path = tmp_path / "result.json"
    result = {
        "status": "PASS",
        "expected_commit_sha": sha,
        "tested_commit_sha": sha,
        "test": "tokenizer_round_trip",
        "token_count": 2,
        "round_trip": True,
        "deterministic": True,
    }
    path.write_text(json.dumps(result))
    LOCAL["verify_result"](path, sha)
    result["tested_commit_sha"] = "c" * 40
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        LOCAL["verify_result"](path, sha)
    result["tested_commit_sha"] = sha
    result["status"] = "FAIL"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        LOCAL["verify_result"](path, sha)
