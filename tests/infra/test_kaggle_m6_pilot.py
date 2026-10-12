"""Offline checks for frozen packaging and strict three-arm acceptance."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_m6_pilot.py"))


def test_private_package_requires_exact_frozen_bytes(tmp_path) -> None:
    data = tmp_path / "source"
    data.mkdir()
    hashes = {}
    for name, key in (("train.jsonl", "train_artifact_sha256"),
                      ("validation.jsonl", "validation_artifact_sha256"),
                      ("tokenizer.json", "tokenizer_sha256")):
        content = name.encode()
        (data / name).write_bytes(content)
        hashes[key] = hashlib.sha256(content).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(hashes))
    globals_ = LOCAL["prepare_job"].__globals__
    globals_["MANIFEST"] = manifest
    job = tmp_path / "job"
    LOCAL["prepare_job"](job, data, "wendy_user", "a" * 40)
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] and not metadata["enable_tpu"]
    assert metadata["dataset_sources"] == []
    assert LOCAL["verify_artifacts"](job / "data") == hashes
    script = (job / "remote_m6_pilot.py").read_text()
    assert 'EXPECTED_COMMIT_SHA = "' + "a" * 40 + '"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    compile(script, "remote_m6_pilot.py", "exec")
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, data, "wendy_user", "a" * 40)
    (data / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="train.jsonl hash"):
        LOCAL["prepare_job"](tmp_path / "bad", data, "wendy_user", "a" * 40)


def test_verifier_rejects_partial_and_wrong_mode(tmp_path) -> None:
    sha = "b" * 40
    result = {"status": "PASS", "test": "m6_b_realdata_pilot", "retries": 0,
              "expected_commit_sha": sha, "tested_commit_sha": sha,
              "failed_arm": None, "arms": []}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="three completed"):
        LOCAL["verify_result"](path, sha)
    result["arms"] = [{"arm": "A"}] * 3
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="provenance, mode, or budget"):
        LOCAL["verify_result"](path, sha)
