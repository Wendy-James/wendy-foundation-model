"""Offline checks for frozen packaging and strict three-arm acceptance."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_m6_pilot.py"))


def test_private_package_requires_exact_frozen_bytes(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setitem(globals_, "MANIFEST", manifest)
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


@pytest.mark.parametrize("arm_index", range(3))
@pytest.mark.parametrize("bad_evidence", (
    "missing", "missing_boolean", "false", "zero_changed", "unchanged", "zero_delta",
))
def test_verifier_rejects_missing_or_false_update_evidence(
    tmp_path, arm_index, bad_evidence,
) -> None:
    sha = "c" * 40
    source = json.loads(LOCAL["MANIFEST"].read_text())
    common = {
        "source_commit_sha": sha, "device": "cuda:0", "cuda_device_count": 1,
        "config_sha256": LOCAL["digest"](LOCAL["CONFIG"]),
        "source_manifest_sha256": LOCAL["digest"](LOCAL["MANIFEST"]),
        "train_artifact_sha256": source["train_artifact_sha256"],
        "validation_artifact_sha256": source["validation_artifact_sha256"],
        "tokenizer_sha256": source["tokenizer_sha256"],
        "trainable_parameters": 21_502_464, "microbatch_size": 2,
        "gradient_accumulation_steps": 4, "effective_batch_size": 8,
        "sequence_length": 256, "start_step": 0, "optimizer_steps": 20,
        "measured_steps": 18, "timed_target_tokens": 36_864,
        "timed_seconds": 18.0, "mean_step_seconds": 1.0,
        "tokens_per_second": 2048.0, "wall_seconds": 25.0,
        "training_wall_seconds": 20.0, "sdpa_kernel_observed": None,
        "checkpoint": "checkpoint.pt", "torch_version": "2.5.1",
        "cuda_version": "12.1", "gpu_name": "Tesla T4",
        "train_loss_first_batch": 7.0, "train_loss_last_batch": 6.5,
        "initial_validation_loss": 7.1, "validation_loss": 6.8,
        "peak_cuda_allocated_bytes": 100, "peak_cuda_reserved_bytes": 200,
    }
    arms = []
    for label, backend, precision in LOCAL["ARMS"]:
        arms.append({**common, "arm": label, "attention_backend_requested": backend,
                     "precision": precision, "parameters_updated": True,
                     "parameter_update_witness": {
                         "parameter_name": "final_norm.weight", "elements_compared": 512,
                         "changed_elements": 1, "max_abs_delta": 0.01,
                         "before_sha256": "a" * 64, "after_sha256": "b" * 64}})
    result = {"status": "PASS", "test": "m6_b_realdata_pilot", "retries": 0,
              "expected_commit_sha": sha, "tested_commit_sha": sha,
              "failed_arm": None, "arms": arms}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert LOCAL["verify_result"](path, sha) == result
    target = arms[arm_index]
    if bad_evidence == "missing":
        del target["parameter_update_witness"]
    elif bad_evidence == "missing_boolean":
        del target["parameters_updated"]
    elif bad_evidence == "false":
        target["parameters_updated"] = False
    elif bad_evidence == "zero_changed":
        target["parameter_update_witness"]["changed_elements"] = 0
    elif bad_evidence == "unchanged":
        target["parameter_update_witness"]["after_sha256"] = "a" * 64
    else:
        target["parameter_update_witness"]["max_abs_delta"] = 0.0
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="parameter-update evidence|parameter max_abs_delta"):
        LOCAL["verify_result"](path, sha)
