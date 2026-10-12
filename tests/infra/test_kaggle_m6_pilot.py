"""Offline checks for frozen packaging and strict three-arm acceptance."""

import base64
import hashlib
import json
import runpy
import zlib
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
    assert sorted(path.name for path in job.iterdir()) == [
        "kernel-metadata.json", "remote_m6_pilot.py"]
    script = (job / "remote_m6_pilot.py").read_text()
    assert 'EXPECTED_COMMIT_SHA = "' + "a" * 40 + '"' in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    assert "__WENDYFM_DATA_PAYLOAD__" not in script
    assert 'Path(__file__).resolve().parent / "data"' not in script
    compile(script, "remote_m6_pilot.py", "exec")
    remote = runpy.run_path(str(job / "remote_m6_pilot.py"))
    materialized = tmp_path / "materialized"
    remote["materialize_artifacts"](materialized, hashes, remote["PACKAGED_DATA"])
    assert LOCAL["verify_artifacts"](materialized) == hashes
    assert all((materialized / name).read_bytes() == (data / name).read_bytes()
               for name in remote["PACKAGED_DATA"])
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, data, "wendy_user", "a" * 40)
    (data / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="train.jsonl hash"):
        LOCAL["prepare_job"](tmp_path / "bad", data, "wendy_user", "a" * 40)


def test_packaged_artifacts_are_deterministic_and_reject_tampering(tmp_path, monkeypatch) -> None:
    data = tmp_path / "source"
    data.mkdir()
    hashes = {}
    for name, key in (("train.jsonl", "train_artifact_sha256"),
                      ("validation.jsonl", "validation_artifact_sha256"),
                      ("tokenizer.json", "tokenizer_sha256")):
        raw = (name + "\n").encode()
        (data / name).write_bytes(raw)
        hashes[key] = hashlib.sha256(raw).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(hashes))
    monkeypatch.setitem(LOCAL["prepare_job"].__globals__, "MANIFEST", manifest)
    jobs = [tmp_path / "job1", tmp_path / "job2"]
    for job in jobs:
        LOCAL["prepare_job"](job, data, "wendy_user", "a" * 40)
    assert (jobs[0] / "remote_m6_pilot.py").read_bytes() == (
        jobs[1] / "remote_m6_pilot.py").read_bytes()
    remote = runpy.run_path(str(jobs[0] / "remote_m6_pilot.py"))
    materialize = remote["materialize_artifacts"]
    payload = remote["PACKAGED_DATA"]
    missing = {key: value for key, value in payload.items() if key != "train.jsonl"}
    with pytest.raises(ValueError, match="missing or unexpected"):
        materialize(tmp_path / "missing", hashes, missing)
    corrupt = {**payload, "train.jsonl": "not base64!"}
    with pytest.raises(ValueError, match="payload corrupt"):
        materialize(tmp_path / "corrupt", hashes, corrupt)
    truncated = {**payload, "train.jsonl": base64.b64encode(
        base64.b64decode(payload["train.jsonl"])[:-1]).decode("ascii")}
    with pytest.raises(ValueError, match="payload corrupt"):
        materialize(tmp_path / "truncated", hashes, truncated)
    tampered = {**payload, "train.jsonl": base64.b64encode(
        zlib.compress(b"modified\n", level=9)).decode("ascii")}
    with pytest.raises(ValueError, match="provenance mismatch"):
        materialize(tmp_path / "tampered", hashes, tampered)


def test_failure_record_has_message_and_pre_arm_stage(tmp_path, monkeypatch) -> None:
    source = (ROOT / "scripts/infra/remote_m6_pilot.py").read_text()
    script = tmp_path / "remote_m6_pilot.py"
    script.write_text(source.replace("__WENDYFM_COMMIT_SHA__", "a" * 40).replace(
        "__WENDYFM_DATA_PAYLOAD__", "{}"))
    remote = runpy.run_path(str(script))
    globals_ = remote["main"].__globals__
    output = tmp_path / "result.json"
    monkeypatch.setitem(globals_, "RESULT_PATH", output)

    def fail_clone(*args, **kwargs):
        raise FileNotFoundError("source clone unavailable")

    monkeypatch.setattr(globals_["subprocess"], "run", fail_clone)
    with pytest.raises(SystemExit, match="1"):
        remote["main"]()
    result = json.loads(output.read_text())
    assert result["status"] == "FAIL"
    assert result["failed_arm"] is None
    assert result["failure_stage"] == "clone_source"
    assert result["error_type"] == "FileNotFoundError"
    assert result["error_message"] == "source clone unavailable"


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
