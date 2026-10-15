"""One private, SHA-pinned M7 long-run T4 Kaggle job."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Kaggle's T4 shape exposes two GPUs. Mask before Torch is ever imported.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EXPECTED_COMMIT_SHA = "__WENDYFM_COMMIT_SHA__"
EXPECTED_MANIFEST_SHA256 = "__WENDYFM_MANIFEST_SHA256__"
DATASET_SLUG = "__WENDYFM_DATASET_SLUG__"
PUBLIC_REPO_URL = "https://github.com/Wendy-James/wendy-foundation-model.git"
CONFIG_SHA256 = "71c39117cd56105f892470cd5a20d823c78c64ccdaf88b86721963d9cc523ad4"
TOKENIZER_SHA256 = "c518f4ee754fb30aed2497fd366e4a795f780572e0e1a463ea8f4c01b41a952f"
RESULT_PATH = Path("/kaggle/working/m7-result.json")
TOTAL_TIMEOUT_SECONDS = 3600
RESULT_MARGIN_SECONDS = 60
FILES = frozenset(("manifest.json", "tokenizer.json", "train.jsonl", "validation.jsonl"))
ARTIFACTS = {"train.jsonl": "train_artifact_sha256",
             "validation.jsonl": "validation_artifact_sha256",
             "tokenizer.json": "tokenizer_sha256"}


def sha256(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("M7 job deadline reached during hash verification")
            digest.update(chunk)
    return digest.hexdigest()


def remaining(deadline: float, cap: int | None = None) -> float:
    available = deadline - time.monotonic()
    if available <= 0:
        raise TimeoutError("M7 job deadline reached")
    return min(available, cap) if cap is not None else available


def verify_attachment(data_dir: Path, deadline: float) -> dict:
    """Check frozen manifest and all attachment bytes before CUDA initialization."""
    if not data_dir.is_dir() or {p.name for p in data_dir.iterdir()} != FILES or any(
        not (data_dir / name).is_file() or (data_dir / name).is_symlink() for name in FILES
    ):
        raise ValueError("private attachment must contain exactly four regular files")
    if sha256(data_dir / "manifest.json", deadline) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest SHA256 mismatch")
    source = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("invalid corpus manifest")
    for name, key in ARTIFACTS.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get(key))) or sha256(
            data_dir / name, deadline
        ) != source[key]:
            raise ValueError(f"{name} SHA256 mismatch")
    if source.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("tokenizer differs from frozen M6 artifact")
    if (type(source.get("train_tokens_with_eod")) is not int or
            type(source.get("validation_tokens_with_eod")) is not int or
            source["train_tokens_with_eod"] < 20_000_000 or
            source["validation_tokens_with_eod"] < 200_000):
        raise ValueError("M7 corpus below required token floors")
    return source


def run_command(command: list[str], *, deadline: float, cap: int | None = None,
                cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """No retry; kill the whole child process group if the deadline is reached."""
    timeout = remaining(deadline, cap)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        code = process.wait(timeout=min(timeout, remaining(deadline)))
    except (subprocess.TimeoutExpired, TimeoutError) as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise TimeoutError("M7 child process exceeded its time limit") from error
    if code != 0:
        raise RuntimeError(f"M7 child process exited with status {code}")
    return subprocess.CompletedProcess(command, code)


def compact_error(error: Exception) -> str:
    message = str(error)
    if (not message or len(message) > 200 or
            any(character in message for character in ("/", "\\", "\n", "\r"))):
        return "see private local traceback"
    return message


def collect_result(output_dir: Path, deadline: float) -> tuple[dict, list[dict], dict]:
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    events = [json.loads(path.read_text(encoding="utf-8")) for path in
              sorted((output_dir / "events").glob("step_*.json"))]
    checkpoint = output_dir / "checkpoint_00000489.pt"
    if not checkpoint.is_file():
        raise ValueError("completed checkpoint missing")
    metadata = {"step": 489, "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint, deadline)}
    return plan, events, metadata


def main() -> None:
    started = time.monotonic()
    deadline = started + TOTAL_TIMEOUT_SECONDS - RESULT_MARGIN_SECONDS
    record = {"status": "FAIL", "test": "m7_1m_private_t4", "retries": 0,
              "expected_commit_sha": EXPECTED_COMMIT_SHA, "tested_commit_sha": None,
              "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256}
    stage = "validate_pin"
    try:
        if (not re.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT_SHA) or
                not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_MANIFEST_SHA256) or
                not re.fullmatch(r"[a-z0-9][a-z0-9-]*", DATASET_SLUG)):
            raise ValueError("invalid executable, manifest or dataset pin")
        with tempfile.TemporaryDirectory(prefix="wendyfm-m7-") as directory:
            workspace = Path(directory)
            repo = workspace / "wendyfm"
            stage = "clone_source"
            run_command(["git", "-c", "credential.helper=", "clone", "--quiet",
                         PUBLIC_REPO_URL, str(repo)], deadline=deadline, cap=120)
            stage = "checkout_source"
            run_command(["git", "-C", str(repo), "checkout", "--quiet", "--detach",
                         EXPECTED_COMMIT_SHA], deadline=deadline, cap=30)
            actual = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True,
                                    timeout=remaining(deadline, 10)).stdout.strip()
            if actual != EXPECTED_COMMIT_SHA:
                raise ValueError("checkout SHA mismatch")
            record["tested_commit_sha"] = actual
            stage = "verify_wrapper"
            template = (repo / "scripts/infra/remote_m7.py").read_text(encoding="utf-8")
            expected = template.replace("__WENDYFM_" + "COMMIT_SHA__", EXPECTED_COMMIT_SHA)
            expected = expected.replace("__WENDYFM_" + "MANIFEST_SHA256__",
                                        EXPECTED_MANIFEST_SHA256)
            expected = expected.replace("__WENDYFM_" + "DATASET_SLUG__", DATASET_SLUG)
            if Path(__file__).read_text(encoding="utf-8") != expected:
                raise ValueError("packaged wrapper differs from pinned executable")
            config = repo / "configs/m7_t4_1m.json"
            stage = "verify_config"
            if sha256(config, deadline) != CONFIG_SHA256:
                raise ValueError("frozen M7 config SHA256 mismatch")
            settings = json.loads(config.read_text(encoding="utf-8"))
            if (settings["precision"] != "fp32" or
                    settings["model"]["attention_backend"] != "sdpa" or
                    settings["train"]["max_steps"] != 489 or
                    settings["long_run"]["target_tokens"] != 1_000_000 or
                    settings["long_run"]["max_wall_seconds"] != 3600):
                raise ValueError("frozen M7 T4 protocol mismatch")
            stage = "verify_attachment"
            data_dir = Path("/kaggle/input") / DATASET_SLUG
            source = verify_attachment(data_dir, deadline)
            # Torch is first imported only after all source and data hash gates pass.
            stage = "cuda_preflight"
            import torch

            if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
                    "T4" not in torch.cuda.get_device_name(0)):
                raise RuntimeError("M7 requires one visible NVIDIA T4")
            stage = "training"
            output = workspace / "run"
            env = os.environ.copy()
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
                       PYTHONPATH=str(repo / "src"))
            run_command([sys.executable, str(repo / "scripts/pretrain_long.py"),
                         "--config", str(config), "--data-dir", str(data_dir),
                         "--output-dir", str(output), "--manifest-sha256",
                         EXPECTED_MANIFEST_SHA256, "--sha", EXPECTED_COMMIT_SHA,
                         "--device", "cuda"], deadline=deadline, cwd=repo, env=env)
            stage = "collect_result"
            plan, events, checkpoint = collect_result(output, deadline)
            if (plan.get("source_commit_sha") != actual or
                    plan.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256 or
                    plan.get("config_sha256") != CONFIG_SHA256 or
                    plan.get("tokenizer_sha256") != TOKENIZER_SHA256 or
                    plan.get("train_artifact_sha256") != source["train_artifact_sha256"] or
                    plan.get("validation_artifact_sha256") !=
                    source["validation_artifact_sha256"] or
                    plan.get("precision") != "fp32" or
                    plan.get("attention_backend") != "sdpa" or
                    plan.get("max_steps") != 489 or
                    plan.get("tokens_per_step") != 2048 or
                    plan.get("cuda_device_count") != 1 or
                    "T4" not in str(plan.get("gpu_name")) or
                    [event.get("step") for event in events] != [0, 489] or
                    events[-1].get("event") != "complete" or
                    events[-1].get("cumulative_target_tokens") != 1_001_472 or
                    any(event.get("peak_cuda_reserved_bytes", 12 * 1024**3) >=
                        12 * 1024**3 for event in events)):
                raise ValueError("runner provenance or completion mismatch")
            record.update(status="PASS", plan=plan, events=events, checkpoint=checkpoint)
    except Exception as error:
        record.update(error_type=type(error).__name__, error_message=compact_error(error),
                      failure_stage=stage)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "failure_stage": record.get("failure_stage")}))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
