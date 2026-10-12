"""One private, bounded, sequential M6 real-data Kaggle experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

EXPECTED_COMMIT_SHA = "__WENDYFM_COMMIT_SHA__"
PUBLIC_REPO_URL = "https://github.com/Wendy-James/wendy-foundation-model.git"
RESULT_PATH = Path("/kaggle/working/m6-pilot-result.json")
TOTAL_TIMEOUT_SECONDS = 3600
ARM_TIMEOUT_SECONDS = 900
MEMORY_LIMIT_BYTES = 12 * 1024**3
ARMS = (("A", "reference", "fp32"), ("B", "sdpa", "fp32"),
        ("C", "sdpa", "fp16"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remaining(started: float, cap: int) -> float:
    available = TOTAL_TIMEOUT_SECONDS - (time.monotonic() - started)
    if available <= 0:
        raise TimeoutError("M6 pilot exceeded its total runtime limit")
    return min(available, cap)


def main() -> None:
    started = time.monotonic()
    record = {"status": "FAIL", "test": "m6_b_realdata_pilot", "retries": 0,
              "expected_commit_sha": EXPECTED_COMMIT_SHA, "tested_commit_sha": None,
              "arms": [], "failed_arm": None}
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT_SHA):
            raise ValueError("invalid commit pin")
        with tempfile.TemporaryDirectory(prefix="wendyfm-m6-") as directory:
            workspace = Path(directory)
            repo = workspace / "wendyfm"
            subprocess.run(["git", "-c", "credential.helper=", "clone", "--quiet",
                            PUBLIC_REPO_URL, str(repo)], check=True, capture_output=True,
                           text=True, timeout=remaining(started, 120))
            subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--detach",
                            EXPECTED_COMMIT_SHA], check=True, capture_output=True,
                           text=True, timeout=remaining(started, 30))
            actual = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True,
                                    timeout=remaining(started, 10)).stdout.strip()
            if actual != EXPECTED_COMMIT_SHA:
                raise ValueError("checkout SHA mismatch")
            record["tested_commit_sha"] = actual
            template = (repo / "scripts/infra/remote_m6_pilot.py").read_text()
            if Path(__file__).read_text() != template.replace(
                    "__WENDYFM_" + "COMMIT_SHA__", EXPECTED_COMMIT_SHA):
                raise ValueError("packaged wrapper differs from pinned commit")
            manifest = repo / "data/manifests/fineweb_edu_m6.json"
            source = json.loads(manifest.read_text())
            data_dir = Path(__file__).resolve().parent / "data"
            for name, key in (("train.jsonl", "train_artifact_sha256"),
                              ("validation.jsonl", "validation_artifact_sha256"),
                              ("tokenizer.json", "tokenizer_sha256")):
                if sha256(data_dir / name) != source[key]:
                    raise ValueError(f"{name} provenance mismatch")
            env = os.environ.copy()
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0")
            env["PYTHONPATH"] = str(repo / "src")
            if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
                    "T4" not in torch.cuda.get_device_name(0)):
                raise RuntimeError("pilot requires one visible free T4")
            for index, (label, backend, precision) in enumerate(ARMS, 1):
                record["failed_arm"] = label
                output = workspace / f"arm-{label}"
                command = [sys.executable, str(repo / "scripts/pretrain_real.py"),
                           "--config", str(repo / "configs/m6_real_pilot.json"),
                           "--source-manifest", str(manifest),
                           "--data-dir", str(data_dir), "--output-dir", str(output),
                           "--device", "cuda", "--attention-backend", backend,
                           "--precision", precision, "--max-cuda-reserved-bytes",
                           str(MEMORY_LIMIT_BYTES)]
                try:
                    subprocess.run(command, cwd=repo, env=env, check=True,
                                   capture_output=True, text=True,
                                   timeout=remaining(started, ARM_TIMEOUT_SECONDS))
                finally:
                    result_path = output / "result.json"
                    if result_path.exists():
                        record["arms"].append({"arm": label, **json.loads(result_path.read_text())})
                if len(record["arms"]) != index:
                    raise RuntimeError("arm produced no result")
                arm = record["arms"][-1]
                if (arm["source_commit_sha"] != actual or
                        arm["attention_backend_requested"] != backend or
                        arm["precision"] != precision or
                        arm["peak_cuda_reserved_bytes"] >= MEMORY_LIMIT_BYTES):
                    raise ValueError("arm provenance, mode, or memory gate failed")
            record["failed_arm"] = None
            record["status"] = "PASS"
    except Exception as error:
        record["error_type"] = type(error).__name__
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "completed_arms": len(record["arms"])}))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
