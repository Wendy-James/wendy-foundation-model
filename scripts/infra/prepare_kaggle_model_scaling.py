"""Package and verify one private, SHA-pinned M5 Phase C1 Kaggle sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.gpu import _documents
from wendyfm.training.scaling import MODELS, load_sweep_config, parameter_counts

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path(__file__).with_name("remote_model_scaling.py")
CONFIG = ROOT / "configs/m5_phase_c1_fp32.json"
MARKER = "__WENDYFM_COMMIT_SHA__"
SLUG = "wendyfm-m5-c1-fp32-scaling"


def valid_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("commit SHA must be 40 lowercase hexadecimal characters")
    return value


def prepare_job(job_dir: Path, user: str, sha: str) -> None:
    valid_sha(sha)
    load_sweep_config(CONFIG)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", user):
        raise ValueError("invalid Kaggle username")
    if job_dir.resolve().is_relative_to(ROOT):
        raise ValueError("job directory must be outside repository")
    if job_dir.exists() and any(job_dir.iterdir()):
        raise ValueError("job directory must be empty")
    source = REMOTE.read_text(encoding="utf-8")
    if source.count(MARKER) != 1:
        raise ValueError("remote script must contain one SHA marker")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / REMOTE.name).write_text(source.replace(MARKER, sha), encoding="utf-8")
    metadata = {"id": f"{user}/{SLUG}", "title": SLUG, "code_file": REMOTE.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                  encoding="utf-8")


def verify_result(path: Path, sha: str) -> dict:
    valid_sha(sha)
    settings = load_sweep_config(CONFIG)
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if any(wrapper.get(key) != value for key, value in {
        "status": "PASS", "test": "m5_phase_c1_fp32", "expected_commit_sha": sha,
        "tested_commit_sha": sha, "retries": 0}.items()):
        raise ValueError("wrapper status or SHA mismatch")
    cells = wrapper.get("cells")
    if not isinstance(cells, list) or len(cells) != 3:
        raise ValueError("expected three completed cells")
    baseline = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_gpu_profile.py"))
    verify_manifest = baseline["verify_manifest"]
    counts = parameter_counts()
    train_path = (CONFIG.parent / settings["train_text"]).resolve()
    tokenizer = BPETokenizer.train(_documents(train_path), vocab_size=272,
                                   special_tokens=(settings["end_of_document"],))
    tokenizer_bytes = (json.dumps(tokenizer.spec.serialization_dict(), sort_keys=True,
                                  ensure_ascii=False) + "\n").encode("utf-8")
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    for index, (cell, model, count) in enumerate(zip(cells, MODELS, counts, strict=True), 1):
        if (cell.get("cell") != index or cell.get("model") != model
                or cell.get("parameter_count") != count
                or cell.get("tokenizer_sha256") != tokenizer_sha):
            raise ValueError("cell model, parameter count, or tokenizer hash mismatch")
        verify_manifest(cell, sha, CONFIG)
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--job-dir", type=Path, required=True)
    prepare.add_argument("--user", required=True)
    prepare.add_argument("--sha", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--sha", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_job(args.job_dir, args.user, args.sha)
            print(f"Prepared private M5 C1 sweep job: {args.job_dir}")
        else:
            verify_result(args.result, args.sha)
            print("PASS: M5 Phase C1 result verified")
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
