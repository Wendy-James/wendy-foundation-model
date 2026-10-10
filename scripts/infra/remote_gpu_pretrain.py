"""Private Kaggle entry point for bounded FP32 pretraining and exact resume."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_COMMIT_SHA = "__WENDYFM_COMMIT_SHA__"
PUBLIC_REPO_URL = "https://github.com/Wendy-James/wendy-foundation-model.git"
RESULT_PATH = Path("/kaggle/working/gpu-pretrain-result.json")


def verify_checkout(repo: Path, expected_sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("invalid pinned commit SHA")
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual != expected_sha:
        raise ValueError("checked-out commit differs from pin")
    return actual


def _equal_tree(left, right) -> bool:
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal_tree(first, second) for first, second in zip(left, right, strict=True)
        )
    return left == right


def run_pretraining(repo: Path, directory: Path) -> dict:
    """Run eight full steps and a separate four-plus-four continuation."""
    import torch

    script = repo / "scripts/pretrain_gpu.py"
    config = repo / "configs/gpu_fp32.json"

    def command(output: Path, *flags: str) -> dict:
        completed = subprocess.run(
            [sys.executable, str(script), "--config", str(config),
             "--output-dir", str(output), *flags],
            cwd=repo, check=True, capture_output=True, text=True,
        )
        return json.loads(completed.stdout)

    full_dir = directory / "full"
    split_dir = directory / "split"
    full = command(full_dir)
    partial = command(split_dir, "--end-step", "4")
    resumed = command(split_dir, "--resume")
    if full["optimizer_steps"] != resumed["optimizer_steps"] or (
        partial["optimizer_steps"] != resumed["start_step"]
    ):
        raise ValueError("invalid resume step accounting")
    uninterrupted = torch.load(full_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    continued = torch.load(split_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if not _equal_tree(uninterrupted, continued):
        raise ValueError("checkpoint continuation differs from uninterrupted training")
    if full["validation_loss"] != resumed["validation_loss"]:
        raise ValueError("validation loss differs after checkpoint continuation")
    return {"manifest": full, "checkpoint_resume_exact": True}


def main() -> None:
    result: dict[str, object] = {
        "status": "FAIL", "expected_commit_sha": EXPECTED_COMMIT_SHA,
        "tested_commit_sha": None, "test": "fp32_single_gpu_pretrain",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="wendyfm-gpu-pretrain-") as directory:
            workspace = Path(directory)
            repo = workspace / "wendyfm"
            subprocess.run(
                ["git", "-c", "credential.helper=", "clone", "--quiet",
                 PUBLIC_REPO_URL, str(repo)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--quiet", "--detach",
                 EXPECTED_COMMIT_SHA],
                check=True, capture_output=True, text=True,
            )
            result["tested_commit_sha"] = verify_checkout(repo, EXPECTED_COMMIT_SHA)
            result.update(run_pretraining(repo, workspace))
            result["status"] = "PASS"
    except Exception as error:
        # Dependency errors can contain credentials; publish only the exception class.
        result["error_type"] = type(error).__name__
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
