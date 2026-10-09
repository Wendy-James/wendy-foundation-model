"""Kaggle entry point for one measured CUDA forward/backward/AdamW step."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXPECTED_COMMIT_SHA = "__WENDYFM_COMMIT_SHA__"
PUBLIC_REPO_URL = "https://github.com/Wendy-James/wendy-foundation-model.git"
RESULT_PATH = Path("/kaggle/working/model-smoke-result.json")


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


def run_model_check(repo: Path) -> dict[str, object]:
    """Import only the pinned model implementation and run one tiny CUDA update."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    sys.path.insert(0, str(repo / "src"))
    from wendyfm.model import DecoderOnlyTransformer, ModelConfig

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    device = torch.device("cuda:0")
    config = ModelConfig(
        vocab_size=32, d_model=32, n_heads=4, n_layers=1,
        max_seq_len=8, intermediate_size=64,
    )
    model = DecoderOnlyTransformer(config).to(device)
    ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = model.next_token_loss(ids)
    if not bool(torch.isfinite(loss)):
        raise ValueError("loss is nonfinite")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "test": "model_cuda_adamw",
        "loss": float(loss.detach().item()),
        "cuda_device": torch.cuda.get_device_name(device),
        "step_time_s": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def main() -> None:
    result: dict[str, object] = {
        "status": "FAIL", "expected_commit_sha": EXPECTED_COMMIT_SHA,
        "tested_commit_sha": None,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="wendyfm-source-") as directory:
            repo = Path(directory) / "wendyfm"
            subprocess.run(
                ["git", "-c", "credential.helper=", "clone", "--quiet", PUBLIC_REPO_URL, str(repo)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--quiet", "--detach", EXPECTED_COMMIT_SHA],
                check=True, capture_output=True, text=True,
            )
            result["tested_commit_sha"] = verify_checkout(repo, EXPECTED_COMMIT_SHA)
            result.update(run_model_check(repo))
            result["status"] = "PASS"
    except Exception as error:
        # Do not serialize exception messages: subprocess and dependency errors can contain secrets.
        result["error_type"] = type(error).__name__
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
