"""Kaggle entry point; the preparation command replaces the pinned SHA."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_COMMIT_SHA = "__WENDYFM_COMMIT_SHA__"
PUBLIC_REPO_URL = "https://github.com/Wendy-James/wendy-foundation-model.git"
RESULT_PATH = Path("/kaggle/working/tokenizer-smoke-result.json")


def run_tokenizer_check(repo: Path) -> dict[str, object]:
    """Import the fetched source and check repeatable training and round-trip encoding."""
    sys.path.insert(0, str(repo / "src"))
    from wendyfm.tokenizer import BPETokenizer

    corpus = ["hello world\n", "你好 world\n", "hello 世界!\n"]
    sample = "hello 世界!\n"
    first = BPETokenizer.train(corpus, vocab_size=280)
    second = BPETokenizer.train(corpus, vocab_size=280)
    deterministic = first.spec == second.spec
    encoded = first.encode(sample)
    round_trip = first.decode(encoded) == sample
    assert deterministic, "training produced different specifications"
    assert encoded and round_trip, "tokenizer round trip failed"
    return {
        "test": "tokenizer_round_trip",
        "token_count": len(encoded),
        "round_trip": round_trip,
        "deterministic": deterministic,
    }


def main() -> None:
    result: dict[str, object] = {
        "status": "FAIL",
        "expected_commit_sha": EXPECTED_COMMIT_SHA,
        "tested_commit_sha": None,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="wendyfm-source-") as directory:
            repo = Path(directory) / "wendyfm"
            subprocess.run(
                ["git", "-c", "credential.helper=", "clone", "--quiet", PUBLIC_REPO_URL, str(repo)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--quiet", "--detach", EXPECTED_COMMIT_SHA],
                check=True,
                capture_output=True,
                text=True,
            )
            actual_sha = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            result["tested_commit_sha"] = actual_sha
            assert actual_sha == EXPECTED_COMMIT_SHA, "checked-out commit differs from pin"
            result.update(run_tokenizer_check(repo))
            result["status"] = "PASS"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
