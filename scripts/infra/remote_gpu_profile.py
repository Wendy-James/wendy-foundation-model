"""Private Kaggle wrapper for one SHA-pinned, bounded M5 Phase A baseline."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_COMMIT_SHA = "__WENDYFM_COMMIT_SHA__"
PUBLIC_REPO_URL = "https://github.com/Wendy-James/wendy-foundation-model.git"
RESULT_PATH = Path("/kaggle/working/m5-profile-result.json")


def verify_checkout(repo: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT_SHA):
        raise ValueError("invalid commit pin")
    actual = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if actual != EXPECTED_COMMIT_SHA:
        raise ValueError("checkout SHA mismatch")


def main() -> None:
    record: dict = {"status": "FAIL", "expected_commit_sha": EXPECTED_COMMIT_SHA,
                    "tested_commit_sha": None, "test": "m5_phase_a_fp32"}
    try:
        with tempfile.TemporaryDirectory(prefix="wendyfm-m5-") as directory:
            repo = Path(directory) / "wendyfm"
            subprocess.run(["git", "-c", "credential.helper=", "clone", "--quiet",
                            PUBLIC_REPO_URL, str(repo)], check=True, capture_output=True,
                           text=True, timeout=60)
            subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--detach",
                            EXPECTED_COMMIT_SHA], check=True, capture_output=True,
                           text=True, timeout=30)
            verify_checkout(repo)
            record["tested_commit_sha"] = EXPECTED_COMMIT_SHA
            output = Path(directory) / "result.json"
            try:
                subprocess.run([sys.executable, str(repo / "scripts/profile_gpu.py"),
                                "--expected-sha", EXPECTED_COMMIT_SHA,
                                "--output", str(output)], cwd=repo, check=True,
                               capture_output=True, text=True, timeout=300)
            finally:
                if output.exists():
                    record["manifest"] = json.loads(output.read_text(encoding="utf-8"))
            manifest = json.loads(output.read_text(encoding="utf-8"))
            if (manifest.get("status") != "PASS"
                    or manifest.get("source_commit_sha") != EXPECTED_COMMIT_SHA):
                raise ValueError("profile result SHA or status mismatch")
            record["manifest"] = manifest
            record["status"] = "PASS"
    except Exception as error:
        # Exception strings can contain remote data or credentials.
        record["error_type"] = type(error).__name__
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
