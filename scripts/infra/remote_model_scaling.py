"""Private, bounded sequential Kaggle wrapper for three M5 Phase C1 cells."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXPECTED_COMMIT_SHA = "__WENDYFM_COMMIT_SHA__"
PUBLIC_REPO_URL = "https://github.com/Wendy-James/wendy-foundation-model.git"
RESULT_PATH = Path("/kaggle/working/m5-scaling-result.json")
CELL_TIMEOUT_SECONDS = 240
TOTAL_TIMEOUT_SECONDS = 900


def remaining_seconds(started: float, stage_limit: int) -> float:
    remaining = TOTAL_TIMEOUT_SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("M5 scaling wrapper exceeded its runtime budget")
    return min(remaining, stage_limit)


def main() -> None:
    started = time.monotonic()
    record: dict = {"status": "FAIL", "test": "m5_phase_c1_fp32",
                    "expected_commit_sha": EXPECTED_COMMIT_SHA,
                    "tested_commit_sha": None, "cells": [], "retries": 0}
    active_cell = None
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT_SHA):
            raise ValueError("invalid commit pin")
        with tempfile.TemporaryDirectory(prefix="wendyfm-m5-c1-") as directory:
            repo = Path(directory) / "wendyfm"
            subprocess.run(["git", "-c", "credential.helper=", "clone", "--quiet",
                            PUBLIC_REPO_URL, str(repo)], check=True, capture_output=True,
                           text=True, timeout=remaining_seconds(started, 60))
            subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--detach",
                            EXPECTED_COMMIT_SHA], check=True, capture_output=True,
                           text=True, timeout=remaining_seconds(started, 30))
            actual = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True,
                                    timeout=remaining_seconds(started, 10)).stdout.strip()
            if actual != EXPECTED_COMMIT_SHA:
                raise ValueError("checkout SHA mismatch")
            record["tested_commit_sha"] = actual
            profile_env = os.environ.copy()
            profile_env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0")
            for cell in (1, 2, 3):
                active_cell = cell
                output = Path(directory) / f"cell-{cell}.json"
                try:
                    subprocess.run([sys.executable, str(repo / "scripts/profile_scaling.py"),
                                    "--expected-sha", EXPECTED_COMMIT_SHA,
                                    "--cell", str(cell), "--output", str(output)],
                                   cwd=repo, check=True, capture_output=True, text=True,
                                   env=profile_env,
                                   timeout=remaining_seconds(started, CELL_TIMEOUT_SECONDS))
                finally:
                    if output.exists():
                        record["cells"].append(json.loads(output.read_text(encoding="utf-8")))
                if len(record["cells"]) != cell:
                    raise RuntimeError("cell returned without a result")
                if record["cells"][-1].get("status") != "PASS":
                    raise ValueError("cell failed")
            record["status"] = "PASS"
    except Exception as error:
        record["error_type"] = type(error).__name__
        record["failed_cell"] = active_cell
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": record["status"], "cells": len(record["cells"])}))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
