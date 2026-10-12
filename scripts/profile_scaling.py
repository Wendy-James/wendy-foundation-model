"""Run one frozen M5 Phase C1 cell from a clean pinned checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wendyfm.training.scaling import run_cell  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--cell", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/m5_phase_c1_fp32.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_cell(args.config, args.output, expected_sha=args.expected_sha, cell=args.cell)
    print(json.dumps({"status": result["status"], "cell": args.cell, "output": str(args.output)}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
