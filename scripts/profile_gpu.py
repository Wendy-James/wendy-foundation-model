"""Execute the frozen M5 Phase A FP32 baseline from a clean pinned checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wendyfm.training.profile import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/m5_phase_a_fp32.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output, expected_sha=args.expected_sha)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
