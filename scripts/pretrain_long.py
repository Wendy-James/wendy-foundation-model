"""Run a bounded M7 real-data segment or resume from its last checkpoint."""

import argparse
from pathlib import Path

from wendyfm.training.long_run import run_long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m7_t4_1m.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--sha", required=True, help="full pushed executable commit SHA")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_long(args.config, args.data_dir, args.output_dir,
                      expected_manifest_sha256=args.manifest_sha256,
                      expected_commit_sha=args.sha, device_name=args.device,
                      resume=args.resume)
    print(f"{result['event'].upper()}: step={result['step']} "
          f"target_tokens={result['cumulative_target_tokens']}")


if __name__ == "__main__":
    main()
