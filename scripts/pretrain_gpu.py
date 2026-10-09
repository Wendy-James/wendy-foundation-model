"""Run bounded FP32 single-GPU pretraining from a repository checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wendyfm.training.gpu import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gpu_fp32.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpu_fp32"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, device_name=args.device,
                         end_step=args.end_step, resume=args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
