"""Run the M6 real-data single-device pilot from prepared local artifacts."""

import argparse
import json
from pathlib import Path

from wendyfm.training.real_pilot import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m6_real_pilot.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fineweb_edu_m6"))
    parser.add_argument("--source-manifest", type=Path,
                        default=Path("data/manifests/fineweb_edu_m6.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m6_real_pilot"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--attention-backend", choices=("reference", "sdpa"))
    parser.add_argument("--max-cuda-reserved-bytes", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.data_dir, args.source_manifest, args.output_dir,
                         device_name=args.device, precision=args.precision,
                         end_step=args.end_step, resume=args.resume,
                         attention_backend=args.attention_backend,
                         max_cuda_reserved_bytes=args.max_cuda_reserved_bytes),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
