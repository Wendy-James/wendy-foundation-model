"""Prepare or verify an ignored, revision-pinned M7 FineWeb-Edu corpus."""

import argparse
from pathlib import Path

from wendyfm.training.long_data import file_sha256, prepare_corpus, verify_corpus
from wendyfm.training.real_data import fineweb_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, default=Path("data/raw/fineweb_edu_m7"))
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument("--min-train-tokens", type=int, default=20_000_000)
    prepare.add_argument("--min-validation-tokens", type=int, default=200_000)
    prepare.add_argument("--max-documents", type=int, default=50_000)
    prepare.add_argument("--max-text-bytes", type=int, default=120_000_000)
    verify = commands.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_corpus(fineweb_rows(), args.output_dir, args.tokenizer,
                                min_train_tokens=args.min_train_tokens,
                                min_validation_tokens=args.min_validation_tokens,
                                max_documents=args.max_documents,
                                max_text_bytes=args.max_text_bytes)
        print(f"PREPARED: {result['selected_documents']} documents; run verify before training")
    else:
        result = verify_corpus(args.output_dir)
        print(f"PASS: {result['selected_documents']} documents; "
              f"{result['train_tokens_with_eod']} train and "
              f"{result['validation_tokens_with_eod']} validation tokens")
        print(f"Manifest SHA256: {file_sha256(args.output_dir / 'manifest.json')}")


if __name__ == "__main__":
    main()
