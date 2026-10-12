"""Prepare a bounded FineWeb-Edu pilot corpus (requires optional datasets)."""

import argparse
from pathlib import Path

from wendyfm.training.real_data import fineweb_rows, prepare_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/fineweb_edu_m6"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/fineweb_edu_m6.json"))
    parser.add_argument("--max-documents", type=int, default=256)
    parser.add_argument("--max-text-bytes", type=int, default=1_000_000)
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()
    prepare_documents(fineweb_rows(), args.output_dir, args.manifest,
                      max_documents=args.max_documents, max_text_bytes=args.max_text_bytes,
                      vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
