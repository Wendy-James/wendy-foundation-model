"""Reproducible local benchmark for WendyFM's naive and incremental BPE trainers."""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wendyfm.tokenizer import BPETokenizer, TokenizerConfig, pretokenize  # noqa: E402
from wendyfm.tokenizer.training import train_naive, train_optimized  # noqa: E402


VOCAB_SIZE = 300
SPECIAL_TOKENS = ("<|eos|>",)
REPETITIONS = 320


def build_corpora() -> dict[str, list[str]]:
    english = (
        "The small language model learns patterns from carefully prepared text. "
        "Training should be reproducible, measurable, and easy to explain.\n"
    )
    chinese = "小型语言模型从经过准备的文本中学习模式。训练应该可复现、可测量、易于解释。\n"
    code = (
        "def update_model(tokens):\n"
        "    counts = {}\n"
        "    for token in tokens:\n"
        "        counts[token] = counts.get(token, 0) + 1\n"
        "    return counts\n"
    )
    mixed = "Model 模型: loss=0.42; print('你好🙂'); newline\n"
    return {
        "english": [english * REPETITIONS],
        "chinese": [chinese * REPETITIONS],
        "code": [code * REPETITIONS],
        "mixed": [mixed * REPETITIONS],
    }


def measure_trainer(trainer, texts: list[str], config: TokenizerConfig):
    tracemalloc.start()
    started = time.perf_counter()
    spec = trainer(texts, config)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return spec, elapsed, peak


def main() -> None:
    output = ROOT / "benchmarks" / "tokenizer" / "results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    config = TokenizerConfig(VOCAB_SIZE, SPECIAL_TOKENS)

    for domain, texts in build_corpora().items():
        naive_spec, naive_time, naive_peak = measure_trainer(train_naive, texts, config)
        optimized_spec, optimized_time, optimized_peak = measure_trainer(
            train_optimized, texts, config
        )
        assert naive_spec == optimized_spec
        tokenizer = BPETokenizer(optimized_spec)
        sample = texts[0]
        started = time.perf_counter()
        encoded = tokenizer.encode(sample)
        encoding_time = time.perf_counter() - started
        pieces = pretokenize(sample, config.special_tokens)
        characters = len(sample)
        utf8_bytes = len(sample.encode("utf-8"))
        rows.append(
            {
                "domain": domain,
                "corpus_characters": characters,
                "corpus_utf8_bytes": utf8_bytes,
                "pretokenized_pieces": len(pieces),
                "configured_vocab_size": config.vocab_size,
                "actual_vocab_size": tokenizer.vocab_size,
                "merge_count": len(optimized_spec.merges),
                "naive_training_seconds": f"{naive_time:.9f}",
                "optimized_training_seconds": f"{optimized_time:.9f}",
                "speedup": f"{naive_time / optimized_time:.6f}",
                "spec_equal": True,
                "naive_peak_python_bytes": naive_peak,
                "optimized_peak_python_bytes": optimized_peak,
                "encoding_characters_per_second": f"{characters / encoding_time:.3f}",
                "token_count": len(encoded),
                "characters_per_token": f"{characters / len(encoded):.6f}",
                "utf8_bytes_per_token": f"{utf8_bytes / len(encoded):.6f}",
            }
        )

    fieldnames = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    for row in rows:
        print(
            f"{row['domain']}: naive={row['naive_training_seconds']}s "
            f"optimized={row['optimized_training_seconds']}s speedup={row['speedup']}x "
            f"tokens={row['token_count']}"
        )


if __name__ == "__main__":
    main()
