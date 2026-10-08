# WendyFM Tokenizer Benchmark Report

## 1. Implementation

WendyFM implements a from-scratch byte-level BPE tokenizer. IDs `0..255` are
bytes, learned IDs are assigned in merge order, and special tokens occupy the
top of the configured vocabulary range. The tokenizer uses a versioned,
human-readable JSON representation.

## 2. Algorithm

Training starts with UTF-8 byte IDs in each minimal pre-tokenized piece. It
counts adjacent token pairs, selects one pair, creates a token whose bytes are
the concatenation of its children, replaces non-overlapping occurrences, and
repeats until capacity or exhaustion.

The deterministic rule is maximum frequency, then lexicographically smallest
`(left_token_id, right_token_id)`.

## 3. Pre-tokenization

`unicode_runs_v1` keeps whitespace, word-like Unicode text, and punctuation or
symbol runs separate. Configured special tokens are isolated first, with the
longest match winning and declaration order breaking equal-length ties. This is
deliberately not GPT-2 or tiktoken compatibility.

## 4. Naive and optimized trainers

The naive trainer recounts every adjacent pair across every sequence after each
merge, making it a clear correctness oracle but repeatedly rescanning the
corpus.

The optimized trainer maintains:

- a current token sequence and pair counter per pre-tokenized sequence;
- global pair counts;
- a `pair -> sequence IDs` inverted index.

Only sequences containing the selected pair are removed from the global counts,
updated with the same left-to-right replacement operation, and reinserted.
Global zero-count pairs and stale index memberships are removed immediately.

## 5. Differential correctness

Every benchmark run asserted exact equality of the complete `TokenizerSpec`
before any speedup was reported. Unit tests also compare naive and optimized
training on empty, repeated, overlapping, multilingual, code, whitespace,
special-token, capacity, tie, and fixed-seed randomized corpora.

## 6. Benchmark methodology

The benchmark uses deterministic local templates repeated 320 times. It covers
English-like text, Chinese, code, and mixed multilingual text. Configuration is
`vocab_size=300` with one special token, `<|eos|>`. No external data or network
access is used. Each domain runs both trainers under `tracemalloc`; encoding is
measured using the optimized tokenizer.

## 7. Results

| Domain | Characters | UTF-8 bytes | Pieces | Actual vocab | Merges | Naive (s) | Optimized (s) | Speedup | Naive peak | Optimized peak | Spec equal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| English | 43840 | 43840 | 13440 | 300 | 43 | 2.749475791 | 0.379323167 | 7.248373x | 2925589 | 8061095 | true |
| Chinese | 12160 | 35840 | 2880 | 300 | 43 | 0.884050083 | 1.268940167 | 0.696684x | 1071148 | 7018037 | true |
| Code | 42880 | 42880 | 15040 | 294 | 37 | 1.809561416 | 0.397594958 | 4.551269x | 3167958 | 8106697 | true |
| Mixed | 13760 | 17280 | 6080 | 291 | 34 | 0.650472292 | 0.142392125 | 4.568176x | 1284010 | 3578917 | true |

The machine-readable source is `benchmarks/tokenizer/results.csv`. All four
domains produced equal naive and optimized specifications.

## 8. Interpretation

The optimized trainer was 7.25x faster on English, 4.55x faster on code, and
4.57x faster on mixed text. Chinese was slower at 0.70x, showing that
the inverted-index bookkeeping has overhead when the corpus and number of
affected sequences do not amortize it.

`tracemalloc` peak Python allocations were higher for the optimized trainer in
all four runs because it retains per-sequence counters and the inverted index.
These are Python allocation measurements, not process RSS.

| Domain | Encoding chars/s | Tokens | Characters/token | UTF-8 bytes/token |
|---|---:|---:|---:|---:|
| English | 278041.962 | 24000 | 1.826667 | 1.826667 |
| Chinese | 109259.708 | 17280 | 0.703704 | 2.074074 |
| Code | 335214.312 | 15040 | 2.851064 | 2.851064 |
| Mixed | 259529.192 | 6080 | 2.263158 | 2.842105 |

These are WendyFM measurements only and are not comparisons with other
tokenizers.

## 9. Limitations and intentional non-optimizations

The benchmark is small and CPU-local. Timing is sensitive to Python runtime,
machine load, and `tracemalloc`. The optimized implementation still rebuilds
pair counters for affected sequences. No multiprocessing, native extension,
GPU path, external tokenizer compatibility, or large external corpus was used.

## 10. Interview takeaways

This milestone demonstrates byte-level reversibility, deterministic BPE merge
selection, a simple pre-tokenization boundary contract, differential testing of
an optimized implementation, and the tradeoff between rescanning cost and
index memory. It also demonstrates that an optimization can regress on a small
or unfavorable workload and should be evaluated by measurement.
