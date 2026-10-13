# WendyFM

**A reproducible, from-scratch foundation-model training and systems lab.**

WendyFM is a learning-driven engineering project for understanding language-model training **from implementations and measurements**, not from API wrappers. Each milestone aims to include code, tests, an experiment protocol, quantitative results, and limitations.

> **Current status — Milestone 1 complete:** byte-level BPE tokenizer.  
> **Not yet implemented or claimed:** end-to-end Transformer pretraining, RL post-training, distributed training, or scaling results.

## Milestone status

| Area | Status | Evidence |
| --- | --- | --- |
| Byte-level BPE tokenizer | **Complete** | [Technical report](report/tokenizer.md), [reproducible benchmark](benchmarks/tokenizer/results.csv), [tests](tests/tokenizer/) |
| Decoder-only Transformer (RoPE, RMSNorm, SwiGLU) | Planned | Future implementation and unit tests |
| Training loop, optimizer, evaluation | Planned | Future baseline and loss/quality measurements |
| GPU/attention performance profiling | Planned | Future throughput and memory benchmarks |
| Distributed training & scaling experiments | Planned | Future reproducible multi-device experiments |

## Milestone 1 · Byte-level BPE

### What was built

- UTF-8 byte-based vocabulary with deterministic pair-selection tie breaking.
- Multilingual pre-tokenization and explicit special-token isolation.
- A **naive reference trainer** and an **incrementally maintained pair-index trainer**.
- Differential correctness tests: both approaches produce **exactly the same tokenizer specification** on the benchmark cases.
- Serialization, encoding, decoding, and tests for tricky boundaries.

### Results — small CPU-local, synthetic corpora

| Corpus | Indexed trainer vs. naive | Interpretation |
| --- | ---: | --- |
| English | **7.25×** | Lower training time on this benchmark |
| Code | **4.55×** | Lower training time |
| Mixed multilingual | **4.57×** | Lower training time |
| Chinese | **0.70×** | **Regression**, index maintenance outweighed savings |

The indexed trainer also had **higher measured peak Python allocations** in all four runs. These figures are specific to the benchmark's short, repeated synthetic templates and `tracemalloc` settings—**not** a large-corpus performance claim, production tokenizer comparison, or process-RSS measurement.

**Inspect the evidence:** [method, environment and limitations](report/tokenizer.md) · [CSV data](benchmarks/tokenizer/results.csv) · [benchmark source](scripts/benchmark_tokenizer.py).

## Reproduce the tokenizer milestone

Prerequisites: Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Wendy-James/wendy-foundation-model.git
cd wendy-foundation-model
uv sync --group dev
uv run pytest -q
uv run python scripts/benchmark_tokenizer.py
```

The benchmark script overwrites `benchmarks/tokenizer/results.csv` with measurements from **your own machine**. Runtime and allocation measurements may vary by environment; exact tokenizer-spec equality is checked before speedups are reported.

## Next experiments

1. Implement a small decoder-only Transformer with unit-tested layers.
2. Establish a reproducible training/evaluation baseline before performance optimization.
3. Measure throughput, memory, and optimization regressions on accessible compute.
4. Record configuration, seed, comparison baseline, metrics, and failure analysis for every result.

These are **planned experiments**; this section should not be read as a list of completed capabilities.

## Principles

- **Correctness before optimization.** Check numerical and behavioral equivalence where appropriate.
- **Show negative results.** Regressions and failure cases are research evidence too.
- **Reuse high-quality open source.** Credit sources and comply with licenses.
- **Make public artifacts safe.** Use publishable/synthetic data; do not commit private credentials, company code, or internal datasets.

[Project status](PROJECT_STATUS.md) · [Technical report](report/tokenizer.md) · [Benchmark data](benchmarks/tokenizer/results.csv) · [Researcher profile](https://github.com/Wendy-James)
