# WendyFM

WendyFM is a from-scratch foundation-model training and systems project.

The goal is to build and understand the major components of modern language-model training rather than wrapping an existing model API.

## Planned scope

- Byte-level BPE tokenizer
- Decoder-only Transformer
- RoPE / RMSNorm / SwiGLU
- AdamW training
- Pretraining experiments
- Evaluation
- GPU profiling
- Attention optimization
- Distributed training
- Scaling experiments

## Status

Milestone 1 — Tokenizer is complete. See [the technical report](report/tokenizer.md)
and [benchmark results](benchmarks/tokenizer/results.csv).
