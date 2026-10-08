# WendyFM Project Status

## Goal

Build an interview-ready foundation-model training and systems project.

## Next milestone

Milestone 2 — Decoder-only Transformer.

## Completed milestones

### Milestone 1 — Tokenizer

Complete, committed, and pushed to GitHub. Includes deterministic byte-level
BPE, naive and optimized trainers, serialization, tests, and a local benchmark
report:

- `report/tokenizer.md`
- `benchmarks/tokenizer/results.csv`

## Done

- Mac development environment checked
- Git configured
- GitHub SSH authentication configured
- uv installed
- Python 3.11 project environment created
- Hugging Face CLI authentication and `whoami` verified
- W&B CLI login verified; authentication stored in `~/.netrc`
- Mac to W&B online metric logging verified
- Kaggle CPU tokenizer smoke test passed on pinned commit
  `5f810f4e1d6a7f83a68aa350a175b7a18ff469ef`; downloaded result verified
  on Mac (`benchmarks/infra/kaggle_tokenizer_smoke_20261008.json`)
- DSW local tool inventory completed (local inventory only)

## In progress

- Infrastructure integration and connectivity audit: HF model upload, DSW
  end-to-end permissions, and full GPU pretraining remain unverified

## Next

1. Verify Hugging Face model upload and DSW connectivity and end-to-end permissions
2. Run and evaluate full GPU pretraining on Kaggle
3. Implement the decoder-only Transformer and add model unit tests
4. Start small-scale pretraining

## Research / engineering principles

Every experiment should record:

- hypothesis
- baseline
- change
- configuration
- metric
- result
- interpretation
