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

## In progress

- Infrastructure integration and connectivity audit

## Next

1. Verify Hugging Face model upload and complete the DSW audit
2. Verify the full GitHub-to-Kaggle training pipeline
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
