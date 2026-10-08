# WendyFM Project Status

## Goal

Build an interview-ready foundation-model training and systems project.

## Current milestone

Milestone 2 — Decoder-only Transformer.

## Completed milestones

### Milestone 1 — Tokenizer

Complete. Includes deterministic byte-level BPE, naive and optimized trainers,
serialization, tests, and a local benchmark report:

- `report/tokenizer.md`
- `benchmarks/tokenizer/results.csv`

## Done

- Mac development environment checked
- Git configured
- GitHub SSH authentication configured
- uv installed
- Python 3.11 project environment created

## In progress

- Decoder-only Transformer design and implementation

## Next

1. Implement the decoder-only Transformer
2. Add model unit tests
3. Start small-scale pretraining

## Research / engineering principles

Every experiment should record:

- hypothesis
- baseline
- change
- configuration
- metric
- result
- interpretation
