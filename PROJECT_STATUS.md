# WendyFM Project Status

## Goal

Build an interview-ready foundation-model training and systems project.

## Completed

- Milestone 1: deterministic byte-level BPE tokenizer, tests, and local benchmark
  ([report](report/tokenizer.md), [results](benchmarks/tokenizer/results.csv)).
- Milestone 2: decoder-only Transformer with RoPE, RMSNorm, SwiGLU, and model
  component tests. A Kaggle model smoke was prepared; see
  [Kaggle model smoke](docs/kaggle-model-smoke.md) for its actual status.
- Milestone 3 CPU pilot: frozen configuration, synthetic train and held-out
  validation fixtures, four optimizer steps, measured metrics, and exact
  checkpoint-resume regression test. See [pilot report](docs/cpu-pretraining-pilot.md)
  and [manifest](experiments/cpu_pilot_20261009.json).
- Milestone 3 GPU model smoke: one private Kaggle Tesla T4 forward, backward,
  and AdamW step passed on pinned commit
  `39bc1b9cc857115f788f6b604845178677fc2ad0`, with finite gradients
  and an actual parameter update. See [report](docs/kaggle-model-smoke.md)
  and [verified result](benchmarks/infra/gpu_model_smoke_20261009.json).
- M5 Phase C1: three controlled FP32 model-size cells passed offline result
  verification on pinned source SHA `33ab89b5eb36339fad4bcce76330c5208e2a100c`.
  See the [experiment report](docs/m5-phase-c1-results.md) and
  [exact result](benchmarks/infra/m5_c1_fp32_scaling_20261012.json).

## Infrastructure evidence

- Mac development environment, Git, GitHub SSH, and `uv` configured.
- Hugging Face CLI authentication and `whoami` verified.
- W&B CLI login and Mac online metric logging verified.
- Kaggle CPU tokenizer smoke passed on pinned commit
  `5f810f4e1d6a7f83a68aa350a175b7a18ff469ef`; result downloaded and
  verified locally ([result](benchmarks/infra/kaggle_tokenizer_smoke_20261008.json)).
- DSW local tool inventory completed; end-to-end permissions remain unverified.

## Next

1. Review the bounded M5 C1 comparison and decide whether to run separately
   controlled context-length or batch-size sweeps.
2. Plan real-data pretraining and systems optimization without treating the
   tiny synthetic fixture as evidence of language quality or a scaling law.

## Research / engineering principles

Every experiment should record its hypothesis, baseline, change, configuration,
metric, result, and interpretation. Do not claim results that were not measured.
