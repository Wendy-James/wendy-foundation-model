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

## Infrastructure evidence

- Mac development environment, Git, GitHub SSH, and `uv` configured.
- Hugging Face CLI authentication and `whoami` verified.
- W&B CLI login and Mac online metric logging verified.
- Kaggle CPU tokenizer smoke passed on pinned commit
  `5f810f4e1d6a7f83a68aa350a175b7a18ff469ef`; result downloaded and
  verified locally ([result](benchmarks/infra/kaggle_tokenizer_smoke_20261008.json)).
- DSW local tool inventory completed; end-to-end permissions remain unverified.

## Next

1. Add and validate a GPU execution path; the current training loop explicitly
   requires CPU tensors and a CPU model.
2. Run a permitted small GPU pretraining experiment and record its actual
   configuration, measurements, and validation results.
3. Profile training, test a controlled systems optimization, then design
   scaling experiments.

## Research / engineering principles

Every experiment should record its hypothesis, baseline, change, configuration,
metric, result, and interpretation. Do not claim results that were not measured.
