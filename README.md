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

The tokenizer and decoder-only Transformer are implemented and tested. A bounded
CPU pretraining pilot now exercises tokenization, next-token sampling, training,
held-out validation, and exact checkpoint resume. See the
[CPU pilot report](docs/cpu-pretraining-pilot.md) and
[measured manifest](experiments/cpu_pilot_20261009.json).

From a checkout with `uv` installed, reproduce the CPU pilot with:

```sh
uv sync --group dev
uv run python scripts/pretrain_cpu.py --config configs/cpu_pilot.json --output-dir outputs/cpu_pilot
```

The output directory holds the checkpoint, serialized tokenizer, and run
manifest. It is ignored by Git. For an interrupted run, add `--end-step 2`,
then rerun the same command with `--resume` to finish the frozen four-step
schedule.
