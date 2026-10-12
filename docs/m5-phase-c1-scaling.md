# M5 Phase C1: controlled FP32 model-size scaling

This is the code-only phase for Issue #14. No Phase C1 GPU measurements are
included. The frozen config is [`configs/m5_phase_c1_fp32.json`](../configs/m5_phase_c1_fp32.json).
The three cells change only model width, depth, and FFN width:

| Cell | `d_model` | Heads | Layers | FFN width | Trainable P |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (shared baseline) | 32 | 4 | 2 | 64 | 38,048 |
| 2 | 64 | 4 | 4 | 128 | 199,232 |
| 3 | 128 | 4 | 6 | 256 | 1,054,336 |

All use the same synthetic training fixture and tokenizer, seed 20261009,
`B=4`, `T=16`, AdamW, FP32 reference attention, 0.003 constant learning rate,
and 0.01 weight decay. There is no AMP, compile, SDPA substitution, or DDP.
The existing M4 sampler shifts targets once; the existing M4 `train()` path
times CPU sampling through the optimizer step with CUDA synchronization at
the interval boundaries. Each fresh repeat runs 5 warmup and 30 measured
steps, scoring `30 × 4 × 16 = 1920` targets. A cell has three independent
repeats with a newly seeded model, sampler, and optimizer each time.

Each cell records actual trainable parameter count, elapsed and mean step
seconds, target tokens per second, warmup-end and measured peak PyTorch
allocated and reserved bytes, per-step finite loss and gradient norm, and
whether an optimizer update changed a parameter. The sweep reports the median
throughput and `(max - min) / median` spread per cell. Spread above 10% is
flagged unstable and must not be silently ranked. Counts include embeddings
and norms. CUDA allocator bytes are not whole-device memory.

The private Kaggle wrapper clones the exact executable SHA, masks the T4×2
allocation to one logical T4 before importing PyTorch, and runs cells in
order. Each cell has a 240-second subprocess bound; the entire wrapper has a
900-second bound. It stops on the first timeout, OOM, nonfinite update, or
other error and never retries. The result retains completed cells and a
truthful failure type. Queue time is separate from in-job time. A provider
watchdog is still required when execution is authorized.

## Later execution gate

After review, freeze the full executable commit SHA and perform the Issue #14
B1 preflight before any GPU job. Only then prepare the private Kaggle package
outside the repository:

```sh
uv run --no-sync python scripts/infra/prepare_kaggle_model_scaling.py prepare \
  --job-dir /tmp/wendyfm-m5-c1-job --user KAGGLE_USERNAME --sha FULL_COMMIT_SHA
```

Use one free T4 job, a 900-second provider timeout, zero retries, and the
private metadata. Verify its downloaded JSON offline with:

```sh
uv run --no-sync python scripts/infra/prepare_kaggle_model_scaling.py verify \
  --result /path/to/m5-scaling-result.json --sha FULL_COMMIT_SHA
```

The verifier checks all three models and actual parameter counts, fixture,
tokenizer, config, executable SHA, software and GPU identity, complete step
observations, memory bounds, and time/token arithmetic. The fixture's loss
cannot establish model quality, and three cells cannot establish a scaling law.
