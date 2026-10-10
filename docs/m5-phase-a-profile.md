# M5 Phase A: single GPU FP32 baseline

This implements only the baseline cell from Issue #11 and the protocol in PR
#6. It contains no GPU results. The M4 eight-step smoke result is not a
steady-state measurement.

## Frozen cell and reference

The executable baseline is [`configs/m5_phase_a_fp32.json`](../configs/m5_phase_a_fp32.json):
`d_model=32`, four heads, two layers, FFN width 64, batch four, and 16 scored
targets per example. It uses the M4 training fixture, training-only BPE
tokenizer, explicit sampler targets, FP32 AdamW, constant 0.003 learning rate,
0.01 weight decay, and M4 gradient clipping. The seed is 20261009. The M4
reference commit is `99264e0b8039979f11ea42839c25d002a1a65cc0`; the
actual profiler commit must be pinned separately and is reported with every
result.

The harness calls the existing `train()` function for each complete optimizer
step. Before timing, it checks that sampled targets are shifted exactly once,
that explicit cross-entropy equals `next_token_loss`, and that one M4 reference
step changes a parameter. The reference model is discarded. Each repeat
creates a fresh model, AdamW optimizer, and sampler with the same seed.

Each repeat runs five warmup steps, then 30 measured steps. M4 `train()`
synchronizes CUDA and resets memory peaks before starting its clock, then
synchronizes after the last step. The measured interval includes CPU sampling,
host-to-device transfer, forward/loss/backward, finite checks, clipping,
observer recording, and AdamW. It excludes tokenization, model setup,
warmup, reference checks, and result serialization. Existing M4 per-step
scalar and finite checks synchronize as part of that path. There is no extra
per-step timing synchronization. Step latency is total measured elapsed time
divided by 30; it is not a distribution of individual CUDA step latencies.

One complete repeat scores `30 × 4 × 16 = 1920` target tokens. Throughput is
1920 divided by measured elapsed seconds. Peak allocated and reserved memory
are PyTorch CUDA allocator statistics during the measured interval, including
live model and optimizer state. Warmup-end allocated and reserved baselines
are reported separately. These values are not total device memory use.
Every measured loss, gradient norm, and parameter is checked for finiteness;
the measured run must change a parameter. The result includes all
30 per-step loss and gradient-norm observations per repeat. The reported
throughput median and `(max - min) / median` spread retain all three repeats;
spread above 10% marks the baseline unstable.
Kaggle's T4 accelerator allocates two physical T4 cards. The private wrapper
sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0` in the profiler
child environment before Python imports PyTorch. This uses one logical T4 from
that T4×2 allocation; it is not a two-GPU run or a separate T4×1 SKU. The
runner still requires exactly one visible CUDA device and FP32 parameters on
`cuda:0`. It queries physical GPU index 0 with `nvidia-smi -i 0`, checks its
name against PyTorch's selected device, and verifies index 0 is first in PCI
order. It fails if the mapping differs. Local result verification requires the
pinned fixture hash, one visible CUDA device, software and GPU identity, and
consistent elapsed-time arithmetic.

## Run later, after Phase B approval

Use a clean, public checkout of the committed profiler SHA. Prepare a private
Kaggle script outside the repository:

```sh
python scripts/infra/prepare_kaggle_gpu_profile.py prepare \
  --job-dir /tmp/wendyfm-m5-job --user KAGGLE_USERNAME --sha FULL_COMMIT_SHA
```

Before starting a job, check free T4 quota, private metadata with
`machine_shape=NvidiaTeslaT4`, and the pinned SHA. The package contains only
the wrapper and metadata; it clones the public pinned commit, uses the committed
local fixture, and has no dataset attachments or credentials. When Phase B is
authorized, use the Kaggle CLI's documented `--accelerator NvidiaTeslaT4`
and `--timeout 300` options for **one** free private job, with zero retries
and no paid compute. The wrapper enforces a
300-second total deadline across clone, checkout, and profiling; set
a 300-second external job watchdog as well, covering Kaggle startup time.
Stop on a SHA, device-count, metadata, alignment,
finiteness, update, or timeout failure. The wrapper writes
`/kaggle/working/m5-profile-result.json`, including a failure record when the
profile process writes one. Do not resubmit the same failed job automatically.

After downloading the JSON, verify it locally:

```sh
python scripts/infra/prepare_kaggle_gpu_profile.py verify \
  --result /path/to/m5-profile-result.json --sha FULL_COMMIT_SHA
```

The SHA, config, and fixture hash checks fail closed. The verifier checks the completed
repeat counts and token arithmetic. Keep downloaded results outside Git until
reviewed for the separate Phase B evidence PR. Do not interpret this fixture
run as a language-quality or scaling-law result.

## Offline validation

`uv run pytest -q`, `uv run ruff check .`, and `git diff --check` require no
GPU or large downloads. CUDA acceptance remains unrun in Phase A.
