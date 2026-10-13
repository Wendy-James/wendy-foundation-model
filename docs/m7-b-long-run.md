# M7-B: bounded long-run trainer, offline implementation

This branch adds a thin interval runner around WendyFM's existing `train()`,
`validation_loss()`, `save_checkpoint()`, `load_checkpoint()`,
`NextTokenBatchSampler`, model, and tokenizer. It does not change the frozen
M6 executable or the M7-A corpus preflight. There is no GPU result here.
The design follows PyTorch's explicit optimizer/RNG state handling and the
evaluation/checkpoint cadence in [LitGPT pretraining](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/pretrain.py)
without importing Fabric or replacing WendyFM's trainer.

## One process, one schedule

After M7-A's full offline preflight, `run_long()` checks the ignored corpus
against the externally supplied manifest SHA256, artifact and document hashes,
frozen tokenizer, split rule, and token totals. It then loads the token tensors,
sampler, model, and AdamW optimizer **once per process**. The real-data path
requires 20M train and 200k validation tokens; synthetic CPU tests use a
tokenizer override unavailable in the CLI. It fixes
`TrainConfig.max_steps` before the first update and calls `train()` over
consecutive global step intervals with the same optimizer and sampler.
The learning-rate schedule therefore uses global steps. Resume reconstructs
the objects once, checks the frozen plan and source hashes, and restores
model, optimizer, sampler, Python/Torch/CUDA RNG, and global step from the
existing atomic checkpoint format. CPU exact equality across an uninterrupted
and an interrupted/resumed run is tested; CUDA exact equality is unmeasured.

The output directory must be outside Git or under ignored `outputs/`.
It contains `plan.json`, atomic `events/step_XXXXXXXX.json` records, the
latest `checkpoint_XXXXXXXX.pt`, and a concise private `failure.json` if a
run fails after plan creation. Each step event is written only after its
checkpoint succeeds. The previous checkpoint is removed only after the
new event succeeds. If interrupted between those writes, resume uses the
last matching event/checkpoint pair and repeats at most that interval.
There are no automatic retries.

The plan freezes the executable commit, config, manifest, train/validation
artifact and tokenizer SHA256 values, revision, device/GPU and PyTorch/CUDA
identity, precision, attention backend, parameter count, token budget,
interval, memory, and wall bounds. Each interval records
global step, cumulative scored target tokens, per-step train loss and
gradient norm, validation loss, measured target tokens/time and tokens/sec,
full-run effective tokens/sec, startup/training/validation/checkpoint and
cumulative work time, CUDA allocated/reserved peaks, and final learning rate.
Step zero records validation before any update. `train()` already checks
nonfinite loss/gradients/parameters, GradScaler behavior, and CUDA memory;
the wrapper checks that `final_norm.weight` changes over each interval.

## Frozen first T4 protocol

The [config](../configs/m7_t4_1m.json) keeps M6's 21,502,464-parameter
model, SDPA FP32, T=256, microbatch 2, accumulation 4, and seed 20261012.
At 2,048 scored target tokens per update, 489 steps score **1,001,472**
tokens, the smallest whole-step budget meeting 1M. Validate the entire
held-out corpus at step zero and step 489; save at both points. The run has
zero automatic retries, a 12 GiB CUDA reserved-memory stop, and a provisional
3,600-second wall bound. The operator must confirm this bound after real
corpus preparation and arrange an outer process timeout: the in-process
checks run at step and phase boundaries and cannot interrupt a stalled
download, validation call, or checkpoint write.

The M6-B SDPA FP32 arm measured 16,602.380302 target tokens/s over just
18 timed steps. At that rate, 1,001,472 tokens take about 1.01 minutes of
timed training. M7-A's explicit overhead allowance yields about **4.31
minutes** of free T4 quota for the 1M pilot, **$0 direct GPU charge** if
free quota is available. Startup, full 200k-token validation, checkpoint
time, and sustained throughput remain unmeasured for M7; this is an estimate.

After the M7-A corpus is independently prepared and strictly verified, a
local CPU command can exercise a small config, and a separately authorized
private T4 package can use the same CLI with `--device cuda`:

```sh
PYTHONPATH=src uv run --no-sync python scripts/pretrain_long.py \
  --config configs/m7_t4_1m.json \
  --data-dir data/raw/fineweb_edu_m7 \
  --output-dir outputs/m7_t4_1m \
  --manifest-sha256 FULL_VERIFIED_MANIFEST_SHA256 \
  --sha FULL_PUSHED_EXECUTABLE_SHA \
  --device cuda
```

Use `--resume` only with the exact same config, corpus, tokenizer, device,
and output. The CLI checks that its full executable Git SHA matches and the
checkout is clean. In the private job, pin that SHA and supply
the verified corpus through a private attachment outside source control;
never depend on a sibling path relative to the executed script. The job
must verify the attachment and SHA before model/CUDA initialization, use a
single visible T4, enforce the outer wall deadline, and export only compact
events and the run plan. This packaging and the real corpus are still gated;
this branch does not upload or submit anything.

## Remaining gates before a GPU run

1. B1 prepares the real M7 corpus from the pinned source, verifies its
   >=20M train and >=200k validation token floors and artifact hashes, and
   records preparation wall time and manifest SHA256 outside Git.
2. Review a private SHA-pinned Kaggle package and corpus attachment path,
   free T4 availability, one-device isolation, outer timeout, and result
   export. Never place the corpus, tokenizer artifact, checkpoints, weights,
   credentials, or bulky logs in Git.
3. Obtain separate approval for the one free-T4 job. This offline PR does
   not authorize or claim a GPU measurement.
