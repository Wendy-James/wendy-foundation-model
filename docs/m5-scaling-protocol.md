# M5 single-GPU profiling and scaling protocol

## Question and scope

How do WendyFM's steady-state training throughput and CUDA memory change with
model size, context length, and microbatch size on one GPU? The baseline is the
M4 decoder and AdamW step in FP32. Each sweep changes one named dimension while
holding the others fixed. The synthetic M3 fixtures exercise the training path;
they do not support claims about language-model quality or data scaling.

This is a prospective protocol, not a benchmark result. No M5 GPU run or BF16
measurement is claimed here. The M4 eight-step manifest's `tokens_per_second`
includes its first step and cannot be used as a steady-state M5 data point.

## Freeze and prerequisites

- Reference source commit: `99264e0b8039979f11ea42839c25d002a1a65cc0`
  (the current M4 code commit). Record the full executed SHA and require it to
  equal this value for the reference FP32 series. Freeze the profiler's code
  and configuration before measurements too; record its SHA and any uncommitted
  diff. If a profiler cannot be run from that commit without code changes, pin
  a **new** full SHA for the complete FP32 series and label the old SHA as the
  design reference. Never pool results from different executable SHAs.
- Use one CUDA device (`cuda:0`) with no other user workload. Record GPU name,
  total memory, driver, CUDA and PyTorch versions, power limit, software SHA,
  configuration hash, fixture/tokenizer hashes, seed, and timestamps. Keep
  deterministic-algorithm and TF32 settings from `training/gpu.py`; log any
  change. Use the same machine and settings for every comparison.
- Use the M4 training and validation fixtures and training-only tokenizer,
  with vocabulary capacity 272 and seed 20261009. There are 390 training and
  160 validation tokens in the recorded M3 pilot; check the actual tokenized
  lengths before running each context (`length >= T + 1`). Do not treat
  overlapping windows as independent evaluation examples.
- The current `scripts/pretrain_gpu.py` performs validation and writes a
  checkpoint around an eight-step FP32 run. Its timing does not separate
  warmup, and it has no BF16 option. A future profiling harness must execute
  the same sampler, explicit once-shifted targets, loss, gradient checks,
  clipping, and AdamW update as `training/loop.py`, while adding the measurement
  boundaries below. Verify its FP32 step against the existing path before
  using its numbers. Keep profiling output and checkpoints outside Git.

## Predeclared sweep

`B` is the per-step microbatch, `T` is the number of **scored target tokens per
example**, and `P` is the instantiated trainable parameter count (record the
actual `sum(p.numel() for p in model.parameters())`). One step scores `B × T`
tokens. No gradient accumulation, activation checkpointing, compilation,
fused optimizer, or attention-kernel substitution is permitted in this series.
Keep vocabulary, tokenizer, optimizer and schedule, sampler, seed, and
precision fixed within each FP32 comparison. Use a fresh model and optimizer
per cell and per repeat. Fix the learning rate rather than comparing cells at
different positions in a decay schedule; the profiling harness should use the
same constant AdamW learning rate, weight decay, and gradient clipping for all
cells, recorded before execution.

| Sweep | Frozen dimensions | Values (in order) |
| --- | --- | --- |
| Model | `T=16`, `B=4` | `(d_model, n_heads, n_layers, intermediate_size)` = `(32,4,2,64)` baseline; `(64,4,4,128)`; `(128,4,6,256)` |
| Context | Baseline model, `B=4` | `T=16, 32, 64, 128` |
| Microbatch | Baseline model, `T=16` | `B=1, 2, 4, 8` |

The shared baseline is one cell, so this is nine distinct FP32 cells. `T` is
also `max_seq_len` in `ModelConfig`; the sampler must use the same `T`. These
sizes are candidate settings, not promises that they fit a particular GPU.
Run one cell at a time in the table order, recording skipped cells and why.
The three sweeps do not establish interactions among `P`, `T`, and `B`; a
follow-up factorial experiment needs a separate preregistration.

## Measurement contract

For each cell, perform three independent repeats with the same seed and a
fresh process or fully released CUDA state. Within each repeat, run five full
optimizer steps as warmup, then 30 full steps as the measured interval. Warmup
includes the first AdamW state allocation and CUDA kernel setup. Synchronize
`cuda:0` after warmup, reset CUDA peak-memory statistics, then start a monotonic
wall clock immediately before the first measured batch draw. Synchronize after
the final optimizer step and stop the clock. Do not synchronize between steps
except for the existing finite-value checks and scalar reads in the reference
loop. Record elapsed seconds and completed measured steps, including any
partial run as an invalid measurement. Use the median of the three valid
repeats for the main table and retain all repeat-level values.

- **Training throughput:** `sum(target_ids.numel()) / elapsed_seconds`, in
  scored target tokens/s. For a complete repeat this is `30 × B × T / elapsed`.
  The interval includes CPU batch sampling, host-to-device copies, forward,
  cross-entropy, backward, finite checks, clipping, and AdamW. It excludes
  tokenization, model creation, warmup, validation, checkpoint I/O, process
  startup, and result serialization. Record median step time as
  `elapsed_seconds / 30`. Never call this GPU-only compute throughput.
- **Peak allocated CUDA memory:** call `torch.cuda.max_memory_allocated(0)`
  after the final synchronization, in bytes, following the reset after warmup.
  This peak includes live model and optimizer state as well as training
  temporaries; record `memory_allocated(0)` immediately after warmup as the
  live baseline. Also record peak and warmup-baseline **reserved** bytes using
  `max_memory_reserved(0)` and `memory_reserved(0)`. Allocated is PyTorch
  tensor memory; reserved is the caching allocator's pool. Neither is total
  device use reported by `nvidia-smi`, and neither includes every non-PyTorch
  allocation. State these boundaries whenever reporting memory.
- **Stability:** record the loss, gradient norm, and update completion for
  every measured step. Require all losses, gradients, norms, and parameters to
  remain finite. Record minimum/maximum loss and whether the last update
  changed a parameter. The loss on different sampled windows is not a
  convergence comparison. For each cell report repeat throughput spread
  `(max - min) / median` and raw step times. If spread exceeds 10%, mark the
  throughput unstable and do not rank that cell; investigate contention or
  clocks before any separately labeled rerun. Do not silently discard slow
  repeats.

## Hypotheses and decisions

Before looking at measurements, test these directional predictions against
the baseline on the same device: increasing model size raises peak allocated
memory and lowers target-token throughput; increasing context at fixed `B`
raises peak allocated memory; increasing `B` from 1 to 4 improves throughput
until another limit intervenes. Treat equal or opposite medians, OOM, or
unstable repeats as outcomes that fail to support a prediction, not as data
to omit. Report absolute values, ratios to the shared baseline, repeat spread,
and an explanation grounded in profiles. These small tests cannot establish
an asymptotic scaling law.

Profile one valid baseline repeat and the largest valid cell of each sweep
with a short CPU/CUDA operator trace outside the timing interval. Attribute
time and allocated bytes to attention score/softmax, projections and MLP,
cross-entropy, optimizer, copies, and Python/launch overhead where visible.
The reference attention materializes FP32 `[B, heads, T, T]` scores and
weights, so a context-memory hypothesis is particularly relevant. The trace
is diagnostic evidence, not a substitute for the timing contract. Record
profiler overhead separately and do not compare profiled times to unprofiled
throughput.

## Failure handling and resource ceiling

- Preflight each cell's model/config validity and corpus length. If CUDA OOM
  occurs during setup, warmup, or measurement, record the cell, failing phase,
  exception, and memory observations available before failure. Synchronize if
  possible and release the process. Skip all larger settings in that sweep;
  continue with the next sweep's baseline only if it has not already failed.
  Do not repeatedly retry the same OOM cell, reduce its batch in place, or
  turn it into a successful measurement.
- Stop the whole series on a nonfinite value, invalid target alignment,
  failed parameter update, incorrect device count, unexpected source SHA,
  another GPU workload, thermal/power throttling that invalidates comparison,
  or a failed FP32 reference-step check. Preserve the failure record.
- Hard ceiling: **$0 paid compute or services** and **60 minutes of actual
  single-GPU session time**, including setup, warmup, failed cells, and
  profiling. Check free quota before starting; stop before either limit is
  crossed. Stop after the declared cells and traces finish; no exploratory
  expansion inside this series. Any later paid experiment needs a separate
  approved budget and protocol.

## Precision gate: FP32 before BF16

Complete and inspect the FP32 series first. Only then consider BF16, and only
if the device reports BF16 support and a separately tested implementation
exists. The present M4 runner is FP32 only. BF16 work changes executable code,
so pin its new full SHA and rerun an FP32 control on that **same** SHA, machine,
cell, and measurement contract before a BF16 comparison. Use autocast for the
forward/loss path with FP32 master parameters and AdamW state; preserve the
attention module's explicit FP32 score/softmax path unless an independently
specified attention experiment changes it. Check finite loss, gradients,
parameters, and a completed update before timing. Do not interpret differences
between the old FP32 SHA and new BF16 SHA as a precision effect. BF16 is a
separate gated series under the same $0 and 60-minute ceiling, with its own
preregistered cells and stop record. The 60-minute limit is shared across
FP32 and BF16 work, not restarted for the second series.

## Result record

Publish a machine-readable row per cell/repeat, including failed and skipped
cells: source/profiler SHA, config and data hashes, hardware/software identity,
precision, `P/B/T`, seed, warmup and measured step counts, elapsed seconds,
target-token count, throughput, allocated/reserved baselines and peaks in
bytes, per-step stability observations, status/reason, and session time used.
The report should state baseline, change, metric, observed result, and
interpretation for each sweep. Leave values blank until measured; never fill
them with estimates or reuse the M4 pilot's bounded-run numbers.
