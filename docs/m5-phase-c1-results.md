# M5 Phase C1: measured FP32 model-size comparison

**Question.** With batch size and context fixed, how do complete training-step
time, scored-target-token throughput, and PyTorch CUDA allocator memory change
as the model grows? Cell 1 is the same-run baseline. Cells 2 and 3 change model
width, layer count, and FFN width together; this is a model-size comparison, not
an isolation of any one architectural dimension.

## Provenance and controls

- Executable source SHA: `33ab89b5eb36339fad4bcce76330c5208e2a100c`.
  The verified [result](../benchmarks/infra/m5_c1_fp32_scaling_20261012.json)
  has SHA-256 `6d0c18f6f791724f17ed249d9c8c0a32567b39249c8eacb86851b796e4a36d52`.
  Its wrapper and all cells report `PASS`, with zero retries.
- Frozen [config](../configs/m5_phase_c1_fp32.json) SHA-256:
  `31d136934ea88f6261517d0d7deff23aac22a2d8a1683e11e9998006bff7f026`.
  Synthetic M4 training fixture SHA-256:
  `5022a371ea0812869a8563b9f3f4e979ae2d55c0fea5e56d48c4292ecde2b2bd`;
  training-tokenizer SHA-256:
  `7ba2b4461232d3dc17fd399c2da0f0b944d23dc217e1bf74e26613de0a247627`.
  The encoded fixture has 390 tokens.
- One visible Tesla T4 (`cuda:0`, 70 W power limit), PyTorch `2.11.0+cu128`,
  CUDA `12.8`, driver `580.178.04`; seed `20261009`, one CPU sampling thread.
- All cells use `vocab_size=272`, four attention heads, `B=4`, `T=16`, FP32
  reference attention, AdamW, constant learning rate `0.003`, weight decay
  `0.01`, and the same fixture and training-only tokenizer. There is no AMP,
  `torch.compile`, SDPA substitution, or distributed training.
- Each cell has three fresh model, sampler, and optimizer repeats. Each repeat
  runs five warmup and 30 measured optimizer steps, scoring
  `30 × 4 × 16 = 1,920` target tokens. CUDA is synchronized around the measured
  interval. Timing includes CPU sampling, host-to-device copy, forward, loss,
  backward, finite checks, gradient clipping, observer recording, and AdamW;
  it excludes setup, warmup, and result serialization.

## Comparison

Throughput is the median of three repeat rates (`1,920 / elapsed seconds`).
Spread is `(highest rate − lowest rate) / median rate`. Step time is the median
of each repeat's elapsed time divided by 30; it is **not** a distribution of
individual step latencies. Memory columns are the **maximum** measured peak
among the three repeats, in bytes, from PyTorch's CUDA allocator.

| Cell | Width / layers / FFN | Trainable parameters | Parameters vs cell 1 | Median target tokens/s | Throughput vs cell 1 | Spread | Median step time (ms) | Peak allocated (bytes) | Peak reserved (bytes) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1: baseline | 32 / 2 / 64 | 38,048 | 1.00× | 3,316.974 | — | 6.912% | 19.295 | 68,143,616 | 69,206,016 |
| 2 | 64 / 4 / 128 | 199,232 | 5.24× | 1,812.206 | −45.37% | 8.290% | 35.316 | 71,109,120 | 71,303,168 |
| 3 | 128 / 6 / 256 | 1,054,336 | 27.71× | 1,284.357 | −61.28% | 3.326% | 49.830 | 88,200,704 | 90,177,536 |

All three cells passed the run checks and stayed below the preregistered 10%
spread threshold, so none is flagged unstable. At fixed `B` and `T`, measured
step time and peak allocator memory rose while scored-target throughput fell
across these cells. The observation supports only this bounded T4 workload.

## Limits and next decision

The corpus is a tiny synthetic fixture, so its loss cannot establish language
quality. Three small models, one short sequence length, one batch size, and one
GPU do not establish an LLM scaling law or extrapolate to larger pretraining.
The widths and depths change together. Allocated and reserved bytes exclude
memory outside PyTorch's CUDA allocator. The same-run cell 1 is the comparison
baseline; measurements from another run or executable SHA are not substituted.
Context-length and batch-size sweeps require their own frozen controls and
separate execution decision.
