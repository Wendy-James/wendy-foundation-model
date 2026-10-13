# M6-B T4 pilot evidence — 2026-10-13

The [exact downloaded result](../benchmarks/infra/m6_b_t4_pilot_20261013.json)
is a byte-for-byte copy of the latest successful
`~/wendyfm-overnight/m6-b-final-20261013-162153/m6-pilot-result.json`.
Its SHA256 is
`589690b85dfec85b003bb1a0d53d0099c23e801ed393fc50ab958504e0794d4d`.
The strict repository verifier returned `PASS: M6-B result verified` for the
original and archived files with executable commit
`8eba5c0e006696ad6b1cfcb786b70ff00dc354ac`. The earlier failed result
was excluded. The archived JSON is the source for every number below.

## Controlled comparison

A is reference attention in FP32. B changes only the requested attention
backend to PyTorch SDPA in FP32. C uses SDPA with FP16 autocast and
GradScaler. Each is a fresh sequential arm in one private Kaggle T4 job,
with seed 20261012, the same frozen FineWeb-Edu train/validation split and
tokenizer, 21,502,464 trainable parameters, sequence length 256, effective
batch 8, and 20 optimizer steps. The first two steps warm up timing; each
arm scores 36,864 target tokens over 18 timed steps. This is one run per
configuration, not three replicates.

| Arm | Requested attention / precision | Timed target tokens/s | Peak CUDA allocated (MiB) | Peak CUDA reserved (MiB) | Final validation loss | Training time (s) | Timed steps (s) | Arm wall time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | reference / FP32 | 15,361.64 | 520.5 | 546 | 4.676410 | 3.135 | 2.400 | 39.93 |
| B | SDPA / FP32 | 16,602.38 | 486.2 | 526 | 4.676410 | 2.727 | 2.220 | 34.01 |
| C | SDPA / FP16 | 16,129.38 | 483.2 | 508 | 4.676228 | 2.813 | 2.286 | 34.23 |

Throughput is `timed_target_tokens / timed_seconds`, excluding the two
warmup steps. `training_wall_seconds` is the measured elapsed time of the
20-step training loop, including warmup. `wall_seconds` covers the arm's
setup, validation, training, and checkpoint work; it is not Kaggle
queue time. Memory is the PyTorch CUDA allocator peak, not total GPU use.

| Comparison | Timed throughput ratio | Timed throughput change | Peak allocated change | Peak reserved change |
| --- | ---: | ---: | ---: | ---: |
| B / A | 1.0808× | +8.08% | −6.59% | −3.66% |
| C / A | 1.0500× | +5.00% | −7.16% | −6.96% |
| C / B | 0.9715× | −2.85% | −0.62% | −3.42% |

For this short 21.5M-parameter, T=256 pilot, B has the highest observed
throughput. C lowers allocator memory but does not improve throughput over B.
The final validation losses are close; the tiny differences do not establish
a quality advantage. The latest PR #19 result comment transcribed A and B
throughput as 15,341.64 and 16,682.38 tokens/s. The exact verified JSON
contains 15,361.64 and 16,602.38, respectively, so the ratios here use the
JSON values.

## Provenance and limits

The result records `PASS`, zero retries, all three completed arms, one
visible `Tesla T4` (`cuda:0`), PyTorch `2.11.0+cu128`, and CUDA `12.8`.
Each arm records `parameters_updated: true` and a before/after SHA256,
changed-element count, and maximum delta for `final_norm.weight`; the strict
verifier checks these witnesses. The common source manifest, config, train,
validation, and tokenizer SHA256 values are checked against the pinned
repository files. The source is FineWeb-Edu `sample-10BT` at revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; the bounded split contains
57 training and 7 validation documents. Neither raw documents nor tokenizer
artifacts are archived here.

| Frozen input | SHA256 recorded in each arm |
| --- | --- |
| Config | `4465374cf4bb67887f616c6dc9b273c1d87609eb569b913d4ea9a535ab3a356c` |
| Source manifest | `e775b782ecc5779905fb58bd93b13acabc6b561b9ea911382d9dd9b6c18cac1c` |
| Training artifact | `ccef606d2e7542bd545dff10212e70e850f0fcd85af639254686cfa4d4ba84bd` |
| Validation artifact | `2fc7aaab12a93759dacac849041226d179143a39c358d55945615cef49c03ccc` |
| Tokenizer | `c518f4ee754fb30aed2497fd366e4a795f780572e0e1a463ea8f4c01b41a952f` |

The result reports `sdpa_kernel_observed: null`; no fused-kernel identity can
be claimed. One short run per configuration cannot quantify run-to-run noise,
statistical significance, large-model behavior, quality, or scaling laws.
The strict verifier checks recorded fields and arithmetic, but cannot
independently remeasure GPU work from the JSON. The `checkpoint.pt` field is
a filename in the result; no weights or checkpoint are included in Git.

## M7 direction

Use SDPA FP32 as the measured throughput baseline for a controlled M7 systems
study. First repeat the same bounded configuration enough times to report
spread and identify the selected SDPA kernel. Then vary one scale dimension
at a time and record throughput, allocator memory, validation loss, and full
training time. Any additional GPU work requires its own authorization and
execution plan; this evidence branch launches none.

To reproduce the offline verification:

```sh
uv run --no-sync python scripts/infra/prepare_kaggle_m6_pilot.py verify \
  --result benchmarks/infra/m6_b_t4_pilot_20261013.json \
  --sha 8eba5c0e006696ad6b1cfcb786b70ff00dc354ac
```
