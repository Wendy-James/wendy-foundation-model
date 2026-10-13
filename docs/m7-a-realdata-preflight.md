# M7-A: longer real-data pretraining preparation

M7-A adds an offline data preparation and verification path. It does not
download the M7 corpus during CI, train a model, run a GPU job, or change the
frozen M6 executable. The actual M7 source manifest and its train,
validation, and tokenizer artifacts belong under ignored `data/raw/` or
outside the repository. The first completed preparation must be verified and
its manifest SHA256 recorded before an M7 training run is proposed.

## Reuse and upstream review

| Component | Compatibility and choice | License and maintenance checked 2026-10-13 |
| --- | --- | --- |
| [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), AdamW, RNG and tensor serialization | Already used by WendyFM's model, `train()`, and exact checkpoint APIs; keep SDPA FP32 as the M6 throughput baseline at the tested scale. No new framework. | PyTorch's [license file](https://github.com/pytorch/pytorch/blob/main/LICENSE) is BSD-style with third-party notices; repository activity was current on review. No upstream source copied. |
| [LitGPT pretrain/DataModule](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/pretrain.py) | Its explicit data module, evaluation interval, save interval, and throughput monitor are useful design references. Direct adoption would bring Fabric and replace WendyFM's own model, tokenizer, sampler, and checkpoint contract. Keep the existing WendyFM APIs and use a thin segment orchestrator in the next phase. | [Apache-2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.md); repository activity was current on review. No source copied. |
| [FineWeb-Edu `sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | The existing `fineweb_rows()` range-reads `id` and `text` from one Parquet shard at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. M7 reuses that pinned iterator and the M6 document split rule; no `datasets` dependency or unpinned viewer API. | Dataset card at the pinned revision declares ODC-BY and the `sample-10BT` train split. The dataset was last modified 2025-07-11 when checked. Respect source attribution and keep text outside Git. |

Local API reuse: `real_data.fineweb_rows()` and `split_for_id()` supply pinned
source order and document-level partitioning; `BPETokenizer` supplies the
frozen vocabulary; `prepare_token_ids()` and `NextTokenBatchSampler` supply
aligned windows; `real_pilot.run()` uses `train()`, `validation_loss()`, and
`save_checkpoint()`/`load_checkpoint()` with sampler and RNG state. The M7
manifest retains the fields that `real_pilot.run()` validates, and an offline
test exercises its segment/resume path. M7-A does not change these APIs.

## Corpus contract and preflight

Use the M6 tokenizer bytes from ignored local storage. Its small 1,024-ID
vocabulary may be inefficient for a longer run; measure tokenization cost and
quality before a later tokenizer experiment. The CLI requires its SHA256 to
match the committed M6 manifest; a test-only API override permits synthetic
tokenizers. The M7 preparation streams source rows in pinned shard order and
selects the shortest prefix that
reaches at least **20,000,000 training tokens** and **200,000 held-out
validation tokens**, counting one EOD token per document. It caps scanned
source rows at 50,000 and selected UTF-8 text at 120,000,000 bytes. If either
floor is unmet within those caps or the pinned first shard ends, preparation
fails; expanding the source requires a separately reviewed pinned shard
contract. No silent resampling or source switch occurs.

The existing SHA256-of-ID split rule keeps M6 validation IDs in validation.
Exact duplicate text is skipped across both splits, and duplicate source IDs
fail preparation. The output holds `train.jsonl`, `validation.jsonl`, the
byte-copied frozen `tokenizer.json`, and `manifest.json`. The manifest records
the revision and shard, source row count, skipped duplicate count, each
document's ID/split/text hash/byte count/token count, split token totals,
and artifact SHA256 values. Preparation stages files in a temporary
directory and publishes the directory only after both floors pass. The
preflight rehashes artifacts, retokenizes each document, checks split and
content separation, and recomputes totals. It cannot prove that supplied
rows came from Hugging Face without independently reading the pinned source;
the source iterator and manifest must be reviewed together.

```sh
PYTHONPATH=src uv run --no-sync --with 'pyarrow>=20,<25' \
  --with 'fsspec>=2025' --with 'aiohttp>=3' --with 'requests>=2' \
  python scripts/prepare_m7_data.py prepare \
  --output-dir data/raw/fineweb_edu_m7 \
  --tokenizer data/raw/fineweb_edu_m6/tokenizer.json

PYTHONPATH=src uv run --no-sync python scripts/prepare_m7_data.py verify \
  --output-dir data/raw/fineweb_edu_m7
```

Both commands are for a future local data-preparation step. M7-A tests use
synthetic rows only. The `data/raw/` path is ignored by Git; the API also
rejects a repository-local output path outside that ignored directory.

## Token and GPU budget

The [verified M6-B evidence in Draft PR #20](https://github.com/Wendy-James/wendy-foundation-model/pull/20)
records **16,602.380302 target tokens/s** for SDPA FP32, 21,502,464
parameters, T=256, effective batch 8, and 2,048 scored target tokens per
optimizer step on one T4. This is 18 timed steps in a short pilot, so it is
a planning baseline, not a sustained M7 benchmark. M6's 119,426 training
tokens would be reused roughly 8.4, 41.9, or 167.5 times by the budgets
below; the new 20M-token floor reduces nominal token-budget/corpus ratios
to about 0.05, 0.25, and 1.00. Random-window sampling with replacement
can still revisit positions; those ratios do not measure actual coverage.

| Requested target tokens | Whole steps | Actual scored tokens | M6-rate training minutes | Planning GPU minutes |
| ---: | ---: | ---: | ---: | ---: |
| 1M | 489 | 1,001,472 | 1.01 | 4.31 |
| 5M | 2,442 | 5,001,216 | 5.02 | 13.53 |
| 20M | 9,766 | 20,000,768 | 20.08 | 48.10 |

Planning minutes use `actual_tokens / 16,602.380302 / 60 × 1.30 + 2 + N`,
where 30% covers sustained-throughput loss, 2 minutes covers setup and
initial validation, and `N` is one minute per 1M-token validation/checkpoint
interval, including the final interval. These allowances are **assumptions**,
not M7 measurements. In particular, 200k-token validation cost and longer
run throughput have not been measured. Stop or replan if observed timing
breaks the bound. On an available free Kaggle T4 the proposed 1M pilot has
**$0 direct GPU charge** and about **4–5 minutes** of planned GPU quota;
paid GPU use is outside this plan. The 5M/20M figures are comparisons, not
authorized jobs.

## Next training contract and stop criteria

For M7-B, freeze one config and manifest before launch. Use SDPA FP32 and a
thin runner around the existing `train()`, `validation_loss()`,
`save_checkpoint()` and `load_checkpoint()` APIs, with the same data hashes,
tokenizer, sampler, and output directory on resume. `real_pilot.run()` already
proves exact segment continuation, but calling it at every interval would
reload and retokenize the whole corpus and repeat initial validation. Keep
the corpus and model alive across intervals in one job. The final
`TrainConfig.max_steps` must be fixed up front because the cosine schedule
and checkpoint fingerprint depend on it. Write each interval's metrics to
an ignored append-only log before the next interval; checkpoints stay ignored.

At step zero and approximately every 1M target tokens (489 steps), evaluate
the full held-out set with the existing nonoverlapping-window validation
function. Record global step, cumulative scored tokens, training loss curve
(per-step or window mean), validation loss, `timed_target_tokens / timed_seconds`,
training/validation/checkpoint wall times, peak CUDA allocated and reserved
bytes, config/source/tokenizer hashes, and checkpoint/resume provenance.
Report both timed throughput and full-run effective throughput. A 1M pilot
should first test this cadence and actual validation/checkpoint overhead;
only then decide whether 5M or 20M is justified.

Proposed hard stops for a later authorized T4 run: nonfinite loss/gradient,
failed parameter update, data or resume hash mismatch, CUDA OOM, 12 GiB
reserved allocator peak, 60-minute job wall bound, or unavailable free GPU
quota. Compare validation loss against its previous recorded value and stop
for review if it rises in two consecutive evaluations; a short noisy rise
alone is not evidence of overfitting. No GPU job is launched in M7-A.
