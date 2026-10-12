# M6-A real-data pilot contract

This phase prepares code and offline tests. No GPU measurements are reported here.
The real corpus is created by `scripts/prepare_fineweb.py` under ignored
`data/raw/fineweb_edu_m6`; only its source manifest belongs in Git. The
preparation uses HTTP range reads against the pinned first Parquet shard, not
a full sample download. The dataset-viewer rows API does not enforce the
`revision` query value and is unsuitable for this source contract.
The committed [source manifest](../data/manifests/fineweb_edu_m6.json) records
64 real document IDs, 246,033 UTF-8 text bytes, 57 train / 7 validation
documents, and 119,426 / 8,738 tokens with EOD respectively. Raw text and
the frozen tokenizer stay under ignored `data/raw/`.

## Upstream provenance

- PyTorch `torch.nn.functional.scaled_dot_product_attention` provides the
  optimized causal attention API. WendyFM's existing reference implementation
  remains the correctness oracle. PyTorch is BSD-style licensed; no PyTorch
  source was copied. PyTorch chooses an SDPA kernel at runtime, so the runner
  reports the requested API and leaves actual kernel observation `null` until
  a profiler captures it. [SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- Hugging Face `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split `train`,
  revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` is ODC-BY. The
  preparation command uses optional PyArrow and fsspec range reads and stops at
  64 selected documents or 250,000 UTF-8 text bytes for the checked pilot
  manifest. It does not request the full sample. Source IDs, text hashes,
  token counts, and artifact hashes are recorded. [Dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [LitGPT pretrain.py](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/pretrain.py)
  (Apache-2.0) informed explicit parameter counting and throughput fields.
  WendyFM keeps its own loop and uses observed target token counts divided by
  measured wall time; it does not copy LitGPT code or report FLOP/MFU guesses.
- [TorchTitan](https://github.com/pytorch/torchtitan) (BSD-3-Clause) informed
  the resume fingerprint, sampler state, and allocated/reserved CUDA memory
  fields. WendyFM keeps its existing single-device checkpoint and sampler;
  no TorchTitan code or distributed machinery was copied.

## Frozen workload

`configs/m6_real_pilot.json` defines 6 layers, width 512, 8 heads, SwiGLU
intermediate width 1536, untied 1,024-token embedding and output head, RoPE,
and sequence length 256. The actual model has **21,502,464 trainable
parameters**, counted with `sum(p.numel() for p in model.parameters() if
p.requires_grad)`. Microbatch 2 and 4 accumulation steps give effective batch
8 and 2,048 target tokens per optimizer step. The 20-step pilot is bounded at
40,960 target tokens. Warmup excludes the first 2 steps from throughput.
Reference attention and SDPA are separate explicit config choices. FP32 is the
baseline; `--precision fp16` requires CUDA autocast and GradScaler.

A conservative planning allowance for a 16 GiB T4 is **4 GiB**: FP32 weights,
gradients, and two AdamW moment tensors are `16 × 21,502,464 = 344,039,424`
bytes (328.1 MiB). The remaining roughly 3.7 GiB allowance covers untied
logits, reference attention score/probability tensors, other saved activations,
optimizer temporaries, CUDA context, and allocator fragmentation. This is a
planning estimate, not a measured peak or a proof of fit. B1 should stop if
peak reserved memory approaches 12 GiB; actual allocated and reserved peaks
are recorded separately. FP16 still keeps FP32 parameters and AdamW states.

## Preparation and run

```sh
PYTHONPATH=src uv run --no-sync --with 'pyarrow>=20,<25' --with 'fsspec>=2025' --with 'aiohttp>=3' --with 'requests>=2' python scripts/prepare_fineweb.py --max-documents 64 --max-text-bytes 250000 --vocab-size 1024
PYTHONPATH=src uv run --no-sync python scripts/pretrain_real.py --device cpu
```

The preparation step trains WendyFM BPE once on **train documents only** and
serializes it beside the ignored raw text. Validation IDs are a stable hash
partition, not a random window split. The runner checks every source document
hash and frozen tokenizer hash before training. The existing sampler shifts
next-token targets once. Validation scores nonoverlapping held-out windows
independently of optimizer updates. Resume checks source/config/tokenizer
fingerprints, restores the sampler and RNG, and restores GradScaler state for
FP16. FP32 CPU continuation is tested exactly; CUDA and FP16 remain unmeasured.

The source manifest and raw artifacts must travel together to the later private
pilot. The source commit hash in a result identifies the checked-out executable
code. A matched A/B run needs separate output directories and the same frozen
source, seed, schedule, model dimensions, and token budget. Only the explicit
attention backend or precision should differ.
