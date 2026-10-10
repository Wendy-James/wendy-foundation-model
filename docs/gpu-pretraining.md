# M4 bounded FP32 single-GPU pretraining — Issue #2

## Question and baseline

Can the existing WendyFM tokenizer, decoder, sampler, AdamW loop, and exact
checkpoint continuation run eight FP32 optimizer steps on one CUDA device?
The baseline is the same seeded model before updates; the change is eight
updates on sampled training windows. This uses the M3 synthetic text fixtures
to validate the pipeline. It is not a language-model quality benchmark.

The [frozen configuration](../configs/gpu_fp32.json) sets vocabulary capacity
272, model width 32, two decoder layers, sequence length 16, batch size 4,
eight updates, and seed 20261009. The tokenizer trains only on training text.
The sampler returns CPU integer inputs and targets shifted once; both move to
CUDA before loss computation. The model scores explicit targets without a
second shift. Validation reads held-out text without updating parameters.

Training throughput counts 512 target tokens divided by synchronized wall
time around all eight updates. It includes sampling, host-to-device copies,
forward and backward computation, gradient checks, clipping, and AdamW. It
excludes tokenization, validation, and checkpoint writing. This small run
includes first-step overhead; report it as bounded-run throughput, not
steady-state performance. Peak memory is PyTorch peak **allocated** bytes
on `cuda:0` during the training interval. No GPU throughput, memory, or loss
measurements exist for M4 until a verified job returns them.

## Local validation

```sh
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run python scripts/pretrain_gpu.py --device cpu --config configs/gpu_fp32.json --output-dir outputs/m4_cpu_dry_run
```

CUDA tests skip on machines without CUDA. The CPU command exercises the new
entry point; its measurements must not be presented as GPU evidence.
Checkpoints and tokenizers stay in ignored `outputs/`.

## Next private Kaggle experiment

Run only after the M4 branch is pushed. Set `KAGGLE_USER` to the account that
owns the private job, and confirm **GPU T4 x1** in Kaggle settings. The
metadata requests a GPU but cannot specify its model or count.

```sh
cd ~/workspaces/projects/wendyfm-m4-gpu
M4_SHA="$(git rev-parse HEAD)"
JOB_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wendyfm-fp32-job.XXXXXX")"
uv run python scripts/infra/prepare_kaggle_gpu_pretrain.py prepare --job-dir "$JOB_DIR" --user "$KAGGLE_USER" --sha "$M4_SHA"
kaggle kernels push -p "$JOB_DIR"
```

The private script clones the pinned public commit, verifies HEAD before
importing WendyFM, runs one eight-step FP32 training path and a separate
four-plus-four continuation, then compares their complete checkpoints. It
publishes only `gpu-pretrain-result.json`; temporary checkpoints are deleted.
Poll `kaggle kernels status "$KAGGLE_USER/wendyfm-fp32-pretrain"` in separate
calls. After success, download the result and verify it locally:

```sh
RESULT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wendyfm-fp32-result.XXXXXX")"
kaggle kernels output "$KAGGLE_USER/wendyfm-fp32-pretrain" -p "$RESULT_DIR"
uv run python scripts/infra/prepare_kaggle_gpu_pretrain.py verify --result "$RESULT_DIR/gpu-pretrain-result.json" --sha "$M4_SHA"
```

Only after that verification should a small results manifest and measured
report be committed. No credentials, private logs, or model weights belong
in Git.
