# M6-B private T4 pilot package

This stacked branch prepares one private Kaggle script job for Issue #16. It
does not contain GPU evidence. The generated Python job embeds compressed,
byte-identical M6-A train, validation, and tokenizer artifacts; the raw
artifacts stay outside Git. The committed source manifest fixes their SHA256
hashes and the pinned FineWeb-Edu revision. At runtime the wrapper materializes
all three files in its temporary workspace and verifies their hashes before
CUDA or model initialization.

The wrapper checks out one full commit SHA, compares its own code with that
commit, masks T4x2 to one visible T4, then runs fresh A/B/C subprocesses in
sequence with distinct output directories. A is reference attention FP32; B is
PyTorch SDPA FP32; C is PyTorch SDPA FP16 autocast and GradScaler. All use the
same config bytes, 20261012 seed, 21,502,464-parameter model, 256-token
sequence, effective batch 8, and 20 optimizer steps (40,960 target tokens).
The runner rejects nonfinite loss, validation loss, gradients, gradient norm,
or parameters, a skipped GradScaler update, and reserved CUDA memory at 12
GiB. Any subprocess failure or 900-second arm timeout stops later arms. There
are zero retries and a 3600-second total timeout. OOM and other failures
produce a single `FAIL` JSON with the failed arm, concise failure stage, error
type, and error message; no partial result passes verification. The final
`PASS` JSON contains all three per-arm records.

The measured fields include train and held-out validation loss, target
tokens/sec, measured step time, peak CUDA allocated and reserved bytes, device,
GPU name, and PyTorch/CUDA versions. `sdpa_kernel_observed` is `null` because
the current runner has no reliable kernel observation; it does not infer a
FlashAttention backend from the API request. This short run measures systems
behavior and correctness, not language quality.

After the stacked PR has its final pushed commit SHA, prepare the package in
an empty directory outside Git:

```sh
uv run --no-sync python scripts/infra/prepare_kaggle_m6_pilot.py prepare \
  --job-dir /Users/admin/wendyfm-m6-b-job \
  --data-dir /Users/admin/workspaces/projects/wendyfm-m6-pretrain/data/raw/fineweb_edu_m6 \
  --user wendyzhan040513 --sha FULL_PUSHED_COMMIT_SHA
```

The future B1 submission is one command, with no automated retry:

```sh
kaggle kernels push -p /Users/admin/wendyfm-m6-b-job
```

Download the private job's `m6-pilot-result.json` and verify it offline:

```sh
uv run --no-sync python scripts/infra/prepare_kaggle_m6_pilot.py verify \
  --result /path/to/m6-pilot-result.json --sha FULL_PUSHED_COMMIT_SHA
```

Submission remains gated on B1 review. The package requests a private free T4
and no paid GPU, W&B, or dataset attachment.
