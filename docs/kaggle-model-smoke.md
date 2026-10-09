# Kaggle single-T4 model smoke preparation

This prepares a **private** Kaggle GPU script pinned to an exact public Git
commit. The first job completed successfully on 2026-10-09. Kaggle's GPU setting must be
confirmed as **GPU T4 x1** in the kernel settings before launch; the metadata
requests GPU, but does not encode a GPU model or count. The job needs internet
to clone the public repository. It uses no dataset and no training-loop code.

The remote script checks out the pinned commit, verifies `git rev-parse HEAD`
against the 40-character SHA **before importing WendyFM**, then performs one
tiny CUDA forward, next-token loss, backward, and AdamW update. Its JSON output
records the actual loss, CUDA device name, synchronized step time in seconds,
and peak allocated GPU bytes. This is a functional smoke check, not a throughput
benchmark or full pretraining run.

## Verified result — 2026-10-09

The [verified result](../benchmarks/infra/gpu_model_smoke_20261009.json) reports
`PASS` for expected and tested commit
`39bc1b9cc857115f788f6b604845178677fc2ad0`. The repository verifier
accepted the downloaded JSON, including both commit SHAs and the assertions
that gradients were finite and AdamW updated model parameters.

| Observation | Measured value |
| --- | ---: |
| CUDA device | Tesla T4 |
| Next-token loss | 3.5073606967926025 |
| Synchronized one-step time | 1.3913469169999928 s |
| Peak allocated GPU memory | 17,345,024 bytes |
| Finite gradients | true |
| Parameter update | true |

This is one tiny forward, backward, and optimizer step. Its timing includes
first-step overhead and is **not** steady-state pretraining throughput. No
training-loop throughput or validation loss was measured by this job.

## Terminal-first workflow

Run locally from the repository root after the model code is available at a
public commit. Use a public SHA that contains `src/wendyfm/model`; local
uncommitted changes cannot be tested by this job.

```sh
KAGGLE_USER="your-kaggle-username"
EXPECTED_SHA="<exact-40-character-public-commit-sha>"
JOB_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wendyfm-model-job.XXXXXX")"
python3 scripts/infra/prepare_kaggle_model_smoke.py prepare --job-dir "$JOB_DIR" --user "$KAGGLE_USER" --sha "$EXPECTED_SHA"
cat "$JOB_DIR/kernel-metadata.json"
```

Inspect the generated script and metadata. Confirm `is_private: true`,
`enable_gpu: true`, `enable_internet: true`, and empty source lists. Confirm
**GPU T4 x1** in Kaggle settings. The prepare command only writes two files
outside the repository; it does not contact Kaggle.

```sh
kaggle kernels push -p "$JOB_DIR"
kaggle kernels status "$KAGGLE_USER/wendyfm-model-smoke"
```

Repeat the status command in separate terminal calls until complete. If it
fails, inspect Kaggle's kernel log. Download only after completion:

```sh
RESULT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wendyfm-model-result.XXXXXX")"
kaggle kernels output "$KAGGLE_USER/wendyfm-model-smoke" -p "$RESULT_DIR"
python3 scripts/infra/prepare_kaggle_model_smoke.py verify --result "$RESULT_DIR/model-smoke-result.json" --sha "$EXPECTED_SHA"
cat "$RESULT_DIR/model-smoke-result.json"
```

The verifier requires PASS, both matching SHAs, the expected test name, a
nonempty CUDA device, positive finite loss, time, and memory measurements,
finite gradients, and a parameter update. A failure records only an error
class to avoid exposing credentials in logs or output.

## Credentials and optional W&B design

The job needs no Kaggle API token inside its source. Keep Kaggle CLI credentials
on the local machine; never package `kaggle.json`, `.netrc`, W&B keys, private
data, or checkpoints. The script makes no W&B call, so this smoke run can work
without a W&B account.

For a later opt-in W&B instrumented run, store `WANDB_API_KEY` as a **Kaggle
Secret** attached to the private kernel, retrieve it at runtime through
`kaggle_secrets.UserSecretsClient().get_secret("WANDB_API_KEY")`, and pass it
directly to `wandb.login(key=...)`. Install or confirm the `wandb` package in
that run, then send only non-sensitive scalar metrics (loss, step time, GPU
memory) and the public commit SHA. Do not print the key, put it in command
arguments, write it to JSON, or upload environment/config files. Keep W&B
disabled by default and treat network/auth failure as optional telemetry rather
than a reason to conceal a failed model smoke check.
