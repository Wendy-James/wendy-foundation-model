# M7-C: private Kaggle delivery, offline package

This branch adds only a private Kaggle packager and remote wrapper around the
existing `scripts/pretrain_long.py`. It does not include a corpus, tokenizer
artifact, weights, checkpoint, GPU run, or measured M7 result. Issue #22's
M6-B SDPA FP32 16,602.380302 target tok/s was measured over 18 steps; the
roughly 4.31-minute, $0-direct-charge free-T4 1M plan remains an estimate.

## Gates before a single submission

1. B1 completes M7-A's full FineWeb-Edu preparation and `prepare_m7_data.py
   verify` using the frozen M6 tokenizer. Record the exact `manifest.json`
   SHA256, the three artifact SHA256 values, 20M train and 200k validation
   token floors, source revision, local byte count, and CPU preparation time.
   The 7-document smoke is too small for this job. The full manifest SHA is
   intentionally a required packager input; no value is assumed here.
2. Keep the verified four files together in an ignored local corpus directory.
   Create a **private** Kaggle Dataset under the same account used for the
   kernel. Its attached root must have exactly `manifest.json`,
   `tokenizer.json`, `train.jsonl`, and `validation.jsonl`. Check its privacy
   in Kaggle's dataset UI before attaching it. The CLI's `datasets create`
   default is private; never pass `--public`. Do not put these files in the
   kernel source directory or Git. If upload staging needs
   `datasets-metadata.json`, Kaggle must omit it from the mounted file root.
3. Push this M7-C branch and use its **full commit SHA** as `--sha`. The
   packager checks the frozen config SHA256
   `71c39117cd56105f892470cd5a20d823c78c64ccdaf88b86721963d9cc523ad4`
   and tokenizer SHA256
   `c518f4ee754fb30aed2497fd366e4a795f780572e0e1a463ea8f4c01b41a952f`.
   The exact source and wrapper are rechecked from that commit on Kaggle.
4. Confirm free T4 quota, private dataset access, GitHub access from the
   Kaggle session, and one intended submission. Obtain separate B1 approval
   before launching. There are zero retries and no paid accelerator.

## Package and launch commands for B1

Set these values from the actual verified corpus and pushed commit:

```sh
export M7_CORPUS_DIR=/absolute/ignored/path/to/fineweb_edu_m7
export M7_MANIFEST_SHA256=FULL_64_HEX_VERIFIED_MANIFEST_SHA256
export M7_COMMIT_SHA=FULL_40_HEX_PUSHED_M7_C_COMMIT_SHA
export KAGGLE_USER=YOUR_KAGGLE_USERNAME
export M7_DATASET_SLUG=wendyfm-m7-corpus-private
export M7_JOB_DIR=/absolute/path/outside/git/m7-kaggle-job

PYTHONPATH=src uv run --no-sync python scripts/prepare_m7_data.py verify \
  --output-dir "$M7_CORPUS_DIR"
PYTHONPATH=src uv run --no-sync python scripts/infra/prepare_kaggle_m7.py prepare \
  --job-dir "$M7_JOB_DIR" --data-dir "$M7_CORPUS_DIR" \
  --user "$KAGGLE_USER" --dataset-slug "$M7_DATASET_SLUG" \
  --sha "$M7_COMMIT_SHA" --manifest-sha256 "$M7_MANIFEST_SHA256"
```

For the private dataset, stage only those four corpus files in a separate
directory outside Git, run `kaggle datasets init -p DATASET_STAGE`, set the
metadata ID to `$KAGGLE_USER/$M7_DATASET_SLUG`, then make **one** private
`kaggle datasets create -p DATASET_STAGE` call without `--public`. Inspect the
resulting mounted file list and visibility before the kernel launch. Dataset
upload is a separate B1 action; this code never uploads it.

After all gates pass and B1 authorizes the one free GPU job:

```sh
kaggle kernels push -p "$M7_JOB_DIR" --timeout 3600
kaggle kernels status "$KAGGLE_USER/wendyfm-m7-1m-private"
kaggle kernels output "$KAGGLE_USER/wendyfm-m7-1m-private" \
  -p /absolute/path/outside/git/m7-result-download
PYTHONPATH=src uv run --no-sync python scripts/infra/prepare_kaggle_m7.py verify \
  --result /absolute/path/outside/git/m7-result-download/m7-result.json \
  --data-dir "$M7_CORPUS_DIR" --sha "$M7_COMMIT_SHA" \
  --manifest-sha256 "$M7_MANIFEST_SHA256"
```

Do not repeat `kernels push` automatically if the status is running or fails.
The Kaggle CLI's `--timeout 3600` is the outer hard limit. The wrapper stops
its child by 3,540 seconds to leave time for a compact failure record. It
masks Kaggle's T4×2 to one visible T4 before importing Torch, validates the
commit, wrapper, config, manifest and all three artifact hashes before CUDA,
then calls the existing runner once with SDPA FP32, 489 steps and 1,001,472
scored target tokens. In-process nonfinite, memory, provenance and wall checks
remain in M7-B. CUDA behavior, sustained speed, full validation overhead,
and exact GPU resume have not been measured by these offline tests.

Only `m7-result.json` leaves the temporary workspace: run plan, step-zero
and completion events, checkpoint **metadata** (size and SHA256), or a
sanitized error stage/type/message. The checkpoint weights and private corpus
are never copied to Kaggle output. A 20M-token corpus does not imply all 20M
tokens are consumed in this 1M scored-token run; report actual sampler use
and repeats when interpreting the result.
