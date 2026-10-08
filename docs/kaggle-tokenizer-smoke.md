# GitHub-to-Kaggle tokenizer smoke test

This test ran as a private, CPU-only Kaggle script with internet access. It
cloned the public WendyFM repository, checked out an exact commit SHA, imported
`BPETokenizer` from that checkout, trained twice on a tiny fixed corpus, and
asserted deterministic specifications and an encode/decode round trip. The
downloaded `tokenizer-smoke-result.json` was verified on the Mac.

## Observed result — 2026-10-08

**PASS** for the remote tokenizer smoke test. The expected and tested commit
SHAs both equal `5f810f4e1d6a7f83a68aa350a175b7a18ff469ef`; `round_trip`
and `deterministic` are both `true`, and `token_count` is `5`. The downloaded
evidence is saved at `benchmarks/infra/kaggle_tokenizer_smoke_20261008.json`.
This job used CPU. Full GPU pretraining remains unverified.

## Mac terminal commands

Run from the WendyFM repository root. Set your own Kaggle username. The SHA is
read from GitHub's public `main` ref, so local uncommitted changes are excluded.

```sh
KAGGLE_USER="your-kaggle-username"
EXPECTED_SHA="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
JOB_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wendyfm-kaggle-job.XXXXXX")"
python3 scripts/infra/kaggle_smoke.py prepare --job-dir "$JOB_DIR" --user "$KAGGLE_USER" --sha "$EXPECTED_SHA"
cat "$JOB_DIR/kernel-metadata.json"
```

Inspect the metadata: `is_private` must be `true`, `enable_gpu` and
`enable_tpu` must be `false`, and `enable_internet` must be `true`. Submit only
after that check:

```sh
kaggle kernels push -p "$JOB_DIR"
kaggle kernels status "$KAGGLE_USER/wendyfm-tokenizer-smoke"
```

Check status again with the same command until Kaggle reports completion. Then
download and verify the result in a second temporary directory:

```sh
RESULT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wendyfm-kaggle-result.XXXXXX")"
kaggle kernels output "$KAGGLE_USER/wendyfm-tokenizer-smoke" -p "$RESULT_DIR"
python3 scripts/infra/kaggle_smoke.py verify --result "$RESULT_DIR/tokenizer-smoke-result.json" --sha "$EXPECTED_SHA"
```

The verifier prints `PASS` only when the run succeeded, both recorded SHAs
equal the pinned SHA, and the tokenizer assertions succeeded. A failure exits
nonzero; inspect the downloaded JSON and Kaggle job log.

## Result JSON

The saved Kaggle result contains these values; `token_count` was calculated by
the checked-out source:

```json
{
  "status": "PASS",
  "expected_commit_sha": "5f810f4e1d6a7f83a68aa350a175b7a18ff469ef",
  "tested_commit_sha": "5f810f4e1d6a7f83a68aa350a175b7a18ff469ef",
  "test": "tokenizer_round_trip",
  "token_count": 5,
  "round_trip": true,
  "deterministic": true
}
```

A failed run writes `status: "FAIL"` with an `error` field when possible.

## Data and credential boundary

The generated job directory contains only the pinned public-repository URL,
commit SHA, script, and metadata. It contains no repository credentials or
local source copy. Kaggle API authentication stays on the Mac. Do not copy
`~/.netrc`, API keys, Hugging Face or W&B credentials, or company DSW data into
the job or output directories. These directories are created outside Git.
