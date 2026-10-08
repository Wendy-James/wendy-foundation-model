# Environment connectivity runbook

This records the WendyFM connectivity checks reported as complete. It does not
establish a full training pipeline or account access beyond the checks listed.

## Verified

| Check | Result |
| --- | --- |
| Mac to GitHub SSH and API | PASS |
| Mac to Codex CLI | PASS |
| Mac to Kaggle API | PASS |
| Kaggle dual Tesla T4 CUDA matrix multiplication | PASS |
| Kaggle results downloaded to Mac | PASS |
| Kaggle reads the public WendyFM GitHub README | PASS |
| Kaggle runs the pinned WendyFM tokenizer source on CPU; round trip and deterministic training | PASS |
| Hugging Face CLI authentication | PASS |
| Hugging Face `whoami` | PASS |
| W&B CLI login | PASS |
| W&B authentication stored in `~/.netrc` | PASS |
| Mac to W&B online metric logging | PASS |

For the W&B logging check, `wandb.init(mode="online")` and
`run.log({"test_metric": 1.0})` succeeded, five files synchronized, and the
run finished successfully.

The tokenizer smoke job ran on Kaggle CPU at commit
`5f810f4e1d6a7f83a68aa350a175b7a18ff469ef`. Its downloaded result was
verified on the Mac: `status`, `round_trip`, and `deterministic` passed, and the
expected and tested commit SHAs matched. Evidence:
`benchmarks/infra/kaggle_tokenizer_smoke_20261008.json`. See
`docs/kaggle-tokenizer-smoke.md` for the procedure and result.

## DSW local tool inventory

The DSW local tool inventory is complete. This records only an inventory of
locally available tools; it does not establish DSW connectivity, execution, or
end-to-end permissions.

## Not yet verified

| Check | Status |
| --- | --- |
| Hugging Face model upload | NOT YET VERIFIED |
| DSW connectivity and end-to-end permissions | NOT YET VERIFIED |
| Full GPU pretraining on Kaggle | NOT YET VERIFIED |

## Platform responsibilities

| Platform | Responsibility |
| --- | --- |
| Mac | Development |
| Codex | Implementation |
| GitHub | Version control |
| Kaggle | Personal GPU experiments |
| Hugging Face | Models and datasets |
| W&B | Experiment tracking |
| DSW | Authorized company workloads only |

## Terminal-first workflow

1. Develop and test WendyFM locally from the terminal. Review changes before
   publishing the public repository to GitHub.
2. Use the terminal to submit public-research jobs through the Kaggle API.
   Verify the required code and inputs are available in the Kaggle environment.
3. Download results to the Mac, inspect them locally, and record the command,
   configuration, metrics, and interpretation for reproducible experiments.
4. Treat each unverified service or end-to-end connection above as a separate
   check; record an observed result before marking it verified.

WendyFM is personal, public research. Keep its code, data, credentials, and
results separate from company DSW resources. Use DSW only for authorized
company work under company access rules; do not move company code, data, or
outputs into WendyFM or its public GitHub repository.

## Credential safety

Never commit `~/.netrc`, credentials, or tokens. Keep authentication files and
secret values out of the repository, logs, reports, and experiment artifacts.
