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

## Not yet verified

| Check | Status |
| --- | --- |
| Hugging Face account authentication | NOT YET VERIFIED |
| W&B account authentication and cloud logging | NOT YET VERIFIED |
| DSW connectivity and permissions | NOT YET VERIFIED |
| Full GitHub-to-Kaggle training pipeline | NOT YET VERIFIED |

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
