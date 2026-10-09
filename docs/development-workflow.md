# WendyFM development and research workflow

**Adopted:** 2026-10-09  
**Scope:** WendyFM foundation-model engineering only. Post-training SFT/GRPO research runs in a separate project/workflow.

This is an operating agreement, not evidence that every service or end-to-end pipeline has been tested. See [environment connectivity](environment-connectivity.md) and experiment manifests for individual verification.

## Responsibility boundaries

| Component | Responsibility |
| --- | --- |
| ChatGPT with authorized GitHub connector | Planning, architecture, review, task decomposition, repository inspection, PR review. Direct GitHub writes only on explicitly scoped files/branches; no direct assumption of access to Mac terminal state. |
| Mac terminal (B1) | Reproduce tests, run small CPU jobs, submit/monitor permitted GPU experiments, inspect logs and manifests, verify Git sync and CI. |
| Mac Codex CLI (B2) | Core WendyFM implementation, bug fixes, tests, scoped commits, branch publication. |
| Codex Cloud | Optional independent review, documentation, and simple isolated tasks; not a dependency for the critical path. |
| GitHub | Canonical version history, commits, draft PRs, issue tracking and CI (when configured on the relevant branch). |
| Kaggle / independent GPU services | Bounded experiments using pinned source commits and approved budget. |
| Weights & Biases | Optional non-sensitive scalar training metrics and experiment tracking. |
| Hugging Face | Permitted public model artifacts, datasets and research outputs. |
| Company DSW | **Outside WendyFM.** Only separately authorized company work; never move company source, data, internal metrics, model weights or credentials to this repository. |

A separate pair of research windows (A1/A2) belongs to the SFT/GRPO paper. WendyFM operates from B1/B2 only. These are human workflow roles, not continuously running autonomous services.

## Verified vs. pending as of 2026-10-09

### Reported and/or repository-evidenced

- Mac Git/GitHub CLI/SSH and Codex CLI can access the authorized repository.
- Independent Git worktrees, normal commits, pushes, and draft PRs have been used.
- ChatGPT's GitHub connector can inspect the repository and comment on authorized PRs/issues.
- A GitHub Actions **integration smoke** workflow exists on branch `test/chatgpt-github-link-20261009` (draft PR #1); it is **not on `main`** as of this review. The workflow has previously run its syntax and pytest checks according to operator evidence. Do not imply it protects all branches or runs by default.
- Codex Cloud has accessed the repository for independent review/lightweight work, with prior reported tests and Ruff validation.
- Mac → Kaggle API, pinned Git SHA Kaggle tokenizer test, results download/verification, Mac → W&B online metric logging and Hugging Face CLI authentication were previously validated; see [connectivity runbook](environment-connectivity.md).
- WendyFM one-step private Kaggle Tesla T4 CUDA/AdamW smoke has a saved result: [verified JSON](../benchmarks/infra/gpu_model_smoke_20261009.json). A one-step GPU smoke is **not** a full GPU pretraining benchmark.

### Not automatically established

- Codex Cloud's fresh-task automatic `.venv` recovery is unresolved: a previous `uv sync` encountered network access errors while fetching a dependency. Do **not** spend engineering time on repeated recovery attempts without a task-specific need.
- Mac Codex CLI is API-key authenticated in the current setup; its use must **not** be assumed to charge ChatGPT Plus Codex quota.
- GitHub Actions coverage beyond the integration workflow's actual trigger/branch, full GPU pretraining, GPU resume correctness, W&B cloud-to-GPU integration and Hugging Face model uploads must be verified independently before claiming completion.

## Development rules

1. First inspect `AGENTS.md`, existing branches/worktrees and `git status`. Preserve uncommitted changes. Use a distinct branch for each substantial task; do not recreate or overwrite an existing worktree/branch.
2. Assign non-overlapping responsibilities to B1 and B2. Avoid running two writers against the same paths.
3. For core changes, implement a minimal correctness-first approach; use existing tokenizer, decoder and training interfaces rather than rebuilding them.
4. Before pushing: run the relevant tests and Ruff, check `git diff --check`, inspect staged paths and secret/size boundaries; publish a descriptive commit and draft PR. **Never auto-merge a PR.**
5. Cite a real immutable SHA for remote experiments. Specify config, dataset/source fingerprints, hardware, steps, measurement definitions, and limits. Use a private bounded GPU job and no automatic retries.
6. Preserve small verified experiment evidence (manifest, scalar metrics, scripts, and interpretation) in Git; do not commit credentials, private datasets, large model weights, giant logs, company information or sensitive paths.
7. Do not claim successful training, GPU performance, scaling, exact resume, HF upload or W&B synchronization without corresponding measured evidence.
8. Keep Codex Cloud optional. Do not reinstall or debug the already validated local toolchain without a concrete failure.
9. Expense guard: free/available compute first; local correctness test first; request explicit approval before renting paid GPU.

## Mainline progression

1. **M3 closeout:** preserve real CPU pretraining manifest, exact CPU resume proof, and one-step Kaggle CUDA smoke JSON in Git.
2. **M4 single-GPU FP32:** implement/test bounded device-aware next-token training, synchronized step timing, true target-token throughput, peak allocated memory, and checkpoint/resume; verify a real remote run, not only an offline mock.
3. **Profiling and controlled optimization:** measure real bottlenecks; isolate baseline and single changes with matched batch/context/model settings.
4. **Scaling and multi-GPU:** proceed only after single-GPU baseline and a reviewed cost/benefit plan.

The goal is evidence of real pretraining, training-system understanding, GPU performance analysis, and ultimately scaling and distributed training for foundation-model core algorithm internships.
