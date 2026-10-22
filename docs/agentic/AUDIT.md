# P0 engineering audit — 2026-10-22

Planning: [Issue #27](https://github.com/Wendy-James/wendy-foundation-model/issues/27).
Read repository AGENTS.md; this task explicitly expands the original pretraining scope.
No paper repository was opened or modified.

## Baseline and ancestry

Selected **088d63a1e8bc0c78f49b41445aab867e95ffecbb**, the M3 feature head.
This is the smallest existing verified core with BPE tokenizer, explicit RoPE/RMSNorm/
SwiGLU decoder, aligned next-token sampling, AdamW loop and exact CPU checkpoint tests.
Agentic code adds no dependency and does not import Torch. Baseline tests were rerun:
152 passed (4.09s); warning: the existing Torch installation has no NumPy installed.

Main 2891aa0 documents M1 and is not an ancestor of this stack's later README change.
Do not combine the separate model PR or historical evidence automatically.
Consolidation proposal: review/merge the core stack separately, then retarget or rebase
this focused PR under explicit review; current Draft PR targets M3 to show only P0 changes.

Inspected actual core source/test paths: `model/transformer.py`, `attention.py`,
`training/loop.py`, `checkpoint.py`, and their tests; later `training/profile.py`,
`scaling.py`, `real_pilot.py`, `long_data.py`, `long_run.py`, interval tests, and
`scripts/infra/prepare_kaggle_m7.py` / `tests/infra/test_kaggle_m7.py` on remote refs.
Reuse is architectural contracts/provenance discipline; no GPU launcher is invoked.

Verified stack (ancestor to descendant): M3 088d63a → M4 99264e0 → M5 profiling
48e0c3e → M5 scaling 33ab89b → M6 pilot 8eba5c0 → M7-A c559452 → M7-B 98bb78c
→ M7-C 6599a26 → verifier f8d7956. M6 evidence ee8526a is a sibling of M7-A.

| PR | Head reviewed | Base | Scope / evidence limits |
| --- | --- | --- | --- |
| #20 | ee8526ab24b7e9bd34dbf4ecdbe0e3ed2aa3cc53 | M6 pilot | Archived M6-B evidence; executable 8eba5c0e006696ad6b1cfcb786b70ff00dc354ac; result SHA256 589690b85dfec85b003bb1a0d53d0099c23e801ed393fc50ab958504e0794d4d. Historical measurements, not rerun here. |
| #21 | c5594528a2fbb27e4489d3aefbd445062a058522 | M6 pilot | Larger corpus preparation; prepared tokens are not trained tokens. |
| #23 | 98bb78c5cef6cdcdf4ec88bf29acec79eb81d1da | M7-A | Resumable interval runner, offline tests. |
| #25 | 6599a2637e3bd0a96f009e0a8b793fe9c82b8952 | M7-B | Private delivery preparation; no confirmed M7 training. |
| #26 | 18cee0dc595c16751c03e9f26325190634d327f3 (PR response); remote f8d7956 | M7-C | Accounting/tamper verifier; remote includes later strict integer follow-up. |

All 21 registered local worktrees were clean at audit time; no branch was overwritten.
Fetched remote refs before selection. New unique worktree:
`/Users/admin/workspaces/projects/wendyfm-agentic-p0`, branch
`feat/agentic-p0-20261022`. No existing worktree or ignored artifact was changed.
No running task was interrupted. No prototype candidate was found in the project-directory
name inventory or tracked repository paths; an externally prepared prototype remains unavailable.
