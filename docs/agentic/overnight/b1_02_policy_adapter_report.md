# B1 / 02 — Model policy reliability, offline audit

Date: 2026-10-22 (Asia/Shanghai). Outcome: PASS for the bounded offline acceptance conditions: meaningful narrow improvements validated by tests. Genuine Qwen inference remains UNVERIFIED.

## Evidence and scope

- Worktree: `/Users/admin/workspaces/projects/wendyfm-30h-v3-dev-20261022_205603`.
- Source HEAD and pinned baseline: `ce66312aad0e47c38aae7225a9ab124dcd03880f`. Changes are uncommitted; HEAD alone does not identify the modified implementation.
- Read `/Users/admin/Downloads/wendyfm_30h_guard_v3/evidence/CONTEXT.md` before source work. Its issue #27 / PR #28 / PR #29 evidence is historical, not fresh approval.
- `LIVE_GITHUB_DISCUSSION=NOT_REFRESHED`: no live GitHub connector/search capability was exposed in this session. No fresh comments or review approval claimed.
- Earlier B1 directory contained only `02_policy_adapter.before.git.txt` and `02_policy_adapter.codex.log`; inspected both. No completed B1 01 report existed there. The before-state file recorded the worktree branch with no changes.
- Modified only the policy adapter, its tests, and this report. Environment, contracts, runtime, cleanup helper and verifier were read for boundary review, not modified. No trainer, inference server, PR, push or merge added.

Tested file SHA-256:

| File | SHA-256 |
| --- | --- |
| `src/wendyfm/agentic/model_policy.py` | `9775928d6610e29aef987c9bb03b28c84ccb6c0b5bbe557898eb14c517d4ad4a` |
| `tests/agentic/test_model_policy.py` | `0f6ca1db4c5628246e2d150e09be48c0a3d5e3c8047be406b5bfd95589788830` |

## Narrow changes and interpretation

1. **Transactional policy history.** Previously, generation exceptions, malformed output and empty output consumed the initial user message or appended feedback before yielding a valid action; retrying the same observation then failed as stale. Build a candidate transcript, isolate it from backend mutations, and commit transcript/feedback state only after successful validation. Caller-directed retry now works without duplicate tool responses. Production `_rollout` still terminates on policy errors; this change does not introduce automatic retries or unbounded recovery.
2. **Feedback continuity.** A pending tool response must preserve the original instruction, tool schema, and complete previous feedback prefix. Same-length rewritten history and changed tool registrations are rejected before inference. This protects transcript integrity; it is not a general untrusted-environment sandbox.
3. **Token metrics and final extraction.** Require integer input counts in `1..4096` and integer output counts in `1..max_new_tokens`; booleans, zero output and over-bound metrics fail. Reject stray `</think>` as well as existing thinking/tool/special-token markup checks. A plain nonempty answer remains legal; the parser does not require every turn to contain a tool call.
4. **Session recovery classification.** A successfully decoded response clears the previous session error classification. A spawned mock worker can recover after a parse error without restart. Cancellation still uses the existing bounded cleanup helper.

Baseline/change/metric/result: baseline full suite passed 198 tests. Eleven new retry/history-continuity regressions run against the original adapter failed, confirming observable gaps rather than tests that merely mirror the fix. With the modified adapter the full suite passed 233 tests, an increase of 35 cases; the adapter file now contains 54 collected cases. These are correctness results, not model performance or reward-improvement measurements.

## Template, mask, context and tool boundary review

- Factory pins `Qwen/Qwen3-0.6B` and model/tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`; local-only snapshot and pretrained loading, `trust_remote_code=False`, CPU float32 evaluation. No download attempted.
- Backend retains Transformers `apply_chat_template` with tools, generation prompt, `enable_thinking=False`, tensor/dictionary output. Returned inputs, including attention mask, go directly to built-in `generate`; deterministic generation uses the configured token bound. No second tokenizer or custom template was introduced.
- Input count comes from templated tensor width; inputs over 4096 fail before generation. Tests include exactly 4096 and 4097. Decoder-only output is sliced after the prompt; output count includes returned special tokens, whereas decoding skips them. Counts are generated token counts, not a claim of answer-only token counts. Empty decoded output and zero returned tokens fail at the policy boundary.
- Strict parsing accepts one full `<tool_call>` JSON envelope with exact `name`/`arguments` fields or one plain answer. Missing fields/tags, duplicate names/argument keys, NaN, multiple calls and ordinary text mixed with calls fail closed. Existing repeated `lookup` names on successive turns remain supported and verified by the four-step mock rollout.
- `registered_tools` is not the execution authorization boundary. Unknown names and ill-typed arguments remain recorded invalid actions for feedback/replay. The host-owned `LookupEnvironment.step` exclusively authorizes `lookup` with exactly one string `key` of at most 128 characters; values must be keys in its public world. No shell, eval, arbitrary tool dispatcher or model-written Python is executed.
- Invalid actions receive fixed public error messages without argument echo. Successful observations contain public synthetic lookup values only. New tests cover extra fields, overlong keys, nested arguments and tool markup as a key. Contract decoding rejects unknown observation fields; the adapter passes only instruction, public tool descriptions and feedback to its backend, not the verifier, reward object or hidden oracle. Existing tests check that the first model input does not contain the final lookup value and that trusted replay independently grades the resulting episode.
- Tool response order remains user → assistant tool call → tool feedback → next assistant action. Candidate-history validation prevents feedback duplication or rewriting. Sanitization is specific to the current trusted synthetic environment; arbitrary future tool observations would require their own validation before this adapter could safely consume them.

## Actual validation commands

An existing environment was reused read-only, with bytecode and pytest/Ruff caches disabled; no dependency installation or edits to that environment/other worktree occurred. Interpreter: `/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python`; pytest 9.1.1, torch 2.14.1, Ruff 0.16.9. All test commands ran from this report's worktree.

```sh
# Before edits: exit 0, 198 passed, 1 warning, 7.07s.
PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest -q -p no:cacheprovider

# Intermediate targeted validation: exit 0, 47 passed, 1 warning, 3.96s.
PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest tests/agentic/test_model_policy.py -q -p no:cacheprovider

# Regression control: exit 1 as expected, 11 failed, 43 deselected, 0.08s.
# Copied src/wendyfm into this isolated /private/tmp directory, then replaced
# only model_policy.py with git show ce66312...:src/wendyfm/agentic/model_policy.py.
PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest tests/agentic/test_model_policy.py -q -p no:cacheprovider -o pythonpath=/private/tmp/wendyfm-b1-02-baseline-20261022/src -k 'failed_turn_preserves or feedback_cannot_rewrite' > /private/tmp/wendyfm-b1-02-baseline-20261022/regression.log 2>&1

# Final implementation: exit 0, 233 passed, 1 warning, 7.31s.
PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest -q -p no:cacheprovider

# Exit 0, All checks passed!
/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/ruff check --no-cache .

# Exit 0, no whitespace errors; source/test diff manually inspected.
git diff --check
git diff -- src/wendyfm/agentic/model_policy.py
git diff -- tests/agentic/test_model_policy.py
shasum -a 256 src/wendyfm/agentic/model_policy.py tests/agentic/test_model_policy.py
```

The warning is PyTorch's unavailable NumPy initialization; it did not fail any test. No installation attempted. Contract doubles test template arguments, tensor slicing, attention-mask forwarding and CPU generation bounds. Worker lifecycle tests spawn actual OS processes with mock backends, including timeout, SIGTERM resistance, abrupt exit, parse recovery and cancellation. None uses Qwen weights.

## Explicit limitations and independent paper boundary

- `PINNED_QWEN_TEMPLATE=UNVERIFIED`, `REAL_QWEN_INFERENCE=UNVERIFIED`. Default HF cache has no pinned snapshot; both default Python and the reused test interpreter lack Transformers and huggingface_hub. Thus the exact revision's template text, tool envelope emission, special-token behavior, model context configuration and real multi-turn tool use could not be validated. No arbitrary output format or alternate revision was substituted. Real validation needs the authorized pinned local artifacts and dependencies; fetching/installing them is outside this task's authorization (`NEEDS_AUTH`).
- The tests establish the adapter's strict accepted protocol, not that the unavailable model obeys it. Mixed prose/tool output is deliberately rejected, not heuristically extracted. No genuine model latency, quality, training, RL weight update or reward improvement is claimed.
- Worker cleanup covers the direct model process. Descendant-process containment remains unverified; no broader process supervision change was made in this adapter-only audit.
- No GPU, paid compute, model download, dependency install, network credential access, private upload, PR creation/push or merge occurred. No private model/paper caches or artifacts were inspected.
- WendyFM owns infrastructure correctness and reproducible policy/tool/replay contracts. The independent paper project owns hypotheses, private/public research datasets, reward science, ablations, statistics and publication. Its worktrees, experiments and results were not changed or validated. B2 read-only worktree availability: N/A.

TASK_STATUS: PASS
