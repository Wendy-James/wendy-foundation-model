# B1 / 05 — Final developer handoff

Date: 2026-10-22, Asia/Shanghai. **Offline handoff acceptance: PASS.** This is a local engineering audit, not model-inference or training acceptance.

## Identity, authority and evidence

- Owned worktree: `/Users/admin/workspaces/projects/wendyfm-30h-v3-dev-20261022_205603`.
- Branch: `ops/agentic-v3-dev-20261022_205603`.
- HEAD, pinned source and merge base with that pin: `ce66312aad0e47c38aae7225a9ab124dcd03880f`.
- All changes remain uncommitted for human review; no staging, commit, push, new PR or merge performed. HEAD identifies the baseline, not the modified tested tree.
- `LIVE_GITHUB_DISCUSSION=NOT_REFRESHED`. No GitHub connector/search/fetch capability was exposed in the inspected tool inventory. No live discussion or fresh review approval is claimed; this does not block offline acceptance.
- Read `/Users/admin/Downloads/wendyfm_30h_guard_v3/evidence/CONTEXT.md` before work. Issue #27 and PR #28/#29 therein are dated historical evidence only.
- Read all available detailed B1 reports: `b1_02_policy_adapter_report.md`, `b1_03_local_inference_gate_report.md`, `b1_04_trl_readiness_report.md`, `b1_06_replay_provenance_report.md` under `docs/agentic/overnight/`.
- Read all available completion messages `02_policy_adapter.last.md`, `03_local_inference_gate.last.md`, `04_trl_readiness.last.md`, `06_replay_provenance.last.md` and `progress.json` under `/Users/admin/wendyfm-overnight/agentic-30h-v3-20261022_205603/B1`. All four recorded offline PASS. No B1 01 completion report exists in the enumerated directory. The ongoing 05 log is not a completed review.
- Inspected complete tracked diff and untracked provenance implementation/tests. Staged diff is empty. Added only this handoff and its evidence manifest during B1 05.

`docs/agentic/overnight/b1_05_evidence_manifest.json` records exact SHA-256 and byte counts for every prior changed file/report and unchanged contract/runtime/verifier/runner/upstream evidence anchors. It excludes itself and this report to avoid self-reference. Together with the baseline and tracked diff/new files, it identifies the dirty implementation; it is not an execution attestation or model-artifact manifest.

## Status separation

| Category | Evidence and limits |
| --- | --- |
| Completed / verified locally | Transactional adapter history, feedback continuity, strict parsing and token bounds, session error recovery; opt-in replay receipt and fixed-verifier tamper checks. Final full suite: 245 passed. Ruff and diff check pass. Historical Episode v0.1 retained. |
| Mocked-only | Template/mask/generation tensor tests use doubles. Four-step model-policy rollout is mock-generated. OS worker tests spawn real processes with mock backends. Synthetic receipt hashes and mock token counts are not measured Qwen artifacts/tokens. |
| Planned | Thin TRL facade/host registry/reward callback and API smoke tests in B1 04; research-fixture v0.1 specification in PAPER_INTERFACE. Neither is implemented. No second trainer added. |
| Gated / UNVERIFIED / NEEDS_AUTH | Genuine pinned Qwen artifacts, template and CPU load/inference; dependencies; exact TRL release compatibility and trace/loss-mask hooks; independent research fixture. These are narrower runtime/integration gates, not local task BLOCKED status. |
| BLOCKED | No essential local correctness failure found in final checks. Historical supervisor reports blocked on unavailable discussion were review-availability failures, not current test failures. |
| Unknown | Latest GitHub discussion/approval; absent B1 01 work; real runtime capacity/headroom; descendant-process containment; actual model tool-use quality; research validity. B2 read-only worktree availability: N/A as supplied. |

B1 06 says “13 collected receipt cases”; current collection is **12**, verified below. Its recorded historical 93-test agentic result is not a fresh result here. Final full-suite count is authoritative for this tree: 245, consistent with prior B1 02's 233 plus 12 new provenance cases. No source correction was needed; preserve prior report as historical evidence with this explicit correction.

## Exact changed-file summary

All paths below are relative to the owned worktree above; the evidence manifest includes their exact identities.

| Path | Change / evidence |
| --- | --- |
| `src/wendyfm/agentic/model_policy.py` | Retry-safe transcript commit, detached backend input, continuity checks, token bounds, stray thinking-tag rejection, session error reset. Full suite validates adapter and lifecycle regressions. |
| `tests/agentic/test_model_policy.py` | Parser, retry/history, context/token edge cases, sanitized tool feedback, worker recovery/cancellation tests. B1 02 reports 11 regressions failed against original source; not rerun as a baseline experiment in B1 05. |
| `src/wendyfm/agentic/provenance.py` | New host-owned source/hash sidecar; compares independent receipt and manifest then fixed verifier. Not wired into CLI replay; hashes/source assertions do not prove inference. |
| `tests/agentic/test_provenance.py` | 12 collected synthetic cases: identities, tampering, forbidden fields, source/artifact constraints and verifier bypass rejection. Included in final full suite. |
| `docs/agentic/PAPER_INTERFACE.md` | Optional sidecar trust limitations and future protected research-fixture specification. Documentation, no private labels or external verifier execution. |
| `docs/agentic/overnight/b1_02_policy_adapter_report.md` | Adapter findings and historical test/baseline evidence. |
| `docs/agentic/overnight/b1_03_local_inference_gate_report.md` | Missing artifacts/dependencies, capacity limitations, future provisioning gate. |
| `docs/agentic/overnight/b1_04_trl_readiness_report.md` | Historical upstream evidence versus unresolved release/API compatibility; proposed thin adapter only. |
| `docs/agentic/overnight/b1_06_replay_provenance_report.md` | Receipt evidence and replay limitations; case count corrected above. |
| `docs/agentic/overnight/b1_05_handoff_report.md` | This handoff, final checks, restart and transfer instructions. |
| `docs/agentic/overnight/b1_05_evidence_manifest.json` | Local file hashes/sizes; no real model hashes. |

Tracked pre-handoff diff: PAPER_INTERFACE +35 lines; model_policy +25/-12; test_model_policy +138/-1. Untracked files must be reviewed separately because ordinary `git diff` omits them. No secrets, model weights, private data or large artifacts added; nothing staged.

## Actual final checks and restart commands

Run from this worktree. Existing interpreter was reused read-only; bytecode, pytest and Ruff caches disabled. No other worktree was edited.

```sh
cd /Users/admin/workspaces/projects/wendyfm-30h-v3-dev-20261022_205603
git branch --show-current
git rev-parse HEAD
git merge-base HEAD ce66312aad0e47c38aae7225a9ab124dcd03880f
git status --porcelain=v1
git diff
git diff --cached --stat
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest -q -p no:cacheprovider
/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/ruff check --no-cache .
git diff --check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest tests/agentic/test_provenance.py --collect-only -q -p no:cacheprovider
```

Executed results: full pytest exit 0, **245 passed, 1 warning in 7.15s**; Ruff exit 0, **All checks passed!**; diff check exit 0; provenance collection exit 0, **12 tests collected in 0.02s**. Warning: installed torch cannot initialize absent NumPy. It did not fail tests; no installation attempted. Timings are local test timings, not model latency or training benchmarks. After writing handoff/manifest, Ruff and diff check are repeated; no production code changed after the full suite.

Validate manifest bytes without executing model code:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path('docs/agentic/overnight/b1_05_evidence_manifest.json').read_text())
for name, expected in m['files'].items():
    data = Path(name).read_bytes()
    assert len(data) == expected['bytes'], name
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
print('manifest verified')
PY
```

## Genuine Qwen gate and next smallest authorized run

`REAL_QWEN_INFERENCE=UNVERIFIED`; zero genuine trajectories available. B1 03 found no default cached `Qwen/Qwen3-0.6B` snapshot at revision `c1899de289a04d12100db370d81485cdf75e47ca`; test interpreter lacks Transformers, huggingface_hub, safetensors and NumPy. Those earlier readiness observations are historical, not re-probed model readiness in this task. Configuration coherence and mocks cannot validate actual template behavior. Do not substitute a mock or alternate model to close the gate.

Separate explicit human authorization must specify: approved artifact source/license and exact revision; model/config/generation-config/tokenizer/template/card/license download or supplied local path; isolated project environment and reviewed pinned CPU torch/Transformers/Hub/safetensors/NumPy plus necessary transitive dependencies; storage destination and download/disk budget; CPU memory headroom and overall wall-time bound. B1 03's ~1.40 GiB weight size is historical, not locally measured. Resolve versions and validate exact upstream artifacts; minimum version ranges are not compatibility proof. TRL/PEFT/vLLM and GPU are unnecessary for this gate. Never mutate the reused test environment or paper project.

The smallest subsequent run is **one** existing public synthetic CPU closed loop, after human review establishes a clean committed code SHA (not an auto-commit in this task) and provisions the prerequisites. Hash actual weights/tokenizer/config/template/license, exercise actual tool chat template, retain approved environment versions and independently trusted manifest. Preserve any failure rather than repeatedly retrying.

Future commands below are **not executed**. Set `approved_python` to the human-provisioned interpreter and `reviewed_sha` to its clean reviewed checkout SHA. Use a fresh authorized output outside Git, e.g. `/private/tmp/wendyfm-qwen-human-smoke.json`. An external supervisor must enforce a separate overall wall limit covering cache resolution, hashing and evidence write; the existing runtime's 300-second inference budget does not cover the whole CLI. Do not assume a portable `timeout` binary or descendant containment exists.

```sh
# Prerequisites: explicit provisioning/run authorization and clean reviewed tree.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$approved_python" scripts/run_agentic_model.py --output /private/tmp/wendyfm-qwen-human-smoke.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$approved_python" scripts/run_agentic_model.py --verify /private/tmp/wendyfm-qwen-human-smoke.json --expected-sha "$reviewed_sha"
```

Existing bounds: CPU float32, greedy generation, thinking disabled, four policy decisions, <=128 new tokens/decision, <=4096 input tokens, 60-second startup/decision bounds and bounded direct-worker cleanup. Require actual model-generated lookup actions consuming prior tool feedback followed by final answer and trusted replay before claiming a closed loop. A guessed final answer or successful replay alone is insufficient. CLI `--verify` validates only embedded Episode; independently check outer artifact metadata against measured files. Opt-in receipt needs an independent host recorder; never promote the trace's own receipt to trusted evidence. Record failures/partial steps/error_type honestly, with no RL update.

## Versioned handoff to A1/A2 and paper boundary

1. Human reviewer records reviewed source SHA plus this dirty-tree manifest/diff and new files; later pin the approved implementation revision. A1/A2 receive public interface/code/test evidence only, with explicit source labels and gates. No automatic message/upload or assumed lane-specific responsibility occurs here.
2. Pin `wendyfm.episode.v0.1` and optional `wendyfm.replay-provenance.v0.1`; preserve exact decoder semantics, fixture digest, policy references, ordered tool results, terminal states and measured/null token semantics. Run offline tests and manifest checks before integration. Do not invent a separate trainer or alter frozen Episode fields silently.
3. Transfer only public synthetic fixtures/tool schemas/observations, adapter signatures and negative acceptance cases. Trusted host retains manifest and receipt independently. Policy-facing data must omit reward, hidden labels, oracle, verifier implementation and protected registry; hashes are bindings, not access control. Same-user Python workers are not a private-data sandbox.
4. A1/A2 can review B1 04's proposed thin framework seam, but exact TRL pin/dispatch, masking and generation hooks remain UNVERIFIED. Proposed `tests/agentic/test_trl_contract.py` does not exist; do not advertise its specification as passing tests. Reuse mature upstream generation/optimization only after separate release and inference gates.
5. Any research fixture needs a separately reviewed versioned public-input/protected-label registry boundary as specified in PAPER_INTERFACE. WendyFM's current lookup verifier cannot grade arbitrary research tasks. The independent paper project must supply/review fixture ownership, label/reward/verifier revisions and scientific acceptance through its own authorization. No hidden scoring labels, paper datasets or private artifacts belong in the public transfer.

WendyFM owns engineering interfaces, execution, tests, provenance and replay. The independent paper project owns hypotheses, datasets/splits, reward science, ablations, statistics and publication. No paper worktree, experiment or artifact was read, copied, modified or validated. No GPU, paid compute, dependency install, model download, private upload, credential exposure, PR push/creation, merge or auto-commit occurred. No actual Qwen trajectory, RL weight update, reward improvement, M7 long-run completion or 20M-token training result is claimed. Offline handoff acceptance is satisfied despite the explicit later gates.

TASK_STATUS: PASS
