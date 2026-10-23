# B1 / 06 — Replay provenance integrity, bounded local audit

Date: 2026-10-22, Asia/Shanghai. Offline acceptance: PASS for a narrow opt-in integrity extension and verified regressions. No inference or research result is claimed.

## Evidence and worktree identity

Read the required `/Users/admin/Downloads/wendyfm_30h_guard_v3/evidence/CONTEXT.md` before source work. Read existing B1 02, 03 and 04 completion reports in `/Users/admin/wendyfm-overnight/agentic-30h-v3-20261022_205603/B1`, and their detailed reports under this worktree's `docs/agentic/overnight`. No completed B1 01 or 05 report was present in the enumerated B1 directory. The snapshot is dated offline evidence, not current approval.

`LIVE_GITHUB_DISCUSSION=NOT_REFRESHED`: available tool metadata exposed no live GitHub/search connector. No website polling, fresh discussion review or approval claim.

Owned worktree: `/Users/admin/workspaces/projects/wendyfm-30h-v3-dev-20261022_205603`.
Pinned source and current HEAD: `ce66312aad0e47c38aae7225a9ab124dcd03880f`.
Existing B1 02 dirty adapter/test modifications and B1 reports were preserved. This task adds `provenance.py`, its tests, the optional interface documentation and this report. Changes remain uncommitted; HEAD alone does not identify this tested implementation.

## Audit: assertions versus verified boundaries

| Boundary | Existing local behavior | Remaining limit |
| --- | --- | --- |
| Expected policy identity | `verify_episode` compares complete Episode manifest with a separately trusted decoded manifest, including kind, policy/adapter/model/tokenizer revisions | Does not execute policy or prove recorded actions came from it; revision strings do not authenticate loaded artifacts |
| Execution SHA | Manifest equality rejects a different expected SHA; manifest validates 40 lowercase hexadecimal characters | Caller can supply an arbitrary valid SHA; replay does not measure historical executed bytes. Dirty-worktree CLI guard can be explicitly bypassed in the scripted demo with trusted SHA |
| Model/tokenizer revisions | Trusted manifest equality checks both; model factory compares configured revisions | Model revision format is not generally enforced as an immutable repository SHA in frozen contract; real local files and inference absent |
| Tool outputs | Deterministic replay compares each tool result with host environment output, not just a JSON digest | No per-output hashes in Episode; timing/error assertions are not deterministically reproduced |
| Immutable task | Literal environment/task identity and `FIXTURE_SHA` equality against local canonical fixture | Current digest covers the public lookup mapping, not all instruction/tool-schema/source bytes; broader research task provenance absent |
| Verifier identity/reward | Literal verifier identity, fixed host `reward_for`, comparison with independently recomputed reward and verified flag | Identity is a label, not executable-byte attestation; Python workers share user/filesystem privileges |
| Outer model evidence | Runner writes model/tokenizer file hashes and wall time | `scripts/run_agentic_model.py --verify` reads only embedded Episode, ignoring outer schema/model/hash/timing claims; no explicit mock/real source field |

Generated JSON SHA alone is insufficient: an attacker can edit and rehash a trace. Passing a trace's manifest or receipt back as the trusted input is not an independent trust boundary. Existing verifier does useful environment/reward replay, not historical execution authentication. The public synthetic answer can be guessed and is not a hidden research oracle.

## Narrow change and actual evidence

Added separate `wendyfm.replay-provenance.v0.1` host receipt, without modifying `wendyfm.episode.v0.1`, its decoder, runtime or verifier. `bind_provenance` binds canonical Episode digest, ordered non-null tool result digests, full policy reference, execution SHA, fixture task revision, verifier identity/digest and optional model/tokenizer artifact hashes. `verify_bound_episode` requires separately retained trusted receipt plus trusted manifest, compares the receipt and Episode binding, then calls the existing fixed verifier.

Explicit source values: scripted, mock, real. Scripted pairs only with scripted policy; mock/real with model policy. Real requires artifact hashes; mock/scripted cannot claim them. Unknown fields/sources/verifiers and malformed digests are rejected through strict decoding. All hashes in the new unit fixtures are synthetic values, not measured model artifacts.

Baseline/change/metric/result: existing replay has no optional receipt API and cannot bind a mock/real source or verifier artifact digest. Added 13 collected receipt cases; full agentic suite now **93 passed**. Tests exercise changed code/task/policy/model/tokenizer/verifier identities, artifact/output hashes, source relabeling, rehashed timing tampering, historical Episode compatibility, prohibited oracle/reward/verifier fields, and recomputed-receipt attempts to bypass tool/reward verification. These are local correctness checks; no baseline-failure experiment was run and no performance/reward improvement is inferred.

This extension is opt-in and not wired into legacy CLI replay. It checks evidence consistency against a host receipt, not signatures or actual inference. `bind_provenance` does not measure file hashes, authenticate source labels, attest verifier bytes, or validate a reward itself. The recorder must independently measure and retain those values; a verifier digest supplied by a caller is an assertion until independently measured. Arbitrary Python policy code remains trusted engineering code, not OS-contained malicious code.

## Timing, tokens and reward trust

No metric is synthesized or modified. Runtime uses `time.monotonic` elapsed measurements; scripted token counts remain null. Real backend counts input tensor width and generated tensor length; model mock counts are supplied backend-double values, not genuine tokenizer measurements. Existing replay requires integer counts on successful model decisions and null on failed decisions. The sidecar hashes existing measured-or-null values unchanged; this detects mutation against a retained receipt, not truth of original measurement. Historical Episode elapsed time remains required float rather than adding null to its frozen schema.

Policy API receives only detached Observation and returns strict Action. No reward, hidden expected answer or external Verifier field is accepted. `verify_bound_episode` accepts no reward function, expected-answer argument or executable verifier. Even an internally consistent rehashed receipt cannot make a modified tool result or reward pass the fixed verifier, as tested. This confirms interface separation only: it is not proof that a malicious same-user Python process cannot inspect public source/filesystem. No private oracle exists in the current fixture.

## Independent research-project boundary

Updated `PAPER_INTERFACE.md` to specify the required future `wendyfm.research-fixture.v0.1` interface: immutable public task/tool/input/split identity plus separately protected host label/verifier/reward-specification registry; negative cases for wrong answers, stale revision, changed output, self-awarded reward and policy-selected verifier. This is a specification, not an implemented fixture or validated research reward. Lookup-only replay cannot accept new research tasks without a separately reviewed interface change.

WendyFM owns engineering, execution, provenance and interface testing. The independent paper project owns hypotheses, independent labels, reward science, ablations, statistics and publication. No paper repository, private artifact, dataset or experiment was read, copied, changed or validated. Independent fixture remains UNVERIFIED/NEEDS_AUTH for later separately authorized integration. B1 worktree available for B2 review: N/A as supplied.

## Actual commands and results

Executed in the owned worktree, using the existing test environment read-only with bytecode/test/lint caches disabled:

```sh
git status --short
git rev-parse HEAD
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest tests/agentic -q -p no:cacheprovider
/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/ruff check --no-cache src/wendyfm/agentic tests/agentic scripts/run_agentic_model.py
git diff --check
shasum -a 256 src/wendyfm/agentic/provenance.py tests/agentic/test_provenance.py src/wendyfm/agentic/verifier.py src/wendyfm/agentic/contracts.py
git diff --stat
```

Intermediate suite: 91 passed, 1 warning, 6.22s. After adding two fixed-verifier bypass regressions: **93 passed, 1 warning, 6.24s**, exit 0. Warning: installed torch lacks NumPy initialization. Ruff: **All checks passed**, exit 0. `git diff --check`: exit 0. Inspected tracked diff and new files; no secrets, weights or large artifacts added/staged. No installs, downloads, GPU, RL, paid compute, uploads, credential inspection, PR creation/push/merge or other-worktree edits occurred.

| Tested file | SHA-256 |
| --- | --- |
| `src/wendyfm/agentic/provenance.py` | `63e58cda7139f0e68f5488d98ab18ec3b478d4549533cbe2d1b0f7d1989a31a7` |
| `tests/agentic/test_provenance.py` | `0500e22796fcf3673c79addad34110d47f8b67f8970b37e84debf7e869433572` |
| `src/wendyfm/agentic/verifier.py` (unchanged) | `c9135d68b690cc2442c08fb4ffa04e57c506d237b93ea813ce65957204e0ded1` |
| `src/wendyfm/agentic/contracts.py` (unchanged) | `80c794af3a1f0ed7f29838417eb85fab77bb3f39eddc63389c2319731415b2a0` |

Real Qwen inference/artifacts/dependencies remain UNVERIFIED per preceding B1 03 gate; provisioning NEEDS_AUTH. No real-model timing/token measurement or scientific benchmark is claimed. Those narrower gates do not prevent this local acceptance.

TASK_STATUS: PASS
