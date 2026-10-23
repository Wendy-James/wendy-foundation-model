# B1 / 04 — Post-training thin-adapter readiness

Date: 2026-10-22, Asia/Shanghai. **Offline acceptance: PASS** for an interface specification and explicit gates, not framework integration or training readiness.

`LIVE_GITHUB_DISCUSSION=NOT_REFRESHED`
`TRL_RELEASE_API_COMPATIBILITY=UNVERIFIED`
`REAL_MODEL_INFERENCE=UNVERIFIED`
`INSTALLATION_DOWNLOAD=NEEDS_AUTH` (not performed)

## Evidence and scope

Read the required historical snapshot at `/Users/admin/Downloads/wendyfm_30h_guard_v3/evidence/CONTEXT.md` before work. Read both available B1 02/03 completion reports in `/Users/admin/wendyfm-overnight/agentic-30h-v3-20261022_205603/B1`, and their detailed reports in this worktree. No B1 01 completed report was listed. The snapshot's Issue #27 and PR #28/#29 are dated evidence, not latest discussion or approval. Available tool metadata exposed no live GitHub capability; no website polling or network request was attempted.

Owned worktree: `/Users/admin/workspaces/projects/wendyfm-30h-v3-dev-20261022_205603`. HEAD and pinned source: `ce66312aad0e47c38aae7225a9ab124dcd03880f`. Existing uncommitted B1 02 model-policy/test changes and B1 reports were preserved. This task adds only this document, with no production/test code changes. HEAD alone does not identify the modified tested tree. No trainer, optimizer, framework dependency, PR, commit, push or merge was added.

Locally reviewed documentation: `docs/agentic/UPSTREAM_REUSE.md`, `upstream_api_evidence.json`, `upstream_inventory.json`, `PAPER_INTERFACE.md`, and B1 02/03. The first three contain historical documentation/registry excerpts, not independently available TRL release source or installed APIs. The API evidence points to https://huggingface.co/docs/trl/grpo_trainer without a release pin. Its environment API explicitly says experimental. Repository HEAD metadata and registry release metadata are different artifacts. Consequently **every framework compatibility row below is UNVERIFIED**; excerpts support a design hypothesis only.

## Existing WendyFM interface audit

| Boundary | Actual local contract | Adapter implication |
| --- | --- | --- |
| Reset | `LookupEnvironment.reset(seed) -> Observation`, clearing feedback and copying the public fixture | TRL-facing reset must return instruction string or None, not Observation; map seed from trusted host context |
| Tool execution | `step(ToolCall) -> ToolResult`; `observe() -> Observation` | Expose only typed `lookup(key: str) -> str`; preserve status and output, including invalid-action feedback |
| Policy | `ToolCallingPolicy.decide(Observation) -> (Action, input_tokens, output_tokens)`; `ModelSession.decide(observation, timeout)`; `rollout_model(manifest, factory, max_new_tokens=128) -> Episode` | Existing policy is an inference boundary, not a TRL `PreTrainedModel`; reuse validated HF model/tokenizer with upstream generation rather than embedding a second policy loop |
| Tool schema | Exactly `lookup`, one string `key`, no extra properties, key length <=128 | Annotation/docstring and runtime validation both required; TRL dispatch failures must also be captured |
| Trajectory | `Episode` v0.1 contains ordered `Step` list, manifest, terminal and reward; no distinct Trajectory schema | Reuse v0.1 only if full semantics can be preserved; do not introduce a divergent schema or invent measurements |
| Terminal decision | `Action` has exactly one tool call or final answer; default four decisions include final | Completion finalization is host-owned, not a model-visible `finish` tool; multiple/parallel tool calls currently unsupported |
| Reward | `reward_for(manifest, steps, terminal) -> RewardResult`; `verify_episode(episode, separately_trusted_manifest) -> RewardResult` | Trusted callback recomputes replay and reward; never accept a model-supplied reward or use trace manifest as trust anchor |
| Metrics/provenance | Actual per-decision token counts for model steps; null for scripted steps; immutable revisions required | Tool-return strings alone cannot construct a valid model Episode: need generation IDs, timing, terminal and policy provenance hooks |

Current replay checks order, fixture, action/output consistency and recomputed reward. It does not authenticate timings, prove model reasoning, or validate a training gradient. The fixed synthetic answer is public; correct reward is not held-out generalization. Worker cleanup is bounded for direct workers; arbitrary process descendants/security isolation remain unverified.

## Version and dependency compatibility matrix

Observed now using `importlib.metadata.version`, without installs or model loading:

- Default interpreter `/Users/admin/miniforge3/bin/python3`: Python **3.13.13**.
- Read-only existing test interpreter `/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python`: Python **3.11.16**. No files/caches were written to that environment or its worktree.

“Candidate target” means a recorded version/range for later resolution, **not a tested or selected lockfile**. Exact target pins remain unset where no defensible pin is available.

| Component | Default installed | Test interpreter installed | Candidate target from local evidence | Readiness |
| --- | --- | --- | --- | --- |
| TRL | absent | absent | historical release 1.16.0; exact integration pin NOT_SELECTED | UNVERIFIED: environment API not verified against that artifact |
| Transformers | absent | absent | recorded TRL minimum >=4.56.2; inference report historical Qwen minimum >=4.51.0; exact pin NOT_SELECTED | UNVERIFIED: lower bounds do not prove template/generation compatibility |
| PEFT | absent | absent | historical 0.21.2; TRL optional extra >=0.13.0; defer initially | UNVERIFIED: no LoRA round trip/base-module validation |
| torch | absent | 2.14.1 | project range >=2.5,<3; retain 2.14.1 only as offline test baseline, integration pin NOT_SELECTED | Local tests pass; TRL/model/PEFT stack UNVERIFIED |
| vLLM (optional) | absent | absent | disabled for first CPU seam; historical inventory 0.32.0 conflicts with TRL extra >=0.22.0,<=0.31.0 | UNVERIFIED; do not install latest or infer an older compatible pin |
| Accelerate / datasets | absent / absent | absent / absent | historical TRL >=1.10.1 / >=4.7.0 | UNVERIFIED, unresolved transitive dependencies |
| NumPy | 2.5.2 | absent | historical TRL >=1.26.0 | Test interpreter lacks NumPy; torch emits warning in offline tests |

Historical PEFT 0.21.2 requires torch >=1.13.0, Transformers, Accelerate >=0.21.0 and safetensors (plus other requirements in inventory). Historical vLLM 0.32.0 requires torch ==2.13.0 and Transformers >=5.16.1,<5.20.0: it does not match installed test torch 2.14.1 or recorded TRL extra ceiling. Repository HEAD's separate build metadata has another torch requirement; do not mix it with registry release requirements. Python satisfies the recorded TRL >=3.11 floor, but native wheels and full dependency resolution are unverified. Local inventory is neither a fresh package index nor a complete compatible environment.

## TRL workflow comparison — all release compatibility UNVERIFIED

| Historical documentation claim | Proposed mapping | Unverified release/API question |
| --- | --- | --- |
| `environment_factory` accepts zero-argument factory, or named factory dictionary | Single zero-argument factory returning tool facade; defer multi-task dictionary | Constructor signature, exact reset kwargs and instance allocation |
| `reset` returns None/string; string appended to last user message | Return public instruction; do not repeat it in dataset prompt | When reset runs, forwarded columns, prompt concatenation and reuse |
| Environment methods discovered as tools | Public facade exposes only `reset` and typed/documented `lookup` | Discovery exclusions, private method behavior, schema generation and error mapping |
| Instances isolated per rollout and reused/reset across steps | Fresh host episode identity on every reset; clear all prior state | Distinct identities for simultaneous completions and reward timing before next reset |
| Reward callback gets `environments`, accepts dataset kwargs, returns one float per completion | Host callback accepts `completions, environments, **kwargs`, uses separately trusted registry | Ordering, duplicate objects, batching and completion format |
| Optional `get_reward` supplies environment reward | Deliberately omit it; external verifier callback only | Confirm absence and that reward funcs are actually used |
| Tool outputs inserted into conversation; non-tool token accounting | Serialize `status=...; output=...`; preserve assistant/tool association | Actual pinned Qwen template, IDs, masking, call IDs, truncation and non-tool loss mask |

Correction to earlier reuse wording: not necessarily a newly constructed environment every completion. The local excerpt explicitly describes reuse with reset. Isolation must mean distinct active state and reset-safe host records, tested on the exact release. A factory must not expose verifier/reward methods or oracle/dataset-label fields. A Python facade is not an OS security boundary.

## Minimal adapter specification (not implemented)

Keep three small responsibilities, using upstream TRL for generation and optimization:

1. **Policy-facing facade**: factory takes no arguments and constructs an object whose only public callable methods are `reset` and `lookup`. `reset(**kwargs) -> str` clears public lookup state, validates only supported public row fields, and asks a host registry to allocate a fresh episode generation key. Reject unexpected fields rather than forwarding `ground_truth`, reward, verifier or a trusted manifest. Seed/provenance come from host configuration. `lookup(key: str) -> str` has a schema-generating docstring, delegates to existing `LookupEnvironment.step(ToolCall('lookup', {'key': key}))`, records original ToolCall/ToolResult through host instrumentation, and returns exactly `status={status}; output={output}`. Invalid arguments/unknown keys remain invalid-action feedback. Before reset or beyond host decision bounds, fail closed. No public `step`, `observe`, `get_reward`, finalizer or evaluator method.
2. **Host registry/finalizer**: keyed by environment identity AND reset generation, holding trusted manifest, generation events, monotonic timing and tool dispatch events. Attach final answer after actual generation completes, never via an exposed tool. Count every policy decision, including final and malformed output; enforce four-decision/128-new-token/4096-input-token limits and worker deadlines consistent with the existing runtime. Capture errors/truncation as their actual terminal states. Snapshot completed records before reset/reuse; reject stale keys. Do not fabricate an Episode when per-turn counts/timing/order are unavailable. Current TRL excerpt alone provides no verified event hook for this: **trace instrumentation is an explicit readiness gate**. Preserve existing inference/replay path while investigating the exact upstream hook; no new rollout engine.
3. **Trusted external callback**: `reward_func(completions, environments, **kwargs) -> list[float]`. Require equal list lengths, uniquely identified active episodes and one finalized trace per completion; fail on missing/reused/out-of-order records. Decode strict Episode, match separately trusted manifest, call `verify_episode`, return verified scalar in original completion order. Never grade untrusted `episode.reward` directly. Dataset prompts must contain no labels; reward callback alone may access independent labels in a future separately authorized task. Reward mismatch/provenance corruption is a correctness error, not a silent zero reward.

Facade tool return strings do not include rewards, trusted manifest or future independent labels. The public toy tool world is visible by design. Completion text may be standard string or conversational messages in the historical docs: select exactly one format after pinned API verification and reject other shapes; do not silently guess final text from arbitrary messages. v0.1 has no call-ID field and supports one action per turn; fail closed on parallel calls and retain raw trainer transcript in separate evidence. If new framework events cannot fit v0.1, specify a reviewed schema revision instead of silently discarding evidence.

The current `ToolCallingPolicy` is **not** the policy supplied to `GRPOTrainer`: upstream should receive a genuine compatible HF causal model/tokenizer, with optional PEFT later. WendyFM provides execution, trace and reward translation only. Do not implement PPO/GRPO loss, advantage normalization, optimizer, sampling engine or a second trainer.

## Exact future API-contract smoke acceptance

These are test specifications, **NOT run tests or existing executable test files**. Implement the proposed facade/registry only after B1 03 genuine inference acceptance and independently checking an exact locally supplied TRL release. No install/train command is authorized by this document.

Proposed command after those gates and test implementation:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 <approved-project-python> -m pytest tests/agentic/test_trl_contract.py -q -p no:cacheprovider
```

Required assertions for that test file:

- `test_release_contract`: check installed versions equal a reviewed lock; `inspect.signature(GRPOTrainer.__init__)` has `environment_factory`, `reward_funcs`, `processing_class`; bind zero-arg factory. Inspect the exact release tool discovery/reset/reward dispatch APIs and invoke them directly where supported. Assert discovered tool names are exactly `{'lookup'}` and schema has only required string key, no extra properties. No `train()` call.
- `test_facade_reset_and_feedback`: factory creates distinct A/B; reset A and B returns INSTRUCTION string and no history. On A, `lookup('start')`, `lookup('drawer')`, `lookup('cabinet')` returns exactly `status=ok; output=drawer`, `status=ok; output=cabinet`, `status=ok; output=violet`. B's history remains empty. A unknown key returns `status=invalid_action; output=unknown key`. Reset A clears its history and advances registry generation; stale reward identity is rejected. Assert long/non-string/extra arguments cannot execute a valid lookup. This is fixture contract testing, not inference.
- `test_reward_alignment_and_trust`: finalize two separate fixture records, one final `violet`, one wrong final, with SCRIPTED/null token semantics. Callback yields `[1.0, 0.0]`; reversing completions AND associated environments yields `[0.0, 1.0]`. Unequal lengths, duplicate active identities, stale reset record, mismatched manifest, altered tool output or altered reward raise errors. Policy tool inventory contains no oracle/reward method. Fixtures remain explicitly scripted; never claim these counts as model events.
- `test_real_generation_trace_no_update`: prerequisite is hashed real Qwen artifact/template and successful existing offline inference/replay. Use only the independently verified pinned TRL inference/rollout entry point (NOT `train`/`training_step`) with same genuine HF model, thinking disabled and bounded CPU worker. Do not invent a private TRL entry-point name from these excerpts. If no supported bounded generation-only hook exists, this test remains UNVERIFIED and thin-adapter readiness does not advance. Record actual generated assistant/tool transcript and tool dispatch, capture call association and token mask; require three genuine model lookup actions followed by final answer for the positive tool-use smoke. An answer alone, scripted outputs, tool-text included in policy loss, missing per-turn metrics, timeouts or parser failures do not pass. Verify resulting strict model Episode against trusted manifest; compare parameter hashes before/after to confirm no update.

The last test must run once with an outer wall bound (300-second inference budget, existing 60-second startup/decision bounds plus bounded cleanup). Preserve failures; no indefinite retries. It must also check original instruction is appended exactly once, tool outputs reach the next generation, simultaneous rollout state cannot cross-contaminate, completed records survive reset and callback ordering. Unit tests establish local translation; only release-dispatch plus genuine generation can establish framework compatibility. GPU, vLLM, PEFT and training are unnecessary for this contract gate.

## Actual offline validation and limitations

Executed from the owned worktree:

```sh
git status --short
git rev-parse HEAD
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest tests/agentic/test_model_policy.py tests/agentic/test_agentic.py -q -p no:cacheprovider
/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/ruff check --no-cache src/wendyfm/agentic tests/agentic/test_model_policy.py tests/agentic/test_agentic.py scripts/run_agentic_model.py
git diff --check
shasum -a 256 src/wendyfm/agentic/model_policy.py tests/agentic/test_model_policy.py docs/agentic/upstream_api_evidence.json docs/agentic/upstream_inventory.json
```

Also executed inline read-only Python metadata probes for Python, TRL, Transformers, PEFT, torch, vLLM, Accelerate, datasets, NumPy, pytest and Ruff under the interpreters above. Default pytest/Ruff also absent; test interpreter pytest 9.1.1, Ruff 0.16.9.

Targeted tests: **78 passed, 1 warning, 4.55s**, exit 0. Warning is torch initialization without NumPy. These test the existing contracts and model backend doubles, not TRL or real Qwen. Ruff and whitespace results are recorded below after final checks. No full-suite rerun necessary for documentation-only scope.

Narrow readiness gates: genuine model/config/tokenizer artifacts absent per B1 03; Transformers/Hub stack absent; exact TRL release and dispatch hooks absent; trace/loss-mask instrumentation unverified; PEFT and optional vLLM deferred. Provisioning requires separate authorization. This task did not load a model, install dependencies, download weights, use GPU/paid compute, upload private data, expose credentials, access paper artifacts or change other worktrees. No model success, training, gradient update, reward improvement or benchmark is claimed. Offline acceptance does not depend on those later runtime gates.

## Independent paper-project boundary

WendyFM owns execution, policy/tool interfaces, trace provenance/replay and thin training adapters. The independent paper project owns the research question, independent labels, reward definitions, experimental design, ablations, statistics and claims. WendyFM's synthetic verifier is an infrastructure fixture, not a research reward definition or validation of the paper. Future scientific labels/rewards require an explicit separately trusted interface and versioned provenance; no private labels should enter policy prompts/tools. No paper experiment, data, worktree or claim was inspected, changed or validated. B1 worktree availability for B2 review: N/A as supplied.

## Final checks and artifact identities

Ruff: **All checks passed**, exit 0. `git diff --check`: exit 0, no whitespace errors. Production/test files retain B1 02 identities:

| Artifact | SHA-256 |
| --- | --- |
| `src/wendyfm/agentic/model_policy.py` | `9775928d6610e29aef987c9bb03b28c84ccb6c0b5bbe557898eb14c517d4ad4a` |
| `tests/agentic/test_model_policy.py` | `0f6ca1db4c5628246e2d150e09be48c0a3d5e3c8047be406b5bfd95589788830` |
| `docs/agentic/upstream_api_evidence.json` | `e481232721a0b586910d79847b960a4e263b9baf50f6804d543722907c83c1ea` |
| `docs/agentic/upstream_inventory.json` | `b55ee082eaebad05604cc8c2bc39eecbd9f5319347324f5d40522f89118c0b9e` |

TASK_STATUS: PASS
