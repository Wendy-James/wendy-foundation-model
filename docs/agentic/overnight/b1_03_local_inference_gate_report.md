# B1 / 03 — Genuine model inference gate, local CPU audit

Date: 2026-10-22 (Asia/Shanghai). Offline audit acceptance: PASS. The real-model gate is blocked by absent artifacts and dependencies; this is not successful model inference.

- `LIVE_GITHUB_DISCUSSION=NOT_REFRESHED`
- `REAL_MODEL_INFERENCE: UNVERIFIED`
- `LOCAL_MODEL_IDENTITY: UNVERIFIED`
- `ACTUAL_CHAT_TEMPLATE_SUPPORT: UNVERIFIED`
- `MODEL_RUNTIME_CAPACITY: UNVERIFIED`
- `INSTALLATION_DOWNLOAD: NEEDS_AUTH` (not performed)

## Evidence and provenance

Read the required dated snapshot `/Users/admin/Downloads/wendyfm_30h_guard_v3/evidence/CONTEXT.md` before source inspection. Read the existing B1 02 completion message at `/Users/admin/wendyfm-overnight/agentic-30h-v3-20261022_205603/B1/02_policy_adapter.last.md` and the detailed local `b1_02_policy_adapter_report.md`. No B1 01 completion report was present in that directory. Enumerating available tool metadata returned no GitHub capability. No live review, fresh approval or current discussion is claimed.

Owned worktree: `/Users/admin/workspaces/projects/wendyfm-30h-v3-dev-20261022_205603`. HEAD equals pinned source `ce66312aad0e47c38aae7225a9ab124dcd03880f`. Existing B1 02 modifications to the adapter/tests and untracked overnight reports were present before this task and preserved. This task adds only this report; HEAD alone does not identify the tested modified adapter.

Tested SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `src/wendyfm/agentic/model_policy.py` | `9775928d6610e29aef987c9bb03b28c84ccb6c0b5bbe557898eb14c517d4ad4a` |
| `tests/agentic/test_model_policy.py` | `0f6ca1db4c5628246e2d150e09be48c0a3d5e3c8047be406b5bfd95589788830` |
| `src/wendyfm/agentic/contracts.py` | `80c794af3a1f0ed7f29838417eb85fab77bb3f39eddc63389c2319731415b2a0` |
| `src/wendyfm/agentic/runtime.py` | `3842f4b661dd1fc83ebdf31f68fc6c747e650daf09693c0bec209cd3533cf918` |
| `src/wendyfm/agentic/verifier.py` | `c9135d68b690cc2442c08fb4ffa04e57c506d237b93ea813ce65957204e0ded1` |
| `scripts/run_agentic_model.py` | `d9a8adfd334a1691068393f75ca99d9df9836760540c2e22805a933f96069655` |

## Local readiness findings

Candidate: `Qwen/Qwen3-0.6B`, approved recorded revision `c1899de289a04d12100db370d81485cdf75e47ca`.

1. **Capacity:** `sysctl hw.memsize` was denied by the sandbox. A read-only Python `os.sysconf` fallback returned 1,048,576 physical pages × 16,384 bytes = 16 GiB RAM. `SC_AVPHYS_PAGES` was unsupported. `vm_stat` reported 3,694 free pages (~57.7 MiB), 823 speculative pages, 179,515 inactive pages (~2.74 GiB), and 435,170 compressor-occupied pages (~6.64 GiB). These are VM counters, not guaranteed allocatable model headroom; runtime capacity is unverified. `df -h . /Users/admin/.cache /private/tmp` reported 109 GiB available on the same Data volume. Disk appears adequate for the recorded artifact size, but no allocation/load was attempted.
2. **Dependencies:** default `/Users/admin/miniforge3/bin/python3` lacks torch, Transformers, huggingface_hub and safetensors. Existing read-only test interpreter `/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python` has torch 2.14.1 and pytest 9.1.1, but lacks Transformers, huggingface_hub, safetensors and NumPy. Checks used `importlib.util.find_spec` and `importlib.metadata.version`; no packages were installed. Tests actually imported torch successfully with a NumPy warning.
3. **Cache:** `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `XDG_CACHE_HOME` were all unset (only these non-secret path settings were inspected). `/Users/admin/.cache/huggingface/hub` exists, but `models--Qwen--Qwen3-0.6B` and its exact revision snapshot do not. The legacy `/Users/admin/.cache/huggingface/transformers` directory does not exist. No arbitrary private caches or other model/paper directories were searched. An unconfigured alternative location elsewhere is not ruled out; it cannot establish readiness for this loader.
4. **Coherence:** `NEXT_MODEL.json`, `MODEL_REVISION`, factory model revision and tokenizer revision agree on the recorded pin. Both AutoTokenizer and AutoModel load from the same locally resolved snapshot, with `local_files_only=True`, `trust_remote_code=False`, CPU float32, and built-in Transformers generation. This verifies configuration coherence, not local artifact identity. With no snapshot, there are no model/config/tokenizer/license bytes to hash or validate.
5. **Actual template:** the adapter calls upstream `apply_chat_template` with tools, `enable_thinking=False`, generation prompt and tensor/dictionary output, forwards the attention mask, and rejects input beyond 4,096 tokens. Existing documentation records a tool-aware template and Transformers >=4.51.0 requirement, but those are historical assertions, not direct local verification of this revision. Mock template tests cannot establish actual Qwen template syntax, special tokens, context limits or emitted tool actions.

**Decision:** prerequisites fail; zero real inference attempts and zero real model traces were produced. There is no genuine model verifier outcome, latency, reward measurement or tool-use claim. No weights were fetched and no substitute model was used. The existing runner also requires a clean committed worktree; this dirty audited state was not falsely attributed to HEAD or bypassed.

## Actual offline validation

Commands executed from the owned worktree:

```sh
git status --short
git rev-parse HEAD
sysctl hw.memsize
vm_stat
df -h . /Users/admin/.cache /private/tmp

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/python -m pytest tests/agentic/test_model_policy.py tests/agentic/test_agentic.py -q -p no:cacheprovider
/Users/admin/workspaces/projects/wendy-foundation-model/.venv/bin/ruff check --no-cache src/wendyfm/agentic tests/agentic/test_model_policy.py tests/agentic/test_agentic.py scripts/run_agentic_model.py
git diff --check
shasum -a 256 src/wendyfm/agentic/contracts.py src/wendyfm/agentic/runtime.py src/wendyfm/agentic/verifier.py scripts/run_agentic_model.py
```

Additional inline read-only Python probes ran under both interpreters: module availability/version checks described above; under default Python, `os.sysconf('SC_PHYS_PAGES')`, `os.sysconf('SC_PAGE_SIZE')`, and `os.sysconf('SC_AVPHYS_PAGES')`; under the test interpreter, configured/default HF cache paths and exact snapshot existence checks plus SHA-256 of the adapter and its test file. No Hub API or model loader was invoked.

Results: **78 passed, 1 warning in 4.63s**, exit 0. Warning: torch could not initialize absent NumPy. Ruff: **All checks passed**, exit 0. Whitespace check passed. No unnecessary full suite rerun.

Tests include strict parser rejection, revision mismatch, chat-template/mask forwarding doubles, token/context bounds, feedback continuity, contract validation and trusted episode replay. `test_feedback_driven_four_step_rollout_and_trusted_replay` passed: a **mock** backend supplies lookup(start) → lookup(drawer) → lookup(cabinet) → final violet; environment outputs drawer → cabinet → violet; independent verifier returns 1.0. These are actual assertions over scripted outputs, not traced Qwen actions. Spawned mock worker failure/timeout tests do not verify real model load or native-library behavior.

## Minimal future human-approved provisioning and smoke plan

This is documentation only, not an installation request or action. After separate explicit authorization:

1. Use an isolated environment owned for this project, leaving existing environments and independent paper experiments untouched. Resolve compatible CPU torch, Transformers (historically recorded minimum 4.51.0), Hub, safetensors and required transitive packages; pin the tested versions. Qwen support and API compatibility must be checked, not assumed from a lower version bound. Avoid TRL/PEFT, GPU extras and a duplicate trainer for this inference gate.
2. Fetch only the approved revision's config, generation config, tokenizer files/template, model weights and model card/license. Recorded weight size is 1,503,300,328 bytes (~1.40 GiB), not locally measured. Budget at least several GiB disk for artifacts and environment overhead; CPU float32 weights alone may be approximately 3 GB, with additional loading, KV cache and runtime memory. Recheck memory pressure and establish conservative headroom (e.g. 8 GiB available as a planning threshold, not a measured requirement) before any load. Dependency supply-chain, native compatibility and disk/cache duplication remain risks. The recorded Apache-2.0 license must be checked against the exact fetched license/card and attribution obligations.
3. Verify the exact snapshot revision and file completeness, hash model/tokenizer/config/template files, reject alternate revisions or broken links, directly inspect and exercise the pinned chat template with tool messages. Establish a clean reviewed code SHA without committing unrelated work or secrets. No automatic commits or downloads are authorized here.
4. Run **one** existing `scripts/run_agentic_model.py` smoke with offline flags, CPU float32, greedy generation, thinking disabled, four decisions, at most 128 new tokens per decision, 4,096 input tokens, 60-second startup/decision bounds and 300-second inference deadline. Allow bounded direct-worker cleanup; CLI snapshot checks/hashing/output are outside that inference deadline and need a separate overall wall limit. Store evidence outside Git in an authorized temporary path. Preserve timeout/parse/model failure as evidence; do not repeatedly retry. Independently replay using `--verify` and a trusted `--expected-sha`; inspect actual actions and feedback before claiming tool use. A correct final answer alone is insufficient. Direct-worker cleanup does not verify containment of descendants.

No GPU, paid compute, dependency installation, model download, private upload, credential access, PR creation, push or merge occurred. The only external-worktree use was reading the already installed test interpreter; bytecode and test/lint caches were disabled.

## Independent paper-project boundary

WendyFM owns implementation, policy/tool contracts, provenance and replay correctness. The independent paper project owns hypotheses, datasets/splits, reward science, ablations, statistics and publication. No paper worktree, private artifact, dataset or experiment was changed or validated. This public synthetic fixture supports infrastructure checks only; it provides no research generalization, training, RL update, reward-improvement or benchmark evidence. B1 worktree availability for B2 review: N/A as supplied.

TASK_STATUS: PASS
