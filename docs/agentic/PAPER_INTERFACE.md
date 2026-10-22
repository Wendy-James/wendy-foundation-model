# WendyFM engineering interface

Version: `wendyfm.episode.v0.1`. This is an offline correctness milestone.
WendyFM owns contracts, runtime, observability, and reproducibility. The separate paper
project owns hypotheses, datasets, reward research, experiments, statistics, and
publication. Its data, experiments, and claims are not copied here.

## Callable modules and boundaries

- `contracts`: frozen `Episode`, `Step`, `ToolCall`, `ToolResult`, `Observation`,
  `Action`, `PolicyRef`, `RewardResult`, `RunManifest`; `decode`, `canonical`, `digest`.
- `environment.LookupEnvironment.reset(seed) -> Observation`,
  `step(ToolCall) -> ToolResult`, `observe() -> Observation`.
- `runtime.rollout(RunManifest, policy_callable) -> Episode`.
  Policy receives only detached Observation and returns one Action; no reward callback,
  manifest, expected answer, verifier identity/code, or evaluator seed is passed.
- `verifier.verify_episode(Episode, separately_trusted_manifest) -> RewardResult`.
  Checks provenance, deterministic action replay, terminal state and recomputed reward.
  The host owns this boundary. Never trust rewards supplied by the policy or a trace.

Only `lookup-chain.v1` / `synthetic-chain-001` is supported. It is a public synthetic
three-link lookup task followed by a final answer. `FIXTURE_SHA` pins canonical fixture
bytes. No held-out data or research reward is included. Lookup arguments require exactly
one string key, no extra properties, with bounded lengths; unknown keys/tools produce
invalid-action feedback. Four default steps include the final answer. Max bound is 100.
Each policy decision uses a fresh spawned process, terminated on deadline; the fixed
in-memory tool is bounded, and elapsed step time is checked after tool execution.

## Episode / trajectory JSON

An Episode contains `manifest`, ordered `steps`, `terminal_state`, and `reward`.
The steps list is the trajectory; no separate divergent trajectory schema exists.
Each Step stores action/tool arguments, result/status/output, index, elapsed wall seconds,
and input/output token counts. Scripted token counts are null, not estimated.
Terminal states: final, max_steps, timeout, policy_error. RewardResult records scalar
0.0/1.0, verifier identity `lookup-chain-verifier.v1`, and verified flag.

Manifest records schema, code provenance SHA, environment version, task ID, local RNG
seed, step bound, timeout, fixture SHA256 and PolicyRef. PolicyRef records model,
tokenizer, adapter and policy revisions. Model/tokenizer/adapter are null for scripted
execution. Real models require immutable resolved repository SHAs and artifact SHA256s;
any adapter must pin its revision, base-model compatibility and artifact digest. Future
datasets require immutable source revision, manifest/artifact SHA256s and held-out split
provenance in a new versioned manifest. P0 has only the frozen public synthetic fixture.
Unknown fields, wrong types, booleans as integers, duplicate keys and unsupported versions
are rejected. JSON is canonical and finite; trace input size is bounded.

## Exact CPU baseline and replay

From a clean checkout of the delivered commit (replace `<PR_SHA>` with the Draft PR head):

```sh
git checkout --detach <PR_SHA>
PYTHONPATH=src python3 -m wendyfm.agentic.demo --output /tmp/wendyfm-p0-episode.json
PYTHONPATH=src python3 -m wendyfm.agentic.demo --replay /tmp/wendyfm-p0-episode.json --expected-sha <PR_SHA>
```

Output file must not already exist. Python >=3.11; no Torch/framework needed for this demo.
Expected baseline: final, 4 steps, reward 1.0, verified true. The SHA is measured from
Git at runtime. Replay requires a trusted SHA/manifest, never one taken only from the
untrusted trace. The synthetic seed and task are fixed in `baseline_manifest`.

## Limitations and next boundary

This is an interface-isolated synthetic tool world, **not an OS security sandbox**.
Python callables are trusted engineering code: spawned processes share filesystem and
user permissions. Before private evaluation or arbitrary agent code, use separate
restricted service/container identities, protected evaluator storage, resource limits,
and authenticated trace provenance. No hidden evaluation dataset exists in this demo.
A public toy answer can be guessed; reward indicates answer correctness, not proof the
policy reasoned. Replay validates environment consistency, not policy determinism or
cryptographic authenticity of timings/timeout/error assertions. Deadlines include process
startup; token/timing reproducibility has these explicit limits. No arbitrary tools,
network tools, distributed rollout, real-model inference, adapter loading or weight update.

The next smallest inference experiment: one locally available, revision-pinned small
instruction/tool-capable Transformers model, greedy CPU generation, this same four-step
public task, capped generated tokens and wall time. Parse tool calls strictly and record
actual tokenizer token counts, success, invalid actions, latency, model/tokenizer digests.
Do not use the short-trained 21.5M WendyFM model as a claimed capable agent. No GPU or
training is needed; approve model choice/license and artifact source before download.
