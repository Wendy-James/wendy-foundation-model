# P1 model policy adapter (CPU-first, no weight update)

Status: **mocked adapter verified; real-model inference unverified**. This change is
stacked on P0 `892f7b5f7c325b43d819704540172dce282e5786`. It reuses P0's
`Observation`, `Action`, `Episode`, `LookupEnvironment`, rollout step loop and trusted
`verify_episode`. P0's scripted demo and schema remain unchanged.

## Policy interface and limits

The candidate is `Qwen/Qwen3-0.6B` at model **and tokenizer** revision
`c1899de289a04d12100db370d81485cdf75e47ca` (Apache-2.0). The pinned
`tokenizer_config.json` has a `tools` branch and `<tool_call>` formatting. Its model
card requires Transformers >=4.51.0; no Transformers version has passed actual
model inference here. The selected local loader requires the exact pinned snapshot,
uses `local_files_only=True`, `trust_remote_code=False`, CPU float32, greedy
generation, `enable_thinking=False`, and at most 128 generated tokens per decision.
It never automatically downloads weights. The published model weight file is
1,503,300,328 bytes; actual runtime RAM is not yet measured.

`ToolCallingPolicy` turns the policy-visible instruction and registered lookup schema
into Qwen chat messages. On each turn it records the assistant tool call and passes
the host's returned `ToolResult` as a `tool` message before generating again. It
accepts exactly one JSON `<tool_call>` or a plain final answer. Duplicate JSON keys,
malformed or partial calls, unclosed tags, and thinking/special tokens are parse
errors. Unknown tools and invalid arguments are sent only to P0's registered
`LookupEnvironment`, which returns `invalid_action`; there is no shell, network or
arbitrary Python tool dispatch. The policy receives no verifier, reward, world object
or expected answer.

One spawned child owns the loaded model across all decisions. The host retains the
environment and trusted verifier; it sends only detached Observations to the child.
Each decision has a 60-second maximum in the real-model command, with four decisions,
128 new tokens each, a bounded model-start period and a 300-second overall deadline.
The child is terminated on deadline/exit. This process separation does **not** provide
an OS sandbox against malicious local Python code; private evaluation needs stronger
isolation. The synthetic task has no held-out split, and its answer is public.
Model-worker shutdown escalates from SIGTERM to SIGKILL after a one-second grace
period; a SIGTERM-resistant mock is covered by an offline regression test. This
does not change P0's separate scripted-policy worker implementation.

The unchanged Episode terminal states are `final`, `max_steps`, `timeout` and
`policy_error`. Model decisions now require integer input/output token counts; P0
scripted decisions still require null counts. `rollout_model` separately reports
`parse_error`, `stale_observation`, `inference_failure` or `timeout`; tool validation
is in each Step's `ToolResult.status=invalid_action`. The trusted verifier checks
trace provenance, action replay, reward and count **type**, not whether model token
counts or latency were honestly measured. The proposed run JSON stores the actual
model/tokenizer revision, weight/tokenizer SHA256, Episode, error type and wall time.
No model-run JSON exists until a real run succeeds.

## Reproduce offline checks

From the P1 worktree with the existing P0 Python environment:

```sh
/Users/admin/workspaces/projects/wendyfm-m3-trainer/.venv/bin/python -m pytest -q tests/agentic
/Users/admin/workspaces/projects/wendyfm-m3-trainer/.venv/bin/ruff check src/wendyfm/agentic tests/agentic scripts/run_agentic_model.py
git diff --check
```

`tests/agentic/test_model_policy.py` uses mock generated text and no model files.
It covers four feedback-dependent actions, messages, reward replay, unknown tools,
malformed calls, stale observations, inference failure, timeout and revision mismatch.

## Local model preflight and future command

At review time this Mac had 16 GiB RAM and ~110 GiB free disk, but no cached
`Qwen3-0.6B` snapshot. The existing P0 environment had Torch but lacked
Transformers, Hugging Face Hub, safetensors, TRL and PEFT. Thus no real-model
inference was attempted, and no large dependency stack or 1.50 GB weights were
downloaded. RAM/disk capacity alone is not a verified model-runtime preflight.

After the exact model snapshot and compatible Transformers/Hub installation are
separately approved in a new isolated environment, run from a **clean committed**
P1 checkout, with an output path outside Git:

```sh
PYTHONPATH=src python scripts/run_agentic_model.py --output /tmp/wendyfm-qwen3-p1.json
PYTHONPATH=src python scripts/run_agentic_model.py --verify /tmp/wendyfm-qwen3-p1.json --expected-sha <P1_COMMIT_SHA>
```

The first command refuses an uncached model and never downloads it. The second
command replays P0's trusted verifier with an independently supplied code SHA;
it does not rerun inference. Before any scientific claim, inspect the trace to
establish that the actual model made multiple tool calls after seeing feedback.
A correct final answer alone is insufficient. No GRPO/PEFT training is included.

## Research-project handoff

Callable boundary: `rollout_model(RunManifest, backend_factory) -> (Episode,
error_type)`; `verify_episode(Episode, separately_trusted_manifest) -> RewardResult`.
The model factory must expose immutable `model_revision` and `tokenizer_revision`.
The paper project supplies frozen task/split definitions, independent truth rules,
train-reward semantics, malformed/tool-failure controls and analysis. This public
three-link lookup task validates engineering flow only. P0 `PAPER_INTERFACE.md`
remains the authoritative Episode v0.1 contract. TRL's `environment_factory`
integration and all RL weight updates remain separate, unverified gates.
