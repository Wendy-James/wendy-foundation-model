# Upstream API and dependency review — 2026-10-22

Observed registry versions are inventory, **not tested integration pins**. Full direct
requirements, Python bounds, source HEAD SHAs and license excerpts are in
`upstream_inventory.json`; documented API excerpts/URLs in `upstream_api_evidence.json`.
PyPI and raw documentation were accessible; unauthenticated GitHub REST hit a rate limit,
so source SHAs were resolved with `git ls-remote`. Registry releases and repository HEAD
are recorded separately; latest docs may differ from release APIs. No framework installed.

| Upstream / observed package | License | Relevant API / implementation | Dependencies and compatibility risk | Estimated integration effort |
| --- | --- | --- | --- | --- |
| TRL 1.16.0 | Apache-2.0 | GRPOTrainer, tools, environment_factory, external reward_funcs | Python >=3.11, Transformers >=4.56.2, Accelerate >=1.10.1, datasets >=4.7.0, NumPy/Pandas/PyArrow; experimental environment API; latest vLLM exceeds TRL extra's <=0.31.0 bound | 1–2 days for adapter and smoke tests after model inference |
| PEFT 0.21.2 | Apache-2.0 | LoraConfig, get_peft_model, PeftModel.from_pretrained | Torch >=1.13, Transformers, Accelerate, safetensors, Hub; target module names/base revisions must match; WendyFM decoder lacks Transformers generation contract | ~1 day for an HF model load/adapter round trip, more for custom decoder |
| OpenEnv / openenv-core 0.3.0 | BSD-3-Clause | Environment reset/step and client/server protocol | FastAPI, Pydantic, WebSockets, HTTPX, OpenAI, Gradio, fastmcp; substantial service dependency surface for a three-link CPU fixture | 1–2 days for transport wrapper; defer service |
| verifiers 0.4.0 | MIT | MultiTurnEnv, Rubric, rollout/evaluation lifecycle | Async HTTP/model clients, datasets, Pydantic and orchestration dependencies (full list in inventory); distinct async state and endpoint assumptions | 2–3 days for state/reward mapping and integration tests |
| verl 0.9.1 | Apache-2.0 | Ray actors, FSDP/Megatron workers, rollout engine | Python <3.13 (current local interpreter 3.13), Ray, Torch/backend extras; distributed GPU stack and strict backend matrix | 1–2 weeks plus separately reviewed GPU infrastructure |
| OpenRLHF 0.11.2 | Apache-2.0 | Ray/vLLM/DeepSpeed PPO and agent execution | DeepSpeed ==0.19.6, flash-attn ==2.8.3, Ray ==2.55.0, Transformers ==5.15.0; CUDA-oriented dependencies; vLLM pin differs from latest | 1–2 weeks plus separately reviewed GPU infrastructure |
| vLLM 0.32.0 | Apache-2.0 | OpenAI-compatible tools/tool_choice and model-specific tool-call parser | Torch ==2.13.0, Transformers >=5.16.1,<5.20, CUDA/kernel dependencies; incompatible with observed OpenRLHF Transformers pin and TRL vLLM extra ceiling | 1–3 days for inference service on a verified platform; defer for CPU P0 |

Estimates are engineering judgments, not measured effort or compatibility claims.
Licenses must be rechecked on exact release artifacts before any code redistribution.

## Selected minimal approach

Use the local `LookupEnvironment.reset/step` boundary with strict WendyFM contracts,
consistent with OpenEnv's lifecycle but without its HTTP server dependencies. No upstream
code is copied; retain an explicit migration seam to an OpenEnv transport if tasks need it.
Avoid implementing a trainer. For the trainer adapter, choose TRL's **environment_factory**
with a fresh/reset environment per completion and a separate trusted reward function.

The documented TRL reset method returns None/string, while our reset returns Observation:
the future wrapper must return the instruction string, expose only a documented
`lookup(key: str) -> str` method, and record actions/results in host-owned state. Do not
expose `get_reward`, verifier methods, private task fields, or dataset `ground_truth`
columns as policy tools. TRL's external reward callback receives per-completion
`environments`; use a trusted registry keyed by completion to verify and return one float
per completion. Never place the evaluator on the policy-facing environment object.
TRL's tool method discovery and reset semantics need release-specific integration tests.

A future HF adapter should load revision-pinned AutoModelForCausalLM/AutoTokenizer,
map chat-template tool calls to Action, map ToolResult to tool-role messages, enforce
step/token/deadline bounds, and populate actual token metrics. Begin with inference only;
PEFT and GRPO weight updates follow a tested dependency matrix and separate experiment.
`NEXT_MODEL.json` records the selected Qwen3-0.6B immutable revision and bounded CPU plan.
No external API call, model download, inference, RL update or performance claim is made.
