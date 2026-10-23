# Qwen CPU inference gate — 2026-10-23

Status: **actual pinned tokenizer/template verified; model inference blocked**.
No GPU, large weight download, model load, generation, GRPO or weight update.

## Preserved overnight engineering

Read all five detailed B1 reports and five completion messages from
`agentic-30h-v3-20261022_205603`, plus its progress/summary/evidence manifest.
There was no completed B1 01 report. PR #29's live remote head was
`ce66312aad0e47c38aae7225a9ab124dcd03880f`; comments and reviews were empty.
The 30h development branch still had 11 changed/untracked files on that same base.

Created `/Users/admin/workspaces/projects/wendyfm-agentic-inference-20261023`,
branch `feat/agentic-inference-gate-20261023`, stacked on PR #29. Applied the tracked
diff and copied the untracked files without changing the original development or QA
worktrees. All 11 preserved files are byte-identical to the originals; all 15 B1
evidence-manifest entries match. Commit `ed87db8` preserves them for review.

The preserved implementation adds transactional policy history, feedback continuity,
metric bounds, parser rejection and explicit session recovery. Production rollout
still stops on policy errors; it does not retry automatically. The opt-in provenance
sidecar compares independently retained host evidence before fixed-verifier replay;
it does not prove actual inference. Frozen Episode v0.1 is unchanged. Historical
reports retain their dates, approval restrictions and limitations. The current user
instruction authorizes this isolated dependency setup; earlier overnight reports
did not authorize installs. B1 05 corrects B1 06's sidecar count to 12 cases.

Fresh focused checks on the byte-identical preserved source:

```text
Existing read-only Python 3.11.16 test environment:
tests/agentic: 93 passed, 1 existing Torch missing-NumPy warning in 7.52s
Ruff: All checks passed!
git diff --check: passed
```

The historical 245 full-suite result was not rerun. B2's overnight review was read
as dated independent evidence; it did not accept the later finalized sidecar.
No fresh independent QA acceptance of this PR is claimed.

## Capacity and isolated environment

Read-only host inventory: 16 GiB physical RAM, 104 GiB free disk. VM counters showed
approximately 6.72 GiB compressor occupancy; swap usage was 19,969.50 MiB.
`memory_pressure -Q` reported 32% system-wide memory free. These are separate VM
observations, not a demonstrated safe FP32 model load or an allocatable-memory
guarantee. Recheck pressure/headroom before any future model load.

Created `/Users/admin/.cache/wendyfm/envs/qwen3-cpu-20261023`, CPython 3.11.16, without
changing any existing environment. Torch 2.14.1 was reused from the package cache
offline. Installed only inference dependencies; exact 25-package pins are in
`configs/agentic/qwen3-cpu-macos-arm64.txt`. No TRL, PEFT, vLLM or test stack.

The initial UV offline full Torch resolution failed on missing cached dependencies.
After fetching small dependencies, UV stalled/restarted the Transformers wheel
download; it was interrupted after 9m31s. A bounded curl download of the exact
same PyPI wheel succeeded and its 11,993,498 bytes matched SHA256
`4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550`.
Installed that wheel locally and the remaining dependencies from cache.
`uv pip check` exposed missing Torch setuptools; installing cached setuptools
84.2.1 resolved it. Final check: **all installed packages compatible**.
Imports of AutoTokenizer, AutoModelForCausalLM and Qwen3ForCausalLM passed; a CPU
Torch tensor returned `[1.0]`. This is import/CPU validation, not model execution.

Reproduce the environment on macOS arm64 with a supplied CPython 3.11.16:

```sh
uv venv --python 3.11.16 /tmp/wendyfm-qwen-cpu
uv pip install --python /tmp/wendyfm-qwen-cpu/bin/python -r configs/agentic/qwen3-cpu-macos-arm64.txt
uv pip check --python /tmp/wendyfm-qwen-cpu/bin/python
```

These install packages, not model weights. Other platforms/Python versions have not
been validated. Exact pins do not guarantee compatibility with every platform.

## Actual tokenizer/template evidence

Remote metadata at immutable revision
`c1899de289a04d12100db370d81485cdf75e47ca` confirms Apache-2.0 and a
1,503,300,328-byte `model.safetensors`. The remote artifact reference is saved in
`evidence/qwen3_remote_reference_20261023.json`; its expected weight SHA256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
This is remotely observed metadata; no local weight bytes were hashed or fetched.

Through upstream `hf_hub_download`, fetched only `config.json`,
`generation_config.json`, `tokenizer_config.json`, `tokenizer.json` and `README.md`
at that pin, using public access (`token=False`), with an explicit <12 MB file
allowlist. The tokenizer's measured SHA256 matches the remote LFS reference.
No tokenizer/model blobs are committed; only public JSON hashes and measurements.

Ran `scripts/preflight_agentic_model.py` once in offline mode using the actual
AutoTokenizer and its pinned chat template. It never loads/generates/downloads a
model. The assistant lookup call and tool feedback are supplied fixtures, clearly
marked `source=tokenizer_fixture` and `real_model_inference=false`.

| Actual check | Result |
| --- | --- |
| Transformers / Hub / tokenizers | 4.57.6 / 0.36.0 / 0.22.2 |
| Torch / safetensors / NumPy | 2.14.1 / 0.6.2 / 2.3.5 |
| Initial fixture input | 173 tokens |
| Tool/feedback fixture input | 212 tokens |
| Assistant `<tool_call>` and tool `<tool_response>` | Preserved |
| Attention mask and input tensor shape | Match |
| Thinking-disabled generation prompt | Verified |
| Tokenizer preflight wall time | 0.145451 seconds |
| Actual generated tokens / model latency | null / null |
| Cached weights / genuine model episodes | absent / zero |

Complete measured hashes, fixture messages and versions are in
`evidence/qwen3_tokenizer_preflight_20261023.json`. The template's measured SHA256 is
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
The local script preserves failures as JSON with a nonzero exit, and writes evidence
exclusively outside Git; the reviewed public result was copied here afterward.

With those small files cached, reproduce without network/model execution:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /tmp/wendyfm-qwen-cpu/bin/python scripts/preflight_agentic_model.py \
  --output /tmp/wendyfm-qwen-tokenizer-new.json
```

The preflight asserts the actual Qwen template protocol required by the existing
adapter; it does not generate or infer tool actions. A successful template check,
mock rollout, provenance receipt or reward replay is insufficient for the real gate.

## Blockers and next action

Weights are absent from the configured/default Hub cache. No alternative local
snapshot path was supplied at review. Dependencies and pinned template are now ready,
but genuine **model → tool → feedback → model → final answer** remains unverified.
There is no real trajectory, generated-token count, model latency or inference
failure to report; blocked preflight is distinct from attempted model inference.

Next: obtain explicit approval to cache the pinned 1.50 GB weights, or supply the
existing exact snapshot locally. Reduce/recheck memory pressure before loading.
Then run exactly one existing CPU FP32 greedy rollout from a clean source SHA:
four decisions, <=128 new tokens/decision, <=4096 input tokens, thinking disabled,
60-second startup/decision bounds, 300-second inference budget and bounded cleanup.
Use a separate outer supervisor deadline for snapshot/hash/write work; preserve any
failure rather than automatically retry. Inspect the actual actions, feedback,
final answer, token counts and timings, and independently replay the Episode.

The legacy runner's `--verify` checks only Episode consistency, not its outer
evidence metadata or proof of inference. The optional receipt is still opt-in.
Direct-worker cleanup does not contain arbitrary descendants. Those documented
limits remain open. GRPO and new TRL integration stay deferred until the real-model
gate passes. No research datasets, private scorers or scientific claims were used.
