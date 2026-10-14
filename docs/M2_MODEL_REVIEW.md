# Milestone 2 review — minimal decoder-only Transformer (draft)

**Status:** engineering prototype on a review branch. This is **not** a completed model-training, pretraining, or benchmark milestone.

## Implementation

- Torch implementation in `src/wendyfm/model.py`: token embeddings, RMSNorm, explicit causal multi-head attention, RoPE, SwiGLU MLP, residual blocks, and tied output head.
- Shape and input checks; cross-entropy with caller-provided next-token labels. The caller, not the model, shifts next-token labels.
- `tests/test_model.py`: shape/loss, parameter tying, causal isolation, gradients, invalid inputs, and toy-sequence overfit test.
- `scripts/smoke_tiny_lm.py`: CPU-only, deterministic toy-data smoke output. No checkpoints/data artifacts published.

## Local test evidence (2026-10-14)

Execution was performed in an isolated CPU tool environment, **not** in the user's Mac/DSW/Kaggle environment.

- Python 3.13.5; torch 2.10.0+cpu; CPU threads = 1.
- `PYTHONPATH=src pytest -q tests/test_model.py`: **5 passed**.
- `PYTHONPATH=src python scripts/smoke_tiny_lm.py`: **passed**.
- Config: vocab 24, width 32, 4 heads, 2 blocks, context 12; tied embeddings.
- Unique parameters: 33,696.
- Repeated synthetic sequence, seed 7, 65 AdamW steps, LR 0.003.
- Loss: initial `28.28857994`, final `0.00031974`. This demonstrates toy fitting, **not generalization**.

## How to rerun after checkout

```bash
uv sync --group dev
uv pip install 'torch>=2.3'
PYTHONPATH=src uv run --no-sync pytest -q tests/test_model.py
PYTHONPATH=src uv run --no-sync python scripts/smoke_tiny_lm.py
```

**Dependency note:** this is a draft while a PyTorch optional-dependency group and refreshed `uv.lock` are still to be reviewed. The standard tokenizer-only `uv sync --group dev` should not be forced to install GPU/torch extras; model tests use `pytest.importorskip` until the extra is formalized.

## Review gate before merging

- [ ] Re-run on the owner's local Mac (Python 3.11 or pinned supported version); verify `uv` commands and lock strategy.
- [ ] Format with `ruff check` and `ruff format --check`; inspect gradient and causal-mask behavior.
- [ ] Decide whether to use reference attention or SDPA for larger models, keeping explicit tests.
- [ ] Add numerical correctness tests vs. a reference for RoPE and attention; handle dtype/device cases.
- [ ] Add real public-text tokenization and train/validation split, establish loss baseline, and log config/seed.
- [ ] Record GPU/CPU hardware, tokens/s, peak memory, model/data revisions, and reproducibility conditions.
- [ ] Review with the repository owner; do not claim Milestone 2 completed until the above is independently verified.

Inspired by the publicly available [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics); implementation and tests should be studied and explained, not merely accepted because they were AI-assisted.
