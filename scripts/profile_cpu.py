"""Bounded CPU training measurement using WendyFM's model, sampler, and loop.

Run from a checkout with ``python scripts/profile_cpu.py``. The synthetic token
stream and model initialization are seeded; wall time and throughput are actual
measurements and will vary between runs. No GPU statistics are collected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wendyfm.model import DecoderOnlyTransformer, ModelConfig  # noqa: E402
from wendyfm.training.data import NextTokenBatchSampler  # noqa: E402
from wendyfm.training.loop import TrainConfig, train  # noqa: E402


def _bounded(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def run(
    output_path: Path,
    *,
    warmup_steps: int = 1,
    measured_steps: int = 3,
    batch_size: int = 2,
    seq_len: int = 8,
    seed: int = 20261010,
) -> dict[str, int | float | str]:
    """Warm up, then time optimizer steps and save one small JSON result.

    A target token is one element of the explicit shifted target tensor, so
    each measured step scores ``batch_size * seq_len`` targets. The existing
    training loop times sampling, forward/backward, and optimizer work.
    """
    _bounded(warmup_steps, "warmup_steps", 5)
    _bounded(measured_steps, "measured_steps", 20)
    _bounded(batch_size, "batch_size", 4)
    _bounded(seq_len, "seq_len", 32)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")

    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(seed)
    model_config = ModelConfig(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=seq_len,
        intermediate_size=32,
    )
    token_ids = torch.arange(256, device="cpu", dtype=torch.long) % model_config.vocab_size
    sampler = NextTokenBatchSampler(
        token_ids,
        seq_len=seq_len,
        model_vocab_size=model_config.vocab_size,
        seed=seed,
    )
    model = DecoderOnlyTransformer(model_config).cpu()
    total_steps = warmup_steps + measured_steps
    config = TrainConfig(
        max_steps=total_steps,
        learning_rate=0.003,
        min_learning_rate=0.003,
        weight_decay=0.01,
    )

    def provider() -> tuple[torch.Tensor, torch.Tensor]:
        return sampler.sample(batch_size)

    warmup = train(model, provider, config, end_step=warmup_steps)
    measured = train(
        model,
        provider,
        config,
        optimizer=warmup.optimizer,
        start_step=warmup_steps,
        end_step=total_steps,
    )
    if not all(
        math.isfinite(loss)
        for loss in (
            warmup.initial_loss,
            warmup.final_loss,
            measured.initial_loss,
            measured.final_loss,
        )
    ):
        raise FloatingPointError("nonfinite loss in CPU profile")
    elapsed = measured.mean_step_seconds * measured_steps
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("measured wall time must be finite and positive")
    target_tokens = measured_steps * batch_size * seq_len
    result: dict[str, int | float | str] = {
        "device": "cpu",
        "seed": seed,
        "cpu_threads": 1,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "target_token_definition": "one shifted target ID per batch and sequence position",
        "timed_target_tokens": target_tokens,
        "warmup_loss_last_batch": warmup.final_loss,
        "timed_loss_first_batch": measured.initial_loss,
        "timed_loss_last_batch": measured.final_loss,
        "timed_wall_time_seconds": elapsed,
        "mean_step_seconds": measured.mean_step_seconds,
        "target_tokens_per_second": target_tokens / elapsed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/profile_cpu.json"))
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
