"""Bounded CPU-only pretraining pilot using the existing training components."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.checkpoint import load_checkpoint, save_checkpoint
from wendyfm.training.data import NextTokenBatchSampler, prepare_token_ids
from wendyfm.training.loop import TrainConfig, train


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _documents(path: Path) -> list[str]:
    documents = path.read_text(encoding="utf-8").splitlines()
    if not documents or any(not document.strip() for document in documents):
        raise ValueError(f"expected nonempty, one-document-per-line text: {path}")
    return documents


@torch.no_grad()
def validation_loss(model: DecoderOnlyTransformer, ids: torch.Tensor, seq_len: int) -> float:
    """Score every full held-out window once, without updates or random draws."""
    if ids.numel() < seq_len + 1:
        raise ValueError("validation corpus needs at least seq_len + 1 tokens")
    model.eval()
    losses = []
    for start in range(0, ids.numel() - seq_len):
        window = ids[start : start + seq_len + 1]
        losses.append(model.next_token_loss(window[:-1][None, :], window[1:][None, :]))
    return float(torch.stack(losses).mean())


def run(config_path: Path, output_dir: Path, *, end_step: int | None = None,
        resume: bool = False) -> dict:
    """Run a fixed-size segment and write a checkpoint and factual JSON manifest."""
    started = time.perf_counter()
    config_bytes = config_path.read_bytes()
    settings = json.loads(config_bytes)
    model_config = ModelConfig(**settings["model"])
    train_config = TrainConfig(**settings["train"])
    batch_size = settings["batch_size"]
    seed = settings["seed"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    threads = settings["cpu_threads"]
    if type(threads) is not int or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    torch.set_num_threads(threads)
    random.seed(seed)
    torch.manual_seed(seed)
    train_path = (config_path.parent / settings["train_text"]).resolve()
    validation_path = (config_path.parent / settings["validation_text"]).resolve()
    if train_path == validation_path:
        raise ValueError("training and validation paths must differ")
    train_documents = _documents(train_path)
    validation_documents = _documents(validation_path)
    if set(train_documents) & set(validation_documents):
        raise ValueError("training and validation documents must be disjoint")

    tokenizer = BPETokenizer.train(
        train_documents, vocab_size=model_config.vocab_size,
        special_tokens=(settings["end_of_document"],),
    )
    train_ids = prepare_token_ids(
        train_documents, tokenizer, model_vocab_size=model_config.vocab_size,
        end_of_document=settings["end_of_document"],
    )
    validation_ids = prepare_token_ids(
        validation_documents, tokenizer, model_vocab_size=model_config.vocab_size,
        end_of_document=settings["end_of_document"],
    )
    sampler = NextTokenBatchSampler(
        train_ids, seq_len=model_config.max_seq_len,
        model_vocab_size=model_config.vocab_size, seed=seed,
    )
    model = DecoderOnlyTransformer(model_config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    start_step = 0
    if resume:
        start_step = load_checkpoint(
            checkpoint_path, model, optimizer, sampler=sampler,
            config=train_config, batch_size=batch_size,
        )
    if end_step is None:
        end_step = train_config.max_steps
    if type(end_step) is not int or not start_step < end_step <= train_config.max_steps:
        raise ValueError("end_step must be after the checkpoint step and at most max_steps")

    initial_validation_loss = validation_loss(model, validation_ids, model_config.max_seq_len)
    result = train(
        model, lambda: sampler.sample(batch_size), train_config,
        optimizer=optimizer, start_step=start_step, end_step=end_step,
    )
    final_validation_loss = validation_loss(model, validation_ids, model_config.max_seq_len)
    save_checkpoint(
        checkpoint_path, model, result.optimizer, result.step,
        sampler=sampler, config=train_config, batch_size=batch_size,
    )
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    manifest = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "train_text_sha256": _sha256(train_path),
        "validation_text_sha256": _sha256(validation_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "device": "cpu",
        "seed": seed,
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "train_tokens": train_ids.numel(),
        "validation_tokens": validation_ids.numel(),
        "validation_windows": validation_ids.numel() - model_config.max_seq_len,
        "start_step": start_step,
        "optimizer_steps": result.step,
        "segment_steps": result.step - start_step,
        "train_loss_first_batch": result.initial_loss,
        "train_loss_last_batch": result.final_loss,
        "initial_validation_loss": initial_validation_loss,
        "validation_loss": final_validation_loss,
        "tokens_per_second": result.tokens_per_second,
        "training_wall_time_seconds": result.mean_step_seconds * (result.step - start_step),
        "wall_time_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_path.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cpu_pilot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cpu_pilot"))
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, end_step=args.end_step,
                         resume=args.resume), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
