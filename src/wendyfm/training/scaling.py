"""Frozen three-cell M5 Phase C1 model-size experiment contract."""

from __future__ import annotations

import json
from pathlib import Path

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.training.loop import TrainConfig
from wendyfm.training.profile import BASELINE, MODEL, TRAIN, run_profile

MODELS = (
    MODEL,
    {**MODEL, "d_model": 64, "n_layers": 4, "intermediate_size": 128},
    {**MODEL, "d_model": 128, "n_layers": 6, "intermediate_size": 256},
)
TRAIN_TEXT = "../data/fixtures/cpu_pilot_train.txt"


def load_sweep_config(path: Path) -> dict:
    settings = json.loads(path.read_text(encoding="utf-8"))
    if set(settings) != set(BASELINE) | {"models", "train", "train_text"}:
        raise ValueError("Phase C1 config keys differ from the frozen sweep")
    if any(type(settings[key]) is not type(value) or settings[key] != value
           for key, value in BASELINE.items()):
        raise ValueError("Phase C1 baseline bounds differ")
    if (type(settings["models"]) is not list or settings["models"] != list(MODELS)
            or any(type(model) is not dict or any(type(value) is not int
                   for value in model.values()) for model in settings["models"])):
        raise ValueError("Phase C1 models differ from the strict three-cell allowlist")
    if settings["train"] != TRAIN or any(type(value) is not type(TRAIN[key])
                                        for key, value in settings["train"].items()):
        raise ValueError("Phase C1 optimizer differs from the frozen baseline")
    if settings["train_text"] != TRAIN_TEXT:
        raise ValueError("Phase C1 corpus differs from the M4 fixture")
    TrainConfig(**settings["train"])
    for model in settings["models"]:
        ModelConfig(**model)
    return settings


def parameter_counts() -> list[int]:
    """Count actual trainable tensors, including embeddings and norms."""
    return [sum(parameter.numel() for parameter in
                DecoderOnlyTransformer(ModelConfig(**model)).parameters()
                if parameter.requires_grad) for model in MODELS]


def run_cell(config_path: Path, output_path: Path, *, expected_sha: str,
             cell: int) -> dict:
    settings = load_sweep_config(config_path)
    if type(cell) is not int or not 1 <= cell <= len(MODELS):
        raise ValueError("cell must be 1, 2, or 3")
    model = settings["models"][cell - 1]
    result = run_profile(config_path, output_path, expected_sha=expected_sha,
                         settings={**settings, "model": model})
    result["cell"] = cell
    result["model"] = model
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return result
