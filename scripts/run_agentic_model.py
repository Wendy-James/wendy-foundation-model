"""Bounded, local-only Qwen3 CPU rollout or independent episode replay."""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from wendyfm.agentic.contracts import Episode, PolicyRef
from wendyfm.agentic.demo import baseline_manifest
from wendyfm.agentic.model_policy import (
    MODEL_ID,
    MODEL_REVISION,
    LocalHFBackendFactory,
    rollout_model,
)
from wendyfm.agentic.verifier import verify_episode


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(sha):
    base = baseline_manifest(sha)
    return replace(base, timeout_seconds=60.0,
                   policy=PolicyRef("model", MODEL_REVISION, MODEL_REVISION,
                                    sha, MODEL_REVISION))


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--output", type=Path)
    choice.add_argument("--verify", type=Path)
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.verify:
        if not args.expected_sha:
            parser.error("--verify requires an independently trusted --expected-sha")
        record = json.loads(args.verify.read_text(encoding="utf-8"))
        episode = Episode.from_json(json.dumps(record["episode"]))
        verify_episode(episode, _manifest(args.expected_sha))
        print(f"verified=true terminal={episode.terminal_state} "
              f"steps={len(episode.steps)} reward={episode.reward.value}")
        return

    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                               text=True).strip():
        parser.error("a clean committed worktree is required")
    if args.output.resolve().is_relative_to(root):
        parser.error("write model evidence outside Git")
    if args.output.exists():
        parser.error("output already exists")
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                          local_files_only=True))
    except (ImportError, OSError) as exc:
        parser.error(
            f"pinned model unavailable locally; no download attempted: {type(exc).__name__}"
        )
    weights = snapshot / "model.safetensors"
    tokenizer = snapshot / "tokenizer.json"
    if not weights.is_file() or not tokenizer.is_file():
        parser.error("pinned local snapshot lacks reviewed model/tokenizer files")

    started = time.monotonic()
    episode, error_type = rollout_model(_manifest(sha), LocalHFBackendFactory())
    verify_episode(episode, _manifest(sha))
    record = {"schema": "wendyfm.model-rollout.v0.1", "model_id": MODEL_ID,
              "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
              "model_weights_sha256": _sha256(weights),
              "tokenizer_sha256": _sha256(tokenizer),
              "wall_seconds": time.monotonic() - started,
              "error_type": error_type, "episode": asdict(episode)}
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(record, destination, sort_keys=True, allow_nan=False)
        destination.write("\n")
    print(f"terminal={episode.terminal_state} steps={len(episode.steps)} "
          f"reward={episode.reward.value} error={error_type}")


if __name__ == "__main__":
    main()
