"""Run or replay the public synthetic baseline without importing Torch."""

import argparse
import subprocess
from pathlib import Path

from .contracts import VERSION, Episode, PolicyRef, RunManifest
from .environment import FIXTURE_SHA
from .runtime import rollout, scripted_policy
from .verifier import verify_episode


def baseline_manifest(sha):
    return RunManifest(VERSION, sha, "lookup-chain.v1", "synthetic-chain-001", 17, 4, 5.0,
                       PolicyRef("scripted", None, None, None, sha), FIXTURE_SHA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sha = args.expected_sha or subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if not args.expected_sha and subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        parser.error("clean checkout required, or supply an explicitly trusted --expected-sha")
    manifest = baseline_manifest(sha)
    episode = (Episode.from_json(args.replay.read_text()) if args.replay
               else rollout(manifest, scripted_policy))
    verify_episode(episode, manifest)
    if args.output:
        with args.output.open("x") as output:
            output.write(episode.to_json() + "\n")
    print(f"terminal={episode.terminal_state} steps={len(episode.steps)} "
          f"reward={episode.reward.value} verified=true sha={sha}")


if __name__ == "__main__":
    main()
