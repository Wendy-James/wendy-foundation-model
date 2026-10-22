"""Trusted host-only verifier. Never exported through the policy observation."""

from dataclasses import asdict

from .contracts import Episode, RewardResult, RunManifest, decode
from .environment import FIXTURE_SHA, LookupEnvironment


def reward_for(manifest, steps, terminal):
    # Compute the oracle independently of recorded tool feedback.
    env = LookupEnvironment()
    env.reset(manifest.seed)
    key = "start"
    for _ in range(3):
        key = env.world[key]
    success = (terminal == "final" and bool(steps) and
               steps[-1].action is not None and steps[-1].action.final_answer == key)
    return RewardResult(float(success), "lookup-chain-verifier.v1", True)


def verify_episode(episode, trusted_manifest):
    """Require a separately trusted manifest; a trace's self-declared SHA is insufficient.

    Replay deterministic actions/results, then recompute reward. Timings are measured
    metadata, not cryptographically authenticated or deterministically reproduced.
    """
    episode = Episode.from_json(episode.to_json())
    trusted_manifest = decode(RunManifest, asdict(trusted_manifest))
    if episode.manifest != trusted_manifest or trusted_manifest.fixture_sha256 != FIXTURE_SHA:
        raise ValueError("provenance mismatch")
    if not 1 <= len(episode.steps) <= trusted_manifest.max_steps:
        raise ValueError("step count mismatch")
    env = LookupEnvironment()
    env.reset(trusted_manifest.seed)
    terminal = "max_steps"
    for index, step in enumerate(episode.steps):
        if step.index != index or terminal != "max_steps":
            raise ValueError("step ordering mismatch")
        counts = (step.input_tokens, step.output_tokens)
        if trusted_manifest.policy.kind == "scripted":
            if any(value is not None for value in counts):
                raise ValueError("scripted token metrics must be null")
        elif step.action is not None and any(
            type(value) is not int or value < 0 for value in counts
        ):
            raise ValueError("model decisions require integer token counts")
        elif step.action is None and any(value is not None for value in counts):
            raise ValueError("failed model decisions cannot claim token counts")
        if step.result is not None and step.result.status == "timeout":
            if step.result.output != "step deadline exceeded":
                raise ValueError("timeout mismatch")
            terminal = "timeout"
        elif step.action is None:
            if step.result is not None:
                raise ValueError("policy error mismatch")
            terminal = "policy_error"
        elif step.action.tool_call is not None:
            if env.step(step.action.tool_call) != step.result:
                raise ValueError("tool output mismatch")
        else:
            if step.result is not None:
                raise ValueError("final result mismatch")
            terminal = "final"
    if terminal == "max_steps" and len(episode.steps) != trusted_manifest.max_steps:
        raise ValueError("premature termination")
    if episode.terminal_state != terminal:
        raise ValueError("terminal mismatch")
    reward = reward_for(trusted_manifest, episode.steps, terminal)
    if reward != episode.reward:
        raise ValueError("reward tampering")
    return reward
