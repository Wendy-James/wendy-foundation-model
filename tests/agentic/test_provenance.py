"""Synthetic receipts only: no genuine inference or artifact measurement."""

import inspect
from dataclasses import asdict, replace

import pytest

from wendyfm.agentic.contracts import PolicyRef, decode
from wendyfm.agentic.demo import baseline_manifest
from wendyfm.agentic.provenance import (
    ReplayProvenance,
    bind_provenance,
    verify_bound_episode,
)
from wendyfm.agentic.runtime import rollout, scripted_policy


@pytest.fixture(scope="module")
def recorded():
    episode = rollout(baseline_manifest("a" * 40), scripted_policy)
    receipt = bind_provenance(episode, source="scripted", verifier_sha256="b" * 64)
    return episode, receipt


def test_historical_episode_and_bound_replay(recorded):
    episode, receipt = recorded
    assert "source" not in asdict(episode)
    assert verify_bound_episode(episode, receipt, episode.manifest, receipt) == episode.reward
    assert all(step.input_tokens is None for step in episode.steps)


@pytest.mark.parametrize("change", [
    {"execution_sha": "c" * 40}, {"task_revision": "c" * 64},
    {"verifier_sha256": "c" * 64}, {"episode_sha256": "c" * 64},
    {"tool_results_sha256": ["c" * 64]},
    {"policy": PolicyRef("scripted", None, None, None, "c" * 40)},
])
def test_changed_receipt_rejected(recorded, change):
    episode, receipt = recorded
    with pytest.raises(ValueError, match="trusted provenance"):
        verify_bound_episode(episode, replace(receipt, **change), episode.manifest, receipt)


def test_rehashed_tampering_cannot_replace_host_receipt(recorded):
    episode, receipt = recorded
    steps = list(episode.steps)
    steps[0] = replace(steps[0], elapsed_seconds=steps[0].elapsed_seconds + 1.0)
    altered = replace(episode, steps=steps)
    forged = bind_provenance(altered, source="scripted", verifier_sha256="b" * 64)
    with pytest.raises(ValueError, match="trusted provenance"):
        verify_bound_episode(altered, forged, episode.manifest, receipt)
    with pytest.raises(ValueError, match="episode binding"):
        verify_bound_episode(altered, receipt, episode.manifest, receipt)


def test_mock_real_source_and_artifact_binding(recorded):
    episode, _ = recorded
    policy = PolicyRef("model", "d" * 40, "e" * 40, None, "f" * 40)
    episode = replace(episode, manifest=replace(episode.manifest, policy=policy))
    mock = bind_provenance(episode, source="mock", verifier_sha256="b" * 64)
    with pytest.raises(ValueError, match="requires artifact"):
        bind_provenance(episode, source="real", verifier_sha256="b" * 64)
    real = bind_provenance(episode, source="real", verifier_sha256="b" * 64,
                           model_sha256="1" * 64, tokenizer_sha256="2" * 64)
    for forged in (real, replace(mock, policy=replace(policy, model_revision="3" * 40)),
                   replace(mock, policy=replace(policy, tokenizer_revision="3" * 40)),
                   replace(real, model_sha256="3" * 64),
                   replace(real, tokenizer_sha256="3" * 64)):
        trusted = mock if forged.model_sha256 is None or forged == real else real
        with pytest.raises(ValueError, match="trusted provenance"):
            verify_bound_episode(episode, forged, episode.manifest, trusted)


def test_no_policy_verifier_or_oracle_extension(recorded):
    _, receipt = recorded
    for field in ("reward", "expected_answer", "verifier", "source_override"):
        with pytest.raises(ValueError, match="fields"):
            decode(ReplayProvenance, {**asdict(receipt), field: "untrusted"})
    assert list(inspect.signature(verify_bound_episode).parameters) == [
        "episode", "provenance", "trusted_manifest", "trusted_provenance"]
    with pytest.raises(ValueError, match="literal"):
        decode(ReplayProvenance, {**asdict(receipt), "source": "unknown"})
    with pytest.raises(ValueError, match="literal"):
        decode(ReplayProvenance, {**asdict(receipt), "verifier_identity": "external"})


@pytest.mark.parametrize("target", ["reward", "tool"])
def test_receipt_never_bypasses_fixed_verifier(recorded, target):
    episode, _ = recorded
    if target == "reward":
        altered = replace(episode, reward=replace(episode.reward, value=0.0))
    else:
        steps = list(episode.steps)
        steps[0] = replace(steps[0], result=replace(steps[0].result, output="forged"))
        altered = replace(episode, steps=steps)
    receipt = bind_provenance(altered, source="scripted", verifier_sha256="b" * 64)
    with pytest.raises(ValueError, match="reward tampering|tool output mismatch"):
        verify_bound_episode(altered, receipt, episode.manifest, receipt)
