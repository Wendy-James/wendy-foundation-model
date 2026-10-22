"""CPU contract, rollout, isolation, contamination and tamper acceptance tests."""

import json
import time
from dataclasses import asdict, replace

import pytest

from wendyfm.agentic.contracts import Action, Episode, ToolCall, decode
from wendyfm.agentic.demo import baseline_manifest
from wendyfm.agentic.environment import LookupEnvironment
from wendyfm.agentic.runtime import rollout, scripted_policy
from wendyfm.agentic.verifier import verify_episode

SHA = "a" * 40


def invalid_policy(observation):
    return Action(ToolCall("lookup", {"key": 2}), None)


def slow_policy(observation):
    time.sleep(5)
    return Action(None, "unused")


def broken_policy(observation):
    raise RuntimeError("private exception text")


@pytest.fixture(scope="module")
def episode():
    return rollout(baseline_manifest(SHA), scripted_policy)


def test_state_reset_and_isolation():
    a, b = LookupEnvironment(), LookupEnvironment()
    initial = a.reset(17)
    b.reset(17)
    a.step(ToolCall("lookup", {"key": "start"}))
    a.world["start"] = "modified"
    assert b.observe() == initial
    assert a.reset(17) == initial
    view = a.observe()
    view.tools.clear()
    view.feedback.append("modified")
    assert a.observe() == initial


def test_feedback_multiturn_and_serialization(episode):
    assert episode.terminal_state == "final"
    assert [s.result.output for s in episode.steps[:-1]] == ["drawer", "cabinet", "violet"]
    assert [s.action.tool_call.arguments["key"] for s in episode.steps[:-1]] == [
        "start", "drawer", "cabinet"]
    assert episode.steps[-1].action.final_answer == "violet"
    assert Episode.from_json(episode.to_json()) == episode
    assert verify_episode(episode, baseline_manifest(SHA)).value == 1.0


@pytest.mark.parametrize("call", [ToolCall("shell", {}), ToolCall("lookup", {}),
                                 ToolCall("lookup", {"key": False}),
                                 ToolCall("lookup", {"key": "start", "extra": 1}),
                                 ToolCall("lookup", {"key": "missing"})])
def test_invalid_arguments(call):
    env = LookupEnvironment()
    env.reset(17)
    assert env.step(call).status == "invalid_action"


def test_max_step_termination():
    manifest = replace(baseline_manifest(SHA), max_steps=2)
    episode = rollout(manifest, invalid_policy)
    assert episode.terminal_state == "max_steps"
    assert len(episode.steps) == 2
    assert all(s.result.status == "invalid_action" for s in episode.steps)
    assert verify_episode(episode, manifest).value == 0.0


def test_timeout_and_worker_cleanup():
    import multiprocessing
    before = multiprocessing.active_children()
    manifest = replace(baseline_manifest(SHA), timeout_seconds=0.1)
    start = time.monotonic()
    episode = rollout(manifest, slow_policy)
    assert time.monotonic() - start < 3
    assert episode.terminal_state == "timeout"
    assert verify_episode(episode, manifest).value == 0.0
    assert multiprocessing.active_children() == before


def test_policy_error_is_sanitized():
    episode = rollout(baseline_manifest(SHA), broken_policy)
    assert episode.terminal_state == "policy_error"
    assert "private exception" not in episode.to_json()
    assert verify_episode(episode, episode.manifest).value == 0.0


def test_deterministic_replay(episode):
    again = rollout(baseline_manifest(SHA), scripted_policy)
    left, right = asdict(episode), asdict(again)
    for record in (left, right):
        for step in record["steps"]:
            step.pop("elapsed_seconds")
    assert left == right
    assert verify_episode(again, baseline_manifest(SHA)) == episode.reward


def test_altered_tool_output_detection(episode):
    steps = list(episode.steps)
    steps[0] = replace(steps[0], result=replace(steps[0].result, output="altered"))
    with pytest.raises(ValueError, match="tool output"):
        verify_episode(replace(episode, steps=steps), baseline_manifest(SHA))


def test_reward_tampering(episode):
    with pytest.raises(ValueError, match="reward tampering"):
        verify_episode(replace(episode, reward=replace(episode.reward, value=0.0)),
                       baseline_manifest(SHA))
    with pytest.raises(ValueError, match="reward tampering"):
        verify_episode(replace(episode, reward=replace(episode.reward, verified=False)),
                       baseline_manifest(SHA))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(extra="unknown"),
    lambda d: d["manifest"].update(schema="wendyfm.episode.v9"),
    lambda d: d["manifest"].update(seed=True),
    lambda d: d["manifest"].update(max_steps=0),
    lambda d: d["steps"][0].update(elapsed_seconds=float("nan")),
    lambda d: d["steps"][0]["action"].update(final_answer="also final"),
    lambda d: d["reward"].update(value=True),
])
def test_schema_validation(episode, mutation):
    data = asdict(episode)
    mutation(data)
    with pytest.raises(ValueError):
        Episode.from_json(json.dumps(data))


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Episode.from_json('{"manifest":1,"manifest":2}')


def test_provenance(episode):
    for manifest in (replace(episode.manifest, provenance_sha="b" * 40),
                     replace(episode.manifest, seed=18),
                     replace(episode.manifest, fixture_sha256="b" * 64),
                     replace(episode.manifest, policy=replace(
                         episode.manifest.policy, policy_revision="b" * 40))):
        with pytest.raises(ValueError, match="provenance"):
            verify_episode(episode, manifest)


def test_no_hidden_label_leakage():
    env = LookupEnvironment()
    view = asdict(env.reset(17))
    assert set(view) == {"instruction", "feedback", "tools"}
    assert "violet" not in json.dumps(view)
    assert "verifier" not in json.dumps(view)
    assert "seed" not in json.dumps(view)
    assert "world" not in json.dumps(view)
    assert decode(type(env.observe()), view) == env.observe()


def test_terminal_and_order_tampering(episode):
    for altered in (replace(episode, terminal_state="max_steps"),
                    replace(episode, steps=episode.steps[:-1]),
                    replace(episode, steps=[replace(episode.steps[0], index=1)])):
        with pytest.raises(ValueError):
            verify_episode(altered, episode.manifest)
