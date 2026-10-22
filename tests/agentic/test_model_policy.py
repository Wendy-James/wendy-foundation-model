"""Offline P1 model-policy tests; no model files or Transformers are imported."""

import json
import time
from dataclasses import dataclass, replace

import pytest

from wendyfm.agentic.contracts import PolicyRef, ToolCall
from wendyfm.agentic.demo import baseline_manifest
from wendyfm.agentic.environment import LookupEnvironment
from wendyfm.agentic.model_policy import (
    MODEL_REVISION,
    GeneratedTurn,
    PolicyAdapterError,
    ToolCallingPolicy,
    parse_turn,
    rollout_model,
)
from wendyfm.agentic.verifier import verify_episode


def model_manifest(timeout=5.0):
    base = baseline_manifest("a" * 40)
    return replace(base, timeout_seconds=timeout,
                   policy=PolicyRef("model", MODEL_REVISION, MODEL_REVISION,
                                    "a" * 40, MODEL_REVISION))


@dataclass(frozen=True)
class MockFactory:
    turns: tuple[str, ...]
    sleep_seconds: float = 0.0
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = MODEL_REVISION

    def __call__(self):
        return MockBackend(self.turns, self.sleep_seconds)


class MockBackend:
    def __init__(self, turns, sleep_seconds=0.0):
        self.turns = iter(turns)
        self.sleep_seconds = sleep_seconds
        self.calls = []

    def generate(self, messages, tools, max_new_tokens):
        self.calls.append((json.loads(json.dumps(messages)), tools, max_new_tokens))
        time.sleep(self.sleep_seconds)
        return GeneratedTurn(next(self.turns), 24 + len(messages), 8)


def call(key):
    return '<tool_call>{"name":"lookup","arguments":{"key":"' + key + '"}}</tool_call>'


def test_feedback_driven_four_step_rollout_and_trusted_replay():
    manifest = model_manifest()
    factory = MockFactory((call("start"), call("drawer"), call("cabinet"), "violet"))
    episode, error = rollout_model(manifest, factory)
    assert error is None
    assert episode.terminal_state == "final"
    assert len(episode.steps) == 4
    assert [s.action.tool_call.arguments["key"] for s in episode.steps[:-1]] == [
        "start", "drawer", "cabinet"]
    assert [s.result.output for s in episode.steps[:-1]] == ["drawer", "cabinet", "violet"]
    assert all(s.input_tokens > 0 and s.output_tokens == 8 for s in episode.steps)
    assert verify_episode(episode, manifest).value == 1.0


def test_policy_messages_include_tool_feedback_but_no_verifier():
    backend = MockBackend((call("start"), call("drawer")))
    policy = ToolCallingPolicy(backend)
    env = LookupEnvironment()
    first = env.reset(17)
    action, _, _ = policy.decide(first)
    assert action.tool_call == ToolCall("lookup", {"key": "start"})
    assert "violet" not in json.dumps(backend.calls[0])
    env.step(action.tool_call)
    policy.decide(env.observe())
    messages, tools, bound = backend.calls[1]
    assert messages[-1] == {"role": "tool", "content": "status=ok; output=drawer"}
    assert messages[-2]["tool_calls"][0]["function"]["arguments"] == {"key": "start"}
    assert [tool["function"]["name"] for tool in tools] == ["lookup"]
    assert bound == 128
    assert "verifier" not in json.dumps(messages)
    with pytest.raises(PolicyAdapterError, match="stale_observation"):
        policy.decide(env.observe())


@pytest.mark.parametrize("text", [
    "<tool_call>{bad}</tool_call>",
    '<tool_call>{"name":"lookup","arguments":{},"name":"shell"}</tool_call>',
    '<tool_call>{"name":"lookup","arguments":{}}</tool_call> trailing',
    "<think>secret</think>violet",
    "",
])
def test_parse_errors_are_not_executed(text):
    with pytest.raises(PolicyAdapterError, match="parse_error"):
        parse_turn(text, {"lookup": {}})


def test_unknown_tool_and_bad_arguments_are_invalid_actions():
    env = LookupEnvironment()
    env.reset(17)
    for text in ('<tool_call>{"name":"shell","arguments":{}}</tool_call>',
                 '<tool_call>{"name":"lookup","arguments":{"key":4}}</tool_call>'):
        action = parse_turn(text, {"lookup": {}})
        assert env.step(action.tool_call).status == "invalid_action"


def test_invalid_model_tool_call_is_recorded_and_recoverable():
    unknown = '<tool_call>{"name":"shell","arguments":{}}</tool_call>'
    episode, error = rollout_model(model_manifest(), MockFactory((
        unknown, call("start"), call("drawer"), "cabinet"
    )))
    assert error is None
    assert episode.steps[0].result.status == "invalid_action"
    assert episode.steps[1].action.tool_call.arguments == {"key": "start"}
    assert episode.terminal_state == "final"
    assert verify_episode(episode, episode.manifest).value == 0.0


def test_inference_failure_is_classified_without_leaking_exception():
    episode, error = rollout_model(model_manifest(), MockFactory(tuple()))
    assert error == "inference_failure"
    assert episode.terminal_state == "policy_error"
    assert "StopIteration" not in episode.to_json()
    assert verify_episode(episode, episode.manifest).value == 0.0


def test_parse_failure_is_classified():
    episode, error = rollout_model(model_manifest(), MockFactory(("<tool_call>{bad}</tool_call>",)))
    assert error == "parse_error"
    assert episode.terminal_state == "policy_error"
    assert verify_episode(episode, episode.manifest).value == 0.0


def test_timeout_kills_model_worker():
    manifest = model_manifest(timeout=0.2)
    episode, error = rollout_model(manifest, MockFactory((call("start"),), 1.0))
    assert error == "timeout"
    assert episode.terminal_state == "timeout"
    assert verify_episode(episode, manifest).value == 0.0


def test_model_provenance_mismatch_rejected():
    with pytest.raises(ValueError, match="provenance"):
        rollout_model(model_manifest(), MockFactory(("violet",), model_revision="b" * 40))
