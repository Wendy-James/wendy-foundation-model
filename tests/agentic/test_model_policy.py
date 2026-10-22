"""Offline P1 model-policy tests; no model files or Transformers are imported."""

import json
import multiprocessing
import os
import signal
import time
from dataclasses import dataclass, replace

import pytest

from wendyfm.agentic.contracts import PolicyRef, ToolCall
from wendyfm.agentic.demo import baseline_manifest
from wendyfm.agentic.environment import LookupEnvironment
from wendyfm.agentic.model_policy import (
    MAX_INPUT_TOKENS,
    MODEL_REVISION,
    GeneratedTurn,
    ModelSession,
    PolicyAdapterError,
    ToolCallingPolicy,
    _HFBackend,
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


@dataclass(frozen=True)
class ResistantFactory(MockFactory):
    def __call__(self):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        return super().__call__()


@dataclass(frozen=True)
class StartupResistantFactory(MockFactory):
    def __call__(self):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30)


@dataclass(frozen=True)
class ExitFactory(MockFactory):
    def __call__(self):
        os._exit(7)


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


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="POSIX signal test")
def test_sigterm_resistant_model_worker_has_bounded_cleanup():
    before = multiprocessing.active_children()
    start = time.monotonic()
    manifest = model_manifest(timeout=0.2)
    episode, error = rollout_model(manifest, ResistantFactory((call("start"),), 5.0))
    assert time.monotonic() - start < 3.0
    assert error == "timeout"
    assert episode.terminal_state == "timeout"
    assert multiprocessing.active_children() == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal test")
def test_resistant_model_startup_timeout_is_bounded():
    before = multiprocessing.active_children()
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="model decision deadline"):
        ModelSession(StartupResistantFactory(tuple()), 128, 1.0)
    assert time.monotonic() - start < 3.0
    assert multiprocessing.active_children() == before


def test_abrupt_model_startup_exit_is_classified_and_cleaned_up():
    before = multiprocessing.active_children()
    with pytest.raises(PolicyAdapterError, match="inference_failure"):
        ModelSession(ExitFactory(tuple()), 128, 5.0)
    assert multiprocessing.active_children() == before


def test_model_session_close_is_idempotent_and_dead_worker_is_classified():
    session = ModelSession(MockFactory(("violet",)), 128, 5.0)
    try:
        session.process.kill()
        session.process.join(timeout=1.0)
        with pytest.raises(PolicyAdapterError, match="inference_failure"):
            session.decide(LookupEnvironment().reset(17), 1.0)
        assert session.error_type == "inference_failure"
    finally:
        session.close()
    session.close()


def test_model_provenance_mismatch_rejected():
    with pytest.raises(ValueError, match="provenance"):
        rollout_model(model_manifest(), MockFactory(("violet",), model_revision="b" * 40))


@pytest.mark.parametrize("length", [12, MAX_INPUT_TOKENS + 1])
def test_hf_backend_uses_upstream_template_mask_and_bounded_generate(length):
    # Contract doubles, not actual Transformers/tokenizer/model acceptance.
    import torch

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "lookup"}]
            assert kwargs == {"tools": [], "tokenize": True, "add_generation_prompt": True,
                              "return_tensors": "pt", "return_dict": True,
                              "enable_thinking": False}
            return {"input_ids": torch.ones((1, length), dtype=torch.long),
                    "attention_mask": torch.ones((1, length), dtype=torch.long)}

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [9, 2]
            assert kwargs == {"skip_special_tokens": True}
            return "violet"

    class Model:
        called = False

        def generate(self, **kwargs):
            self.called = True
            assert kwargs["attention_mask"].shape == (1, length)
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 128
            assert kwargs["pad_token_id"] == 2
            assert not torch.is_grad_enabled()
            return torch.cat((kwargs["input_ids"], torch.tensor([[9, 2]])), dim=1)

    model = Model()
    backend = _HFBackend(model, Tokenizer(), torch)
    messages = [{"role": "user", "content": "lookup"}]
    if length > MAX_INPUT_TOKENS:
        with pytest.raises(ValueError, match="context token limit"):
            backend.generate(messages, [], 128)
        assert not model.called
    else:
        assert backend.generate(messages, [], 128) == GeneratedTurn("violet", length, 2)
        assert model.called
