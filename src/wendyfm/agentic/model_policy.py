"""Local-only HF tool policy; P0 environment and trusted verifier remain host-owned."""

import json
import multiprocessing as mp
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass

from .contracts import Action, Observation, ToolCall, decode
from .processes import cleanup_on_exit, stop_worker
from .runtime import _rollout

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_INPUT_TOKENS = 4096
TOOL_CALL = re.compile(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", re.DOTALL)


class PolicyAdapterError(Exception):
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True)
class GeneratedTurn:
    text: str
    input_tokens: int
    output_tokens: int


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate tool argument")
        result[key] = value
    return result


def parse_turn(text, registered_tools):
    """Accept one Qwen tool call or one plain final answer; never execute text as code."""
    if type(text) is not str or not text.strip() or len(text) > 8192:
        raise PolicyAdapterError("parse_error")
    match = TOOL_CALL.fullmatch(text)
    if match:
        try:
            payload = json.loads(match.group(1), object_pairs_hook=_unique_pairs)
        except (ValueError, TypeError) as exc:
            raise PolicyAdapterError("parse_error") from exc
        if (type(payload) is not dict or set(payload) != {"name", "arguments"} or
                type(payload["name"]) is not str or
                type(payload["arguments"]) is not dict):
            raise PolicyAdapterError("parse_error")
        # Unknown tool names and ill-typed arguments remain visible as invalid actions.
        # LookupEnvironment is the only tool dispatcher.
        try:
            return Action(ToolCall(payload["name"], payload["arguments"]), None)
        except ValueError as exc:
            raise PolicyAdapterError("parse_error") from exc
    if any(tag in text for tag in ("<tool_call>", "</tool_call>", "<think>", "</think>")):
        raise PolicyAdapterError("parse_error")
    if "<|" in text or "|>" in text:
        raise PolicyAdapterError("parse_error")
    try:
        return Action(None, text.strip())
    except ValueError as exc:
        raise PolicyAdapterError("parse_error") from exc


class ToolCallingPolicy:
    """Maintain only policy-visible chat history across feedback-driven decisions."""

    def __init__(self, backend, max_new_tokens=128):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError("invalid generation bound")
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.messages = []
        self.feedback_count = 0
        self.pending_call = False
        self.observation = None

    def decide(self, observation):
        observation = decode(Observation, asdict(observation))
        messages = deepcopy(self.messages)
        if not self.messages:
            if observation.feedback:
                raise PolicyAdapterError("stale_observation")
            messages = [{"role": "user", "content": observation.instruction}]
        elif self.pending_call and len(observation.feedback) == self.feedback_count + 1:
            if (observation.instruction != self.observation.instruction or
                    observation.tools != self.observation.tools or
                    observation.feedback[:-1] != self.observation.feedback):
                raise PolicyAdapterError("stale_observation")
            feedback = observation.feedback[-1]
            messages.append({"role": "tool", "content": (
                f"status={feedback.status}; output={feedback.output}")})
        else:
            raise PolicyAdapterError("stale_observation")
        tools = [{"type": "function", "function": {"name": name,
                  "description": "Look up a key in the task's public tool world.",
                  "parameters": schema}} for name, schema in observation.tools.items()]
        try:
            turn = self.backend.generate(deepcopy(messages), tools, self.max_new_tokens)
        except Exception as exc:
            raise PolicyAdapterError("inference_failure") from exc
        if (not isinstance(turn, GeneratedTurn) or
                type(turn.input_tokens) is not int or
                not 0 < turn.input_tokens <= MAX_INPUT_TOKENS or
                type(turn.output_tokens) is not int or
                not 0 < turn.output_tokens <= self.max_new_tokens):
            raise PolicyAdapterError("inference_failure")
        action = parse_turn(turn.text, observation.tools)
        if action.tool_call is not None:
            call = action.tool_call
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": call.name,
                 "arguments": call.arguments}}
            ]})
        else:
            messages.append({"role": "assistant", "content": action.final_answer})
        # Commit history only after a valid decision; failed turns can be retried
        # without consuming feedback or duplicating tool responses.
        self.messages = messages
        self.feedback_count = len(observation.feedback)
        self.pending_call = action.tool_call is not None
        self.observation = observation
        return action, turn.input_tokens, turn.output_tokens


@dataclass(frozen=True)
class LocalHFBackendFactory:
    """Resolve exactly the pinned local snapshot; never fetch model files."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = MODEL_REVISION

    def __call__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION or (
            self.tokenizer_revision != MODEL_REVISION
        ):
            raise ValueError("unreviewed model or tokenizer revision")
        snapshot = snapshot_download(
            self.model_id, revision=self.model_revision, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32
        ).eval()
        model.to("cpu")
        return _HFBackend(model, tokenizer, torch)


class _HFBackend:
    def __init__(self, model, tokenizer, torch):
        self.model, self.tokenizer, self.torch = model, tokenizer, torch

    def generate(self, messages, tools, max_new_tokens):
        inputs = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=False
        )
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens > MAX_INPUT_TOKENS:
            raise ValueError("model input exceeds context token limit")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
        new_ids = output[0, input_tokens:]
        return GeneratedTurn(self.tokenizer.decode(new_ids, skip_special_tokens=True),
                             input_tokens, int(new_ids.numel()))


def _worker(connection, factory, max_new_tokens):
    try:
        policy = ToolCallingPolicy(factory(), max_new_tokens)
        connection.send({"ready": True})
    except Exception:
        connection.send({"error": "inference_failure"})
        connection.close()
        return
    try:
        while True:
            request = connection.recv()
            if request is None:
                break
            try:
                decision = policy.decide(decode(Observation, request))
                action, input_tokens, output_tokens = decision
                connection.send({"action": asdict(action), "input_tokens": input_tokens,
                                 "output_tokens": output_tokens})
            except PolicyAdapterError as exc:
                connection.send({"error": exc.kind})
            except Exception:
                connection.send({"error": "inference_failure"})
    except EOFError:
        pass
    finally:
        connection.close()


class ModelSession:
    """One persistent model process; kill it on deadline or exit."""

    def __init__(self, factory, max_new_tokens, startup_timeout):
        context = mp.get_context("spawn")
        self.parent, child = context.Pipe(duplex=True)
        self.process = context.Process(target=_worker, args=(child, factory, max_new_tokens))
        self.process.start()
        child.close()
        self.error_type = None
        self.closed = False
        try:
            response = self._receive(startup_timeout)
            if response != {"ready": True}:
                self.error_type = response.get("error", "inference_failure")
                raise PolicyAdapterError(self.error_type)
        except BaseException:
            with cleanup_on_exit(self.close):
                raise

    def _receive(self, timeout):
        if not self.parent.poll(timeout):
            self.error_type = "timeout"
            raise TimeoutError("model decision deadline")
        try:
            return self.parent.recv()
        except (EOFError, OSError) as exc:
            self.error_type = "inference_failure"
            raise PolicyAdapterError(self.error_type) from exc

    def decide(self, observation, timeout):
        try:
            self.parent.send(asdict(observation))
        except (OSError, EOFError) as exc:
            self.error_type = "inference_failure"
            raise PolicyAdapterError(self.error_type) from exc
        response = self._receive(timeout)
        if "error" in response:
            self.error_type = response["error"]
            raise PolicyAdapterError(self.error_type)
        decision = (decode(Action, response["action"]), response["input_tokens"],
                    response["output_tokens"])
        self.error_type = None
        return decision

    def close(self):
        if not self.closed:
            stop_worker(self.process, self.parent)
            self.closed = True


def rollout_model(manifest, factory, *, max_new_tokens=128):
    """Reuse P0 rollout and verifier; return error classification separately."""
    if manifest.policy.kind != "model" or (
        manifest.policy.model_revision != factory.model_revision or
        manifest.policy.tokenizer_revision != factory.tokenizer_revision
    ):
        raise ValueError("model provenance mismatch")
    if manifest.max_steps > 4:
        raise ValueError("model rollout is limited to four decisions")
    deadline = time.monotonic() + min(300.0, (manifest.max_steps + 1) *
                                      manifest.timeout_seconds)
    session = ModelSession(factory, max_new_tokens, manifest.timeout_seconds)
    with cleanup_on_exit(session.close):
        def bounded_decide(observation, timeout):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                session.error_type = "timeout"
                raise TimeoutError("episode deadline")
            return session.decide(observation, min(timeout, remaining))

        episode = _rollout(manifest, bounded_decide)
        return episode, session.error_type
