"""Strict JSON contracts. Unknown fields and coercions are rejected recursively."""

import hashlib
import json
import math
import types
from dataclasses import asdict, dataclass, fields
from typing import Literal, get_args, get_origin, get_type_hints

VERSION = "wendyfm.episode.v0.1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def decode(cls, value):
    if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    hints = get_type_hints(cls)
    return cls(**{k: _typed(hints[k], v) for k, v in value.items()})


def _typed(hint, value):
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        if not any(type(value) is type(x) and value == x for x in args):
            raise ValueError("invalid literal")
    elif origin is types.UnionType:
        for arg in args:
            try:
                return _typed(arg, value)
            except (ValueError, TypeError):
                pass
        raise ValueError("invalid union")
    elif origin is list:
        if type(value) is not list:
            raise ValueError("expected list")
        return [_typed(args[0], x) for x in value]
    elif hasattr(hint, "__dataclass_fields__"):
        return decode(hint, value)
    elif type(value) is not hint:
        raise ValueError(f"expected {hint}")
    return value


@dataclass(frozen=True)
class PolicyRef:
    kind: Literal["scripted", "model"]
    model_revision: str | None
    tokenizer_revision: str | None
    adapter_revision: str | None
    policy_revision: str


@dataclass(frozen=True)
class RunManifest:
    schema: Literal["wendyfm.episode.v0.1"]
    provenance_sha: str
    environment_version: Literal["lookup-chain.v1"]
    task_id: Literal["synthetic-chain-001"]
    seed: int
    max_steps: int
    timeout_seconds: float
    policy: PolicyRef
    fixture_sha256: str

    def __post_init__(self):
        if len(self.provenance_sha) != 40 or any(
            c not in "0123456789abcdef" for c in self.provenance_sha
        ):
            raise ValueError("invalid provenance SHA")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("invalid seed")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 100:
            raise ValueError("invalid step bound")
        if (type(self.timeout_seconds) is not float or
                not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 60):
            raise ValueError("invalid timeout")
        if len(self.fixture_sha256) != 64:
            raise ValueError("invalid fixture digest")
        if self.policy.kind == "model" and (
            not self.policy.model_revision or not self.policy.tokenizer_revision
        ):
            raise ValueError("model and tokenizer revisions required")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResult:
    status: Literal["ok", "invalid_action", "timeout"]
    output: str


@dataclass(frozen=True)
class Observation:
    instruction: str
    feedback: list[ToolResult]
    tools: dict


@dataclass(frozen=True)
class Action:
    tool_call: ToolCall | None
    final_answer: str | None

    def __post_init__(self):
        if (self.tool_call is None) == (self.final_answer is None):
            raise ValueError("action must contain exactly one decision")
        if self.final_answer is not None and len(self.final_answer) > 4096:
            raise ValueError("answer too large")
        if self.tool_call is not None and len(canonical(asdict(self.tool_call))) > 4096:
            raise ValueError("tool call too large")


@dataclass(frozen=True)
class Step:
    index: int
    action: Action | None
    result: ToolResult | None
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None

    def __post_init__(self):
        if self.index < 0 or not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("invalid step metric")
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and value < 0:
                raise ValueError("invalid tokens")


@dataclass(frozen=True)
class RewardResult:
    value: Literal[0.0, 1.0]
    verifier_identity: Literal["lookup-chain-verifier.v1"]
    verified: bool


@dataclass(frozen=True)
class Episode:
    manifest: RunManifest
    steps: list[Step]
    terminal_state: Literal["final", "max_steps", "timeout", "policy_error"]
    reward: RewardResult

    def to_json(self):
        return canonical(asdict(self))

    @classmethod
    def from_json(cls, text):
        if len(text) > 1_000_000:
            raise ValueError("episode too large")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        return decode(cls, json.loads(text, object_pairs_hook=pairs))
