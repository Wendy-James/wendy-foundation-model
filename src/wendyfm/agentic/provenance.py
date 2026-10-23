"""Optional host-owned evidence binding; never authentication by self-declared hash."""

from dataclasses import asdict, dataclass
from typing import Literal

from .contracts import Episode, PolicyRef, decode, digest
from .verifier import verify_episode


@dataclass(frozen=True)
class ReplayProvenance:
    schema: Literal["wendyfm.replay-provenance.v0.1"]
    source: Literal["scripted", "mock", "real"]
    execution_sha: str
    policy: PolicyRef
    task_revision: str
    verifier_identity: Literal["lookup-chain-verifier.v1"]
    verifier_sha256: str
    model_sha256: str | None
    tokenizer_sha256: str | None
    episode_sha256: str
    tool_results_sha256: list[str]

    def __post_init__(self):
        for value, size in ((self.execution_sha, 40), (self.task_revision, 64),
                            (self.verifier_sha256, 64), (self.episode_sha256, 64)):
            _hash(value, size)
        for value in self.tool_results_sha256:
            _hash(value, 64)
        for value in (self.model_sha256, self.tokenizer_sha256):
            if value is not None:
                _hash(value, 64)
        if self.source == "real" and (self.model_sha256 is None or
                                      self.tokenizer_sha256 is None):
            raise ValueError("real source requires artifact hashes")
        if self.source != "real" and (self.model_sha256 is not None or
                                      self.tokenizer_sha256 is not None):
            raise ValueError("non-real source cannot claim model artifacts")
        if (self.source == "scripted") != (self.policy.kind == "scripted"):
            raise ValueError("source/policy mismatch")


def _hash(value, size):
    if type(value) is not str or len(value) != size or any(
        c not in "0123456789abcdef" for c in value
    ):
        raise ValueError("invalid provenance digest")


def bind_provenance(episode, *, source, verifier_sha256,
                    model_sha256=None, tokenizer_sha256=None):
    """Host recorder only. Real requires measured artifacts, never inferred identity.

    This helper cannot establish that inference happened. Its return must be stored
    independently by the trusted recorder, not promoted from policy/trace input.
    """
    episode = Episode.from_json(episode.to_json())
    return decode(ReplayProvenance, asdict(ReplayProvenance(
        "wendyfm.replay-provenance.v0.1", source, episode.manifest.provenance_sha,
        episode.manifest.policy, episode.manifest.fixture_sha256,
        episode.reward.verifier_identity, verifier_sha256, model_sha256,
        tokenizer_sha256, digest(asdict(episode)),
        [digest(asdict(step.result)) for step in episode.steps if step.result is not None],
    )))


def verify_bound_episode(episode, provenance, trusted_manifest, trusted_provenance):
    """Compare an independently retained host receipt, then use the fixed verifier.

    No external verifier, expected-answer or reward callback is accepted. Hashes bind
    recorded metrics but do not authenticate their measurement or execute a policy.
    """
    provenance = decode(ReplayProvenance, asdict(provenance))
    trusted = decode(ReplayProvenance, asdict(trusted_provenance))
    if provenance != trusted:
        raise ValueError("trusted provenance mismatch")
    actual = bind_provenance(
        episode, source=trusted.source, verifier_sha256=trusted.verifier_sha256,
        model_sha256=trusted.model_sha256, tokenizer_sha256=trusted.tokenizer_sha256,
    )
    if actual != trusted:
        raise ValueError("episode binding mismatch")
    return verify_episode(episode, trusted_manifest)
