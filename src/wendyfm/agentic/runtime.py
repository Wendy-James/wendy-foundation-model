"""Bounded CPU rollout. Spawned policy receives only a detached public observation."""

import multiprocessing as mp
import time
from dataclasses import asdict

from .contracts import Action, Episode, Step, ToolResult, decode
from .environment import FIXTURE_SHA, LookupEnvironment
from .processes import cleanup_on_exit, stop_worker


def _decide_worker(policy, observation, connection):
    try:
        connection.send(asdict(policy(observation)))
    except Exception:
        connection.send(None)
    finally:
        connection.close()


def decide(policy, observation, timeout):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_decide_worker, args=(policy, observation, child))
    process.start()
    child.close()
    with cleanup_on_exit(lambda: stop_worker(process, parent)):
        if not parent.poll(timeout):
            raise TimeoutError("policy deadline")
        return decode(Action, parent.recv())


def _rollout(manifest, decide_one):
    # Revalidate even programmatically constructed contracts before execution.
    manifest = decode(type(manifest), asdict(manifest))
    if manifest.fixture_sha256 != FIXTURE_SHA:
        raise ValueError("fixture provenance mismatch")
    env = LookupEnvironment()
    env.reset(manifest.seed)
    steps = []
    terminal = "max_steps"
    for index in range(manifest.max_steps):
        started = time.monotonic()
        action, result = None, None
        try:
            action, input_tokens, output_tokens = decide_one(
                env.observe(), manifest.timeout_seconds
            )
            if action.tool_call is not None:
                result = env.step(action.tool_call)
            if time.monotonic() - started > manifest.timeout_seconds:
                raise TimeoutError("step deadline")
        except TimeoutError:
            terminal, result = "timeout", ToolResult("timeout", "step deadline exceeded")
        except Exception:
            terminal = "policy_error"
        steps.append(Step(index, action, result, float(time.monotonic() - started),
                          input_tokens if action is not None else None,
                          output_tokens if action is not None else None))
        if terminal in ("timeout", "policy_error"):
            break
        if action.final_answer is not None:
            terminal = "final"
            break
    from .verifier import reward_for
    return Episode(manifest, steps, terminal, reward_for(manifest, steps, terminal))


def rollout(manifest, policy):
    """Preserve the P0 spawned-policy boundary and scripted token semantics."""
    return _rollout(manifest, lambda observation, timeout: (
        decide(policy, observation, timeout), None, None
    ))


def scripted_policy(observation):
    from .contracts import ToolCall
    feedback = observation.feedback
    if len(feedback) == 3 and all(x.status == "ok" for x in feedback):
        return Action(None, feedback[-1].output)
    key = feedback[-1].output if feedback else "start"
    return Action(ToolCall("lookup", {"key": key}), None)
