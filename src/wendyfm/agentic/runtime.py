"""Bounded CPU rollout. Spawned policy receives only a detached public observation."""

import multiprocessing as mp
import time
from dataclasses import asdict

from .contracts import Action, Episode, Step, ToolResult, decode
from .environment import FIXTURE_SHA, LookupEnvironment


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
    try:
        if not parent.poll(timeout):
            raise TimeoutError("policy deadline")
        return decode(Action, parent.recv())
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        parent.close()
        process.close()


def rollout(manifest, policy):
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
            action = decide(policy, env.observe(), manifest.timeout_seconds)
            if action.tool_call is not None:
                result = env.step(action.tool_call)
            if time.monotonic() - started > manifest.timeout_seconds:
                raise TimeoutError("step deadline")
        except TimeoutError:
            terminal, result = "timeout", ToolResult("timeout", "step deadline exceeded")
        except Exception:
            terminal = "policy_error"
        steps.append(Step(index, action, result, float(time.monotonic() - started), None, None))
        if terminal in ("timeout", "policy_error"):
            break
        if action.final_answer is not None:
            terminal = "final"
            break
    from .verifier import reward_for
    return Episode(manifest, steps, terminal, reward_for(manifest, steps, terminal))


def scripted_policy(observation):
    from .contracts import ToolCall
    feedback = observation.feedback
    if len(feedback) == 3 and all(x.status == "ok" for x in feedback):
        return Action(None, feedback[-1].output)
    key = feedback[-1].output if feedback else "start"
    return Action(ToolCall("lookup", {"key": key}), None)
