"""Public synthetic tool world; never contains expected answers or reward code."""

import random
from copy import deepcopy

from .contracts import Observation, ToolResult, digest

FIXTURE = {"start": "drawer", "drawer": "cabinet", "cabinet": "violet"}
FIXTURE_SHA = digest(FIXTURE)
TOOLS = {"lookup": {"type": "object", "properties": {"key": {"type": "string"}},
                    "required": ["key"], "additionalProperties": False}}
INSTRUCTION = "Follow three lookup links beginning at start. Return the final value."


class LookupEnvironment:
    def reset(self, seed):
        self.feedback = []
        self.world = deepcopy(FIXTURE)
        self.rng = random.Random(seed)  # Local RNG; never resets process-global state.
        return self.observe()

    def observe(self):
        return Observation(INSTRUCTION, deepcopy(self.feedback), deepcopy(TOOLS))

    def step(self, call):
        args = call.arguments
        if (call.name != "lookup" or set(args) != {"key"} or
                type(args["key"]) is not str or len(args["key"]) > 128):
            result = ToolResult("invalid_action", "lookup requires exactly one string key")
        elif args["key"] not in self.world:
            result = ToolResult("invalid_action", "unknown key")
        else:
            result = ToolResult("ok", self.world[args["key"]])
        self.feedback.append(result)
        return result
