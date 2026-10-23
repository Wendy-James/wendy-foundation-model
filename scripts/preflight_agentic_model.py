"""Offline pinned-tokenizer preflight; never load/generate/download model weights."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from wendyfm.agentic.environment import LookupEnvironment
from wendyfm.agentic.model_policy import MODEL_ID, MODEL_REVISION


def preflight():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    started = time.monotonic()
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                      local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                              trust_remote_code=False)
    observation = LookupEnvironment().reset(17)
    tools = [{"type": "function", "function": {"name": name,
              "description": "Look up a key in the task's public tool world.",
              "parameters": schema}} for name, schema in observation.tools.items()]
    initial = [{"role": "user", "content": observation.instruction}]
    # These are supplied template fixtures, never model-generated actions.
    feedback = [*initial, {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "lookup",
         "arguments": {"key": "start"}}}]},
        {"role": "tool", "content": "status=ok; output=drawer"}]
    rendered = []
    counts = []
    for messages in (initial, feedback):
        rendered.append(tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False))
        inputs = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True)
        if inputs["attention_mask"].shape != inputs["input_ids"].shape:
            raise ValueError("tokenizer attention-mask mismatch")
        counts.append(int(inputs["input_ids"].shape[1]))
    thinking_disabled = rendered[0].endswith("<think>\n\n</think>\n\n")
    if (not thinking_disabled or
            '<tool_call>\n{"name": "lookup", "arguments": {"key": "start"}}\n</tool_call>'
            not in rendered[1] or "<tool_response>\nstatus=ok; output=drawer\n</tool_response>"
            not in rendered[1] or counts[1] <= counts[0]):
        raise ValueError("pinned template did not preserve the supplied tool/feedback fixture")
    artifacts = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "README.md"):
        data = (snapshot / name).read_bytes()
        artifacts[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    weights_present = (snapshot / "model.safetensors").is_file()
    return {"schema": "wendyfm.tokenizer-preflight.v0.1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "source": "tokenizer_fixture", "real_model_inference": False,
            "weights_present": weights_present, "artifacts": artifacts,
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
                "numpy")},
            "fixture_input_tokens": {"initial": counts[0], "with_feedback": counts[1]},
            "fixture_messages": feedback, "template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")).hexdigest(),
            "generation_prompt_thinking_disabled": thinking_disabled,
            "tokenizer_preflight_seconds": time.monotonic() - started,
            "generated_tokens": None, "model_latency_seconds": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("write preflight evidence outside Git")
    try:
        record = preflight()
    except Exception as exc:
        record = {"schema": "wendyfm.tokenizer-preflight.v0.1",
                  "model_revision": MODEL_REVISION, "real_model_inference": False,
                  "error_type": type(exc).__name__, "error": str(exc)}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(record, indent=2))
    if "error_type" in record:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
