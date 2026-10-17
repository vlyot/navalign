"""Read ATIF (v1.7) agent traces into an ordered list of tool calls with their observations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REDACTED = "[removed]"


@dataclass
class ToolCall:
    index: int  # 0-based order across the whole trace
    step_id: int
    name: str
    arguments: dict
    output: str | None  # observation content returned to the model
    author: str | None = None  # agent that made the call (e.g. swe_baseline_agent)
    elapsed_s: float | None = None


@dataclass
class Trace:
    schema_version: str
    agent: dict
    calls: list[ToolCall]
    final_metrics: dict = field(default_factory=dict)
    n_steps: int = 0


def _observation_text(observation, i: int, call_id: str | None) -> str | None:
    """Observation content for the i-th call of a step.

    The notebook harness writes one `{content, extra}` per step; the ATIF `results` list form
    is matched by `source_call_id`, then by position.
    """
    if not observation:
        return None
    results = observation.get("results")
    if results:
        for r in results:
            if call_id and r.get("source_call_id") == call_id:
                return r.get("content")
        return results[i].get("content") if i < len(results) else None
    return observation.get("content") if i == 0 else None


def parse_trace(data: dict) -> Trace:
    calls: list[ToolCall] = []
    steps = data.get("steps") or []
    for step in steps:
        for i, tc in enumerate(step.get("tool_calls") or []):
            extra = tc.get("extra") or {}
            args = tc.get("arguments") or {}
            if isinstance(args, str):  # some writers store arguments as a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(
                index=len(calls),
                step_id=step.get("step_id"),
                name=tc.get("function_name"),
                arguments=args,
                output=_observation_text(step.get("observation"), i, tc.get("tool_call_id")),
                author=extra.get("author"),
                elapsed_s=extra.get("elapsed_s"),
            ))
    return Trace(
        schema_version=data.get("schema_version", ""),
        agent=data.get("agent") or {},
        calls=calls,
        final_metrics=data.get("final_metrics") or {},
        n_steps=len(steps),
    )


def load_trace(path: str | Path) -> Trace:
    with open(path, encoding="utf-8") as f:
        return parse_trace(json.load(f))


def redact_messages(data: dict) -> dict:
    """A copy of a trace whose step messages (the harness's instructions, the task text, and the model's own
    words, which quote both) read "[removed]", for publishing it. Tool calls, their arguments and observations
    and the step ids are kept, so every score and escaping label is unchanged."""
    return data | {"steps": [step | {"message": REDACTED} if "message" in step else step
                             for step in data.get("steps") or []]}
