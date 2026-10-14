"""R3: was a failed `edit_file` caused by escaping?

Tool results reach the model JSON-encoded twice (forum 744272), so quotes, backslashes and
newlines show up as `\\"`, `\\\\` and `\\n`, and the model may copy them into `old_string`.
For each failed edit, the `old_string` is compared with the text the agent had for that file
(its earlier reads, writes and successful edits):

- `stale`     the `old_string` as given appears in that text: the file changed since (or the
              tool rejected it for another reason), not an escaping problem
- `escaping`  it doesn't, but it does once one or two levels of escaping are undone, or once
              its backslashes are doubled (the model dropped a level of escaping)
- `no_match`  neither, or the agent had no text for the file
- `malformed` the call never reached the tool: the agent framework rejected its arguments
              (e.g. `old_string` missing because the model garbled the key, run 1)

Edits the harness blocked because the tool-call budget was spent are not repair failures and
are left out.

The tools are not patched; this is a check after the fact on the official harness's traces.
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

from .explore import normalize_path, observation
from .runs import BUDGET_EXCEEDED, REJECTED_ARGS, load_runs
from .trace import ToolCall, Trace

ESCAPE = re.compile(r"\\([\\\"'nt/])")
UNESCAPED = {"n": "\n", "t": "\t"}


def _unescape_once(s: str) -> str:
    return ESCAPE.sub(lambda m: UNESCAPED.get(m.group(1), m.group(1)), s)


def escaping_variants(s: str) -> list[str]:
    """`s` with one or two levels of JSON string escaping undone, and with its backslashes doubled.

    Variants equal to `s` are dropped.
    """
    once = _unescape_once(s)
    return [c for c in dict.fromkeys([once, _unescape_once(once), s.replace("\\", "\\\\")]) if c != s]


def known_text(calls: list[ToolCall], file: str, before: int) -> list[str]:
    """Text of `file` the agent had before call `before`: read contents, written contents, new strings of successful edits."""
    texts = []
    for c in calls[:before]:
        obs = observation(c)
        if obs.get("status") != "ok" or normalize_path(c.arguments.get("filepath") or obs.get("filepath")) != file:
            continue
        if c.name == "read_file":
            texts.append(obs.get("content") or "")
        elif c.name == "write_file":
            texts.append(c.arguments.get("content") or "")
        elif c.name == "edit_file":
            texts.append(c.arguments.get("new_string") or "")
    return [t for t in texts if t]


def classify_edit_failure(calls: list[ToolCall], i: int) -> dict:
    c = calls[i]
    obs = observation(c)
    file = normalize_path(c.arguments.get("filepath"))
    old = c.arguments.get("old_string") or ""
    texts = known_text(calls, file, i) if file else []
    if REJECTED_ARGS in str(obs.get("error", "")):
        label = "malformed"
    elif old and any(old in t for t in texts):
        label = "stale"
    elif old and any(cand in t for cand in escaping_variants(old) for t in texts):
        label = "escaping"
    else:
        label = "no_match"
    return {
        "index": c.index, "file": file, "label": label, "known_text": bool(texts),
        "error_type": obs.get("error_type"), "error_message": obs.get("error_message") or obs.get("error"),
    }


def edit_failures(trace: Trace, instance_id: str = "") -> list[dict]:
    """One row per failed `edit_file` in the trace."""
    return [
        {"instance_id": instance_id} | classify_edit_failure(trace.calls, i)
        for i, c in enumerate(trace.calls)
        if c.name == "edit_file" and (obs := observation(c)).get("status") != "ok"
        and obs.get("error_type") != BUDGET_EXCEEDED
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R3: label every failed edit_file in a run.")
    parser.add_argument("--run", required=True, action="append", help="run directory; repeat for each shard")
    parser.add_argument("--csv", help="write one row per failed edit here")
    args = parser.parse_args(argv)

    run = load_runs(args.run)
    rows = [r for tid in sorted(run.latest) if (t := run.trace(tid)) for r in edit_failures(t, tid)]
    df = pd.DataFrame(rows, columns=["instance_id", "index", "file", "label", "known_text", "error_type", "error_message"])
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(df.drop(columns="error_message").to_string(index=False))
    counts = df["label"].value_counts()
    print(f"\n{len(df)} failed edits in {df['instance_id'].nunique()} tasks: "
          + ", ".join(f"{label} {counts.get(label, 0)}" for label in ("escaping", "stale", "no_match", "malformed")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
