"""Explored set per tool call (Methodology step 3): what the agent read, saw and edited.

- *read*: `read_file`, and shell commands that print a file (`cat`, `head`, `tail`, `sed -n`,
  `nl`). The lines read come from the `read_file` observation, never its arguments: the tool
  shows at most 150 lines, and garbled argument keys are silently ignored by it.
- *seen*: files in grep/rg/find/ls output, and nodes returned by the graph tools.
- *edit*: `edit_file`, `write_file`, `sed -i` and shell redirects, whether or not they succeeded.

A call the harness refused because the call budget was spent never ran, so it reads, sees and edits
nothing.

SWE-agent traces (converted by `navalign.swe_agent`) use two tools: `bash`, parsed like
`run_command`, and `str_replace_editor`, whose `view` is a read of the line numbers its `cat -n`
output shows and whose `create`, `str_replace`, `insert` and `undo_edit` are edits.

The exploration window is every call before the first edit of a file that exists in the
snapshot or is a target file. Scratch scripts the agent writes first don't end it.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .data import load_jsonl
from .runs import BUDGET_EXCEEDED, load_runs
from .trace import ToolCall, Trace

SANDBOX_PREFIX = re.compile(r"^(?:.*?/)?(?:workspace|testbed)/")  # the harness's sandbox; SWE-bench's /testbed
GRAPH_TOOLS = frozenset({"search_similar_code", "get_code_neighbors", "get_code_subgraph"})
NODE_KEYS = ("node_name", "node_id", "node")
HEREDOC = re.compile(r"(?<!<)<<-?(?!<)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
CONTROL_TOKENS = frozenset({"|", "||", "&&", ";", "&"})
REDIRECTS = frozenset({">", ">>", ">|"})
GREP_COMMANDS = frozenset({"grep", "egrep", "fgrep", "rg"})
GREP_VALUE_OPTIONS = frozenset({"-e", "-f", "-m", "-A", "-B", "-C", "--include", "--exclude", "-g", "-t"})
SED_PRINT = re.compile(r"(\d+)(?:,(\d+))?p")
EDITOR_EDITS = frozenset({"create", "str_replace", "insert", "undo_edit"})
EDITOR_OK = ("File created successfully", "The file ", "Last edit to ")
NUMBERED_LINE = re.compile(r"^[ ]*(\d+)(?:[\t ]|\r?$)", re.MULTILINE)  # a blank line's number has no tab
DIRECTORY_VIEW = "Here's the files and directories"


@dataclass(frozen=True)
class ReadSpan:
    file: str
    start: int | None = None  # None with end None: the whole file
    end: int | None = None
    truncated: bool = False  # the tool cut the read short
    last: int | None = None  # `tail -n N`: the last N lines, position unknown without the file


@dataclass
class CallEvent:
    index: int
    tool: str
    ok: bool
    reads: list[ReadSpan] = field(default_factory=list)
    seen_files: set[str] = field(default_factory=set)
    seen_nodes: set[str] = field(default_factory=set)
    edited_files: set[str] = field(default_factory=set)
    author: str | None = None  # e.g. code_analyzer_agent for the sub-agent's calls

    def to_dict(self) -> dict:
        return {
            "index": self.index, "tool": self.tool, "ok": self.ok, "author": self.author,
            "reads": [asdict(r) for r in self.reads],
            "seen_files": sorted(self.seen_files), "seen_nodes": sorted(self.seen_nodes),
            "edited_files": sorted(self.edited_files),
        }


@dataclass
class Exploration:
    instance_id: str
    events: list[CallEvent]
    window_end: int  # events[:window_end] is the exploration window
    created_files: list[str]  # written paths absent from the snapshot, in order of creation

    @property
    def window(self) -> list[CallEvent]:
        return self.events[: self.window_end]

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id, "window_end": self.window_end,
            "n_calls": len(self.events), "created_files": self.created_files,
            "events": [e.to_dict() for e in self.events],
        }


def normalize_path(path: str | None) -> str | None:
    """Repo-relative POSIX path, or None for paths outside the repo (or the repo root itself)."""
    if not path:
        return None
    p = path.strip().strip("\"'").replace("\\", "/")
    if p.startswith(("/", "workspace/", "testbed/")):
        p = SANDBOX_PREFIX.sub("", p, count=1)
        if p.startswith("/"):
            return None
    p = posixpath.normpath(p)
    return None if p in (".", "..", "workspace", "testbed") or p.startswith("../") else p


def observation(call: ToolCall) -> dict:
    try:
        obs = json.loads(call.output) if call.output else {}
    except json.JSONDecodeError:
        return {}
    return obs if isinstance(obs, dict) else {}


def _collect_nodes(value, out: set[str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if k in NODE_KEYS and isinstance(v, str):
                out.add(v)
            elif k == "nodes" and isinstance(v, list) and all(isinstance(n, str) for n in v):
                out.update(v)  # get_code_subgraph lists node ids
            else:
                _collect_nodes(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_nodes(v, out)


def strip_heredocs(command: str) -> str:
    """The command without its heredoc bodies, operators and delimiters (`cat <<EOF > x.py` → `cat  > x.py`)."""
    lines = command.split("\n")
    kept, i = [], 0
    while i < len(lines):
        delimiters = [m.group(2) for m in HEREDOC.finditer(lines[i])]
        kept.append(HEREDOC.sub("", lines[i]))
        i += 1
        for delimiter in delimiters:
            while i < len(lines) and lines[i].strip() != delimiter:
                i += 1
            i += 1
    return "\n".join(kept)


def _segments(command: str) -> list[list[str]] | None:
    """Simple commands of a shell line (split on pipes, `&&`, `;`, newlines); None if it can't be tokenised."""
    command = strip_heredocs(command).replace("\\\n", " ").replace("\n", " ; ")
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    segments, current = [], []
    for tok in tokens:
        if tok in CONTROL_TOKENS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def _operands(args: list[str], takes_value: frozenset[str] = frozenset()) -> list[str]:
    """Non-option arguments, skipping the values of options in `takes_value`."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in takes_value:
            skip = True
        elif not a.startswith("-") or a == "-":
            out.append(a)
    return out


def _line_count(args: list[str], default: int = 10) -> int:
    for i, a in enumerate(args):
        if a == "-n" and i + 1 < len(args) and args[i + 1].lstrip("+-").isdigit():
            return int(args[i + 1].lstrip("+-"))
        m = re.fullmatch(r"-n?(\d+)", a)
        if m:
            return int(m.group(1))
    return default


def _grep_seen(segment: list[str], stdout: str) -> set[str]:
    """Files in grep/rg output (`path:...`, or bare paths with -l), plus a single searched file."""
    args = segment[1:]
    operands = _operands(args, GREP_VALUE_OPTIONS)
    files = operands if any(a in ("-e", "-f") for a in args) else operands[1:]
    recursive = segment[0] == "rg" or any(re.fullmatch(r"-[a-zA-Z]*[rR][a-zA-Z]*|--recursive", a) for a in args)
    if len(files) == 1 and not recursive:
        return set(files)  # one file searched: output lines carry no path
    list_only = any(a in ("-l", "--files-with-matches") for a in args)
    seen = set()
    for line in stdout.splitlines():
        path = line if list_only else line.split(":", 1)[0] if ":" in line else None
        if path and " " not in path and not path.isdigit() and not line.startswith(("Binary file", "grep:")):
            seen.add(path)
    return seen


def _ls_seen(segment: list[str], stdout: str) -> set[str]:
    operands = _operands(segment[1:]) or ["."]
    seen, current = set(), operands[0] if len(operands) == 1 else "."
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("total "):
            continue
        if line.endswith(":") and line[:-1] in operands:  # header when ls lists several dirs
            current = line[:-1]
            continue
        name = line.split()[-1]
        seen.add(name if name in operands else posixpath.join(current, name))
    return seen


def parse_command(command: str, stdout: str = "") -> tuple[list[ReadSpan], set[str], set[str]]:
    """(reads, seen files, edited files) for one `run_command`, with paths normalised.

    Covers the common forms only; a line that can't be tokenised gives nothing.
    """
    segments = _segments(command)
    if segments is None:
        return [], set(), set()
    reads: list[ReadSpan] = []
    seen: set[str] = set()
    edited: set[str] = set()
    for seg in segments:
        args, i = [], 0
        while i < len(seg):
            tok = seg[i]
            if tok in REDIRECTS or tok in ("<", ">&", "<&"):
                fd = args.pop() if args and args[-1].isdigit() else None  # "2>" redirects a stream
                target = seg[i + 1] if i + 1 < len(seg) else ""
                if tok in REDIRECTS and fd in (None, "1") and target != "/dev/null":
                    edited.add(target)
                i += 2
            else:
                args.append(tok)
                i += 1
        if not args:
            continue
        name = posixpath.basename(args[0])
        rest = args[1:]
        if name in ("cat", "nl"):
            reads += [ReadSpan(f) for f in _operands(rest)]
        elif name == "head":
            n = _line_count(rest)
            reads += [ReadSpan(f, 1, n) for f in _operands(rest, frozenset({"-n", "-c"}))]
        elif name == "tail":
            n = _line_count(rest)
            reads += [ReadSpan(f, last=n) for f in _operands(rest, frozenset({"-n", "-c"}))]
        elif name == "sed":
            in_place = any(a.startswith("-i") or a == "--in-place" for a in rest)
            operands = _operands(rest, frozenset({"-e"}))
            script = next((rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "-e"), None)
            files = operands if script is not None else operands[1:]
            script = script if script is not None else (operands[0] if operands else "")
            if in_place:
                edited.update(files)
            elif "-n" in rest:
                for m in SED_PRINT.finditer(script):
                    start = int(m.group(1))
                    reads += [ReadSpan(f, start, int(m.group(2) or start)) for f in files]
        elif name in GREP_COMMANDS or (name == "git" and rest[:1] == ["grep"]):
            seen |= _grep_seen(args if name != "git" else args[1:], stdout)
        elif name == "find":
            seen |= {line.strip() for line in stdout.splitlines() if line.strip()}
        elif name == "ls":
            seen |= _ls_seen(args, stdout)
    reads = [ReadSpan(p, r.start, r.end, r.truncated, r.last) for r in reads if (p := normalize_path(r.file))]
    return reads, _normalized(seen), _normalized(edited)


def _normalized(paths: set[str]) -> set[str]:
    return {p for p in map(normalize_path, paths) if p}


def _line_runs(numbers: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of line numbers, in order: [3, 4, 5, 9] → [(3, 5), (9, 9)]."""
    runs: list[tuple[int, int]] = []
    for n in numbers:
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        elif not runs or n > runs[-1][1]:
            runs.append((n, n))
    return runs


def _classify_editor(call: ToolCall, event: CallEvent) -> CallEvent:
    """SWE-agent's `str_replace_editor`: observations are plain text, not the harness's JSON."""
    command, output = call.arguments.get("command"), call.output or ""
    path = normalize_path(call.arguments.get("path"))
    if command == "view" and path:
        if output.startswith(DIRECTORY_VIEW):
            event.ok = True
            event.seen_files = _normalized({line.strip() for line in output.splitlines()[1:] if line.strip()})
        elif runs := _line_runs([int(n) for n in NUMBERED_LINE.findall(output)]):
            event.ok = True
            event.reads = [ReadSpan(path, a, b) for a, b in runs]
    elif command in EDITOR_EDITS and path:
        event.ok = output.startswith(EDITOR_OK)
        event.edited_files.add(path)
    return event


def classify_call(call: ToolCall) -> CallEvent:
    if call.name == "str_replace_editor":
        return _classify_editor(call, CallEvent(call.index, call.name, False, author=call.author))
    if call.name == "bash":
        reads, seen, edited = parse_command(call.arguments.get("command", ""), call.output or "")
        return CallEvent(call.index, call.name, True, reads, seen, edited_files=edited, author=call.author)
    obs = observation(call)
    ok = obs.get("status") == "ok"
    event = CallEvent(call.index, call.name, ok, author=call.author)
    if obs.get("error_type") == BUDGET_EXCEEDED:
        return event
    if call.name == "read_file":
        path = normalize_path(obs.get("filepath") or call.arguments.get("filepath"))
        if ok and path:
            event.reads.append(ReadSpan(path, obs.get("start_line"), obs.get("end_line"), bool(obs.get("is_truncated"))))
    elif call.name == "run_command":
        reads, seen, edited = parse_command(call.arguments.get("command", ""), obs.get("stdout") or "")
        event.edited_files = edited
        if ok:
            event.reads, event.seen_files = reads, seen
    elif call.name in GRAPH_TOOLS:
        if ok:
            _collect_nodes(obs.get("results", obs), event.seen_nodes)
    elif call.name in ("edit_file", "write_file"):
        path = normalize_path(call.arguments.get("filepath") or obs.get("filepath"))
        if path:
            event.edited_files.add(path)
    return event


def exploration_window(events: list[CallEvent], snapshot_files: set[str], target_files: set[str] = frozenset()) -> int:
    """Index of the first edit of a snapshot or target file, or len(events) if there is none."""
    real = snapshot_files | target_files
    return next((e.index for e in events if e.edited_files & real), len(events))


def explore_trace(trace: Trace, snapshot_files: set[str], target_files: set[str] = frozenset(),
                  instance_id: str = "") -> Exploration:
    events = [classify_call(c) for c in trace.calls]
    created: list[str] = []
    for e in events:
        if e.ok and e.tool != "edit_file":
            created += [f for f in sorted(e.edited_files - snapshot_files) if f not in created]
    return Exploration(instance_id, events, exploration_window(events, snapshot_files, target_files), created)


def summary_row(ex: Exploration, target_files: set[str]) -> dict:
    """Per-task diagnostics, not scores: where the window ends and when a target file was first read."""
    first = next((e.index for e in ex.window if any(r.file in target_files for r in e.reads)), None)
    return {
        "instance_id": ex.instance_id,
        "n_calls": len(ex.events),
        "window_end": ex.window_end,
        "window_reads": sum(len(e.reads) for e in ex.window),
        "first_target_read": first,
        "created_files": len(ex.created_files),
        "created_target": bool(set(ex.created_files) & target_files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign explore", description="Explored set and exploration window for every trace of a run.")
    parser.add_argument("--run", required=True, action="append", help="run directory; repeat for each shard")
    parser.add_argument("--groundtruth", required=True, help="dir with files.jsonl and targets.jsonl")
    parser.add_argument("--out", help="write one Exploration per task here (JSON lines)")
    parser.add_argument("--csv", help="also write the per-task summary here")
    args = parser.parse_args(argv)

    gt = Path(args.groundtruth)
    if not (gt / "files.jsonl").exists():
        print(f"{gt / 'files.jsonl'} is missing: rebuild the ground truth with this version of navalign.build", file=sys.stderr)
        return 1
    files, targets = load_jsonl(gt / "files.jsonl"), load_jsonl(gt / "targets.jsonl")
    run = load_runs(args.run)
    explorations, rows, problems = [], [], []
    for tid in sorted(run.latest):
        trace = run.trace(tid)
        if trace is None or tid not in files or tid not in targets:
            problems.append(f"{tid}: " + ("no trace" if trace is None else "no ground truth"))
            continue
        target_files = set(targets[tid]["files"])
        ex = explore_trace(trace, set(files[tid]["files"]), target_files, tid)
        explorations.append(ex)
        rows.append(summary_row(ex, target_files))
    if args.out:
        Path(args.out).write_text("".join(json.dumps(ex.to_dict()) + "\n" for ex in explorations), encoding="utf-8")
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["first_target_read"] = summary["first_target_read"].astype("Int64")  # stays an int beside None
    if args.csv:
        summary.to_csv(args.csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
