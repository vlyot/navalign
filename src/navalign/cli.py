"""The `navalign` command: one entry point for every module's command line.

    navalign <command> [args]        e.g. navalign score-trace --demo
    navalign <command> --help
"""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import version

# command: (module, function, summary); modules are imported only when their command runs
COMMANDS = {
    "score-trace": ("score", "trace_main", "navigation scores for one trace (try --demo)"),
    "score": ("score", "main", "scores for every trace of a run, or of a folder of traces"),
    "build": ("build", "main", "ground truth (targets, spans, distances) from tasks, graphs and snapshots"),
    "mapper": ("mapper", "main", "map reference patches to target files and graph nodes"),
    "explore": ("explore", "main", "explored set and exploration window for every trace of a run"),
    "runs": ("runs", "main", "check and summarise a trace-collection run"),
    "escaping": ("escaping", "main", "label every failed edit_file (escaping, stale, no_match)"),
    "grade": ("grade", "main", "summarise a gold-patch control or an agent regrade"),
    "validity": ("validity", "main", "oracle and random baselines, hand-label agreement"),
    "analysis": ("analysis", "main", "outcomes, result tables and figures for a scored run"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: navalign <command> [args]", "", "commands:"]
    lines += [f"  {name:<{width}}  {summary}" for name, (_, _, summary) in COMMANDS.items()]
    return "\n".join([*lines, "", "`navalign <command> --help` lists a command's options."])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] == "--version":
        print(f"navalign {version('navalign')}")
        return 0
    if argv[0] not in COMMANDS:
        print(f"navalign: unknown command {argv[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module, function, _ = COMMANDS[argv[0]]
    return getattr(importlib.import_module(f"navalign.{module}"), function)(argv[1:]) or 0
