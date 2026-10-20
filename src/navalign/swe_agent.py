"""SWE-agent trajectories (`.traj`, SWE-agent 1.x) as a navalign run.

Each trajectory step's action is one tool call: `str_replace_editor <command> <path> [--option
value ...]`, `submit`, or anything else as a `bash` command. The converter writes an ATIF trace per
task plus the files `navalign score` and `navalign analysis` read, so an external run goes through
the same pipeline as the competition's:

- `trace_<id>.json`      ATIF v1.7, one tool call per step, its observation as the call's result
- `results.jsonl`        the submitted patch and exit status (`stopped_by` marks a context-limit exit)
- `grades.jsonl`         pass/fail from SWE-bench's own `results.json` (`resolved` ids)
- `gradable_tasks.csv`   every task, since SWE-bench Verified tasks were validated by hand
- `manifest.json`

Instance ids are renamed to the competition's form, `psf__requests-2931` → `requests_2931`.

    navalign swe-agent --trajs DIR --results results.json --out DIR [--task SWE_ID ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import sys
from pathlib import Path

from .runs import CONTEXT_ERROR

EDITOR = "str_replace_editor"
EDITOR_OPTIONS = {"--view_range": 2, "--old_str": 1, "--new_str": 1, "--file_text": 1, "--insert_line": 1}
EDITOR_HEAD = re.compile(r"^str_replace_editor\s+(\w+)\s+(\S+)")


def navalign_id(swe_id: str) -> str:
    """`psf__requests-2931` → `requests_2931`."""
    repo, number = swe_id.split("__", 1)[1].rsplit("-", 1)
    return f"{repo}_{number}"


def parse_action(action: str) -> tuple[str, dict]:
    """(tool name, arguments) for one SWE-agent action string."""
    action = action.strip()
    if action == "submit" or action.startswith("submit "):
        return "submit", {}
    if not action.startswith(EDITOR + " "):
        return "bash", {"command": action}
    try:
        tokens = shlex.split(action)
    except ValueError:  # unbalanced quotes in an old/new string: the command and path are still readable
        m = EDITOR_HEAD.match(action)
        return EDITOR, {"command": m.group(1), "path": m.group(2)} if m else {"_raw": action}
    args: dict = {"command": tokens[1] if len(tokens) > 1 else None, "path": tokens[2] if len(tokens) > 2 else None}
    i = 3
    while i < len(tokens):
        n = EDITOR_OPTIONS.get(tokens[i])
        if n is None:
            i += 1
            continue
        values = tokens[i + 1: i + 1 + n]
        key = tokens[i].removeprefix("--")
        if key == "view_range":
            args[key] = [int(v) for v in values if v.lstrip("-").isdigit()]
        else:
            args[key] = values[0] if values else ""
        i += 1 + n
    return EDITOR, args


def traj_to_atif(traj: dict) -> dict:
    """An ATIF v1.7 trace with one step, and one tool call, per trajectory step that has an action."""
    info = traj.get("info") or {}
    steps = []
    for step in traj.get("trajectory") or []:
        if not (step.get("action") or "").strip():
            continue
        name, arguments = parse_action(step["action"])
        call_id = f"call_{len(steps)}"
        steps.append({
            "step_id": len(steps) + 1,
            "source": "agent",
            "tool_calls": [{"tool_call_id": call_id, "function_name": name, "arguments": arguments}],
            "observation": {"results": [{"source_call_id": call_id, "content": step.get("observation") or ""}]},
        })
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "swe-agent", "version": info.get("swe_agent_version", "")},
        "steps": steps,
        "final_metrics": {"exit_status": info.get("exit_status"), **(info.get("model_stats") or {})},
    }


def result_row(swe_id: str, traj: dict) -> dict:
    info = traj.get("info") or {}
    exit_status = info.get("exit_status") or ""
    row = {"instance_id": navalign_id(swe_id), "swe_bench_id": swe_id, "agent_patch": info.get("submission") or "",
           "exit_status": exit_status}
    if "context" in exit_status:
        row["stopped_by"] = CONTEXT_ERROR
    return row


def convert_run(trajs: Path, resolved: set[str], out: Path, swe_ids: list[str] | None = None) -> list[str]:
    """Write the run files for every `.traj` in `trajs` (or only `swe_ids`); returns the navalign ids."""
    paths = {p.stem: p for p in sorted(trajs.rglob("*.traj"))}
    wanted = swe_ids or sorted(paths)
    missing = [s for s in wanted if s not in paths]
    if missing:
        raise FileNotFoundError(f"no trajectory for {missing}")
    out.mkdir(parents=True, exist_ok=True)
    ids = []
    with (open(out / "results.jsonl", "w", encoding="utf-8") as f_results,
          open(out / "grades.jsonl", "w", encoding="utf-8") as f_grades):
        for swe_id in wanted:
            traj = json.loads(paths[swe_id].read_text(encoding="utf-8"))
            tid = navalign_id(swe_id)
            (out / f"trace_{tid}.json").write_text(json.dumps(traj_to_atif(traj)), encoding="utf-8")
            f_results.write(json.dumps(result_row(swe_id, traj)) + "\n")
            f_grades.write(json.dumps({"instance_id": tid, "variant": "agent", "passed": swe_id in resolved,
                                       "exit_code": 0}) + "\n")
            ids.append(tid)
    with open(out / "gradable_tasks.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "gradable", "reason"])
        writer.writerows([tid, True, "SWE-bench Verified"] for tid in ids)
    (out / "manifest.json").write_text(json.dumps({"run_id": out.name, "task_ids": ids, "source": str(trajs)}), encoding="utf-8")
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign swe-agent", description="Convert SWE-agent trajectories into a navalign run.")
    parser.add_argument("--trajs", required=True, help="dir with <swe_bench_id>.traj files (searched recursively)")
    parser.add_argument("--results", required=True, help="SWE-bench results.json with a `resolved` id list")
    parser.add_argument("--task", action="append", help="SWE-bench instance id (repeatable); default: every trajectory")
    parser.add_argument("--out", required=True, help="run directory to write")
    args = parser.parse_args(argv)

    resolved = set(json.loads(Path(args.results).read_text(encoding="utf-8"))["resolved"])
    ids = convert_run(Path(args.trajs), resolved, Path(args.out), args.task)
    print(f"{len(ids)} trajectories converted into {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
