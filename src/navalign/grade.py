"""R1 gold-patch control and regrading, through the harness's own test step.

The notebook passes in `verify(task, patch) -> result`, which wraps `swegemma`'s `verify_task`
(the Phase 2 test container, no model). Nothing here imports the harness, so the loop is tested
locally with a fake verifier, like `navalign.collect`.

- Reference control: every task's gold patch and an empty patch. A task is gradable when the
  gold patch passes AND the empty patch fails (`gradable_tasks.csv`).
- Regrade: an agent run's patches, through the same verifier and environment.

    python -m navalign.grade --grades results/gold/repaired --csv results/gold/gradable_tasks.csv
    python -m navalign.grade --grades results/gold/run1-agent --gradable results/gold/gradable_tasks.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .collect import SKIPPED, append_row, as_dict, read_results
from .runs import Run

GRADES = "grades.jsonl"
# The harness reports -1 when the tests never ran: a patch (the agent's or the task's test patch)
# failed to apply, or the test spec was missing. Such an agent grade is not a FAIL of the agent's fix.
TESTS_NOT_RUN = -1
OUTPUT_TAIL = 6000
MISSING_MODULE = re.compile(r"No module named '([^']+)'")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def task_id(task: Any) -> str:
    return task["instance_id"] if isinstance(task, dict) else task.instance_id


def reference_variants() -> dict[str, Callable[[Any], str]]:
    """The gold patch and the empty patch."""
    return {
        "gold": lambda task: task["patch"] if isinstance(task, dict) else task.patch,
        "empty": lambda task: "",
    }


def agent_patches(run: Run) -> Callable[[Any], str | None]:
    """Patch source for regrading: each task's latest agent patch, None if it has no completed row."""
    patches = {tid: row.get("agent_patch") or "" for tid, row in run.latest.items() if not row.get("error")}
    return lambda task: patches.get(task_id(task))


def grade_record(instance_id: str, variant: str, result: Any, started: float, finished: float) -> dict:
    fields = as_dict(result)
    output = fields.get("test_output") or ""
    return {
        "instance_id": instance_id,
        "variant": variant,
        "passed": bool(fields.get("resolved")),
        "exit_code": fields.get("test_exit_code"),
        # swegemma's TaskResult stores `error=` as `error_message`
        "harness_error": fields.get("error") or fields.get("error_message") or "",
        "missing_modules": sorted(set(MISSING_MODULE.findall(output))),
        "test_output_tail": output[-OUTPUT_TAIL:],
        "started": round(started, 3),
        "finished": round(finished, 3),
        "error": "",
    }


def load_grades(grades_dir: str | Path) -> list[dict]:
    return read_results(Path(grades_dir) / GRADES)


def grade_tasks(tasks: dict[str, Any], ids: list[str], verify: Callable[[Any, str], Any], out_dir: Path,
                variants: dict[str, Callable[[Any], str | None]], deadline: float, max_task_s: float) -> list[dict]:
    """Verify each (task, variant) patch and append one row per pair to `out_dir/grades.jsonl`.

    Pairs that already have a row without an error are skipped, so a resumed run continues.
    A variant that returns None for a task (e.g. no agent patch) is skipped. A pair that could
    run past `deadline` gets a `skipped: deadline` row; an exception from `verify` an error row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {(r["instance_id"], r["variant"]) for r in load_grades(out_dir) if not r.get("error")}
    rows = []
    with open(out_dir / GRADES, "a", encoding="utf-8") as f:
        for tid in ids:
            for variant, patch_of in variants.items():
                if (tid, variant) in done or (patch := patch_of(tasks[tid])) is None:
                    continue
                started = time.time()
                if started + max_task_s > deadline:
                    row = {"instance_id": tid, "variant": variant, "started": round(started, 3), "error": SKIPPED}
                else:
                    try:
                        result = verify(tasks[tid], patch)
                    except Exception as exc:  # one broken task must not end the control
                        row = {"instance_id": tid, "variant": variant, "started": round(started, 3),
                               "finished": round(time.time(), 3), "error": f"{type(exc).__name__}: {exc}"}
                    else:
                        row = grade_record(tid, variant, result, started, time.time())
                append_row(f, row)
                rows.append(row)
                status = row["error"] or f"passed={row['passed']} exit={row['exit_code']}"
                print(f"{tid} [{variant}]: {status} ({time.time() - started:.0f}s)", file=sys.stderr)
    return rows


def latest_grades(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["instance_id"], r["variant"]): r for r in rows}


def _outcome(latest: dict, tid: str) -> tuple[bool, bool] | None:
    """(gold passed, empty passed), or None if either is missing or errored."""
    gold, empty = latest.get((tid, "gold")), latest.get((tid, "empty"))
    if gold is None or empty is None or gold.get("error") or empty.get("error"):
        return None
    return gold["passed"], empty["passed"]


def gradable_table(rows: list[dict], replicates: Iterable[tuple[list[dict], Iterable[str] | None]] = ()) -> pd.DataFrame:
    """One row per task: gradable when the gold patch passes and the empty patch fails.

    `replicates` are other control runs of the same environment, each as (rows, repos), where
    `repos` limits it to tasks of those repos (None = all). A task whose (gold, empty) outcome
    differs in any replicate is `unstable` and not gradable: its tests don't give the same answer twice.
    """
    latest = latest_grades(rows)
    others = [(latest_grades(r), set(repos) if repos is not None else None) for r, repos in replicates]
    records = []
    for tid in sorted({tid for tid, _ in latest}):
        gold, empty = latest.get((tid, "gold")), latest.get((tid, "empty"))
        repo = tid.rsplit("_", 1)[0]
        outcome = _outcome(latest, tid)
        if gold is None or empty is None:
            reason = "missing"
        elif gold.get("error") or empty.get("error"):
            reason = "error"
        elif any((o := _outcome(other, tid)) is not None and o != outcome
                 for other, repos in others if repos is None or repo in repos):
            reason = "unstable"
        elif not gold["passed"]:
            reason = "gold_fails"
        elif empty["passed"]:
            reason = "empty_passes"
        else:
            reason = "ok"
        records.append({
            "instance_id": tid,
            "gradable": reason == "ok",
            "reason": reason,
            "gold_passed": bool((gold or {}).get("passed")),
            "gold_exit_code": (gold or {}).get("exit_code"),
            "empty_exit_code": (empty or {}).get("exit_code"),
            "gold_detail": ((gold or {}).get("error") or (gold or {}).get("harness_error") or "")[:200],
            "missing_modules": " ".join((gold or {}).get("missing_modules", [])),
        })
    return pd.DataFrame(records, columns=["instance_id", "gradable", "reason", "gold_passed", "gold_exit_code",
                                          "empty_exit_code", "gold_detail", "missing_modules"])


def missing_modules(rows: list[dict]) -> Counter:
    """How many tasks hit each `ModuleNotFoundError`, across all latest rows."""
    hits = {(m, tid) for (tid, _), r in latest_grades(rows).items() for m in r.get("missing_modules", [])}
    return Counter(m for m, _ in hits)


def snapshot_requirements(pyproject_text: str, names: Iterable[str]) -> dict[str, str]:
    """The requirement each named package has in a snapshot's pyproject.toml (dependencies first, then extras and groups).

    Returns e.g. {"starlette": "starlette>=0.40.0,<0.49.0"} with markers dropped, so a repair can
    pip-install the version that commit declares instead of the newest wheel.
    """
    data = tomllib.loads(pyproject_text)
    project = data.get("project", {})
    sources = [project.get("dependencies", [])]
    sources += list(project.get("optional-dependencies", {}).values())
    sources += [g for g in data.get("dependency-groups", {}).values()]
    wanted = {_normalize(n) for n in names}
    found: dict[str, str] = {}
    for reqs in sources:
        for req in reqs:
            m = REQUIREMENT_NAME.match(req) if isinstance(req, str) else None
            if m and (name := _normalize(m.group(1))) in wanted and name not in found:
                found[name] = re.sub(r"\[[^\]]*\]", "", req.split(";", 1)[0]).replace(" ", "")
    return found


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a gold-patch control or an agent regrade.")
    parser.add_argument("--grades", required=True, action="append", help="dir with grades.jsonl; repeat to combine")
    parser.add_argument("--replicate", action="append", default=[], metavar="DIR[:REPO,...]",
                        help="another control run of the same environment (optionally only for these repos); "
                             "tasks whose outcome differs become 'unstable'")
    parser.add_argument("--gradable", help="gradable_tasks.csv for the agent line (default: this run's own control)")
    parser.add_argument("--csv", help="write gradable_tasks.csv (needs gold and empty rows)")
    parser.add_argument("--agent-csv", help="write the per-task agent grades")
    args = parser.parse_args(argv)

    rows = [r for d in args.grades for r in load_grades(d)]
    latest = latest_grades(rows)
    variants = Counter(v for _, v in latest)
    problems = [f"{tid} [{v}]: {r['error']}" for (tid, v), r in latest.items() if r.get("error")]

    gradable = None
    if {"gold", "empty"} <= set(variants):
        replicates = []
        for spec in args.replicate:
            path, sep, repos = spec.rpartition(":")
            if not sep or not Path(path).is_dir() or "/" in repos or "\\" in repos:
                path, repos = spec, ""  # no repo list (or the colon is a drive letter)
            replicates.append((load_grades(path), repos.split(",") if repos else None))
        table = gradable_table(rows, replicates)
        if args.csv:
            table.to_csv(args.csv, index=False)
        print(table["reason"].value_counts().to_string())
        print(f"\n{len(table)} tasks: gold passes {int(table['gold_passed'].sum())}, "
              f"gradable {int(table['gradable'].sum())}")
        gradable = set(table.loc[table["gradable"], "instance_id"])
    if "agent" in variants:
        agent = pd.DataFrame([{"instance_id": tid, "passed": r.get("passed"), "exit_code": r.get("exit_code"),
                               "tests_ran": r.get("exit_code") != TESTS_NOT_RUN,
                               "harness_error": (r.get("harness_error") or "")[:200]}
                              for (tid, v), r in sorted(latest.items()) if v == "agent" and not r.get("error")],
                             columns=["instance_id", "passed", "exit_code", "tests_ran", "harness_error"])
        if args.gradable:
            gradable = set(pd.read_csv(args.gradable).query("gradable")["instance_id"])
        if gradable is not None:
            agent["gradable"] = agent["instance_id"].isin(gradable)
        if args.agent_csv:
            agent.to_csv(args.agent_csv, index=False)
        ran = agent[agent["tests_ran"]]
        line = f"agent: {int(ran['passed'].sum())}/{len(ran)} passed ({len(agent) - len(ran)} not run)"
        if gradable is not None:
            on = ran[ran["instance_id"].isin(gradable)]
            line += f"; on gradable tasks {int(on['passed'].sum())}/{len(on)}"
        print(line)
    top = missing_modules(rows).most_common(15)
    if top:
        print("\nmissing modules (tasks): " + ", ".join(f"{m} ({n})" for m, n in top))
    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
