"""Per-run outcomes (Methodology step 5): the failure category, and finer outcomes beside it.

Categories, in order of precedence:

- X      the run was ended by the context limit
- A      never read a target file in the exploration window (new-file-only tasks: never created the target)
- B      read one, but never tried to edit a target file (no patch, or edits to other files only)
- C1     edited a target file and failed, with at least one failed edit of a target file caused by escaping (R3)
- C2     edited a target file and failed otherwise
- D      edited a target file and passed
- CD_ng  edited a target file on a task that isn't gradable, or whose tests never ran: C and D merged

"Edited" counts attempts anywhere in the trace, failed ones included (C1 needs them). Whether the
submitted patch touches a target file or node is reported separately. PASS/FAIL comes from the
validated local environment (`navalign.grade`), not the shipped harness's own `resolved`.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .escaping import edit_failures
from .grade import TESTS_NOT_RUN, latest_grades
from .patch import FileChange, parse_patch
from .runs import Run, hit_context_limit
from .score import (
    COUNT_COLUMNS,
    GroundTruth,
    explore_run,
    prefix_scores,
    score_exploration,
)

CATEGORIES = {
    "A": "never read a target file",
    "B": "read one, never edited one",
    "C1": "edited, failed (escaping)",
    "C2": "edited, failed",
    "D": "edited, passed",
    "CD_ng": "edited, not gradable",
    "X": "ended by the context limit",
}
R3_LABELS = ("escaping", "malformed", "stale", "no_match")
PREFIX = 10  # calls; the equal span the prefix scores are taken over
PYTEST_SUMMARY = re.compile(r"^=*\s*((?:\d+ [a-z]+(?:, )?)+) in [\d.]+s\b")
PYTEST_COUNT = re.compile(r"(\d+) ([a-z]+)")


def _base_positions(change: FileChange) -> set[int]:
    """Base-side line of each non-blank changed line: a removed line's own, an added line's preceding one."""
    positions = set()
    for hunk in change.hunks:
        before = hunk.source_start - 1
        for line in hunk:
            if line.source_line_no is not None:
                before = line.source_line_no
            if not line.value.strip():
                continue
            if line.is_removed:
                positions.add(line.source_line_no)
            elif line.is_added:
                positions.add(before)
    return positions


def patch_targets(patch: str, gt: GroundTruth) -> tuple[set[str], set[str]]:
    """Files an agent patch changes, and the target nodes whose spans contain one of its changed lines."""
    if not patch.strip():
        return set(), set()
    files, nodes = set(), set()
    for change in parse_patch(patch):
        files.add(change.path)
        if not change.is_new:
            for line in _base_positions(change):
                nodes |= gt.spans.nodes_in(change.path, line, line) & gt.target_nodes
    return files, nodes


def count_tests(output: str) -> tuple[int, int] | None:
    """(passed, passed + failed + errors) from pytest's last summary line, or None if there is none."""
    for line in reversed(output.splitlines()):
        m = PYTEST_SUMMARY.match(line.strip())
        if m:
            counts = Counter({word: int(n) for n, word in PYTEST_COUNT.findall(m.group(1))})
            passed = counts["passed"]
            return passed, passed + counts["failed"] + counts["error"] + counts["errors"]
    return None


def classify_run(*, found: bool, edited_target: bool, context_limit: bool, graded: bool, passed: bool,
                 escaping: bool) -> str:
    """The run's category; `graded` = the task is gradable and the agent patch's tests ran."""
    if context_limit:
        return "X"
    if not found:
        return "A"
    if not edited_target:
        return "B"
    if not graded:
        return "CD_ng"
    if passed:
        return "D"
    return "C1" if escaping else "C2"


def agent_grades(rows: list[dict]) -> dict[str, dict]:
    """The latest error-free `agent` grade row per task, from a grades.jsonl."""
    return {tid: r for (tid, variant), r in latest_grades(rows).items() if variant == "agent" and not r.get("error")}


def run_outcomes(run: Run, groundtruth: dict[str, GroundTruth], grades: dict[str, dict], gradable: set[str],
                 prefix: int = PREFIX) -> pd.DataFrame:
    """One row per run: its scores (and the same over its first `prefix` calls), category and finer outcomes."""
    explorations, _ = explore_run(run, groundtruth)
    rows = []
    for tid, ex in explorations.items():
        gt, result, grade = groundtruth[tid], run.latest[tid], grades.get(tid)
        score = score_exploration(ex, gt)
        patch = result.get("agent_patch") or ""
        files, nodes = patch_targets(patch, gt)
        labels = Counter(f["label"] for f in edit_failures(run.trace(tid), tid) if f["file"] in gt.target_files)
        found = score.created_target if gt.new_file_only else score.k is not None
        edited = any(e.edited_files & gt.target_files for e in ex.events)
        tests_ran = grade is not None and grade.get("exit_code") != TESTS_NOT_RUN
        passed = bool(grade and grade.get("passed"))
        context = hit_context_limit(run, result)
        tests = count_tests(grade.get("test_output_tail") or "") if tests_ran else None
        rows.append({
            "instance_id": tid,
            "repo": tid.rsplit("_", 1)[0],
            "category": classify_run(found=found, edited_target=edited, context_limit=context,
                                     graded=tid in gradable and tests_ran, passed=passed, escaping=labels["escaping"] > 0),
            "found": found,
            "edited_target": edited,
            "context_limit": context,
            "gradable": tid in gradable,
            "tests_ran": tests_ran,
            "passed": passed if grade is not None else None,
            "patch_produced": bool(patch.strip()),
            "patch_touches_target_file": bool(files & gt.target_files),
            "patch_touches_target_node": bool(nodes) if gt.target_nodes else None,
            "tests_passed": tests[0] if tests else None,
            "tests_total": tests[1] if tests else None,
            "tests_passed_frac": tests[0] / tests[1] if tests and tests[1] else None,
        } | {f"r3_{label}": labels[label] for label in R3_LABELS} | {k: v for k, v in score.row().items() if k != "instance_id"}
          | prefix_scores(score, prefix))
    table = pd.DataFrame(rows)
    for col in ("tests_passed", "tests_total", *COUNT_COLUMNS):
        table[col] = table[col].astype("Int64")
    return table
