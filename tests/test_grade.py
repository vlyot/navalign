import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from navalign.collect import SKIPPED, collect_run
from navalign.data import read_snapshot_files
from navalign.grade import (agent_patches, grade_tasks, gradable_table, load_grades, main, missing_modules,
                            reference_variants, snapshot_requirements)
from navalign.runs import load_run

FAR = time.time() + 3600
SNAPSHOTS = Path(__file__).parent.parent / "data" / "snapshots"


@dataclass
class FakeResult:  # the fields of swegemma's TaskResult that grading reads
    resolved: bool
    test_exit_code: int
    test_output: str = ""
    error: str | None = None


TASKS = {
    "fastapi_1": {"instance_id": "fastapi_1", "patch": "GOLD-1"},
    "rich_1": {"instance_id": "rich_1", "patch": "GOLD-2"},
}


def verifier(outcomes: dict[tuple[str, str], bool], calls: list | None = None):
    """Passes or fails each (task, patch) pair as told; records every call."""
    def verify(task, patch):
        if calls is not None:
            calls.append((task["instance_id"], patch))
        passed = outcomes[(task["instance_id"], patch)]
        output = "" if passed else "E   ModuleNotFoundError: No module named 'typing_inspection'\n1 error"
        return FakeResult(passed, 0 if passed else 2, output)
    return verify


def control(tmp_path, outcomes, **kw) -> list[dict]:
    out = tmp_path / "gold"
    grade_tasks(TASKS, sorted(TASKS), verifier(outcomes, kw.get("calls")), out, reference_variants(),
                kw.get("deadline", FAR), 60)
    return load_grades(out)


def test_gold_pass_and_empty_fail_make_a_task_gradable(tmp_path):
    rows = control(tmp_path, {("fastapi_1", "GOLD-1"): True, ("fastapi_1", ""): False,
                              ("rich_1", "GOLD-2"): True, ("rich_1", ""): False})
    table = gradable_table(rows).set_index("instance_id")
    assert table["gradable"].all() and (table["reason"] == "ok").all()


def test_ungradable_reasons(tmp_path):
    rows = control(tmp_path, {("fastapi_1", "GOLD-1"): False, ("fastapi_1", ""): False,
                              ("rich_1", "GOLD-2"): True, ("rich_1", ""): True})
    table = gradable_table(rows).set_index("instance_id")
    assert table.loc["fastapi_1", "reason"] == "gold_fails"
    assert table.loc["fastapi_1", "missing_modules"] == "typing_inspection"
    assert table.loc["rich_1", "reason"] == "empty_passes"
    assert not table["gradable"].any()


def test_verify_exception_becomes_error_row_and_loop_continues(tmp_path):
    def verify(task, patch):
        if task["instance_id"] == "fastapi_1":
            raise RuntimeError("sandbox died")
        return FakeResult(patch != "", 0)
    out = tmp_path / "gold"
    grade_tasks(TASKS, sorted(TASKS), verify, out, reference_variants(), FAR, 60)
    table = gradable_table(load_grades(out)).set_index("instance_id")
    assert table.loc["fastapi_1", "reason"] == "error"
    assert table.loc["rich_1", "reason"] == "ok"


def test_resume_skips_pairs_already_graded(tmp_path):
    outcomes = {(t, p): p != "" for t, task in TASKS.items() for p in (task["patch"], "")}
    control(tmp_path, outcomes)
    calls = []
    rows = control(tmp_path, outcomes, calls=calls)
    assert calls == [] and len(rows) == 4


def test_deadline_skips_remaining_pairs_and_resume_runs_them(tmp_path):
    outcomes = {(t, p): p != "" for t, task in TASKS.items() for p in (task["patch"], "")}
    rows = control(tmp_path, outcomes, deadline=time.time() + 30)
    assert {r["error"] for r in rows} == {SKIPPED}
    rows = control(tmp_path, outcomes)
    assert gradable_table(rows)["gradable"].all()


def test_agent_patches_come_from_the_latest_completed_rows(tmp_path):
    out = tmp_path / "run"
    results = {"fastapi_1": {"resolved": False, "agent_patch": "AGENT-1", "tool_calls": 0},
               "rich_1": {"resolved": False, "agent_patch": "", "tool_calls": 0}}
    collect_run(TASKS, sorted(TASKS), lambda t: results[t["instance_id"]], out, FAR, 60)
    collect_run(TASKS, ["rich_1"], lambda t: 1 / 0, out, FAR, 60)  # a later failed attempt
    patch_of = agent_patches(load_run(out))
    assert patch_of(TASKS["fastapi_1"]) == "AGENT-1"
    assert patch_of(TASKS["rich_1"]) is None


def test_task_without_agent_patch_is_not_graded(tmp_path):
    calls = []
    out = tmp_path / "agent"
    grade_tasks(TASKS, sorted(TASKS), verifier({("fastapi_1", "AGENT-1"): True}, calls), out,
                {"agent": lambda t: "AGENT-1" if t["instance_id"] == "fastapi_1" else None}, FAR, 60)
    assert calls == [("fastapi_1", "AGENT-1")]


def test_missing_modules_counts_tasks(tmp_path):
    rows = control(tmp_path, {(t, p): False for t, task in TASKS.items() for p in (task["patch"], "")})
    assert missing_modules(rows) == {"typing_inspection": 2}


PYPROJECT = """
[project]
dependencies = ["starlette>=0.40.0,<0.51.0", "pydantic[email] >=2.9 ; python_version >= '3.9'"]
[project.optional-dependencies]
all = ["httpx >=0.23.0"]
[dependency-groups]
tests = ["pytest>=8"]
"""


def test_snapshot_requirements_reads_dependencies_extras_and_groups():
    assert snapshot_requirements(PYPROJECT, ["Starlette", "pydantic", "httpx", "pytest", "absent"]) == {
        "starlette": "starlette>=0.40.0,<0.51.0",
        "pydantic": "pydantic>=2.9",
        "httpx": "httpx>=0.23.0",
        "pytest": "pytest>=8",
    }


@pytest.mark.skipif(not (SNAPSHOTS / "fastapi_14303.tgz").exists(), reason="local data/ subset not present")
def test_snapshot_requirements_on_a_dev_snapshot():
    text = read_snapshot_files(SNAPSHOTS / "fastapi_14303.tgz", ["pyproject.toml"])["pyproject.toml"]
    assert snapshot_requirements(text, ["starlette"]) == {"starlette": "starlette>=0.40.0,<0.51.0"}


def test_cli_writes_gradable_csv(tmp_path, capsys):
    control(tmp_path, {("fastapi_1", "GOLD-1"): True, ("fastapi_1", ""): False,
                       ("rich_1", "GOLD-2"): False, ("rich_1", ""): False})
    csv = tmp_path / "gradable_tasks.csv"
    assert main(["--grades", str(tmp_path / "gold"), "--csv", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "2 tasks: gold passes 1, gradable 1" in out and "typing_inspection (2)" in out
    assert csv.read_text(encoding="utf-8").splitlines()[0].startswith("instance_id,gradable,reason")


def test_cli_reports_agent_passes_on_gradable_tasks(tmp_path, capsys):
    out = tmp_path / "agent"
    grade_tasks(TASKS, sorted(TASKS), lambda t, p: FakeResult(True, 0), out, {"agent": lambda t: "A"}, FAR, 60)
    gradable = tmp_path / "gradable_tasks.csv"
    gradable.write_text("instance_id,gradable\nfastapi_1,True\nrich_1,False\n", encoding="utf-8")
    assert main(["--grades", str(out), "--gradable", str(gradable)]) == 0
    assert "agent: 2/2 passed (0 not run); on gradable tasks 1/1" in capsys.readouterr().out


def test_cli_exits_1_on_error_rows(tmp_path, capsys):
    out = tmp_path / "gold"
    out.mkdir()
    (out / "grades.jsonl").write_text(json.dumps({"instance_id": "rich_1", "variant": "gold", "error": SKIPPED}) + "\n",
                                      encoding="utf-8")
    assert main(["--grades", str(out)]) == 1
    assert "PROBLEM rich_1 [gold]: skipped: deadline" in capsys.readouterr().err


def test_outcome_that_differs_in_a_replicate_is_unstable(tmp_path):
    first = control(tmp_path, {("fastapi_1", "GOLD-1"): True, ("fastapi_1", ""): False,
                               ("rich_1", "GOLD-2"): True, ("rich_1", ""): False})
    second = tmp_path / "second"
    flipped = {("fastapi_1", "GOLD-1"): True, ("fastapi_1", ""): False, ("rich_1", "GOLD-2"): False, ("rich_1", ""): False}
    grade_tasks(TASKS, sorted(TASKS), verifier(flipped), second, reference_variants(), FAR, 60)
    table = gradable_table(first, [(load_grades(second), None)]).set_index("instance_id")
    assert table.loc["rich_1", "reason"] == "unstable" and not table.loc["rich_1", "gradable"]
    assert table.loc["fastapi_1", "reason"] == "ok"
    limited = gradable_table(first, [(load_grades(second), ["fastapi"])]).set_index("instance_id")
    assert limited.loc["rich_1", "reason"] == "ok"  # the replicate only counts for fastapi tasks


def test_cli_replicate_with_repo_list_and_agent_line_from_same_run(tmp_path, capsys):
    out = tmp_path / "run"
    both = reference_variants() | {"agent": lambda t: "A"}
    passes = {("fastapi_1", "GOLD-1"): True, ("fastapi_1", ""): False, ("fastapi_1", "A"): True,
              ("rich_1", "GOLD-2"): True, ("rich_1", ""): False, ("rich_1", "A"): True}
    grade_tasks(TASKS, sorted(TASKS), verifier(passes), out, both, FAR, 60)
    rep = tmp_path / "rep"
    grade_tasks(TASKS, sorted(TASKS), verifier(passes | {("rich_1", "GOLD-2"): False}), rep, reference_variants(), FAR, 60)
    agent_csv = tmp_path / "agent.csv"
    assert main(["--grades", str(out), "--replicate", f"{rep}:rich", "--agent-csv", str(agent_csv)]) == 0
    printed = capsys.readouterr().out
    assert "gradable 1" in printed and "unstable" in printed
    assert "agent: 2/2 passed (0 not run); on gradable tasks 1/1" in printed
    assert agent_csv.read_text(encoding="utf-8").startswith("instance_id,passed,exit_code")


def test_harness_error_message_is_kept_and_unrun_tests_are_not_counted(tmp_path, capsys):
    """swegemma's TaskResult keeps `error` as `error_message`; exit -1 means the tests never ran."""
    out = tmp_path / "agent"
    results = {
        "fastapi_1": {"resolved": True, "test_exit_code": 0, "test_output": "1 passed"},
        "rich_1": {"resolved": False, "test_exit_code": -1, "test_output": "",
                   "error_message": "Failed to apply test_patch: error: patch failed: tests/test_console.py:529"},
    }
    grade_tasks(TASKS, sorted(TASKS), lambda t, p: results[t["instance_id"]], out, {"agent": lambda t: "A"}, FAR, 60)
    rows = {r["instance_id"]: r for r in load_grades(out)}
    assert rows["rich_1"]["harness_error"].startswith("Failed to apply test_patch")
    agent_csv = tmp_path / "agent.csv"
    assert main(["--grades", str(out), "--agent-csv", str(agent_csv)]) == 0
    assert "agent: 1/1 passed (1 not run)" in capsys.readouterr().out
    assert "rich_1,False,-1,False,Failed to apply test_patch" in agent_csv.read_text(encoding="utf-8")
