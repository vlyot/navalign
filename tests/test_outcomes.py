import json
from pathlib import Path

import pytest
from test_escaping import FILE_TEXT, failed_edit, write_run
from test_escaping import read as read_text
from test_explore import call
from test_score import ROWS, ground_truth, read

from navalign.outcomes import (
    CATEGORIES,
    agent_grades,
    classify_run,
    count_tests,
    patch_targets,
    run_outcomes,
)
from navalign.runs import load_run
from navalign.score import GroundTruth

ROOT = Path(__file__).parent.parent
RUN1 = ROOT / "results" / "traces" / "run1"


def diff(path: str, header: str, *lines: str) -> str:
    return "\n".join([f"--- a/{path}", f"+++ b/{path}", header, *lines]) + "\n"


def test_patch_targets_finds_target_nodes_by_base_line():
    patch = diff("pkg/a.py", "@@ -14,3 +14,3 @@", " x", "-y", "+z", " w")
    assert patch_targets(patch, ground_truth()) == ({"pkg/a.py"}, {"pkg.a.f"})


def test_added_line_is_placed_at_the_preceding_base_line():
    gt = ground_truth()
    inside = diff("pkg/a.py", "@@ -19,2 +19,3 @@", " a", " b", "+c")  # after line 20, f's last line
    after = diff("pkg/a.py", "@@ -21,2 +21,3 @@", " a", " b", "+c")  # after line 22, between f and g
    assert patch_targets(inside, gt)[1] == {"pkg.a.f"}
    assert patch_targets(after, gt)[1] == set()


def test_patch_targets_reports_new_and_non_target_files():
    patch = diff("pkg/new.py", "@@ -0,0 +1,2 @@", "+a", "+b") + diff("pkg/a.py", "@@ -30,1 +30,1 @@", "-g1", "+g2")
    assert patch_targets(patch, ground_truth()) == ({"pkg/new.py", "pkg/a.py"}, set())
    assert patch_targets("", ground_truth()) == (set(), set())


@pytest.mark.parametrize("output, counts", [
    ("STDOUT:\n....  [100%]\n- generated xml file: /tmp/x.xml -\n4 passed in 0.55s\n\nSTDERR:\n", (4, 4)),
    ("FAILED tests/t.py::x\n1 failed, 3 passed, 2 warnings in 0.6s\n", (3, 4)),
    ("!!! Interrupted: 1 error during collection !!!\n=========== 1 error in 0.86s ===========\n", (0, 1)),
    ("2 failed, 1 passed, 1 skipped, 1 xfailed in 1.0s\n", (1, 3)),
    ("3 errors in 0.2s\n", (0, 3)),
    ("STDOUT:\n........FFFFF....\nSTDERR:\n", None),
    ("SyntaxError: unexpected character after line continuation character\n", None),
])
def test_count_tests(output, counts):
    assert count_tests(output) == counts


@pytest.mark.parametrize("kwargs, category", [
    ({"context_limit": True, "found": True, "edited_target": True, "graded": True, "passed": True}, "X"),
    ({"found": False, "edited_target": True, "graded": True, "passed": True}, "A"),
    ({"found": True, "edited_target": False, "graded": True, "passed": True}, "B"),
    ({"found": True, "edited_target": True, "graded": False, "passed": True}, "CD_ng"),
    ({"found": True, "edited_target": True, "graded": True, "passed": True, "escaping": True}, "D"),
    ({"found": True, "edited_target": True, "graded": True, "passed": False, "escaping": True}, "C1"),
    ({"found": True, "edited_target": True, "graded": True, "passed": False}, "C2"),
])
def test_classify_run_precedence(kwargs, category):
    assert classify_run(**({"context_limit": False, "escaping": False} | kwargs)) == category
    assert category in CATEGORIES


def test_agent_grades_keeps_the_latest_error_free_agent_rows():
    rows = [{"instance_id": "t", "variant": "agent", "passed": False, "error": ""},
            {"instance_id": "t", "variant": "agent", "passed": True, "error": ""},
            {"instance_id": "t", "variant": "gold", "passed": True, "error": ""},
            {"instance_id": "u", "variant": "agent", "error": "skipped: deadline"}]
    assert agent_grades(rows) == {"t": rows[1]}


def _edit(index: int, path: str = "pkg/a.py"):
    return call(index, "edit_file", {"filepath": path, "old_string": "a", "new_string": "b"}, filepath=path)


def _set_rows(run_dir: Path, fields: dict[str, dict]) -> None:
    path = run_dir / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("".join(json.dumps(r | fields.get(r["instance_id"], {})) + "\n" for r in rows), encoding="utf-8")


def test_run_outcomes_assigns_every_category(tmp_path):
    traces = {
        "pkg_a": [read(0, "pkg/b.py")],
        "pkg_b": [read(0, "pkg/a.py")],
        "pkg_c1": [read_text(0, FILE_TEXT, "pkg/a.py"), failed_edit(1, '    return "a\\0b"', "pkg/a.py")],
        "pkg_c2": [read(0, "pkg/a.py"), _edit(1)],
        "pkg_d": [read(0, "pkg/b.py"), read(1, "pkg/a.py", 10, 20), _edit(2)],
        "pkg_ng": [read(0, "pkg/a.py"), _edit(1)],
        "pkg_nr": [read(0, "pkg/a.py"), _edit(1)],
        "pkg_x": [read(0, "pkg/a.py")],
    }
    run_dir = write_run(tmp_path / "run", traces)
    _set_rows(run_dir, {"pkg_d": {"agent_patch": diff("pkg/a.py", "@@ -14,3 +14,3 @@", " x", "-y", "+z", " w")},
                        "pkg_x": {"extra": {"error_message": "litellm.ContextWindowExceededError"}}})
    gts = {tid: GroundTruth.from_rows(*(row | {"instance_id": tid} for row in ROWS.values())) for tid in traces}
    failed = {"passed": False, "exit_code": 1, "test_output_tail": "1 failed, 2 passed in 0.1s\n"}
    passed = {"passed": True, "exit_code": 0, "test_output_tail": "3 passed in 0.1s\n"}
    grades = {"pkg_b": failed, "pkg_c1": failed, "pkg_c2": failed, "pkg_d": passed, "pkg_ng": passed,
              "pkg_nr": {"passed": False, "exit_code": -1, "test_output_tail": ""}}
    table = run_outcomes(load_run(run_dir), gts, grades, set(traces) - {"pkg_ng"}).set_index("instance_id")

    assert table["category"].to_dict() == {"pkg_a": "A", "pkg_b": "B", "pkg_c1": "C1", "pkg_c2": "C2", "pkg_d": "D",
                                           "pkg_ng": "CD_ng", "pkg_nr": "CD_ng", "pkg_x": "X"}
    d = table.loc["pkg_d"]
    assert (d["k"], d["patch_produced"], d["patch_touches_target_file"], d["patch_touches_target_node"]) == (2, True, True, True)
    assert (d["tests_passed"], d["tests_total"], d["tests_passed_frac"]) == (3, 3, 1.0)
    assert (d["s_file_prefix"], d["focus_prefix"]) == (0.5, 0.5)
    assert table.loc["pkg_c2", "tests_passed_frac"] == pytest.approx(2 / 3)
    assert (table.loc["pkg_c1", "r3_escaping"], table.loc["pkg_c1", "passed"]) == (1, False)
    assert not table.loc["pkg_nr", "tests_ran"]
    assert table.loc["pkg_a", "passed"] is None and table.loc["pkg_a", "repo"] == "pkg"


def test_run1_categories_cover_every_run():
    shards = [RUN1 / f"shard{i}" for i in range(3)]
    grades_dir, gradable_csv = ROOT / "results" / "gold" / "final", ROOT / "results" / "gold" / "gradable_tasks.csv"
    if not all(p.exists() for p in (*shards, grades_dir, gradable_csv)):
        pytest.skip("run 1 or its grades not present")
    import pandas as pd

    from navalign.grade import load_grades
    from navalign.runs import load_runs
    from navalign.score import load_groundtruth

    run = load_runs(shards)
    gradable = set(pd.read_csv(gradable_csv).query("gradable")["instance_id"])
    table = run_outcomes(run, load_groundtruth(ROOT / "results" / "groundtruth", run.latest),
                         agent_grades(load_grades(grades_dir)), gradable)
    assert len(table) == 129 and set(table["category"]) <= set(CATEGORIES)
    assert set(table.loc[table["category"] == "X", "instance_id"]) == {"rich_3469", "rich_3942"}
    graded_passes = table["passed"].fillna(False).astype(bool) & table["gradable"] & table["tests_ran"]
    assert graded_passes.sum() == 46  # roadmap: 46/109 on gradable tasks
