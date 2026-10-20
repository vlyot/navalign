import json

import pytest

from navalign.explore import explore_trace
from navalign.runs import CONTEXT_ERROR, hit_context_limit, load_run
from navalign.swe_agent import convert_run, main, navalign_id, parse_action, traj_to_atif
from navalign.trace import parse_trace

VIEW = "Here's the result of running `cat -n` on /testbed/pkg/a.py:\r\n     1\tdef f():\r\n     2\t    return 1"
TRAJ = {
    "trajectory": [
        {"action": 'grep -rn "def f" /testbed', "observation": "/testbed/pkg/a.py:1:def f():"},
        {"action": "str_replace_editor view /testbed/pkg/a.py  --view_range 1 2", "observation": VIEW},
        {"action": "", "observation": "format error"},
        {"action": "str_replace_editor str_replace /testbed/pkg/a.py   --old_str '    return 1' --new_str '    return 2'",
         "observation": "The file /testbed/pkg/a.py has been edited."},
        {"action": "submit", "observation": "diff --git a/pkg/a.py b/pkg/a.py"},
    ],
    "info": {"submission": "diff --git a/pkg/a.py b/pkg/a.py\n", "exit_status": "submitted", "swe_agent_version": "1.0.1"},
}


def test_navalign_id_uses_the_competition_form():
    assert navalign_id("psf__requests-2931") == "requests_2931"
    assert navalign_id("pytest-dev__pytest-10051") == "pytest_10051"


def test_parse_action_reads_a_view_range():
    assert parse_action("str_replace_editor view /testbed/a.py  --view_range 686 702") == (
        "str_replace_editor", {"command": "view", "path": "/testbed/a.py", "view_range": [686, 702]})


def test_parse_action_reads_old_and_new_strings_with_shell_quoting():
    action = "str_replace_editor str_replace /testbed/a.py --old_str 'x = '\"'\"'a'\"'\"'' --new_str 'x = 1'"
    name, args = parse_action(action)
    assert (name, args["command"], args["old_str"], args["new_str"]) == ("str_replace_editor", "str_replace", "x = 'a'", "x = 1")


def test_parse_action_keeps_command_and_path_when_quotes_do_not_balance():
    assert parse_action("str_replace_editor create /testbed/b.py --file_text 'unclosed")[1] == {
        "command": "create", "path": "/testbed/b.py"}


@pytest.mark.parametrize("action, expected", [("submit", ("submit", {})), ("ls -la", ("bash", {"command": "ls -la"}))])
def test_parse_action_other_tools(action, expected):
    assert parse_action(action) == expected


def test_converted_trajectory_is_read_by_parse_trace():
    trace = parse_trace(traj_to_atif(TRAJ))
    assert [c.name for c in trace.calls] == ["bash", "str_replace_editor", "str_replace_editor", "submit"]
    assert trace.calls[1].output == VIEW and trace.calls[1].arguments["view_range"] == [1, 2]


def test_converted_trajectory_finds_and_edits_the_target():
    ex = explore_trace(parse_trace(traj_to_atif(TRAJ)), {"pkg/a.py"}, {"pkg/a.py"})
    assert ex.window_end == 2
    assert [(r.file, r.start, r.end) for r in ex.events[1].reads] == [("pkg/a.py", 1, 2)]
    assert ex.events[0].seen_files == {"pkg/a.py"}


def write_trajs(tmp_path, trajs: dict[str, dict]):
    d = tmp_path / "trajs"
    d.mkdir()
    for swe_id, traj in trajs.items():
        (d / f"{swe_id}.traj").write_text(json.dumps(traj), encoding="utf-8")
    return d


def test_convert_run_writes_a_run_the_pipeline_loads(tmp_path):
    stopped = TRAJ | {"info": TRAJ["info"] | {"exit_status": "submitted (exit_context)"}}
    trajs = write_trajs(tmp_path, {"psf__requests-1": TRAJ, "psf__requests-2": stopped})
    ids = convert_run(trajs, {"psf__requests-1"}, tmp_path / "run")
    run = load_run(tmp_path / "run")
    assert ids == ["requests_1", "requests_2"] and sorted(run.trace_paths) == ids
    assert run.latest["requests_1"]["agent_patch"].startswith("diff --git")
    assert not hit_context_limit(run, run.latest["requests_1"])
    assert run.latest["requests_2"]["stopped_by"] == CONTEXT_ERROR and hit_context_limit(run, run.latest["requests_2"])
    grades = [json.loads(line) for line in (tmp_path / "run" / "grades.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(g["instance_id"], g["passed"]) for g in grades] == [("requests_1", True), ("requests_2", False)]
    assert (tmp_path / "run" / "gradable_tasks.csv").read_text(encoding="utf-8").count("True") == 2


def test_main_rejects_a_missing_trajectory(tmp_path):
    trajs = write_trajs(tmp_path, {"psf__requests-1": TRAJ})
    (tmp_path / "results.json").write_text(json.dumps({"resolved": []}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        main(["--trajs", str(trajs), "--results", str(tmp_path / "results.json"), "--out", str(tmp_path / "run"),
              "--task", "psf__requests-9"])
