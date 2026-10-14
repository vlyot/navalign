import json
import time
from pathlib import Path

import pytest

from navalign.collect import collect_run, write_manifest
from navalign.runs import check_run, load_run, load_runs, main, summarize_run

FIXTURE = Path(__file__).parent.parent / "fixtures" / "trace_step_example.json"
FAR = time.time() + 3600


def _trace(tool_names: list[str]) -> dict:
    """A trace with one step per tool call, built from the fixture step."""
    step = json.loads(FIXTURE.read_text(encoding="utf-8"))["step"]
    steps = [{"step_id": 1, "source": "system", "message": "sys", "tool_calls": None, "observation": None}]
    for i, name in enumerate(tool_names, 2):
        s = json.loads(json.dumps(step))
        s["step_id"] = i
        s["tool_calls"][0] |= {"function_name": name, "tool_call_id": f"call_{i}"}
        steps.append(s)
    return {"schema_version": "ATIF-v1.7", "agent": {}, "steps": steps}


CALLS = {
    "fastapi_1": ["read_file", "run_command", "read_file"],
    "rich_1": ["search_similar_code", "edit_file"],
}


def fake_agent(out: Path, calls: dict[str, list[str]] = CALLS, reported: dict[str, int] | None = None):
    """Writes traces/trace_<id>.json and logs/<id>.log the way the harness does, then returns a result."""
    def run_one(task):
        tid = task["instance_id"]
        (out / "traces").mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "traces" / f"trace_{tid}.json").write_text(json.dumps(_trace(calls[tid])), encoding="utf-8")
        (out / "logs" / f"{tid}.log").write_text("ok\n", encoding="utf-8")
        n = (reported or {}).get(tid, len(calls[tid]))
        return {"resolved": tid == "rich_1", "test_exit_code": 0, "agent_patch": "x" * 7, "tool_calls": n, "duration_seconds": 3.0}
    return run_one


@pytest.fixture
def run_dir(tmp_path):
    out = tmp_path / "val1"
    ids = sorted(CALLS)
    write_manifest(out, {"run_id": "val1", "task_ids": ids})
    collect_run({t: {"instance_id": t} for t in ids}, ids, fake_agent(out), out, FAR, 60)
    return out


def test_load_run_reads_results_and_traces(run_dir):
    run = load_run(run_dir)
    assert run.manifest["run_id"] == "val1"
    assert sorted(run.latest) == ["fastapi_1", "rich_1"]
    assert [c.name for c in run.trace("fastapi_1").calls] == CALLS["fastapi_1"]
    assert run.trace("ghost") is None


def test_consistent_run_has_no_problems(run_dir):
    assert check_run(load_run(run_dir)) == []


def test_missing_trace_is_flagged(run_dir):
    (run_dir / "traces" / "trace_rich_1.json").unlink()
    assert check_run(load_run(run_dir)) == ["rich_1: no trace file"]


def test_tool_call_mismatch_is_flagged(tmp_path):
    out = tmp_path / "run"
    collect_run({"rich_1": {"instance_id": "rich_1"}}, ["rich_1"], fake_agent(out, reported={"rich_1": 5}), out, FAR, 60)
    assert check_run(load_run(out)) == ["rich_1: result has 5 tool calls, trace has 2"]


def test_submit_patch_is_not_counted_against_the_result(tmp_path):
    out = tmp_path / "run"
    calls = {"rich_1": ["read_file", "edit_file", "submit_patch"]}
    collect_run({"rich_1": {"instance_id": "rich_1"}}, ["rich_1"], fake_agent(out, calls, reported={"rich_1": 2}), out, FAR, 60)
    run = load_run(out)
    assert check_run(run) == []
    assert summarize_run(run).loc[0, "calls_submit_patch"] == 1


def test_task_without_row_and_unfinished_tasks_are_flagged(tmp_path):
    out = tmp_path / "run"
    write_manifest(out, {"task_ids": ["fastapi_1", "rich_1", "rich_9"]})
    tasks = {t: {"instance_id": t} for t in ("fastapi_1", "rich_1")}
    collect_run(tasks, ["fastapi_1"], fake_agent(out), out, time.time() + 30, 60)
    collect_run(tasks, ["rich_1"], lambda task: 1 / 0, out, FAR, 60)
    assert check_run(load_run(out)) == [
        "rich_9: no result row",
        "fastapi_1: not run (deadline)",
        "rich_1: failed: ZeroDivisionError: division by zero",
    ]


def test_resumed_task_uses_its_latest_row(tmp_path):
    out = tmp_path / "run"
    tasks = {"rich_1": {"instance_id": "rich_1"}}
    collect_run(tasks, ["rich_1"], fake_agent(out), out, time.time() + 30, 60)
    collect_run(tasks, ["rich_1"], fake_agent(out), out, FAR, 60)
    run = load_run(out)
    assert len(run.rows) == 2
    assert check_run(run) == []


def test_summary_counts_tools_and_patch_length(run_dir):
    summary = summarize_run(load_run(run_dir)).set_index("instance_id")
    assert summary.loc["fastapi_1", "calls_read_file"] == 2
    assert summary.loc["fastapi_1", "calls_edit_file"] == 0
    assert summary.loc["rich_1", "calls_edit_file"] == 1
    assert summary.loc["rich_1", "patch_chars"] == 7
    assert summary.loc["rich_1", "trace_calls"] == 2
    assert not summary["context_limit"].any()


def test_context_limit_is_read_from_the_log(run_dir):
    (run_dir / "logs" / "rich_1.log").write_text("litellm.ContextWindowExceededError: input too long\n", encoding="utf-8")
    summary = summarize_run(load_run(run_dir)).set_index("instance_id")
    assert summary.loc["rich_1", "context_limit"]
    assert not summary.loc["fastapi_1", "context_limit"]


def test_cli_exits_0_on_a_clean_run_and_writes_csv(run_dir, tmp_path, capsys):
    csv = tmp_path / "summary.csv"
    assert main(["--run", str(run_dir), "--csv", str(csv)]) == 0
    assert "2/2 tasks ran, 1 resolved, 0 hit the context limit" in capsys.readouterr().out
    assert csv.read_text(encoding="utf-8").startswith("instance_id,")


def test_cli_exits_1_and_prints_problems(run_dir, capsys):
    (run_dir / "traces" / "trace_fastapi_1.json").unlink()
    assert main(["--run", str(run_dir)]) == 1
    assert "PROBLEM fastapi_1: no trace file" in capsys.readouterr().err



def test_run_with_no_results_reports_every_task(tmp_path, capsys):
    out = tmp_path / "run"
    write_manifest(out, {"run_id": "dead", "task_ids": ["rich_1"]})
    assert main(["--run", str(out)]) == 1
    captured = capsys.readouterr()
    assert "0/0 tasks ran" in captured.out
    assert "PROBLEM rich_1: no result row" in captured.err


def _shards(tmp_path, ids_per_shard: list[list[str]], run_id: str = "run1") -> list:
    """One run directory per shard, each holding only its own tasks (as `select_tasks` splits them)."""
    dirs = []
    for k, ids in enumerate(ids_per_shard):
        out = tmp_path / f"shard{k}"
        write_manifest(out, {"run_id": run_id, "shard": k, "task_ids": ids})
        collect_run({t: {"instance_id": t} for t in ids}, ids, fake_agent(out), out, FAR, 60)
        dirs.append(out)
    return dirs


def test_load_runs_merges_shards(tmp_path):
    run = load_runs(_shards(tmp_path, [["fastapi_1"], ["rich_1"]]))
    assert run.manifest["run_id"] == "run1" and run.manifest["task_ids"] == ["fastapi_1", "rich_1"]
    assert sorted(run.latest) == ["fastapi_1", "rich_1"]
    assert run.trace("rich_1") is not None
    assert check_run(run) == []


def test_load_runs_flags_tasks_no_shard_covered(tmp_path):
    dirs = _shards(tmp_path, [["fastapi_1"], ["rich_1"]])
    manifest = dirs[1] / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"rich_1"', '"rich_1", "rich_9"'), encoding="utf-8")
    assert check_run(load_runs(dirs)) == ["rich_9: no result row"]


def test_load_runs_flags_a_task_completed_in_two_shards(tmp_path):
    dirs = _shards(tmp_path, [["rich_1"], ["rich_1"]])
    assert check_run(load_runs(dirs)) == ["rich_1: 2 completed rows"]


def test_cli_accepts_repeated_run(tmp_path, capsys):
    dirs = _shards(tmp_path, [["fastapi_1"], ["rich_1"]])
    assert main(["--run", str(dirs[0]), "--run", str(dirs[1])]) == 0
    assert "run1: 2/2 tasks ran" in capsys.readouterr().out


def test_calls_that_never_reached_a_tool_are_not_counted(tmp_path):
    from test_escaping import call, rejected_edit, write_run

    calls = [
        call(0, "read_file", {"filepath": "a.py"}, filepath="a.py", content="x"),
        rejected_edit(1),
        call(2, "read_file", {"filepath": "a.py"}, status="error", error_type="BudgetExceeded"),
        call(3, "get_status", {}),
        call(4, "code_analyzer_agent", {}),
        call(5, "submit_patch", {}),
    ]
    run = load_run(write_run(tmp_path / "run", {"fastapi_1": calls}, reported={"fastapi_1": 1}))
    assert check_run(run) == []
