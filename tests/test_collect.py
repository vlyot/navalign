import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from navalign.collect import SKIPPED, collect_run, load_done, read_results, result_record, select_tasks, write_manifest

IDS = ["rich_2", "fastapi_1", "requests_3", "rich_1", "fastapi_2"]
TASKS = {tid: {"instance_id": tid} for tid in IDS}
FAR = time.time() + 3600


@dataclass
class FakeResult:
    resolved: bool
    test_exit_code: int
    agent_patch: str
    tool_calls: list
    duration_seconds: float
    instance_id: str = ""
    model: str = "main_lora"


def fake_agent(trace_dir: Path | None = None, fail: set[str] = frozenset()):
    def run_one(task):
        tid = task["instance_id"]
        if tid in fail:
            raise RuntimeError("vLLM died")
        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"trace_{tid}.json").write_text("{}", encoding="utf-8")
        return {"resolved": False, "test_exit_code": 1, "agent_patch": "diff", "tool_calls": [{}, {}], "duration_seconds": 1.5}
    return run_one


def test_shards_are_disjoint_and_cover_every_id():
    shards = [select_tasks(IDS, s, 3) for s in range(3)]
    flat = [tid for shard in shards for tid in shard]
    assert sorted(flat) == sorted(IDS)
    assert len(flat) == len(set(flat))


def test_select_tasks_ignores_input_order():
    assert select_tasks(IDS, 1, 2) == select_tasks(list(reversed(IDS)), 1, 2)


def test_select_tasks_drops_done_ids():
    assert select_tasks(IDS, done={"fastapi_1", "rich_1"}) == ["fastapi_2", "requests_3", "rich_2"]


def test_select_tasks_rejects_bad_shard():
    with pytest.raises(ValueError):
        select_tasks(IDS, 2, 2)


def test_load_done_retries_error_rows(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [{"instance_id": "a", "error": ""}, {"instance_id": "b", "error": SKIPPED}, {"instance_id": "c", "error": "RuntimeError: x"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert load_done(path) == {"a"}
    assert load_done(path, retry_errors=False) == {"a", "b", "c"}
    assert load_done(tmp_path / "missing.jsonl") == set()


def test_result_record_accepts_dict_and_object():
    fields = dict(resolved=True, test_exit_code=0, agent_patch="p", tool_calls=[1, 2, 3], duration_seconds=9.0)
    from_obj = result_record("t", FakeResult(**fields, instance_id="t"), 1.0, 2.0)
    from_dict = result_record("t", fields | {"model": "main_lora"}, 1.0, 2.0)
    assert from_obj == from_dict
    assert from_obj["tool_calls"] == 3
    assert from_obj["extra"] == {"model": "main_lora"}
    assert from_obj["error"] == "" and from_obj["started"] == 1.0


def test_result_record_keeps_integer_tool_calls():
    assert result_record("t", {"tool_calls": 10}, 0, 0)["tool_calls"] == 10


def test_unserialisable_values_are_stringified(tmp_path):
    def run_one(task):
        return {"resolved": False, "tool_calls": 0, "when": object()}
    collect_run(TASKS, ["rich_1"], run_one, tmp_path, FAR, 60)
    (row,) = read_results(tmp_path / "results.jsonl")
    assert row["extra"]["when"].startswith("<object object")


def test_collect_run_writes_one_row_per_task_in_order(tmp_path):
    ids = ["rich_1", "fastapi_1"]
    rows = collect_run(TASKS, ids, fake_agent(), tmp_path, FAR, 60)
    assert [r["instance_id"] for r in read_results(tmp_path / "results.jsonl")] == ids
    assert rows[0]["tool_calls"] == 2 and rows[0]["error"] == ""


def test_exception_is_recorded_and_run_continues(tmp_path):
    rows = collect_run(TASKS, ["rich_1", "rich_2"], fake_agent(fail={"rich_1"}), tmp_path, FAR, 60)
    assert rows[0]["error"] == "RuntimeError: vLLM died"
    assert rows[1]["error"] == ""


def test_tasks_past_the_deadline_are_skipped(tmp_path):
    calls = []

    def run_one(task):
        calls.append(task["instance_id"])
        return {}
    rows = collect_run(TASKS, ["rich_1", "rich_2"], run_one, tmp_path, time.time() + 30, 60)
    assert calls == []
    assert [r["error"] for r in rows] == [SKIPPED, SKIPPED]
    assert load_done(tmp_path / "results.jsonl") == set()


def test_collect_run_appends_to_existing_results(tmp_path):
    collect_run(TASKS, ["rich_1"], fake_agent(), tmp_path, FAR, 60)
    collect_run(TASKS, ["rich_2"], fake_agent(), tmp_path, FAR, 60)
    assert [r["instance_id"] for r in read_results(tmp_path / "results.jsonl")] == ["rich_1", "rich_2"]


def test_trace_found_follows_the_trace_file(tmp_path):
    traces = tmp_path / "traces"
    rows = collect_run(TASKS, ["rich_1"], fake_agent(traces), tmp_path, FAR, 60)
    rows += collect_run(TASKS, ["rich_2"], fake_agent(), tmp_path, FAR, 60)
    assert [r["trace_found"] for r in rows] == [True, False]


def test_write_manifest(tmp_path):
    path = write_manifest(tmp_path / "run", {"run_id": "val1", "task_ids": ["rich_1"], "sampling": {"max_output_tokens": 8192}})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "val1"
    assert manifest["task_ids"] == ["rich_1"]
    assert manifest["sampling"]["max_output_tokens"] == 8192
    assert "written_at" in manifest and "navalign_version" in manifest
