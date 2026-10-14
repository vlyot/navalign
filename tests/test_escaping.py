import json
from pathlib import Path

import pytest

from navalign.escaping import edit_failures, escaping_variants
from navalign.trace import ToolCall, Trace, load_trace

VAL1 = Path(__file__).parent.parent / "results" / "traces" / "val1" / "results" / "traces"
FILE_TEXT = 'def f():\n    return "a\\\\0b"\n'  # the file holds a docstring-style `\\0`


def call(index: int, name: str, arguments: dict, **observation) -> ToolCall:
    return ToolCall(index=index, step_id=index + 2, name=name, arguments=arguments,
                    output=json.dumps({"status": "ok"} | observation))


def read(index: int, content: str, path: str = "pkg/mod.py") -> ToolCall:
    return call(index, "read_file", {"filepath": path}, filepath=path, content=content, start_line=1, end_line=2)


def failed_edit(index: int, old: str, path: str = "pkg/mod.py") -> ToolCall:
    return call(index, "edit_file", {"filepath": path, "old_string": old, "new_string": "x"},
                status="error", error_type="FileEditError", error_message="old_string not found")


def labels(calls: list[ToolCall]) -> list[str]:
    return [r["label"] for r in edit_failures(Trace("ATIF-v1.7", {}, calls), "t")]


def test_variants_undo_one_and_two_levels_and_double_backslashes():
    assert escaping_variants('say \\\\\\"hi\\\\\\"') == ['say \\"hi\\"', 'say "hi"', 'say \\\\\\\\\\\\"hi\\\\\\\\\\\\"']
    assert "line1\nline2" in escaping_variants("line1\\nline2")
    assert escaping_variants("plain") == []


def test_escaped_newlines_and_quotes_are_labelled_escaping():
    text = 'x = "a"\nprint(x)\n'
    assert labels([read(0, text), failed_edit(1, 'x = \\"a\\"\\nprint(x)')]) == ["escaping"]


def test_dropped_backslash_level_is_labelled_escaping():
    assert labels([read(0, FILE_TEXT), failed_edit(1, '    return "a\\0b"')]) == ["escaping"]


def test_old_string_present_in_read_text_is_labelled_stale():
    assert labels([read(0, FILE_TEXT), failed_edit(1, "def f():")]) == ["stale"]


def test_no_known_text_is_labelled_no_match():
    rows = edit_failures(Trace("ATIF-v1.7", {}, [failed_edit(0, "anything")]), "t")
    assert [(r["label"], r["known_text"]) for r in rows] == [("no_match", False)]


def test_text_from_other_files_is_ignored():
    assert labels([read(0, FILE_TEXT, "other.py"), failed_edit(1, "def f():")]) == ["no_match"]


def test_write_file_content_counts_as_known_text():
    write = call(0, "write_file", {"filepath": "scratch.py", "content": "a = 1\nb = 2\n"}, filepath="scratch.py")
    assert labels([write, failed_edit(1, "a = 1", "scratch.py")]) == ["stale"]


def test_successful_edits_are_not_reported():
    ok = call(0, "edit_file", {"filepath": "pkg/mod.py", "old_string": "a", "new_string": "b"}, filepath="pkg/mod.py")
    assert labels([ok]) == []


REJECTED = ("Invoking `edit_file()` failed as the following mandatory input parameters are not present:\n"
            "old_string\nYou could retry calling this tool")


def rejected_edit(index: int) -> ToolCall:
    """An edit the agent framework rejected: the model garbled the `old_string` key (run 1)."""
    return ToolCall(index=index, step_id=index + 2, name="edit_file",
                    arguments={"filepath": "pkg/mod.py", "new_string": "x", "if errors": "y"},
                    output=json.dumps({"error": REJECTED}))


def test_rejected_arguments_are_labelled_malformed():
    assert labels([read(0, FILE_TEXT), rejected_edit(1)]) == ["malformed"]


def test_budget_blocked_edits_are_not_repair_failures():
    blocked = call(1, "edit_file", {"filepath": "pkg/mod.py", "old_string": "a", "new_string": "b"},
                   status="error", error_type="BudgetExceeded", error_message="Tool call budget exhausted (50 calls)")
    assert labels([read(0, FILE_TEXT), blocked]) == []


def test_val1_fastapi_15588_failures_are_escaping():
    trace = VAL1 / "trace_fastapi_15588.json"
    if not trace.exists():
        pytest.skip("val1 traces not present")
    rows = edit_failures(load_trace(trace), "fastapi_15588")
    assert [(r["index"], r["label"]) for r in rows] == [(10, "escaping"), (12, "escaping")]


def write_run(out: Path, traces: dict[str, list[ToolCall]], reported: dict[str, int] | None = None) -> Path:
    """A run directory the way collect_run and the harness leave it, with ATIF traces built from `traces`.

    The result row's `tool_calls` is `reported[tid]` if given, else the number of calls.
    """
    (out / "results" / "traces").mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"run_id": "t", "task_ids": sorted(traces)}), encoding="utf-8")
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for tid, calls in traces.items():
            n = (reported or {}).get(tid, len(calls))
            f.write(json.dumps({"instance_id": tid, "tool_calls": n, "error": ""}) + "\n")
            steps = [{"step_id": c.step_id, "observation": {"content": c.output},
                      "tool_calls": [{"function_name": c.name, "arguments": c.arguments, "tool_call_id": f"c{c.index}"}]}
                     for c in calls]
            trace = {"schema_version": "ATIF-v1.7", "agent": {}, "steps": steps}
            (out / "results" / "traces" / f"trace_{tid}.json").write_text(json.dumps(trace), encoding="utf-8")
    return out


def test_cli_labels_failed_edits_and_writes_csv(tmp_path, capsys):
    from navalign.escaping import main

    run = write_run(tmp_path / "run", {"pkg_1": [read(0, FILE_TEXT), failed_edit(1, '    return "a\\0b"'), failed_edit(2, "def f():")]})
    csv = tmp_path / "failures.csv"
    assert main(["--run", str(run), "--csv", str(csv)]) == 0
    assert "2 failed edits in 1 tasks: escaping 1, stale 1, no_match 0" in capsys.readouterr().out
    assert csv.read_text(encoding="utf-8").startswith("instance_id,index,file,label")
