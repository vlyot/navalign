import json
from pathlib import Path

from navalign.trace import parse_trace

FIXTURE = Path(__file__).parent.parent / "fixtures" / "trace_step_example.json"


def _trace_from_fixture() -> dict:
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    system = {"step_id": 1, "source": "system", "message": "sys", "tool_calls": None, "observation": None}
    return {
        "schema_version": fx["trace_top_level"]["schema_version"],
        "agent": fx["trace_top_level"]["agent"],
        "steps": [system, fx["step"]],
        "final_metrics": fx["final_metrics_fastapi_15588"],
    }


def test_parse_fixture_step():
    trace = parse_trace(_trace_from_fixture())
    assert trace.schema_version == "ATIF-v1.7"
    assert trace.n_steps == 2
    (call,) = trace.calls
    assert call.index == 0
    assert call.name == "search_similar_code"
    assert call.arguments == {"query": "Server Sent Events"}
    assert call.author == "swe_baseline_agent"
    assert json.loads(call.output)["count"] == 0


def test_results_list_matched_by_call_id():
    step = {
        "step_id": 3,
        "tool_calls": [
            {"tool_call_id": "a", "function_name": "read_file", "arguments": '{"path": "x.py"}'},
            {"tool_call_id": "b", "function_name": "read_file", "arguments": {"path": "y.py"}},
        ],
        "observation": {"results": [{"source_call_id": "b", "content": "Y"}, {"source_call_id": "a", "content": "X"}]},
    }
    trace = parse_trace({"steps": [step]})
    assert [(c.arguments["path"], c.output) for c in trace.calls] == [("x.py", "X"), ("y.py", "Y")]
