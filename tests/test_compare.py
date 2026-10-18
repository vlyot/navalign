import json
import math
from pathlib import Path

import pandas as pd

from navalign.compare import main, paired, shared_prefix, summary, transitions
from navalign.runs import load_run

COLUMNS = ["instance_id", "category", "gradable", "tests_ran", "passed", "found", "n_calls", "window_end",
           "k", "s_file", "focus", "k_node", "s_node"]


def outcomes(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


A = outcomes([
    ("t1", "D", True, True, True, True, 12, 4, 1, 1.0, 1.0, 1, 1.0),
    ("t2", "B", True, True, False, True, 30, 30, 2, 0.5, 0.5, 2, 0.8),
    ("t3", "A", False, True, False, False, 20, 20, None, 0.0, 0.0, None, None),
])
B = outcomes([
    ("t1", "B", True, True, False, True, 13, 13, 1, 1.0, 0.8, 1, 1.0),
    ("t2", "B", True, True, False, True, 12, 12, 3, 1 / 3, 0.2, 3, 0.6),
    ("t3", "A", False, True, True, False, 11, 11, None, 0.0, 0.0, None, None),
])
T4 = pd.DataFrame([
    ("edited a target file (C/D vs B)", "s_file", 0.75, 0.6, 0.9, 2, 1),
    ("edited a target file (C/D vs B)", "focus", 0.5, 0.4, 0.6, 2, 1),
], columns=["outcome", "predictor", "auc", "ci_low", "ci_high", "n", "n_positive"])


def write_run(root: Path, budget: int, rows: list[dict], calls: dict[str, list[str]]) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"evaluation": {"max_tool_calls": budget}}), encoding="utf-8")
    (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for tid, files in calls.items():
        steps = [{"step_id": i, "tool_calls": [{"tool_call_id": str(i), "function_name": "read_file",
                                                "arguments": {"filepath": f}}],
                  "observation": {"content": "{}"}} for i, f in enumerate(files)]
        (root / f"trace_{tid}.json").write_text(json.dumps({"steps": steps}), encoding="utf-8")
    return root


def runs(tmp_path: Path):
    a = write_run(tmp_path / "ra", 50, [
        {"instance_id": "t1", "resolved": True, "agent_patch": "diff", "tool_calls": 12},
        {"instance_id": "t2", "resolved": False, "agent_patch": "", "tool_calls": 50},
        {"instance_id": "t3", "resolved": False, "agent_patch": "diff", "tool_calls": 20},
    ], {"t1": ["x.py", "y.py", "z.py"], "t2": ["x.py"], "t3": ["a.py"]})
    b = write_run(tmp_path / "rb", 10, [
        {"instance_id": "t1", "resolved": False, "agent_patch": "", "tool_calls": 10},
        {"instance_id": "t2", "resolved": False, "agent_patch": "", "tool_calls": 10},
        {"instance_id": "t3", "resolved": True, "agent_patch": "diff", "tool_calls": 9},
    ], {"t1": ["x.py", "y.py", "q.py"], "t2": ["y.py"], "t3": ["a.py"]})
    return a, b


def test_shared_prefix():
    assert shared_prefix([1, 2, 3], [1, 2, 4, 5]) == 2
    assert shared_prefix([], [1]) == 0
    assert shared_prefix([1, 2], [1, 2]) == 2


def test_summary_counts_outcomes_budget_and_aucs(tmp_path):
    a, _ = runs(tmp_path)
    s = summary(A, T4, load_run(a))
    assert (s["tasks"], s["call budget"], s["resolved (harness)"], s["non-empty patch"]) == (3, 50, 1, 2)
    assert (s["passed (regraded)"], s["passed, gradable tasks whose tests ran"], s["gradable tasks whose tests ran"]) == (1, 1, 2)
    assert (s["category A"], s["category B"], s["category D"], s["category C1"]) == (1, 1, 1, 0)
    assert (s["runs at the call budget"], s["B runs at the call budget"]) == (1, 1)
    assert s["mean s_file"] == 0.5 and s["n s_node"] == 2 and s["median k"] == 1.5
    assert s["AUC s_file -> edited a target file (C/D vs B)"] == 0.75
    assert not any("focus ->" in k for k in s)  # only S_file and -k_node AUCs are compared


def test_paired_agreement(tmp_path):
    a, b = runs(tmp_path)
    p = paired(A, B, load_run(a), load_run(b))
    assert p["tasks in both runs"] == 3 and p["S_file rank correlation n"] == 3
    assert math.isclose(p["S_file rank correlation"], 1.0)  # same order: t1 > t2 > t3 in both
    assert (p["same k"], p["found a target file in both"], p["found in a only"]) == (1, 2, 0)
    assert (p["trace pairs"], p["same first call"], p["max shared leading calls"]) == (3, 2, 2)


def test_transitions():
    t = transitions(A, B, ("a", "b"))
    assert t.loc["D", "B"] == 1 and t.loc["B", "B"] == 1 and t.loc["A", "A"] == 1
    assert int(t.to_numpy().sum()) == 3 and list(t.index) == list(t.columns)


def test_cli_writes_the_three_tables(tmp_path, capsys):
    a, b = runs(tmp_path)
    for name, o in (("run1", A), ("run2", B)):
        (tmp_path / name).mkdir()
        o.to_csv(tmp_path / name / "outcomes.csv", index=False)
        T4.to_csv(tmp_path / name / "t4_predictive.csv", index=False)
    out = tmp_path / "compare"
    assert main(["--a", str(tmp_path / "run1"), "--a-run", str(a), "--b", str(tmp_path / "run2"),
                 "--b-run", str(b), "--out", str(out)]) == 0
    table = pd.read_csv(out / "compare.csv", index_col="measure")
    assert list(table.columns) == ["run1", "run2"] and table.loc["call budget", "run2"] == 10
    assert pd.read_csv(out / "paired.csv", index_col="measure").loc["same first call", "value"] == 2
    assert (out / "transitions.csv").exists()
    assert "Paired tasks (a = run1, b = run2)" in capsys.readouterr().out
