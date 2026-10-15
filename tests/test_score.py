import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from test_escaping import write_run
from test_explore import call, run

from navalign.build import main as build_main
from navalign.data import example_dir
from navalign.explore import ReadSpan, explore_trace
from navalign.score import (
    GroundTruth,
    describe_score,
    load_groundtruth,
    main,
    read_nodes,
    score_exploration,
    score_one,
    score_task,
)
from navalign.trace import Trace, load_trace

ROOT = Path(__file__).parent.parent
GROUNDTRUTH = ROOT / "results" / "groundtruth"
VAL1 = ROOT / "results" / "traces" / "val1"
RUN1 = ROOT / "results" / "traces" / "run1"
DATA = ROOT / "data"

TASK = "pkg_1"
TARGETS = {
    "instance_id": TASK, "files": ["pkg/a.py", "pkg/flat.py"], "target_nodes": ["pkg.a.f"],
    "targets": [{"file": "pkg/a.py", "rule": "exact", "node": "pkg.a.f"}, {"file": "pkg/flat.py", "rule": "file", "node": None}],
}
SPANS = {
    "instance_id": TASK, "n_nodes": 4, "unlocated": [], "skipped_files": [],
    "spans": {"pkg/a.py": [["pkg.a.f", 10, 20], ["pkg.a.g", 30, 40]], "pkg/b.py": [["pkg.b.h", 1, 50]],
              "pkg/flat.py": [["pkg.flat.k", 5, 9]]},
}
DISTANCES = {"instance_id": TASK, "graph_file": "g.json", "targets": ["pkg.a.f"], "cap": 5, "n_nodes": 4,
             "distances": {"pkg.a.f": 0, "pkg.a.g": 1, "pkg.b.h": 2}}
FILES = {"instance_id": TASK, "files": ["README.md", "pkg/a.py", "pkg/b.py", "pkg/flat.py"]}
ROWS = {"targets": TARGETS, "spans": SPANS, "distances": DISTANCES, "files": FILES}


def ground_truth(**targets) -> GroundTruth:
    return GroundTruth.from_rows(TARGETS | targets, SPANS, DISTANCES, FILES)


def read(index: int, path: str, start: int = 1, end: int = 150):
    return call(index, "read_file", {"filepath": path}, filepath=path, start_line=start, end_line=end)


def score(calls, gt: GroundTruth | None = None):
    gt = gt or ground_truth()
    return score_exploration(explore_trace(Trace("ATIF-v1.7", {}, calls), gt.snapshot_files, gt.target_files, TASK), gt)


def test_s_file_is_one_over_k_of_the_first_target_read():
    s = score([run(0, "grep -rn x pkg"), read(1, "pkg/b.py"), read(2, "pkg/a.py")])
    assert (s.k, s.s_file) == (3, pytest.approx(1 / 3))


def test_no_target_read_gives_zero_and_censored_k():
    s = score([read(0, "pkg/b.py"), run(1, "ls")])
    assert (s.k, s.s_file) == (None, 0.0)


def test_reads_after_the_window_are_ignored():
    s = score([read(0, "pkg/b.py"), call(1, "edit_file", {"filepath": "pkg/a.py"}), read(2, "pkg/a.py")])
    assert (s.window_end, s.k, s.read_calls) == (1, None, 1)


def test_focus_counts_read_calls_only():
    s = score([run(0, "grep -rn x pkg"), read(1, "pkg/a.py"), read(2, "pkg/b.py"), run(3, "ls")])
    assert (s.read_calls, s.focus) == (2, 0.5)


def test_scratch_file_reads_count_toward_neither_focus_nor_s_node():
    s = score([call(0, "write_file", {"filepath": "repro.py", "content": "x"}, filepath="repro.py"),
               run(1, "cat repro.py"), read(2, "pkg/a.py", 10, 20)])
    assert (s.read_calls, s.focus, s.s_node) == (1, 1.0, 1.0)
    assert s.steps[1].files == [] and s.steps[1].d is None


def test_target_node_read_gives_d_zero_and_k_node():
    s = score([read(0, "pkg/a.py", 30, 40), read(1, "pkg/a.py", 1, 20)])
    assert [(st.target_node, st.d) for st in s.steps] == [(False, 1), (True, 0)]
    assert (s.k, s.k_node) == (1, 2)


@pytest.mark.parametrize("calls", [[read(0, "README.md")], [read(0, "pkg/a.py", 1, 5)]])
def test_read_showing_no_node_gives_the_cap(calls):
    assert score(calls).steps[0].d == 5


def test_any_read_of_a_target_file_without_target_nodes_gives_d_zero():
    s = score([read(0, "pkg/flat.py", 100, 150)])  # pkg.flat.k is unreachable, and not even shown
    assert (s.steps[0].d, s.steps[0].target_node, s.s_node) == (0, False, 1.0)


def test_call_with_several_reads_takes_the_smallest_d():
    assert score([run(0, "cat pkg/b.py README.md")]).steps[0].d == 2


def test_s_node_is_one_minus_mean_d_over_cap():
    s = score([read(0, "pkg/b.py"), run(1, "grep -rn x ."), read(2, "pkg/a.py", 30, 40)])
    assert (s.mean_d, s.s_node, s.cap) == (1.5, pytest.approx(0.7), 5)


def test_task_without_target_nodes_has_no_node_level():
    s = score([read(0, "pkg/a.py", 10, 20)], ground_truth(target_nodes=[], targets=[{"file": "pkg/a.py", "rule": "file"}]))
    assert (s.s_node, s.k_node, s.cap, s.steps[0].d) == (None, None, None, None)
    assert s.s_file == 1.0


def test_new_file_only_task_has_no_file_score_and_reports_created_target():
    gt = ground_truth(files=["pkg/new.py"], target_nodes=[], targets=[{"file": "pkg/new.py", "rule": "new_file"}])
    s = score([read(0, "pkg/a.py"), call(1, "write_file", {"filepath": "pkg/new.py", "content": "x"}, filepath="pkg/new.py")], gt)
    assert (s.k, s.s_file, s.focus, s.created_target) == (None, None, None, True)


def test_k_seen_counts_files_in_search_results():
    s = score([run(0, "grep -rn f pkg", "pkg/a.py:12:def f():\n"), read(1, "pkg/a.py")])
    assert (s.k_seen, s.k) == (1, 2)


@pytest.mark.parametrize("span, nodes", [
    (ReadSpan("pkg/a.py", 15, 32), {"pkg.a.f", "pkg.a.g"}),
    (ReadSpan("pkg/a.py", 21, 29), set()),
    (ReadSpan("pkg/a.py"), {"pkg.a.f", "pkg.a.g"}),
    (ReadSpan("pkg/a.py", last=12), {"pkg.a.g"}),
    (ReadSpan("pkg/a.py", last=25), {"pkg.a.f", "pkg.a.g"}),
    (ReadSpan("README.md", last=5), set()),
])
def test_read_nodes(span, nodes):
    assert read_nodes(span, ground_truth().spans) == nodes


def _write_groundtruth(directory: Path, rows: dict[str, list[dict]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, items in rows.items():
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")
    return directory


def test_load_groundtruth_reads_the_build_output(tmp_path):
    other = {name: row | {"instance_id": "pkg_2"} for name, row in ROWS.items()}
    gt_dir = _write_groundtruth(tmp_path, {name: [ROWS[name], other[name]] if name != "files" else [ROWS[name]] for name in ROWS})
    gts = load_groundtruth(gt_dir)
    assert list(gts) == [TASK]  # pkg_2 has no files.jsonl row
    gt = gts[TASK]
    assert (gt.target_files, gt.target_nodes, gt.flat_files, gt.new_file_only) == (
        {"pkg/a.py", "pkg/flat.py"}, {"pkg.a.f"}, {"pkg/flat.py"}, False)
    assert gt.distances.cap == 5 and gt.spans.nodes_in("pkg/b.py") == {"pkg.b.h"}
    assert load_groundtruth(gt_dir, ["nope"]) == {}


def test_cli_writes_scores_and_steps(tmp_path, capsys):
    run_dir = write_run(tmp_path / "run", {TASK: [read(0, "pkg/b.py"), read(1, "pkg/a.py", 10, 20)], "pkg_9": [read(0, "x.py")]})
    gt_dir = _write_groundtruth(tmp_path / "gt", {name: [row] for name, row in ROWS.items()})
    out, steps = tmp_path / "out" / "scores.csv", tmp_path / "out" / "steps.jsonl"
    assert main(["--run", str(run_dir), "--groundtruth", str(gt_dir), "--out", str(out), "--steps", str(steps)]) == 1
    assert "PROBLEM pkg_9: no ground truth" in capsys.readouterr().err
    (row,) = pd.read_csv(out).to_dict("records")
    assert (row["instance_id"], row["k"], row["s_file"], row["s_node"]) == (TASK, 2, 0.5, pytest.approx(0.8))
    (line,) = steps.read_text(encoding="utf-8").splitlines()
    assert [s["d"] for s in json.loads(line)["steps"]] == [2, 0]


# --- the bundled toy example ----------------------------------------------------------------

TOY = example_dir()
TOY_TRACE = TOY / "traces" / "trace_toyshop_1.json"
TOY_RAW = {"tasks": TOY / "tasks.jsonl", "graphs": TOY / "graphs", "snapshots": TOY / "snapshots"}


@pytest.fixture
def toy_groundtruth(tmp_path):
    assert build_main(["--data", str(TOY), "--out", str(tmp_path / "gt")]) == 0
    return tmp_path / "gt"


def test_score_one_from_groundtruth_equals_raw_data(toy_groundtruth):
    assert score_one(TOY_TRACE, "toyshop_1", toy_groundtruth) == score_one(TOY_TRACE, "toyshop_1", **TOY_RAW)


@pytest.mark.parametrize("source", ["groundtruth", "raw"])
def test_score_one_names_an_unknown_task(source, toy_groundtruth):
    kwargs = {"groundtruth": toy_groundtruth} if source == "groundtruth" else TOY_RAW
    with pytest.raises(KeyError, match="toyshop_9 is not in"):
        score_one(TOY_TRACE, "toyshop_9", **kwargs)


def test_describe_score_names_the_first_target_read_and_s_file():
    text = describe_score(score_one(TOY_TRACE, "toyshop_1", **TOY_RAW))
    assert text.startswith("toyshop_1: 9 tool calls; the exploration window is calls 1-6")
    assert "first read of a fixed file   call 3     S_file 0.333" in text
    assert "S_node 0.800  (cap 5 hops)" in text


def test_describe_score_says_never_when_no_target_file_is_read():
    text = describe_score(score([read(0, "pkg/b.py"), run(1, "ls")]))
    assert "first read of a fixed file   never" in text
    assert "the exploration window is the whole trace" in text


def test_describe_score_steps_lists_the_window_calls():
    lines = describe_score(score_one(TOY_TRACE, "toyshop_1", **TOY_RAW), steps=True).splitlines()
    assert lines[lines.index("  call  tool         files read          fixed file  fixed code  d") + 4] == \
        "  4     read_file    toyshop/pricing.py  yes         yes         0"


def test_cli_scores_a_plain_trace_folder(tmp_path, toy_groundtruth, capsys):
    traces = tmp_path / "results" / "traces"
    traces.mkdir(parents=True)
    shutil.copy(TOY_TRACE, traces)
    assert main(["--run", str(tmp_path / "results"), "--groundtruth", str(toy_groundtruth)]) == 0
    assert "1 tasks: S_file mean 0.333 over 1" in capsys.readouterr().out


# --- real traces (git-ignored; skipped when absent) ---------------------------------------

def _need(*paths: Path):
    if not all(p.exists() for p in paths):
        pytest.skip("ground truth, traces or dev data not present")


@pytest.mark.parametrize("tid, k, s_file", [
    ("fastapi_14303", 3, 1 / 3), ("fastapi_15588", 4, 1 / 4), ("requests_7427", 2, 1 / 2), ("rich_3944", None, 0.0),
])
def test_val1_k_matches_hand_notes(tid, k, s_file):
    trace = VAL1 / "results" / "traces" / f"trace_{tid}.json"
    _need(trace, GROUNDTRUTH / "files.jsonl")
    gt = load_groundtruth(GROUNDTRUTH, [tid])[tid]
    s = score_exploration(explore_trace(load_trace(trace), gt.snapshot_files, gt.target_files, tid), gt)
    assert (s.k, s.s_file) == (k, pytest.approx(s_file))


def test_val1_new_file_task_is_scored_by_created_target():
    tid = "fastapi_15661"
    trace = VAL1 / "results" / "traces" / f"trace_{tid}.json"
    _need(trace, GROUNDTRUTH / "files.jsonl")
    gt = load_groundtruth(GROUNDTRUTH, [tid])[tid]
    s = score_exploration(explore_trace(load_trace(trace), gt.snapshot_files, gt.target_files, tid), gt)
    assert (s.s_file, s.created_target) == (None, True)


def test_run1_k_agrees_with_explore_first_target_read():
    from navalign.runs import load_runs
    from navalign.score import explore_run

    shards = [RUN1 / f"shard{i}" for i in range(3)]
    _need(*shards, GROUNDTRUTH / "files.jsonl")
    run1 = load_runs(shards)
    gts = load_groundtruth(GROUNDTRUTH, run1.latest)
    explorations, problems = explore_run(run1, gts)
    assert problems == [] and len(explorations) == 129
    for tid, ex in explorations.items():
        gt = gts[tid]
        first = next((e.index for e in ex.window if any(r.file in gt.target_files for r in e.reads)), None)
        k = score_exploration(ex, gt).k
        assert k == (None if first is None or gt.new_file_only else first + 1), tid


def test_score_task_matches_the_prebuilt_ground_truth():
    tid = "requests_7427"
    trace, snapshot = VAL1 / "results" / "traces" / f"trace_{tid}.json", DATA / "snapshots" / f"{tid}.tgz"
    _need(trace, snapshot, GROUNDTRUTH / "files.jsonl")
    from navalign.data import load_tasks

    one_shot = score_task(load_tasks(DATA / "tasks.jsonl")[tid], DATA / "graphs", snapshot, load_trace(trace))
    gt = load_groundtruth(GROUNDTRUTH, [tid])[tid]
    prebuilt = score_exploration(explore_trace(load_trace(trace), gt.snapshot_files, gt.target_files, tid), gt)
    assert one_shot == prebuilt


def test_prefix_scores_cover_only_the_first_n_calls():
    from navalign.score import prefix_scores

    calls = [read(0, "pkg/a.py", 10, 20), read(1, "pkg/b.py"), read(2, "pkg/b.py"), read(3, "pkg/a.py", 30, 40)]
    s = score(calls)
    assert prefix_scores(s, 2) == {"s_file_prefix": 1.0, "focus_prefix": 0.5, "s_node_prefix": pytest.approx(1 - 1 / 5)}
    assert prefix_scores(s, 10)["focus_prefix"] == s.focus == 0.5


def test_prefix_scores_censor_a_first_read_after_n():
    from navalign.score import prefix_scores

    s = score([run(0, "ls"), run(1, "ls"), read(2, "pkg/a.py")])
    assert prefix_scores(s, 2) == {"s_file_prefix": 0.0, "focus_prefix": None, "s_node_prefix": None}
    assert prefix_scores(s, 3)["s_file_prefix"] == pytest.approx(1 / 3)


def test_prefix_scores_are_undefined_for_new_file_only_tasks():
    from navalign.score import prefix_scores

    gt = ground_truth(files=["pkg/new.py"], target_nodes=[], targets=[{"file": "pkg/new.py", "rule": "new_file"}])
    assert set(prefix_scores(score([read(0, "pkg/a.py")], gt), 10).values()) == {None}
