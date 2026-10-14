import json
import tarfile
from pathlib import Path

import pytest

from navalign.data import load_jsonl
from navalign.explore import (
    classify_call,
    exploration_window,
    explore_trace,
    normalize_path,
    parse_command,
    strip_heredocs,
)
from navalign.trace import ToolCall, Trace, load_trace

ROOT = Path(__file__).parent.parent
VAL1 = ROOT / "results" / "traces" / "val1"
SNAPSHOTS = ROOT / "data" / "snapshots"
TARGETS = ROOT / "results" / "groundtruth" / "targets.jsonl"


def call(index: int, name: str, arguments: dict | None = None, **observation) -> ToolCall:
    obs = {"status": "ok"} | observation
    return ToolCall(index=index, step_id=index + 2, name=name, arguments=arguments or {}, output=json.dumps(obs))


def run(index: int, command: str, stdout: str = "", status: str = "ok") -> ToolCall:
    return call(index, "run_command", {"command": command}, status=status, stdout=stdout)


@pytest.mark.parametrize("raw, expected", [
    ("/tmp/swegemma_sandbox_ab12/workspace/src/requests/utils.py", "src/requests/utils.py"),
    ("/workspace/fastapi/sse.py", "fastapi/sse.py"),
    ("./rich/cells.py", "rich/cells.py"),
    ("docs/workspace/notes.md", "docs/workspace/notes.md"),
    ("/usr/lib/python3/os.py", None),
    ("../outside.py", None),
    ("..", None),
    (".", None),
])
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


def test_read_file_range_comes_from_observation_not_arguments():
    c = call(0, "read_file", {"filepath": "src/requests/utils.py", "end_line": 900, 'start_line"': 810},
             filepath="src/requests/utils.py", start_line=1, end_line=150, is_truncated=True, total_lines=1153)
    (read,) = classify_call(c).reads
    assert (read.file, read.start, read.end, read.truncated) == ("src/requests/utils.py", 1, 150, True)


def test_failed_read_file_produces_no_read():
    c = call(0, "read_file", {"filepath": "missing.py"}, status="error", error_message="not found")
    assert classify_call(c).reads == []


@pytest.mark.parametrize("command, expected", [
    ("cat rich/cells.py", [("rich/cells.py", None, None, None)]),
    ("head -n 40 a.py", [("a.py", 1, 40, None)]),
    ("head -20 a.py", [("a.py", 1, 20, None)]),
    ("tail -n 5 a.py", [("a.py", None, None, 5)]),
    ("sed -n '10,30p' a.py", [("a.py", 10, 30, None)]),
    ("nl -ba ./a.py | head -n 50", [("a.py", None, None, None)]),
])
def test_shell_reads(command, expected):
    reads, _, _ = parse_command(command)
    assert [(r.file, r.start, r.end, r.last) for r in reads] == expected


def test_grep_recursive_output_yields_seen_files():
    out = "./rich/_export_format.py:12:_SVG_FONT_FAMILY = 1\n./rich/console.py:42:from x import y\nBinary file x.pyc matches\n"
    _, seen, _ = parse_command('grep -rn "_SVG" .', out)
    assert seen == {"rich/_export_format.py", "rich/console.py"}


def test_grep_single_file_uses_file_argument_not_line_numbers():
    _, seen, _ = parse_command('grep -n "def should_bypass_proxies" src/requests/utils.py', "810:def should_bypass_proxies(url):\n")
    assert seen == {"src/requests/utils.py"}


def test_ls_names_are_joined_to_directory_argument():
    _, seen, _ = parse_command("ls scripts/", "docs.py\nlint.sh\n")
    assert seen == {"scripts/docs.py", "scripts/lint.sh"}
    _, seen, _ = parse_command("ls pyproject.toml", "pyproject.toml\n")
    assert seen == {"pyproject.toml"}


def test_find_output_yields_seen_files():
    _, seen, _ = parse_command('find . -name "*release*"', "./scripts/add_latest_release_date.py\n./docs/en/docs/release-notes.md\n")
    assert seen == {"scripts/add_latest_release_date.py", "docs/en/docs/release-notes.md"}


@pytest.mark.parametrize("command, edited", [
    ("sed -i 's/a/b/' fastapi/x.py", {"fastapi/x.py"}),
    ("echo hi > notes.txt", {"notes.txt"}),
    ("printf x >> a.py && cat a.py", {"a.py"}),
    ("grep -r x . 2>/dev/null", set()),
    ("python t.py 2> err.log", set()),
    ("pytest tests 2>&1 | tail -n 20", set()),
])
def test_shell_edits(command, edited):
    assert parse_command(command)[2] == edited


def test_unparseable_command_gives_nothing():
    assert parse_command("echo 'unbalanced") == ([], set(), set())


def test_heredoc_body_is_not_parsed_as_cat_operands():
    command = "cat <<EOF > reproduce_issue.py\nfrom fastapi import Body\nprint(Body())\nEOF\npython3 reproduce_issue.py"
    assert parse_command(command) == ([], set(), {"reproduce_issue.py"})


@pytest.mark.parametrize("command", [
    "cat <<'EOF' > x.py\nimport y\nEOF",
    "cat << \"END\" > x.py\n  a = 1\nEND",
    "cat <<-EOF > x.py\n\tz = 2\n\tEOF",
])
def test_quoted_and_indented_heredocs_are_stripped(command):
    assert strip_heredocs(command).split() == ["cat", ">", "x.py"]


def test_here_string_is_left_alone():
    assert strip_heredocs('grep x <<< "a b"') == 'grep x <<< "a b"'


def test_newline_separates_commands():
    reads, _, _ = parse_command("cd /workspace\ncat rich/cells.py")
    assert [r.file for r in reads] == ["rich/cells.py"]


def test_graph_tool_results_are_seen_nodes_not_reads():
    c = call(0, "search_similar_code", {"query": "SSE"},
             results=[{"node_name": "tests.test_extra_routes.Item", "code": "class Item:\n    None", "similarity": 1.0}])
    event = classify_call(c)
    assert event.seen_nodes == {"tests.test_extra_routes.Item"} and event.reads == []


def test_code_subgraph_nodes_are_seen():
    c = call(0, "get_code_subgraph", {"nodes": ["rich.markdown.Markdown"]},
             nodes=["rich.markdown.Markdown", "rich.markdown.Heading"], edges=[], node_count=2)
    assert classify_call(c).seen_nodes == {"rich.markdown.Markdown", "rich.markdown.Heading"}


def test_failed_edit_still_counts_as_edit():
    c = call(0, "edit_file", {"filepath": "fastapi/sse.py", "old_string": "x", "new_string": "y"},
             status="error", error_type="FileEditError")
    event = classify_call(c)
    assert not event.ok and event.edited_files == {"fastapi/sse.py"}


SNAPSHOT_FILES = {"fastapi/sse.py", "tests/test_sse.py"}


def test_scratch_write_does_not_end_window_and_is_created():
    trace = Trace("ATIF-v1.7", {}, [
        call(0, "write_file", {"filepath": "repro.py", "content": "x"}, filepath="repro.py"),
        call(1, "read_file", {"filepath": "fastapi/sse.py"}, filepath="fastapi/sse.py", start_line=1, end_line=150),
        call(2, "edit_file", {"filepath": "fastapi/sse.py"}, status="error"),
        call(3, "submit_patch"),
    ])
    ex = explore_trace(trace, SNAPSHOT_FILES)
    assert ex.window_end == 2 and [e.index for e in ex.window] == [0, 1]
    assert ex.created_files == ["repro.py"]


def test_write_of_new_target_file_ends_window():
    events = [classify_call(call(0, "write_file", {"filepath": "scripts/new.py"}, filepath="scripts/new.py"))]
    assert exploration_window(events, SNAPSHOT_FILES) == 1
    assert exploration_window(events, SNAPSHOT_FILES, {"scripts/new.py"}) == 0


def test_window_covers_whole_trace_without_edits():
    trace = Trace("ATIF-v1.7", {}, [run(0, "grep -rn x .", "./a.py:1:x\n"), call(1, "submit_patch")])
    assert explore_trace(trace, SNAPSHOT_FILES).window_end == 2


# --- val1 traces (git-ignored; skipped when absent) ---------------------------------------

def _val1(tid: str):
    trace = VAL1 / "results" / "traces" / f"trace_{tid}.json"
    snapshot = SNAPSHOTS / f"{tid}.tgz"
    if not (trace.exists() and snapshot.exists() and TARGETS.exists()):
        pytest.skip("val1 traces, dev snapshots or ground truth not present")
    with tarfile.open(snapshot) as tf:
        files = {m.name.removeprefix("./") for m in tf if m.isfile()}
    targets = set(load_jsonl(TARGETS)[tid]["files"])
    return explore_trace(load_trace(trace), files, targets, tid), targets


@pytest.mark.parametrize("tid, first", [
    ("fastapi_14303", 2), ("fastapi_15588", 3), ("requests_7427", 1), ("rich_3944", None),
])
def test_val1_first_target_read_matches_hand_notes(tid, first):
    ex, targets = _val1(tid)
    hits = [e.index for e in ex.window if any(r.file in targets for r in e.reads)]
    assert (hits[0] if hits else None) == first


def test_val1_scratch_scripts_do_not_end_window():
    ex, _ = _val1("fastapi_14303")
    assert ex.window_end == 4 and ex.created_files == ["reproduce_bug.py"]
    ex, _ = _val1("requests_7427")
    assert ex.window_end == 8 and ex.created_files == ["test_no_proxy_bug.py"]


def test_val1_new_file_target_is_created():
    ex, _ = _val1("fastapi_15661")
    assert "scripts/prepare_release.py" in ex.created_files


def test_cli_needs_files_jsonl(tmp_path, capsys):
    from navalign.explore import main

    (tmp_path / "gt").mkdir()
    assert main(["--run", str(tmp_path), "--groundtruth", str(tmp_path / "gt")]) == 1
    assert "files.jsonl is missing" in capsys.readouterr().err


def test_cli_summarises_a_run(tmp_path, capsys):
    from navalign.explore import main
    from test_escaping import write_run

    calls = [
        call(0, "write_file", {"filepath": "repro.py", "content": "x"}, filepath="repro.py"),
        call(1, "read_file", {"filepath": "fastapi/sse.py"}, filepath="fastapi/sse.py", start_line=1, end_line=150),
        call(2, "edit_file", {"filepath": "fastapi/sse.py"}, status="error"),
    ]
    run = write_run(tmp_path / "run", {"fastapi_1": calls})
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "files.jsonl").write_text(json.dumps({"instance_id": "fastapi_1", "files": sorted(SNAPSHOT_FILES)}) + "\n", encoding="utf-8")
    (gt / "targets.jsonl").write_text(json.dumps({"instance_id": "fastapi_1", "files": ["fastapi/sse.py"]}) + "\n", encoding="utf-8")
    out, csv = tmp_path / "explored.jsonl", tmp_path / "explored.csv"
    assert main(["--run", str(run), "--groundtruth", str(gt), "--out", str(out), "--csv", str(csv)]) == 0
    (ex,) = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert ex["window_end"] == 2 and ex["created_files"] == ["repro.py"]
    row = csv.read_text(encoding="utf-8").splitlines()[1].split(",")
    assert row[:5] == ["fastapi_1", "3", "2", "1", "1"]
