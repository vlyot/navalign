import json
import shutil
import subprocess
from pathlib import Path

import pytest

from navalign.cli import COMMANDS, main


def test_no_arguments_lists_every_command(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert all(f"  {name} " in out for name in COMMANDS)


def test_unknown_command_exits_2_with_the_list(capsys):
    assert main(["frobnicate"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'frobnicate'" in err and "score-trace" in err


def test_arguments_reach_the_module_main(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["score", "--help"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("usage: navalign score [-h] --run RUN")


def test_score_trace_demo_prints_the_summary(capsys):
    assert main(["score-trace", "--demo"]) == 0
    assert "first read of a fixed file   call 3     S_file 0.333" in capsys.readouterr().out


def test_readme_shows_the_real_demo_output(capsys):
    main(["score-trace", "--demo"])
    summary = capsys.readouterr().out.split("\n\nS_file = ")[0]
    assert f"```\n{summary}\n```" in (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")


def test_score_trace_json_has_the_scores_and_steps(capsys):
    assert main(["score-trace", "--demo", "--json"]) == 0
    score = json.loads(capsys.readouterr().out)
    assert (score["k"], score["s_node"], len(score["steps"])) == (3, pytest.approx(0.8), 6)


def test_score_trace_without_a_source_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["score-trace", "trace_x.json"])
    assert exit_info.value.code == 2
    assert "either --groundtruth DIR or --data DIR" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("navalign") is None, reason="navalign is not installed on PATH")
def test_installed_command_runs_the_demo():
    result = subprocess.run(["navalign", "score-trace", "--demo"], capture_output=True, text=True, check=True)
    assert result.stdout.startswith("toyshop_1: 9 tool calls")
