import pytest

from navalign.build import build_task
from navalign.data import example_dir, load_tasks
from navalign.score import score_task
from navalign.trace import load_trace

TOY = example_dir()
TASK = "toyshop_1"


def build():
    return build_task(load_tasks(TOY / "tasks.jsonl")[TASK], TOY / "graphs", TOY / "snapshots" / TASK, {})


def test_example_dir_has_the_competition_layout():
    assert (TOY / "tasks.jsonl").is_file()
    assert len(list((TOY / "graphs").glob("toyshop_*.json"))) == 1
    assert (TOY / "snapshots" / TASK / "toyshop" / "pricing.py").is_file()
    assert (TOY / "traces" / f"trace_{TASK}.json").is_file()


def test_toy_example_builds_with_full_span_coverage():
    _, spans, distances = build()
    assert spans.coverage == 1.0
    assert spans.unlocated == []
    assert distances.n_reachable == distances.n_nodes == 9


def test_toy_targets_match_the_hand_written_expectation():
    targets, _, distances = build()
    assert targets.files == ["toyshop/pricing.py"]
    assert targets.target_nodes == {"toyshop.pricing.apply_discount"}
    assert dict(targets.rules) == {"file": 2, "exact": 2}  # docstring line, then the bug
    assert distances.cap == 5


def test_toy_trace_scores_match_the_readme():
    task = load_tasks(TOY / "tasks.jsonl")[TASK]
    score = score_task(task, TOY / "graphs", TOY / "snapshots" / TASK, load_trace(TOY / "traces" / f"trace_{TASK}.json"))
    assert (score.n_calls, score.window_end, score.read_calls) == (9, 6, 3)
    assert (score.k, score.k_node, score.k_seen) == (3, 4, 1)
    assert score.s_file == pytest.approx(1 / 3)
    assert score.focus == pytest.approx(2 / 3)
    assert (score.s_node, score.mean_d) == (pytest.approx(0.8), pytest.approx(1.0))
    assert [s.d for s in score.steps] == [None, 2, 1, 0, None, None]
