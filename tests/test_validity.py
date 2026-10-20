import json

import numpy as np
import pandas as pd
import pytest
from test_escaping import write_run
from test_explore import call, run
from test_score import ROWS, SPANS, TASK, ground_truth, read

from navalign.data import example_dir
from navalign.explore import explore_trace
from navalign.runs import load_run
from navalign.score import GroundTruth, score_exploration
from navalign.trace import Trace
from navalign.validity import (
    baselines,
    bm25_ranking,
    checkpoint_passes,
    cohen_kappa,
    issue_mentions,
    label_sheet,
    main,
    oracle_exploration,
    random_exploration,
    ranked_exploration,
    score_level,
    validity_summary,
)

CALLS = [run(0, "grep -rn f pkg"), read(1, "pkg/b.py"), call(2, "write_file", {"filepath": "repro.py", "content": "x"},
         filepath="repro.py"), run(3, "cat repro.py"), read(4, "pkg/a.py", 30, 40), call(5, "edit_file", {"filepath": "pkg/a.py"})]


def explore(calls, gt: GroundTruth, tid: str = TASK):
    return explore_trace(Trace("ATIF-v1.7", {}, calls), gt.snapshot_files, gt.target_files, tid)


def test_oracle_scores_one_on_every_metric():
    gt = ground_truth()
    s = score_exploration(oracle_exploration(explore(CALLS, gt), gt), gt)
    assert (s.window_end, s.s_file, s.focus, s.s_node) == (5, 1.0, 1.0, 1.0)


def test_oracle_without_target_nodes_reads_a_target_file():
    gt = ground_truth(target_nodes=[], targets=[{"file": "pkg/a.py", "rule": "file"}])
    s = score_exploration(oracle_exploration(explore(CALLS, gt), gt), gt)
    assert (s.s_file, s.focus, s.s_node) == (1.0, 1.0, None)


def test_random_exploration_keeps_window_and_read_positions():
    gt = ground_truth()
    ex = explore(CALLS, gt)
    rand = random_exploration(ex, gt, np.random.default_rng(0), ["pkg/a.py", "pkg/b.py", "pkg/flat.py"])
    assert rand.window_end == ex.window_end == 5
    assert [e.index for e in rand.events] == [0, 1, 2, 3, 4]
    assert [len(e.reads) for e in rand.events] == [0, 1, 0, 0, 1]  # the scratch-file read is not navigation
    assert all(r.end - r.start == 149 and r.file.endswith(".py") for e in rand.events for r in e.reads)


def test_random_exploration_is_reproducible():
    gt = ground_truth()
    ex = explore(CALLS, gt)
    files = sorted(f for f in gt.snapshot_files if f.endswith(".py"))

    def draw(seed):
        return [e.reads for e in random_exploration(ex, gt, np.random.default_rng(seed), files).events]

    assert draw(3) == draw(3) != draw(4)


def test_baselines_skip_new_file_tasks_and_keep_every_draw():
    gt = ground_truth()
    new = ground_truth(instance_id="pkg_new", files=["pkg/new.py"], target_nodes=[],
                       targets=[{"file": "pkg/new.py", "rule": "new_file"}])
    explorations = {TASK: explore(CALLS, gt), "pkg_new": explore(CALLS, new, "pkg_new")}
    per_task, random_k = baselines(explorations, {TASK: gt, "pkg_new": new}, draws=20)
    assert per_task["instance_id"].tolist() == [TASK] and len(random_k) == 20
    row = per_task.iloc[0]
    assert (row["oracle_s_file"], row["oracle_s_node"]) == (1.0, 1.0)
    assert row["real_s_file"] == pytest.approx(1 / 5)


def test_summary_and_checkpoint():
    per_task = pd.DataFrame({
        **{f"oracle_{m}": [1.0, 1.0, 1.0] for m in ("s_file", "focus", "s_node")},
        **{f"real_{m}": [0.5, 0.8, 0.9] for m in ("s_file", "focus", "s_node")},
        **{f"random_{m}": [0.0, 0.1, 0.05] for m in ("s_file", "focus", "s_node")},
    })
    summary = validity_summary(per_task)
    assert summary["real_minus_random"].round(3).tolist() == [0.683] * 3
    assert checkpoint_passes(summary)
    per_task.loc[0, "oracle_s_node"] = 0.9
    assert not checkpoint_passes(validity_summary(per_task))


@pytest.mark.parametrize("calls, level", [
    ([read(0, "pkg/b.py")], 0),
    ([read(0, "pkg/a.py", 30, 40)], 1),
    ([read(0, "pkg/a.py", 1, 20)], 2),
])
def test_score_level(calls, level):
    gt = ground_truth()
    assert score_level(score_exploration(explore(calls, gt), gt)) == level


@pytest.mark.parametrize("a, b, kappa", [
    ([0, 1, 2, 2], [0, 1, 2, 2], 1.0),
    ([0, 0, 1, 1], [0, 1, 0, 1], 0.0),
    ([2, 2, 2], [2, 2, 2], 1.0),
])
def test_cohen_kappa(a, b, kappa):
    assert cohen_kappa(np.array(a), np.array(b)) == pytest.approx(kappa)


LEVEL_TRACES = {
    "pkg_0": [read(0, "pkg/b.py"), call(1, "edit_file", {"filepath": "pkg/b.py"})],
    "pkg_1": [read(0, "pkg/a.py", 30, 40), call(1, "edit_file", {"filepath": "pkg/a.py"})],
    "pkg_2": [read(0, "pkg/a.py", 1, 20), call(1, "edit_file", {"filepath": "pkg/a.py"})],
}


def _level_run(tmp_path):
    run_dir = write_run(tmp_path / "run", LEVEL_TRACES)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for name, row in ROWS.items():
        lines = [json.dumps(row | {"instance_id": tid}) for tid in LEVEL_TRACES]
        (gt_dir / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir, gt_dir


def test_label_sheet_samples_every_level_and_hides_scores(tmp_path):
    run_dir, _ = _level_run(tmp_path)
    gts = {tid: GroundTruth.from_rows(*(row | {"instance_id": tid} for row in ROWS.values())) for tid in LEVEL_TRACES}
    explorations = {tid: explore(calls, gts[tid], tid) for tid, calls in LEVEL_TRACES.items()}
    sheet, labels = label_sheet(load_run(run_dir), explorations, gts, n=3)
    assert sorted(labels["instance_id"]) == sorted(LEVEL_TRACES)
    assert labels["label"].isna().all() and list(labels.columns) == ["instance_id", "label", "notes"]
    assert "`pkg.a.f` (pkg/a.py lines 10" in sheet and "lines 30" in sheet
    assert "s_file" not in sheet and "S_node" not in sheet


def test_cli_label_sheet_then_agreement(tmp_path, capsys):
    run_dir, gt_dir = _level_run(tmp_path)
    common = ["--run", str(run_dir), "--groundtruth", str(gt_dir)]
    out = tmp_path / "out"
    assert main(["label-sheet", *common, "--out", str(out), "--n", "3"]) == 0
    assert main(["label-sheet", *common, "--out", str(out)]) == 1  # never overwrites labels
    labels_path = out / "human_labels.csv"
    assert main(["agreement", *common, "--labels", str(labels_path)]) == 1  # still blank
    labels = pd.read_csv(labels_path)
    labels["label"] = labels["instance_id"].str[-1].astype(int)
    labels.to_csv(labels_path, index=False)
    capsys.readouterr()
    assert main(["agreement", *common, "--labels", str(labels_path), "--out", str(out)]) == 0
    assert "kappa 1.000" in capsys.readouterr().out
    assert pd.read_csv(out / "agreement.csv")["scorer"].tolist() == labels["label"].tolist()


NAMED_SPANS = SPANS | {"spans": {"pkg/a.py": [["pkg.a.parse_header", 10, 20], ["pkg.a.g", 30, 40]],
                                 "pkg/b.py": [["pkg.b.render", 1, 50]], "pkg/flat.py": [["pkg.flat.k", 5, 9]]}}


def named_ground_truth() -> GroundTruth:
    return GroundTruth.from_rows(ROWS["targets"], NAMED_SPANS, ROWS["distances"], ROWS["files"])


def test_issue_mentions_follow_the_order_of_mention():
    text = "render() fails when pkg/a.py handles headers via parse_header"
    assert issue_mentions(text, named_ground_truth()) == [("pkg/b.py", 1), ("pkg/a.py", 1), ("pkg/a.py", 10)]


def test_issue_mentions_resolve_module_paths_and_skip_short_names():
    assert issue_mentions("see pkg.b and g", named_ground_truth()) == [("pkg/b.py", 1)]


def test_issue_mentions_skip_names_defined_too_often():
    spans = SPANS | {"spans": {f"pkg/m{i}.py": [[f"pkg.m{i}.render", 1, 5]] for i in range(4)}}
    gt = GroundTruth.from_rows(ROWS["targets"], spans, ROWS["distances"], ROWS["files"])
    assert issue_mentions("render is broken", gt) == []


def test_bm25_ranks_the_node_sharing_issue_words_first():
    texts = {"pkg.a.f": "def f(header):\n    return parse_token(header)", "pkg.a.g": "def g():\n    return 1",
             "pkg.b.h": "def h(html):\n    return render(html)", "pkg.flat.k": "def k():\n    pass"}
    ranking = bm25_ranking("Header parsing breaks on a token", texts, ground_truth())
    assert ranking[0] == ("pkg/a.py", 10) and ("pkg/a.py", 30) not in ranking


def test_ranked_exploration_reads_the_ranking_in_order_then_nothing():
    gt = ground_truth()
    ex = explore(CALLS, gt)
    ranked = ranked_exploration(ex, gt, [("pkg/a.py", 10)])
    assert ranked.window_end == ex.window_end == 5
    assert [[(r.file, r.start, r.end) for r in e.reads] for e in ranked.events] == [[], [("pkg/a.py", 10, 159)], [], [], []]
    s = score_exploration(ranked, gt)
    assert (s.k, s.s_file, s.focus) == (2, 0.5, 1.0)


def test_baselines_with_rankings_add_issue_and_bm25_columns():
    gt = ground_truth()
    rankings = {TASK: {"issue": [["pkg/b.py", 1]], "bm25": [["pkg/a.py", 10], ["pkg/b.py", 1]]}}
    per_task, _ = baselines({TASK: explore(CALLS, gt)}, {TASK: gt}, draws=5, rankings=rankings)
    row = per_task.iloc[0]
    assert (row["issue_s_file"], row["bm25_s_file"], row["bm25_s_node"]) == (0.0, 0.5, pytest.approx(1 - 1 / 5))
    summary = validity_summary(per_task)
    assert {"issue_mean", "real_minus_bm25", "bm25_ci_low"} <= set(summary.columns)


def test_random_label_sheet_skips_excluded_tasks(tmp_path):
    run_dir, _ = _level_run(tmp_path)
    gts = {tid: GroundTruth.from_rows(*(row | {"instance_id": tid} for row in ROWS.values())) for tid in LEVEL_TRACES}
    explorations = {tid: explore(calls, gts[tid], tid) for tid, calls in LEVEL_TRACES.items()}
    _, labels = label_sheet(load_run(run_dir), explorations, gts, n=3, sample="random", exclude=frozenset({"pkg_0"}))
    assert sorted(labels["instance_id"]) == ["pkg_1", "pkg_2"]


def test_cli_rankings_on_the_toy_example(tmp_path):
    from navalign.build import main as build_main

    toy = example_dir()
    assert build_main(["--data", str(toy), "--out", str(tmp_path / "gt")]) == 0
    out = tmp_path / "rankings.jsonl"
    assert main(["rankings", "--tasks", str(toy / "tasks.jsonl"), "--graphs", str(toy / "graphs"),
                 "--groundtruth", str(tmp_path / "gt"), "--out", str(out)]) == 0
    (row,) = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert row["issue"] == [["toyshop/cart.py", 14]]  # "Cart.total(...)" names the method
    assert row["bm25"][0] == ["toyshop/cart.py", 14] and ["toyshop/pricing.py", 10] in row["bm25"]
