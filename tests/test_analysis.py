import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_score import ROWS

from navalign.analysis import (
    auc,
    bootstrap_ci,
    breakdown_table,
    groundtruth_table,
    permutation_p,
    predictive_table,
)
from navalign.outcomes import CATEGORIES
from navalign.score import GroundTruth

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("scores, labels, expected", [
    ([1, 2, 3, 4], [0, 0, 1, 1], 1.0),
    ([4, 3, 2, 1], [0, 0, 1, 1], 0.0),
    ([1, 1, 1, 1], [0, 1, 0, 1], 0.5),
    ([1, 3, 2, 4], [0, 1, 0, 1], 1.0),
    ([1, 2, 2, 3], [0, 0, 1, 1], 0.875),
])
def test_auc(scores, labels, expected):
    assert auc(np.array(scores), np.array(labels)) == expected


def test_auc_is_nan_without_both_classes():
    assert math.isnan(auc(np.array([1, 2]), np.array([1, 1])))


def test_bootstrap_ci_contains_the_point_estimate():
    x = np.random.default_rng(1).normal(size=200)
    lo, hi = bootstrap_ci(lambda a: float(a.mean()), x, n_boot=500)
    assert lo < x.mean() < hi and hi - lo < 0.5


def test_permutation_p_is_small_for_separation_and_large_for_noise():
    labels = np.array([0] * 15 + [1] * 15)
    assert permutation_p(np.arange(30), labels, n_perm=999) < 0.01
    noise = np.random.default_rng(2).normal(size=30)
    assert permutation_p(noise, labels, n_perm=999) > 0.05


def _outcomes() -> pd.DataFrame:
    rows = [
        ("fastapi_1", "A", 0.0, 0.0, 0.5, None, 10, 5, False),
        ("fastapi_2", "B", 0.2, 0.3, 0.6, 9, 10, 20, False),
        ("fastapi_3", "B", 0.25, 0.2, 0.7, None, 10, 18, False),
        ("fastapi_4", "C2", 0.5, 0.6, 0.8, 3, 10, 9, True),
        ("rich_1", "D", 1.0, 0.9, 0.95, 1, 12, 4, True),
        ("rich_2", "D", 0.5, 0.7, None, None, None, 6, None),
        ("rich_3", "CD_ng", 0.33, 0.5, 0.9, 2, 12, 7, True),
        ("rich_4", "X", 0.0, 0.0, 0.4, None, 12, 30, False),
        ("rich_5", "C1", 0.5, 0.5, 0.85, 2, 12, 8, False),
    ]
    cols = ["instance_id", "category", "s_file", "focus", "s_node", "k_node", "cap", "window_end", "patch_touches_target_node"]
    t = pd.DataFrame(rows, columns=cols)
    t["repo"] = t["instance_id"].str.rsplit("_", n=1).str[0]
    t = t.assign(s_file_prefix=t["s_file"], focus_prefix=t["focus"], s_node_prefix=t["s_node"])
    return t.astype({"k_node": "Int64", "cap": "Int64"})


def test_breakdown_counts_sum_to_runs_per_repo():
    t2 = breakdown_table(_outcomes()).set_index("repo")
    assert list(t2.columns) == ["n", *CATEGORIES]
    assert t2.loc["fastapi"].to_dict() == {"n": 4, "A": 1, "B": 2, "C1": 0, "C2": 1, "D": 0, "CD_ng": 0, "X": 0}
    assert t2.loc["all", "n"] == 9 and (t2.drop(columns="n").sum(axis=1) == t2["n"]).all()


def test_predictive_table_uses_non_circular_populations():
    t4 = predictive_table(_outcomes(), n_boot=50, n_perm=50)
    edited = t4[t4["outcome"].str.startswith("edited")].set_index("predictor")
    assert edited.loc["s_file", "n"] == 7  # A and X left out
    assert edited.loc["s_file", "n_positive"] == 5 and edited.loc["s_file", "auc"] == 1.0
    assert edited.loc["s_node", "n"] == 6  # rich_2 has no node level
    assert edited.loc["-k_node", "n"] == 6  # ... and never showing a target node ranks last, not dropped
    assert edited.loc["-k_node", "auc"] == 1.0
    assert edited.loc["s_file_first10", "auc"] == 1.0 and edited.loc["s_node_first10", "n"] == 6
    node = t4[t4["outcome"].str.startswith("patch")]
    assert set(node["n"]) == {6} and node.set_index("predictor").loc["s_file", "n_positive"] == 3
    passed = t4[t4["outcome"].str.startswith("passed")].set_index("predictor")
    assert passed.loc["s_file", "n"] == 4 and passed.loc["s_file", "n_positive"] == 2  # C1, C2, D, D


def test_groundtruth_table_counts_per_repo():
    def gt(tid, **targets):
        rows = {name: row | {"instance_id": tid} for name, row in ROWS.items()}
        rows["targets"] |= targets
        return GroundTruth.from_rows(*rows.values())

    targets = {
        "pkg_1": ROWS["targets"] | {"instance_id": "pkg_1", "node_coverage": 1.0, "rules": {"exact": 3}},
        "pkg_2": ROWS["targets"] | {"instance_id": "pkg_2", "target_nodes": [], "node_coverage": 0.0, "rules": {"file": 2}},
        "lib_1": ROWS["targets"] | {"instance_id": "lib_1", "node_coverage": 0.5, "rules": {"exact": 1, "enclosing": 1}},
    }
    gts = {tid: gt(tid, target_nodes=t["target_nodes"]) for tid, t in targets.items()}
    t1 = groundtruth_table(targets, gts, {"pkg_1"}).set_index("repo")
    assert t1.loc["pkg", ["tasks", "gradable", "with_target_node", "fully_node_covered", "file_level_only",
                          "fallback_rule"]].tolist() == [2, 1, 1, 1, 1, 1]
    assert t1.loc["all", "tasks"] == 3 and t1.loc["all", "fallback_rule"] == 2 and t1.loc["all", "cap_range"] == "5-5"


def test_cli_on_run1_writes_the_tables(tmp_path):
    from navalign.analysis import main

    run1 = ROOT / "results" / "traces" / "run1"
    gold = ROOT / "results" / "gold"
    needed = [run1 / f"shard{i}" for i in range(3)] + [gold / "final", gold / "gradable_tasks.csv"]
    if not all(p.exists() for p in needed):
        pytest.skip("run 1 or its grades not present")
    args = [a for i in range(3) for a in ("--run", str(run1 / f"shard{i}"))]
    args += ["--groundtruth", str(ROOT / "results" / "groundtruth"), "--grades", str(gold / "final"),
             "--gradable", str(gold / "gradable_tasks.csv"), "--out", str(tmp_path)]
    assert main(args) == 0
    t2 = pd.read_csv(tmp_path / "t2_breakdown.csv").set_index("repo")
    assert t2.loc["all", "n"] == 129
    assert len(pd.read_csv(tmp_path / "outcomes.csv")) == 129
    assert pd.read_csv(tmp_path / "t1_groundtruth.csv").set_index("repo").loc["all", "gradable"] == 111
    assert len(pd.read_csv(tmp_path / "t4_predictive.csv")) == 24  # 3 outcomes x 8 predictors


def test_f1_groups_merge_escaping_failures_into_failed_edits():
    pytest.importorskip("matplotlib")
    from navalign.figures import BREAKDOWN, breakdown_counts

    counts = breakdown_counts(_outcomes())
    assert list(counts.columns) == list(BREAKDOWN)
    assert list(counts.index) == ["fastapi", "rich", "all"]
    assert counts.loc["rich", "Edited it, tests failed"] == 1  # rich_5, C1
    assert counts.loc["all", "Edited it, tests failed"] == 2  # + fastapi_4, C2
    assert counts.loc["all"].sum() == len(_outcomes())


def test_figures_render_and_skip_missing_inputs(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from navalign.figures import draw_all, pick_cases

    t = _outcomes()
    t["k"] = [None if s == 0 else round(1 / s) for s in t["s_file"]]
    t["k"] = t["k"].astype("Int64")
    t["read_calls"] = range(len(t))
    assert pick_cases(t) == ["rich_1", "fastapi_3", "fastapi_1"]
    bare = draw_all(tmp_path / "bare", t)
    assert [p.name for p in bare] == ["f1_breakdown.png", "f3_scores_by_outcome.png"]
    assert "skipped F2" in capsys.readouterr().err

    out = tmp_path / "full"
    out.mkdir()
    pd.DataFrame({"instance_id": ["fastapi_1"] * 3, "draw": [0, 1, 2], "k": [None, 7, None]}).to_csv(out / "random_k.csv", index=False)
    steps = [{"instance_id": tid, "steps": [{"index": i, "d": d} for i, d in enumerate([4, 2, 0, None, 1])]}
             for tid in t["instance_id"]]
    (out / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps), encoding="utf-8")
    written = draw_all(out, t)
    assert [p.name for p in written] == ["f1_breakdown.png", "f2_first_read.png", "f3_scores_by_outcome.png",
                                         "f4_trajectories.png"]
    assert all(p.stat().st_size > 10_000 for p in written)
