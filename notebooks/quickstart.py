# Source of the public Kaggle notebook kaic87/navalign-quickstart (CPU, no Internet). Inputs: the
# gemma-4-developer-agent competition data and the kaic87/navalign-src and kaic87/navalign-results
# Datasets. Each `# %%` block is one cell.

# %% [markdown]
# # navalign quickstart
#
# **PASS/FAIL tells you a coding agent failed. navalign tells you where.** It reads the agent's
# trace and scores how fast it reached the code the real fix changed, so you can tell "never found
# the bug" from "found it, fixed it wrong". Code and docs: https://github.com/vlyot/navalign
#
# This notebook
# 1. scores a made-up example trace,
# 2. rebuilds the fix maps for all 129 tasks from the competition data,
# 3. shows the validated grading set,
# 4. shows where each run of the baseline agent ended up (run 1, 50 calls),
# 5. compares it with run 2 (10 calls) and with SWE-agent on SWE-bench Verified,
# 6. scores your own agent's traces.
#
# The fix maps are derived from the competition data, which may not be redistributed (rule 2.4.b),
# so they are built here, in `/tmp`, and are not part of this notebook's output. `navalign-results`
# holds only computed tables.

# %%
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

INPUT = Path("/kaggle/input")
COMPETITION = INPUT / "competitions" / "gemma-4-developer-agent"
SRC = next(INPUT.rglob("navalign/cli.py")).parents[1]
RESULTS = next(INPUT.rglob("navalign_results.md")).parent
GROUNDTRUTH = Path("/tmp/groundtruth")  # derived from the competition data: kept out of /kaggle/working
OUT = Path("/kaggle/working")

sys.path.insert(0, str(SRC))
try:
    import unidiff  # noqa: F401
except ImportError:  # notebooks attached to a competition have no Internet: use the wheel in navalign-src
    wheels = next(INPUT.rglob("unidiff-*.whl")).parent
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wheels), "unidiff"],
                   check=True)

from navalign import cli
from navalign.analysis import groundtruth_table
from navalign.data import load_jsonl
from navalign.figures import draw_all
from navalign.score import load_groundtruth


def navalign(*args):
    """Run a navalign command, as `navalign <args>` would in a terminal."""
    code = cli.main([str(a) for a in args])
    if code:
        raise RuntimeError(f"navalign {args[0]} exited with {code}")


# %% [markdown]
# ## 1. Score a trace
#
# A made-up shop whose discount codes charge the discount instead of taking it off; the fix is in
# `apply_discount` in `toyshop/pricing.py`. The scores cover the exploration window, every call
# before the agent's first edit of a repository file. `--steps` adds a row per call: the files it
# read, whether one was a fixed file, whether the lines it read showed fixed code, and `d`, the
# call-graph distance from what it read to the fixed code.

# %%
navalign("score-trace", "--demo", "--steps")

# %% [markdown]
# ## 2. The fix maps, from the competition data
#
# For each task, `build` maps every changed line of the reference patch to the innermost graph
# function containing it, locates every graph node in the snapshot, and measures each node's
# call-graph distance to the fix. About 8 minutes on CPU. The table is checked against the one the
# paper reports.

# %%
navalign("build", "--data", COMPETITION, "--out", GROUNDTRUTH)
gradable = pd.read_csv(RESULTS / "gradable_tasks.csv")
t1 = groundtruth_table(load_jsonl(GROUNDTRUTH / "targets.jsonl"), load_groundtruth(GROUNDTRUTH),
                       set(gradable.loc[gradable["gradable"], "instance_id"]))
t1.to_csv("/tmp/t1_groundtruth.csv", index=False)
published = pd.read_csv(RESULTS / "run1" / "t1_groundtruth.csv")
pd.testing.assert_frame_equal(pd.read_csv("/tmp/t1_groundtruth.csv"), published)
print("fix maps match the published ground-truth table")
display(published)

# %% [markdown]
# ## 3. The validated grading set
#
# As shipped, the harness passes the reference fix and fails an empty patch on only 71 of the 129
# tasks. In the repaired environment 111 are gradable; `reason` says why each other task is not.

# %%
display(gradable["reason"].value_counts().rename("tasks").to_frame())
display(gradable.loc[~gradable["gradable"], ["instance_id", "reason"]])

# %% [markdown]
# ## 4. Run 1: where every run ended up
#
# The baseline agent on all 129 tasks with 50 tool calls. A: never read a file the fix changed. B:
# read one, never edited one. C1/C2: edited one and failed, C1 when an escaping bug in the harness's
# tools broke an edit. D: edited one and passed. CD_ng: edited one on a task that can't be graded.
# X: ended by the context limit. The figures are drawn from the published per-run outcomes.

# %%
run1 = OUT / "run1"
run1.mkdir(exist_ok=True)
(run1 / "random_k.csv").write_bytes((RESULTS / "run1" / "random_k.csv").read_bytes())
outcomes = pd.read_csv(RESULTS / "run1" / "outcomes.csv")
display(pd.read_csv(RESULTS / "run1" / "t2_breakdown.csv"))
display(pd.read_csv(RESULTS / "run1" / "validity.csv"))
for figure in draw_all(run1, outcomes):
    display(Image(filename=str(figure)))

# %% [markdown]
# ## 5. Run 2, and outside the competition
#
# Run 2 repeats run 1 with the harness's default budget of 10 tool calls. `compare.csv` sets the
# two runs side by side and `transitions.csv` shows how each task's category moved. `swebench/` is
# SWE-agent with SWE-agent-LM-32B on 60 SWE-bench Verified tasks, scored the same way.

# %%
display(pd.read_csv(RESULTS / "compare" / "compare.csv"))
display(pd.read_csv(RESULTS / "compare" / "transitions.csv"))
display(pd.read_csv(RESULTS / "swebench" / "t2_breakdown.csv"))

# %% [markdown]
# ## 6. Score your own agent
#
# Attach the output of your own harness run as an input: any folder with `trace_<instance_id>.json`
# files, such as the harness's `results/traces/`. Set `MY_TRACES` to it and run the cell.
# `my_scores.csv` gets one row per task; `my_steps.jsonl` the per-step reads and distances.

# %%
MY_TRACES = None  # e.g. INPUT / "my-run"
if MY_TRACES:
    navalign("score", "--run", MY_TRACES, "--groundtruth", GROUNDTRUTH, "--out", OUT / "my_scores.csv",
             "--steps", OUT / "my_steps.jsonl")
