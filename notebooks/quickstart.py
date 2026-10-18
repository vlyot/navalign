# Source of the public Kaggle notebook kaic87/navalign-quickstart (CPU, no Internet). Inputs: the
# gemma-4-developer-agent competition data and the kaic87/navalign-src, kaic87/navalign-groundtruth,
# kaic87/navalign-run1-traces and kaic87/navalign-run2-traces Datasets. Each `# %%` block is one cell.

# %% [markdown]
# # navalign quickstart
#
# **PASS/FAIL tells you a coding agent failed. navalign tells you where.** It reads the agent's
# trace and scores how fast it reached the code the real fix changed, so you can tell "never found
# the bug" from "found it, fixed it wrong". Code and docs: https://github.com/vlyot/navalign
#
# This notebook
# 1. scores one trace of the competition's baseline agent,
# 2. gets the same score straight from the raw competition data,
# 3. rebuilds the failure breakdown and the figures for all 129 tasks (run 1),
# 4. compares run 1 with run 2, the same agent on a 10-call budget,
# 5. scores your own agent's traces.
#
# Inputs: `navalign-src` (the toolkit), `navalign-groundtruth` (ground truth for the 129 tasks and
# which tasks can be graded), `navalign-run1-traces` and `navalign-run2-traces` (the baseline
# agent's traces and grades), and the competition data for part 2.

# %%
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

INPUT = Path("/kaggle/input")
COMPETITION = INPUT / "competitions" / "gemma-4-developer-agent"
SRC = next(INPUT.rglob("navalign/cli.py")).parents[1]
GROUNDTRUTH = next(INPUT.rglob("navalign_groundtruth.md")).parent
RUN1 = next(INPUT.rglob("navalign_run1.md")).parent
GRADES = next(RUN1.rglob("grades.jsonl")).parent
OUT = Path("/kaggle/working")

sys.path.insert(0, str(SRC))
try:
    import unidiff  # noqa: F401
except ImportError:  # notebooks attached to a competition have no Internet: use the wheel in navalign-src
    wheels = next(INPUT.rglob("unidiff-*.whl")).parent
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wheels), "unidiff"],
                   check=True)

from navalign import cli


def navalign(*args):
    """Run a navalign command, as `navalign <args>` would in a terminal."""
    code = cli.main([str(a) for a in args])
    if code:
        raise RuntimeError(f"navalign {args[0]} exited with {code}")


# %% [markdown]
# ## 1. Score one trace
#
# `requests_7427`: requests ignored `no_proxy` for some URLs; the fix is in
# `requests.utils.should_bypass_proxies`. The scores cover the exploration window, every call
# before the agent's first edit of a repository file. `--steps` adds a row per call: the files it
# read, whether one was a fixed file, whether the lines it read showed fixed code, and `d`, the
# call-graph distance from what it read to the fixed code.

# %%
navalign("score-trace", next(RUN1.rglob("trace_requests_7427.json")), "--groundtruth", GROUNDTRUTH, "--steps")

# %% [markdown]
# ## 2. The same score from the raw competition data
#
# With `--data`, navalign builds the task's ground truth on the fly from `tasks.jsonl`, the code
# graph and the repository snapshot, instead of reading the prebuilt `navalign-groundtruth`.

# %%
navalign("score-trace", next(RUN1.rglob("trace_requests_7427.json")), "--data", COMPETITION)

# %% [markdown]
# ## 3. Run 1: where every run ended up
#
# The baseline agent on all 129 tasks, one run each. `validity baselines` scores an oracle and a
# random explorer against the same ground truth (the random draws feed figure F2), `score` writes
# the per-step distances (F4), and `analysis` puts each run in an outcome category and draws the
# figures. PASS/FAIL comes from regrading each agent patch in a repaired test environment, and
# only on the tasks whose reference fix passes there (`gradable_tasks.csv`).

# %%
def run_args(root, flag="--run"):
    """`--run <shard>` for every shard of a run Dataset."""
    return [arg for shard in sorted(p.parent for p in root.rglob("manifest.json")) for arg in (flag, shard)]


runs = run_args(RUN1)
navalign("validity", "baselines", *runs, "--groundtruth", GROUNDTRUTH, "--out", OUT / "run1")
navalign("score", *runs, "--groundtruth", GROUNDTRUTH, "--out", OUT / "run1" / "scores.csv",
         "--steps", OUT / "run1" / "steps.jsonl")
navalign("analysis", *runs, "--groundtruth", GROUNDTRUTH, "--grades", GRADES,
         "--gradable", GROUNDTRUTH / "gradable_tasks.csv", "--out", OUT / "run1", "--figures")

# %% [markdown]
# **The failure breakdown (T2).** A: never read a file the fix changed. B: read one, never edited
# one. C1/C2: edited one and failed, C1 when an escaping bug in the harness's tools broke an edit.
# D: edited one and passed. CD_ng: edited one on a task that can't be graded. X: ended by the
# context limit.

# %%
display(pd.read_csv(OUT / "run1" / "t2_breakdown.csv"))
for figure in sorted((OUT / "run1" / "figures").glob("*.png")):
    display(Image(filename=str(figure)))

# %% [markdown]
# ## 4. Run 2: the same agent with a 10-call budget
#
# Run 2 repeats run 1 with the harness's default budget of 10 tool calls instead of 50.
# `compare` sets the two runs side by side: outcomes, scores and AUCs per run, then per-task
# agreement (how close each task's first-read score is across the runs, and how many calls the two
# traces share from the start) and how each task's category moved.

# %%
RUN2 = next((p.parent for p in INPUT.rglob("navalign_run2.md")), None)
if RUN2:
    navalign("analysis", *run_args(RUN2), "--groundtruth", GROUNDTRUTH, "--grades", next(RUN2.rglob("grades.jsonl")).parent,
             "--gradable", GROUNDTRUTH / "gradable_tasks.csv", "--out", OUT / "run2")
    navalign("compare", "--a", OUT / "run1", *run_args(RUN1, "--a-run"), "--b", OUT / "run2", *run_args(RUN2, "--b-run"),
             "--out", OUT / "compare")

# %% [markdown]
# ## 5. Score your own agent
#
# Attach the output of your own harness run as an input: any folder with `trace_<instance_id>.json`
# files, such as the harness's `results/traces/`. Set `MY_TRACES` to it and run the cell.
# `my_scores.csv` gets one row per task; `my_steps.jsonl` the per-step reads and distances.

# %%
MY_TRACES = None  # e.g. INPUT / "my-run"
if MY_TRACES:
    navalign("score", "--run", MY_TRACES, "--groundtruth", GROUNDTRUTH, "--out", OUT / "my_scores.csv",
             "--steps", OUT / "my_steps.jsonl")
