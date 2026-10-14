# nav-alignment-toolkit

Navigation-alignment scoring for coding-agent traces on the Gemma 4 Developer Agent benchmark:
given a task's code graph, its reference patch, and an agent's ATIF trace, score how directly the
agent's exploration converged on the code the real fix touched.

## Status

Built: data loaders, patch parser, `ast` symbol index, patch-to-node mapper, distance maps,
node span index, one-pass ground-truth build, ATIF trace parser, trace-collection bookkeeping and
run checker, explored-set parser, escaping check (R3), gold-patch control and regrading (R1),
scorer, per-run outcomes, validity checks (oracle, random, hand labels), result tables and figures.

## Target mapping

Every non-blank changed line of a reference patch is resolved to one rule:

| Rule | Meaning |
|---|---|
| `exact` | the innermost function/class containing the line is a graph node |
| `enclosing` | the innermost symbol is not a node (async, added by the patch), an enclosing one is |
| `file` | no containing symbol is a node (module-level line, top-level async/added function) |
| `new_file` | the patch creates the file |
| `non_python` | docs, config, etc. |

Graph ids keep enclosing classes and drop enclosing functions, and module names drop `src/` and
`docs_src/`; the mapper follows both conventions.

```
python -m navalign.mapper --data data                       # every task with a local snapshot
python -m navalign.mapper --data data --task rich_3944 --out targets.jsonl
```

## Distance maps

`navalign.distance` gives each graph node's shortest-path distance to the nearest target node on
the `calls` graph, with edge direction ignored (an agent can follow a call either way). Nodes that
reach no target get the cap C = (largest finite distance in the graph) + 1, i.e. the largest
component diameter + 1. The cap belongs to the graph, so all tasks on one commit share it
(dev graphs: fastapi 17, requests 14, rich 13). Tasks with no target node get an empty map and no
node-level score.

## Node span index

The graphs have no line numbers. `navalign.spans` finds each node in the snapshot source with the
same naming rules as the mapper and records `(file, start, end)`, so the scorer can turn a read of
lines a–b into the nodes it covers without the snapshots. Nodes that no file defines, or that two
files both claim, are listed as `unlocated`. Dev coverage: rich 100%, fastapi 99–100%, requests 79%.
The misses are nodes whose enclosing class the graph drops (requests test methods such as
`tests.test_requests.test_unicode_get`, fastapi's pydantic-v1 `Config` classes); library
functions are fully located.

## Building the ground truth

```
python -m navalign.build --data data --out results/groundtruth
python -m navalign.build --data data --task rich_3944 --out results/groundtruth
```

`--tasks`, `--graphs` and `--snapshots` override the paths under `--data`. Snapshots can be `.tgz`
files or extracted `<instance_id>/` folders. Without `--task`, every task with a snapshot is built.
A failing task is recorded in `summary.csv` and the run continues (exit code 1 if any failed).

| File | One JSON line / row per task |
|---|---|
| `targets.jsonl` | mapper output: `files`, `target_nodes`, `node_coverage`, `rules`, `targets` (rule per changed line), `file_nodes` |
| `spans.jsonl` | `n_nodes`, `coverage`, `spans` (`{file: [[node, start, end], ...]}`), `unlocated`, `skipped_files` |
| `distances.jsonl` | `graph_file`, `targets`, `cap`, `n_nodes`, `n_reachable`, `distances` (`{node: hops}`, reachable nodes only; any other node is at `cap`) |
| `files.jsonl` | `files`: every file in the snapshot (used to tell real files from the agent's scratch files) |
| `summary.csv` | counts, coverage, cap, seconds, and `error` (empty on success) |

Load them back with `DistanceMap.from_dict` and `SpanIndex.from_dict`.

## Collecting traces

Traces come from the harness `Evaluator` in a Kaggle GPU notebook (a Copy & Edit of the Getting
Started notebook). `notebooks/collect_traces.py` holds the cells to paste in:

- **A** (first cell): run id, task list or shard (run 1: `TASK_IDS = None`, shard k of 3), `RESUME_FROM`, `max_output_tokens` (8192) and the
  eval budgets (300 s / 50 calls / 10 min / 100 turns), and the clock for the 12-hour cap.
- **B** (after the cell that loads `tasks`, before the model server starts): puts the attached
  `navalign-src` dataset on `sys.path`, writes `max_output_tokens` into `sampling.yaml` and the
  budgets into `eval_config.yaml`.
- **C** (replaces the loop at the end of the Getting Started evaluation cell; the `EvalConfig` /
  `Evaluator` setup above it stays): runs `navalign.collect.collect_run`, which calls
  `run_sync(evaluator.evaluate_task, task=...)` one task at a time and appends a row to
  `results.jsonl` as each task ends.

Run it with Save Version, not a draft session. `collect_run` does not start a task that could run
past the cap (12 h minus 30 min, at most 20 min per task); such tasks get a `skipped: deadline`
row. An exception from the harness becomes an error row and the loop continues. To resume, attach
the earlier version's output as input and set `RESUME_FROM`: finished tasks are skipped, error and
skipped tasks run again. `select_tasks(ids, shard, n_shards)` splits the sorted ids into disjoint
shards; run 1 splits the 129 tasks into three Save Versions (~43 tasks, ~7.5 h each).

| File in `/kaggle/working` | Contents |
|---|---|
| `manifest.json` | run id, shard, task ids, `sampling.yaml` and `evaluation` settings as run, start time |
| `results.jsonl` | per attempt: `resolved`, `test_exit_code`, `agent_patch`, `tool_calls` (count), `duration_seconds`, `started`/`finished`, `trace_found`, `extra` (other result fields), `error` |
| `results/traces/trace_<id>.json`, `results/logs/<id>.log` | written by the harness (the logs were empty in `val1`) |

Download the version output and check it:

```
python -m navalign.runs --run results/traces/val1 --csv results/traces/val1/summary.csv
```

This prints one row per task (latest attempt; `calls_<tool>` counts from the trace, whether the
context limit was hit) and a `PROBLEM` line for each task with no row, not run, failed, without a
trace, or whose `tool_calls` differs from the trace's call count. The harness count leaves out
`submit_patch`, so the check does too. It exits 1 if there are any problems.

A run collected in shards is checked as one, by repeating `--run`; a task that no shard ran, or
that two shards both completed, is a problem:

```
python -m navalign.runs --run results/traces/run1/shard0 --run results/traces/run1/shard1 --run results/traces/run1/shard2
```

## Explored set

`navalign.explore` turns each tool call into read, seen and edit events (Methodology step 3):

| Event | From |
|---|---|
| read | `read_file` (range from the observation's `start_line`/`end_line`; the tool shows at most 150 lines and ignores garbled argument keys); `cat`, `nl` (whole file), `head` (1–N), `sed -n 'a,bp'`, `tail` (`last` = N, position unknown) |
| seen | files in `grep`/`rg` output (or the one file searched), `find` and `ls` output; nodes in `search_similar_code` / `get_code_neighbors` results |
| edit | `edit_file`, `write_file`, `sed -i`, `>` / `>>` redirects, whether or not they succeeded |

Paths are made repo-relative (`/workspace/`, sandbox `/tmp/.../workspace/` and `./` are
stripped; paths outside the repo are dropped). The exploration window is every call before the
first edit of a file that exists in the snapshot (`files.jsonl`) or is a target file, so a
repro script written first doesn't end it. `created_files` lists written paths absent from the
snapshot; for new-file targets it shows whether the agent created the target.

```
python -m navalign.explore --run results/traces/val1 --groundtruth results/groundtruth --out explored.jsonl --csv explored.csv
```

The summary has `window_end`, the reads in the window, `first_target_read` (the first call in
the window that reads a target file) and `created_target`. These are diagnostics, not scores.

Heredoc bodies (`cat <<EOF > repro.py ... EOF`) are dropped before a command is parsed, and a
newline separates commands like `;`, so a script the agent writes isn't read as `cat` operands.

## Scoring

`navalign.score` turns an exploration window into the Methodology step 4 scores. All of it runs
on the ground-truth files; no snapshot or graph is needed.

| Column | Meaning |
|---|---|
| `k`, `s_file` | 1-based position of the first call that reads a target file; S_file = 1/k (0 if none) |
| `focus` | share of the window's read calls that read a target file |
| `s_node`, `mean_d`, `cap` | S_node = 1 − mean(d)/C over read calls, for tasks with a target node |
| `k_node` | first call whose read shows a target node |
| `k_seen` | first call whose results show a target file or node, read or only seen |
| `created_target` | the agent wrote a file the patch creates (new-file-only tasks have no `s_file`) |

d of a read is the hop count of the closest graph node it shows (C if it shows none); a read of a
target file whose changes map to no node (module-level, async or added code, non-Python) gets 0.
A call with several reads takes the smallest d. Reads of paths that are in neither the snapshot
nor the targets (the agent's scratch scripts) count toward neither focus nor S_node. `tail -n N`
is placed at the file's last located node (the ground truth has no file lengths).

```
python -m navalign.score --run results/traces/run1/shard0 --run results/traces/run1/shard1 \
    --run results/traces/run1/shard2 --groundtruth results/groundtruth \
    --out results/analysis/run1/scores.csv --steps results/analysis/run1/steps.jsonl
```

`--steps` writes every window call's files read, target hits and d (the per-step distances).
`score_task(task, graphs_dir, snapshot, trace)` scores one ATIF trace straight from a task, its
graph and its snapshot, building the ground truth on the way:

```python
from navalign.data import load_tasks
from navalign.score import score_task
from navalign.trace import load_trace

task = load_tasks("data/tasks.jsonl")["requests_7427"]
score = score_task(task, "data/graphs", "data/snapshots/requests_7427.tgz",
                   load_trace("results/traces/val1/results/traces/trace_requests_7427.json"))
print(score.k, score.s_file, score.s_node)  # 2 0.5 0.946...
```

## Outcomes

`navalign.outcomes.run_outcomes` gives each run one category, in this order of precedence:

| Category | Rule |
|---|---|
| X | ended by the context limit |
| A | no read of a target file in the window (new-file-only tasks: the target was never created) |
| B | read one, but never tried to edit a target file |
| C1 | edited a target file and failed, with a failed edit of a target file labelled `escaping` (R3) |
| C2 | edited a target file and failed otherwise |
| D | edited a target file and passed |
| CD_ng | edited a target file on a task that isn't gradable, or whose tests never ran |

"Edited" counts attempts (failed ones too) anywhere in the trace. PASS/FAIL is the regrade in
the validated environment. Finer outcomes per run: `patch_produced`,
`patch_touches_target_file`, `patch_touches_target_node` (a changed line of the agent's patch,
placed on the base file, inside a target node's span), `tests_passed`/`tests_total` (pytest's
summary line) and the R3 label counts on target files (`r3_escaping`, `r3_malformed`, ...).

## Validity checks (Checkpoint 2)

```
python -m navalign.validity baselines   --run ... --groundtruth results/groundtruth --out results/analysis/run1
python -m navalign.validity label-sheet --run ... --groundtruth results/groundtruth --out results/analysis/run1 --n 10
python -m navalign.validity agreement   --run ... --groundtruth results/groundtruth \
    --labels results/analysis/run1/human_labels.csv --out results/analysis/run1
```

- `baselines`: an oracle window (every call shows a target node) must score 1 on S_file, focus
  and S_node; a random window (each read call pointed at a random Python file of the snapshot,
  150 lines from line 1 or a node's start, 200 draws) must score clearly below the real runs.
  Writes `baselines.csv` (per task), `validity.csv` (T3) and `random_k.csv`; exits 1 if the check fails.
- `label-sheet`: `human_check.md` with 10 traces' raw window calls (stratified by the scorer's
  level, scores not shown) and a blank `human_labels.csv`. It never overwrites a labels file.
- `agreement`: Cohen's kappa between the hand labels (0 no target file read, 1 target file only,
  2 a target node's lines read) and the scorer's level, and the rank correlation with S_node.

## Analysis

```
pip install -e .[analysis]   # matplotlib, for --figures
python -m navalign.analysis --run ... --groundtruth results/groundtruth --grades results/gold/final \
    --gradable results/gold/gradable_tasks.csv --out results/analysis/run1 --figures
```

| File | Contents |
|---|---|
| `outcomes.csv` | one row per run: scores, category, finer outcomes |
| `t1_groundtruth.csv` | tasks, gradable, target-node coverage, fallback rules, span coverage, caps per repo |
| `t2_breakdown.csv` | runs per category and repo (the main result) |
| `t4_predictive.csv` | AUC (bootstrap 95% CI, permutation p) of S_file, focus, S_node, k_node, the same three scores over each run's first `--prefix` calls (default 10; `s_file_first10`, ...), and window length, for: editing a target file among runs that found one; the patch touching a target node; passing among graded runs that edited one |
| `figures/` | F1 breakdown, F2 first target read vs random (needs `random_k.csv`), F3 scores by outcome, F4 per-step distance for three runs (needs `steps.jsonl`) |

The score is never used to predict A vs the rest (A is its own file-level test). A run that
never edits a target file has a longer window, with more room to stray, so T4 lists window length
as a reference and repeats the scores over an equal span (`score.prefix_scores`: trace positions
1..N of the window; `outcomes.csv` columns `s_file_prefix`, `focus_prefix`, `s_node_prefix`).

## Escaping check (R3)

Tool results reach the model JSON-encoded twice (forum 744272). `navalign.escaping` compares
each failed `edit_file`'s `old_string` with the text the agent had for that file (earlier
reads, writes and successful edits): `stale` if it matches as given (the file changed since),
`escaping` if it only matches with one or two levels of escaping undone or its backslashes
doubled (both directions occur: in `val1` the model wrote `\0` for the file's `\\0`), else
`no_match`. The tools are not patched.

```
python -m navalign.escaping --run results/traces/val1 --csv edit_failures.csv
```

## Gold-patch control (R1)

`navalign.grade` runs patches through the harness's own test step and records one row per task
and variant in `grades.jsonl`. `notebooks/gold_control.py` holds the Kaggle CPU cells: D0 reads
the harness source, D1 holds the settings and repairs, D2 calls `swegemma`'s `verify_task`
with each patch (no model). Variants: `gold` and `empty` (the reference control), or `agent`
(regrade a run's patches).

A task is gradable when its gold patch passes and its empty patch fails. In the notebook the
sandbox is a fresh venv per task that sees the notebook's own site-packages, and the harness's
install steps do nothing, so `ENV = "repaired"` adds a `.pth` file to each sandbox venv. It
puts the workspace's `src/` (so a patched src-layout repo shadows the installed release), the
missing test packages, and per-task pins (the starlette range the snapshot's `pyproject.toml`
declares) ahead of the host's packages. Kaggle allows no Internet in notebooks attached to the
competition, so the packages come from the private dataset `kaic87/navalign-repair-wheels`, built
by `notebooks/build_repair_wheels.sh` (the missing test packages with their dependencies, and
every starlette release since 0.27). D1 installs only the distributions the notebook's Python
lacks, so nothing it already has is shadowed.

```
python -m navalign.grade --grades results/gold/repaired --csv results/gold/gradable_tasks.csv
python -m navalign.grade --grades results/gold/run1-agent --gradable results/gold/gradable_tasks.csv
```

The summary lists the reasons (`ok`, `gold_fails`, `empty_passes`, `unstable`, `error`,
`missing`), the gold pass and gradable counts, and the `ModuleNotFoundError` modules by task
count, which feed the next round of repairs. `--replicate DIR[:repo,...]` adds another control
run of the same environment; a task whose (gold, empty) outcome differs is `unstable`. For agent
grades, exit code −1 means the tests never ran (a patch failed to apply), so those rows are left
out of pass rates (`tests_ran` in `--agent-csv`).

Grading must use the harness version the traces were collected with: run 1 used swegemma 0.2.7
(host wheelhouse Version 28), and D1 asserts it. `kaggle kernels push` can't pin a dataset
version, so pin the wheelhouse input in the notebook's UI before running.

```
python -m navalign.grade --grades results/gold/final --replicate results/gold/repaired-v6 \
    --replicate results/gold/repaired --replicate results/gold/shipped:rich \
    --csv results/gold/gradable_tasks.csv --agent-csv results/gold/agent_grades.csv
```

## Layout

| Path | Contents |
|---|---|
| `src/navalign/` | Toolkit package: `data` (loaders), `patch`, `symbols`, `mapper`, `distance`, `spans`, `build`, `trace`, `collect`, `runs`, `explore`, `escaping`, `grade`, `score`, `outcomes`, `validity`, `analysis`, `figures` |
| `tests/` | Tests (the `val1` and dev-data tests skip when those git-ignored files are absent) |
| `fixtures/trace_step_example.json` | One real ATIF-v1.7 step from the smoke test, for building the trace parser |
| `notebooks/` | `collect_traces.py` (Kaggle GPU cells for trace collection); `gold_control.py` (Kaggle CPU cells for the gold-patch control and regrading); `build_repair_wheels.sh` (builds the offline repair wheels dataset) |
| `data/` | Local competition data subset (git-ignored, see `data/README.md`) |

## Setup (Windows, VS Code terminal)

```
py -m venv .venv
.venv\Scripts\activate
pip install -e .[test,analysis]
pytest
```

Tests on the 5 dev tasks are skipped when `data/` is absent.

## Data policy

Competition rules 2.4.b forbid redistributing Competition Data to non-participants, so `data/`,
`*.tgz` and `*.npz` are git-ignored. Publicly shared competition code must also be shared on Kaggle
(Foundational rule 6.b).
