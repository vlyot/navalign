# nav-alignment-toolkit

Navigation-alignment scoring for coding-agent traces on the Gemma 4 Developer Agent benchmark:
given a task's code graph, its reference patch, and an agent's ATIF trace, score how directly the
agent's exploration converged on the code the real fix touched.

## Status

Built: data loaders, patch parser, `ast` symbol index, patch-to-node mapper, distance maps,
node span index, one-pass ground-truth build, ATIF trace parser, trace-collection bookkeeping and
run checker.
Not yet: explored-set parser (waits for a real trace file), scorer.

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
| `summary.csv` | counts, coverage, cap, seconds, and `error` (empty on success) |

Load them back with `DistanceMap.from_dict` and `SpanIndex.from_dict`.

## Collecting traces

Traces come from the harness `Evaluator` in a Kaggle GPU notebook (a Copy & Edit of the Getting
Started notebook). `notebooks/collect_traces.py` holds the cells to paste in:

- **A** (first cell): run id, task list or shard, `RESUME_FROM`, `max_output_tokens` (8192) and the
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
shards, so the 129 tasks can run as two Save Versions.

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

## Layout

| Path | Contents |
|---|---|
| `src/navalign/` | Toolkit package: `data` (loaders), `patch`, `symbols`, `mapper`, `distance`, `spans`, `build`, `trace`, `collect`, `runs` |
| `tests/` | Tests |
| `fixtures/trace_step_example.json` | One real ATIF-v1.7 step from the smoke test, for building the trace parser |
| `notebooks/` | `collect_traces.py` (Kaggle cells for trace collection); analysis notebooks |
| `data/` | Local competition data subset (git-ignored, see `data/README.md`) |

## Setup (Windows, VS Code terminal)

```
py -m venv .venv
.venv\Scripts\activate
pip install -e .[test]
pytest
```

Tests on the 5 dev tasks are skipped when `data/` is absent.

## Data policy

Competition rules 2.4.b forbid redistributing Competition Data to non-participants, so `data/`,
`*.tgz` and `*.npz` are git-ignored. Publicly shared competition code must also be shared on Kaggle
(Foundational rule 6.b).
