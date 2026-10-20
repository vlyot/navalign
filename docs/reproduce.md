# Reproducing the run-1 results

The fastest route is the Kaggle notebook
[navalign quickstart](https://www.kaggle.com/code/kaic87/navalign-quickstart): it rebuilds the
ground truth from the competition data (in `/tmp`, outside its public output), checks it against the
published ground-truth table, and redraws the failure breakdown (T2) and figures F1–F3 from the
`navalign-results` Dataset. This page is the full pipeline, from the competition data to every table.

Nothing derived from the competition data is published: not the ground truth, the traces (their
tool outputs quote the snapshots' code) or the grade logs. `navalign-results` holds the computed
tables and `gradable_tasks.csv`; `release/build_kaggle.py` (outside the public repository) lists
every file.

Competition rules (2.4.b) forbid redistributing the competition data, so `data/` is git-ignored;
download it from the competition page.

## Setup (Windows, VS Code terminal)

```
py -m venv .venv
.venv\Scripts\activate
pip install -e .[test,analysis]
pytest
```

Tests that need the competition data or run 1 skip when those git-ignored files are absent.

## 1. Build the ground truth

```
navalign build --data data --out results/groundtruth
navalign build --data data --task rich_3944 --out results/groundtruth
```

`--tasks`, `--graphs` and `--snapshots` override the paths under `--data`. Without `--task`, every
task with a snapshot is built. A failing task is recorded in `summary.csv` and the run continues
(exit code 1 if any failed). The output files are described in [inputs.md](inputs.md). The full
129-task build runs in a Kaggle CPU notebook in about 8 minutes.

`navalign mapper --data data [--task ID] [--out targets.jsonl]` runs the patch-to-node mapping
alone.

## 2. Collect traces on Kaggle

Traces come from the harness `Evaluator` in a Kaggle GPU notebook (a Copy & Edit of the Getting
Started notebook). [`notebooks/collect_traces.py`](../notebooks/collect_traces.py) holds the
cells to paste in:

- **A** (first cell): run id, task list or shard (run 1: `TASK_IDS = None`, shard k of 3),
  `RESUME_FROM`, `max_output_tokens` (8192) and the eval budgets (300 s / 50 calls / 10 min /
  100 turns), and the clock for the 12-hour cap.
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
shards; run 1 split the 129 tasks into three Save Versions (~43 tasks, ~7.5 h each).

| File in `/kaggle/working` | Contents |
|---|---|
| `manifest.json` | run id, shard, task ids, `sampling.yaml` and `evaluation` settings as run, start time |
| `results.jsonl` | per attempt: `resolved`, `test_exit_code`, `agent_patch`, `tool_calls` (count), `duration_seconds`, `started`/`finished`, `trace_found`, `extra` (other result fields), `error` |
| `results/traces/trace_<id>.json`, `results/logs/<id>.log` | written by the harness |

Download the version output and check it; a run collected in shards is checked as one:

```
navalign runs --run results/traces/run1/shard0 --run results/traces/run1/shard1 --run results/traces/run1/shard2
```

This prints one row per task (latest attempt; `calls_<tool>` counts from the trace, whether the
context limit was hit) and a `PROBLEM` line for each task with no row, not run, failed, without a
trace, or whose `tool_calls` differs from the trace's call count (the harness count leaves out
`submit_patch`, `get_status` and the call that launches the sub-agent). A task that no shard ran,
or that two shards both completed, is a problem too. It exits 1 if there are any.

## 3. Explored sets and the escaping check

```
navalign explore --run results/traces/val1 --groundtruth results/groundtruth --out explored.jsonl --csv explored.csv
navalign escaping --run results/traces/val1 --csv edit_failures.csv
```

The explore summary has `window_end`, the reads in the window, `first_target_read` and
`created_target`: diagnostics, not scores. The escaping labels are defined in
[methodology.md](methodology.md#escaping-check-r3).

## 4. Gold-patch control and regrading (R1)

`navalign.grade` runs patches through the harness's own test step and records one row per task
and variant in `grades.jsonl`. [`notebooks/gold_control.py`](../notebooks/gold_control.py) holds
the Kaggle CPU cells: D0 reads the harness source, D1 holds the settings and repairs, D2 calls
`swegemma`'s `verify_task` with each patch (no model). Variants: `gold` and `empty` (the
reference control), or `agent` (regrade a run's patches).

A task is gradable when its gold patch passes and its empty patch fails. In the notebook the
sandbox is a fresh venv per task that sees the notebook's own site-packages, and the harness's
install steps do nothing, so `ENV = "repaired"` adds a `.pth` file to each sandbox venv. It puts
the workspace's `src/` (so a patched src-layout repo shadows the installed release), the missing
test packages, and per-task pins (the starlette range the snapshot's `pyproject.toml` declares)
ahead of the host's packages. Kaggle allows no Internet in notebooks attached to the competition,
so the packages come from the private dataset `kaic87/navalign-repair-wheels`, built by
[`notebooks/build_repair_wheels.sh`](../notebooks/build_repair_wheels.sh). D1 installs only the
distributions the notebook's Python lacks, so nothing it already has is shadowed.

```
navalign grade --grades results/gold/final --replicate results/gold/repaired-v6 \
    --replicate results/gold/repaired --replicate results/gold/shipped:rich \
    --csv results/gold/gradable_tasks.csv --agent-csv results/gold/agent_grades.csv
```

The summary lists the reasons (`ok`, `gold_fails`, `empty_passes`, `unstable`, `error`,
`missing`), the gold pass and gradable counts, and the `ModuleNotFoundError` modules by task
count. `--replicate DIR[:repo,...]` adds another control run of the same environment; a task whose
(gold, empty) outcome differs is `unstable`. For agent grades, exit code −1 means the tests never
ran (a patch failed to apply), so those rows are left out of pass rates (`tests_ran` in
`--agent-csv`).

Grading must use the harness version the traces were collected with: run 1 used swegemma 0.2.7
(host wheelhouse Version 28), and D1 asserts it. `kaggle kernels push` can't pin a dataset
version, so pin the wheelhouse input in the notebook's UI before running.

## 5. Scores

```
navalign score --run results/traces/run1/shard0 --run results/traces/run1/shard1 \
    --run results/traces/run1/shard2 --groundtruth results/groundtruth \
    --out results/analysis/run1/scores.csv --steps results/analysis/run1/steps.jsonl
```

`--steps` writes every window call's files read, target hits and d. The columns are defined in
[methodology.md](methodology.md#3-scores).

## 6. Validity checks

```
navalign validity rankings    --tasks data/tasks.jsonl --graphs data/graphs --groundtruth results/groundtruth \
    --out results/groundtruth-ast-out/rankings.jsonl
navalign validity baselines   --run ... --groundtruth results/groundtruth --out results/analysis/run1 \
    --rankings results/groundtruth-ast-out/rankings.jsonl
navalign validity label-sheet --run ... --groundtruth results/groundtruth --out results/analysis/run1 --n 10
navalign validity label-sheet --run ... --groundtruth results/groundtruth --out results/analysis/run1/hand-random \
    --n 25 --sample random --seed 1 --exclude results/analysis/run1/human_labels.csv
navalign validity agreement   --run ... --groundtruth results/groundtruth \
    --labels results/analysis/run1/human_labels.csv --out results/analysis/run1
```

- `baselines`: an oracle window (every call shows a target node) must score 1 on S_file, focus
  and S_node; a random window (each read call pointed at a random Python file of the snapshot,
  150 lines from line 1 or a node's start, 200 draws) must score clearly below the real runs.
  Writes `baselines.csv` (per task), `validity.csv` (T3) and `random_k.csv`; exits 1 if the check
  fails. With `--rankings` it adds two heuristic baselines with the same window: the i-th read
  becomes 150 lines from the i-th entry of a fixed ranking, the files and functions the issue
  names (in order of mention; names under 4 characters or defined more than 3 times skipped), or the
  graph functions ranked by BM25 between the issue text and each function's source. `validity.csv`
  gains `issue_*` and `bm25_*` columns with real − baseline bootstrap CIs.
- `rankings`: those two rankings per task. BM25 needs the graphs' node text, so for the 129 tasks it
  runs in the Kaggle notebook below.
- `label-sheet`: `human_check.md` with 10 traces' raw window calls (stratified by the scorer's
  level, scores not shown) and a blank `human_labels.csv`. It never overwrites a labels file.
  `--sample random` draws uniformly instead; `--exclude` skips tasks an earlier labels file has.
- `agreement`: Cohen's kappa between the hand labels (0 no target file read, 1 target file only,
  2 a target node's lines read) and the scorer's level, and the rank correlation with S_node.

## 7. Analysis

```
navalign analysis --run ... --groundtruth results/groundtruth --grades results/gold/final \
    --gradable results/gold/gradable_tasks.csv --out results/analysis/run1 --figures
```

| File | Contents |
|---|---|
| `outcomes.csv` | one row per run: scores, category, finer outcomes |
| `t1_groundtruth.csv` | tasks, gradable, target-node coverage, fallback rules, span coverage, caps per repo |
| `t2_breakdown.csv` | runs per category and repo (the main result) |
| `t4_predictive.csv` | AUC (bootstrap 95% CI, permutation p) of S_file, focus, S_node, k_node, the same three scores over each run's first `--prefix` calls (default 10; `s_file_first10`, ...), and window length, for: editing a target file among runs that found one; the patch touching a target node; passing among graded runs that edited one |
| `figures/` | F1 breakdown, F2 first target read vs random (needs `random_k.csv`), F3 scores by outcome, F4 per-step distance for three runs (needs `steps.jsonl`) |

`t4_predictive.csv` also has `p_holm`: the permutation p-values Holm-adjusted over every test in
the table except window length (21 tests).

The score is never used to predict A vs the rest (A is its own file-level test). T4 lists window
length as a reference and repeats the scores over an equal span (`outcomes.csv` columns
`s_file_prefix`, `focus_prefix`, `s_node_prefix`).

## 8. Run 2 and the comparison

Run 2 is run 1's setup with `max_tool_calls` 10 instead of 50 (cell A: `RUN_ID = "run2"`, one
shard), regraded with `VARIANT = "agent"` and `RESULTS_DATASET, N_RUN_SHARDS =
"navalign-run2-results", 1` in D1. Sections 3 to 7 then run on `results/traces/run2` with
`--grades results/gold/run2-agent` and `--out results/analysis/run2`, and

```
navalign compare --a results/analysis/run1 --a-run results/traces/run1/shard0 \
    --a-run results/traces/run1/shard1 --a-run results/traces/run1/shard2 \
    --b results/analysis/run2 --b-run results/traces/run2 --out results/analysis/compare
```

| File | Contents |
|---|---|
| `compare.csv` | per run: call budget, resolved, passes (all and on gradable tasks whose tests ran), non-empty patches, call counts, categories, runs at the budget, mean scores, the S_file and −k_node AUCs of T4 |
| `paired.csv` | per-task agreement: S_file rank correlation, same k, found in both or one run, same first call, shared leading calls |
| `transitions.csv` | each task's category in run 1 against run 2 |

## 9. The ast graph (closeness on a rebuilt graph)

The Kaggle CPU notebook `kaic87/navalign-ground-truth-build` (version 4, source
[`notebooks/ast_groundtruth.py`](../notebooks/ast_groundtruth.py); inputs: the competition data,
`navalign-src`, the private `navalign-groundtruth`) builds an `ast` graph per commit, the ground truth on those
graphs, the issue/BM25 rankings, and `graph_compare.csv` (per commit: nodes, edges, share of
shipped edges the ast graph has, isolated nodes). The graphs stay in the notebook's `/tmp`.

```
kaggle kernels output kaic87/navalign-ground-truth-build -p results/groundtruth-ast-out
navalign score --run ... --groundtruth results/groundtruth-ast-out/groundtruth-ast --out results/analysis/run1-ast/scores.csv
navalign validity agreement --run ... --groundtruth results/groundtruth-ast-out/groundtruth-ast \
    --labels results/analysis/run1/human_labels.csv --out results/analysis/run1-ast
```

## 10. Outside the competition: SWE-agent on SWE-bench Verified

```
python notebooks/fetch_swebench.py                     # 60 tasks: requests, flask, pytest, pylint, xarray
E=data/external/swebench
navalign graph --tasks $E/tasks.jsonl --snapshots $E/snapshots --out $E/graphs
navalign build --data $E --out $E/groundtruth
navalign swe-agent --trajs $E/trajs --results $E/results.json --out $E/run
navalign analysis --run $E/run --groundtruth $E/groundtruth --grades $E/run --gradable $E/run/gradable_tasks.csv --out $E/analysis
navalign validity rankings --tasks $E/tasks.jsonl --graphs $E/graphs --groundtruth $E/groundtruth --out $E/rankings.jsonl
navalign validity baselines --run $E/run --groundtruth $E/groundtruth --rankings $E/rankings.jsonl --out $E/validity
```

The trajectories are the SWE-bench experiments submission `20250511_sweagent_lm_32b`
(SWE-agent 1.0.1 with SWE-agent-LM-32B), from the public `swe-bench-submissions` S3 bucket;
pass/fail is that submission's `results.json`. The graphs are `ast` graphs, since SWE-bench ships none.

## Repository layout

| Path | Contents |
|---|---|
| `src/navalign/` | the package: `data` (loaders), `patch`, `symbols`, `mapper`, `distance`, `spans`, `build`, `trace`, `collect`, `runs`, `explore`, `escaping`, `grade`, `score`, `outcomes`, `validity`, `analysis`, `figures`, `compare`, `callgraph`, `swe_agent`, `cli` |
| `src/navalign/examples/toy/` | the bundled toy example (`navalign score-trace --demo`) |
| `tests/` | tests; the ones that need the competition data or run 1 skip when those are absent |
| `fixtures/trace_step_example.json` | one real ATIF-v1.7 step, for the trace parser's tests |
| `notebooks/` | Kaggle cells: `collect_traces.py` (GPU, trace collection), `gold_control.py` (CPU, gold-patch control and regrading), `quickstart.py` (the public notebook), `ast_groundtruth.py` (CPU, ast graphs and baseline rankings); `fetch_swebench.py` (the external SWE-bench set); `build_repair_wheels.sh` (the offline repair wheels) |
| `docs/` | this page, [methodology.md](methodology.md), [inputs.md](inputs.md) |
