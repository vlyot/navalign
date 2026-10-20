# Inputs, and using navalign on another benchmark

navalign was built on the Gemma 4 Developer Agent benchmark, but nothing in it is specific to that
benchmark's tasks. Per task it needs four things: the reference patch, the repository at the base
commit, a call graph, and the agent's trace. The bundled toy example
([`src/navalign/examples/toy/`](../src/navalign/examples/toy)) has one of each and doubles as a
format reference.

## Layout

```
mydata/
  tasks.jsonl                          one task per line
  graphs/<repo>_<base_commit>.json     one call graph per commit
  snapshots/<instance_id>.tgz          or an extracted snapshots/<instance_id>/ folder
```

```
navalign build --data mydata --out groundtruth                 # once per task set
navalign score --run mytraces --groundtruth groundtruth --out scores.csv --steps steps.jsonl
navalign score-trace mytraces/trace_<instance_id>.json --groundtruth groundtruth --steps
```

## Task (`tasks.jsonl`)

One JSON object per line. navalign reads four fields; others are ignored.

| Field | Example | Use |
|---|---|---|
| `instance_id` | `toyshop_1` | names the snapshot and the trace |
| `repo` | `example/toyshop` | the part after the last `/` names the graph file |
| `base_commit` | `0123…4567` | names the graph file |
| `patch` | `diff --git a/… b/…` | the reference fix, a unified diff against the snapshot |

## Snapshot

The repository at `base_commit`, before the fix: a `.tgz` (entries may start with `./`) or a
folder, named after the `instance_id`. Paths in it are repo-relative.

## Call graph

NetworkX node-link JSON with the edge list under `edges` (load it with
`networkx.readwrite.json_graph.node_link_graph(data, edges="edges")`), as in the Gemma 4
benchmark:

- **Nodes**: one per function and class. The id is the dotted module path plus the class chain and
  the name: `toyshop.pricing.apply_discount`, `toyshop.cart.Cart.total`. Enclosing classes are
  kept and enclosing functions dropped (a function nested in `f` is named as if it were at `f`'s
  level). Module names drop a leading `src/` or `docs_src/`. Async functions may be left out; their
  lines then map to the enclosing node or to the file.
- **Edges**: calls between nodes, plus class ↔ method and class ↔ base class. Direction is
  ignored, and every edge in the `edges` list counts (edge `type` is not used; node `text` is used
  only by the BM25 baseline).

`navalign graph --tasks tasks.jsonl --snapshots DIR --out DIR` builds such a graph per commit with
Python's `ast`, async functions included. A call is resolved through the calling function's nested
defs, the module's defs and imports (relative imports and re-exports followed), `self`/`cls`/`super()`
methods through same-repo base classes, and finally a name defined only once in the repository;
anything still ambiguous makes no edge. Edges are typed `calls`, `contains` (class → member) or
`inherits` (class → base). On the Gemma 4 benchmark these graphs hold 97–99% of the shipped graphs'
edges (80% for httpx), and 33–158% more edges in all.

A node the snapshot doesn't define is reported as `unlocated` and never counts as read.

## Trace

ATIF v1.7 JSON, as the Gemma 4 benchmark's harness writes it: `steps`, each with optional
`tool_calls` (`tool_call_id`, `function_name`, `arguments`) and an `observation` whose `content`
is the tool's JSON result (or a `results` list matched by `source_call_id`). The scorer
understands these tools:

| Tool | Read as | Fields used |
|---|---|---|
| `read_file` | a read of lines `start_line`–`end_line` | observation `status`, `filepath`, `start_line`, `end_line`, `is_truncated` |
| `run_command` | reads (`cat`, `head`, `tail`, `sed -n`, `nl`), seen files (`grep`, `rg`, `find`, `ls`), edits (`sed -i`, redirects) | argument `command`, observation `status`, `stdout` |
| `edit_file`, `write_file` | edits | argument `filepath` |
| `search_similar_code`, `get_code_neighbors`, `get_code_subgraph` | seen nodes | node names in the results |

SWE-agent's two tools are understood too (paths under `/testbed/` are made repo-relative):

| Tool | Read as | Fields used |
|---|---|---|
| `bash` | as `run_command` | argument `command`; the observation text as `stdout` |
| `str_replace_editor` | `view`: a read of each run of line numbers its `cat -n` output shows (a directory view is seen files); `create`, `str_replace`, `insert`, `undo_edit`: edits | arguments `command`, `path`; the observation text |

`navalign swe-agent --trajs DIR --results results.json --out RUN` converts SWE-agent `.traj` files
(one action string per step) into ATIF traces plus `results.jsonl`, `grades.jsonl` (pass/fail from
SWE-bench's `resolved` list) and `gradable_tasks.csv`, so `navalign score` and `navalign analysis`
run on them unchanged. Instance ids are renamed `psf__requests-2931` → `requests_2931`.

Any other tool counts as a call (it moves k) but reads nothing.

`navalign score --run DIR` finds every `trace_<instance_id>.json` below `DIR`. If `DIR` has a
`results.jsonl` (written by `navalign.collect` on Kaggle), only the tasks it lists are scored.

## Ground truth (`navalign build` output)

For the benchmark's 129 tasks they are derived from the competition data, so they are built where
the data is (a notebook attached to the competition) and not published.

| File | One JSON line / row per task |
|---|---|
| `targets.jsonl` | `files`, `target_nodes`, `node_coverage`, `rules`, `targets` (rule per changed line), `file_nodes` |
| `spans.jsonl` | `n_nodes`, `coverage`, `spans` (`{file: [[node, start, end], ...]}`), `unlocated`, `skipped_files` |
| `distances.jsonl` | `graph_file`, `targets`, `cap`, `n_nodes`, `n_reachable`, `distances` (`{node: hops}`, reachable nodes only; any other node is at `cap`) |
| `files.jsonl` | `files`: every file in the snapshot (tells real files from the agent's scratch files) |
| `summary.csv` | counts, coverage, cap, seconds, and `error` (empty on success) |

Load them in Python with `navalign.score.load_groundtruth(dir)`.

## Limits

- You need a call graph in the naming scheme above. navalign doesn't build one.
- Line spans come from Python's `ast`, so node-level scores cover Python code only. Other files
  still count at file level.
- `tail -n N` reads are placed at the file's last located node, an estimate.
