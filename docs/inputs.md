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
- **Edges**: calls between nodes, plus class ↔ method. Direction is ignored, and only the `edges`
  list matters (edge `type` and node `text` are not used).

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

Any other tool counts as a call (it moves k) but reads nothing. Traces from agent frameworks with
other tool names (`str_replace_editor`, `bash`, ...) need their calls renamed to these before
scoring.

`navalign score --run DIR` finds every `trace_<instance_id>.json` below `DIR`. If `DIR` has a
`results.jsonl` (written by `navalign.collect` on Kaggle), only the tasks it lists are scored.

## Ground truth (`navalign build` output)

These files are what the `navalign-groundtruth` Kaggle Dataset holds for the benchmark's 129 tasks.

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
