# How navalign scores a trace

navalign works in four steps. The ground truth comes from the task alone (no model): which files
and graph nodes the reference patch changed, how far every node is from them, and where each node
sits in the source. The explored set comes from the trace: what each tool call read, saw or
edited. The scores measure the reads in the exploration window against the ground truth. The
outcome puts each run in one category.

## 1. Ground truth

### Target mapping

Every non-blank changed line of a reference patch is resolved to one rule. Removed lines are
resolved against the base file and added lines against the post-patch file.

| Rule | Meaning |
|---|---|
| `exact` | the innermost function/class containing the line is a graph node |
| `enclosing` | the innermost symbol is not a node (async, added by the patch), an enclosing one is |
| `file` | no containing symbol is a node (module-level line, top-level async/added function) |
| `new_file` | the patch creates the file |
| `non_python` | docs, config, etc. |

Target files are every file the patch touches; target nodes are the nodes of the `exact` and
`enclosing` lines. Node coverage is the share of changed lines that map to a node; tasks with none
get file-level scores only.

Graph ids keep enclosing classes and drop enclosing functions (a function nested in
`should_bypass_proxies` is `requests.utils.get_proxy`), and module names drop `src/` and
`docs_src/`; the mapper follows both conventions.

### Distance maps

`navalign.distance` gives each graph node's shortest-path distance to the nearest target node on
the `calls` graph, with edge direction ignored (an agent can follow a call either way). Nodes that
reach no target get the cap C = (largest finite distance in the graph) + 1, i.e. the largest
component diameter + 1. The cap belongs to the graph, so all tasks on one commit share it
(Gemma 4 benchmark graphs: 13–19). Tasks with no target node get an empty map and no node-level
score.

### Node span index

The graphs have no line numbers. `navalign.spans` finds each node in the snapshot source with the
same naming rules as the mapper and records `(file, start, end)`, so the scorer can turn a read of
lines a–b into the nodes it covers without the snapshots. Nodes that no file defines, or that two
files both claim, are listed as `unlocated`. On the Gemma 4 benchmark's development tasks the
coverage is rich 100%, fastapi 99–100%, requests 79%; the misses are nodes whose enclosing class
the graph drops (requests test methods such as `tests.test_requests.test_unicode_get`, fastapi's
pydantic-v1 `Config` classes). Library functions are fully located.

## 2. Explored set

`navalign.explore` turns each tool call into read, seen and edit events:

| Event | From |
|---|---|
| read | `read_file` (range from the observation's `start_line`/`end_line`; the tool shows at most 150 lines and ignores garbled argument keys); `cat`, `nl` (whole file), `head` (1–N), `sed -n 'a,bp'`, `tail` (`last` = N, position unknown) |
| seen | files in `grep`/`rg` output (or the one file searched), `find` and `ls` output; nodes in `search_similar_code` / `get_code_neighbors` / `get_code_subgraph` results |
| edit | `edit_file`, `write_file`, `sed -i`, `>` / `>>` redirects, whether or not they succeeded |

Paths are made repo-relative (`/workspace/`, sandbox `/tmp/.../workspace/` and `./` are
stripped; paths outside the repo are dropped). The **exploration window** is every call before the
first edit of a file that exists in the snapshot or is a target file, so a repro script written
first doesn't end it. `created_files` lists written paths absent from the snapshot; for new-file
targets it shows whether the agent created the target.

Heredoc bodies (`cat <<EOF > repro.py ... EOF`) are dropped before a command is parsed, and a
newline separates commands like `;`, so a script the agent writes isn't read as `cat` operands.

## 3. Scores

`navalign.score` turns an exploration window into scores. It runs on the ground-truth files; no
snapshot or graph is needed.

| Column | Meaning |
|---|---|
| `k`, `s_file` | 1-based position of the first call that reads a target file; S_file = 1/k (0 if none) |
| `focus` | share of the window's read calls that read a target file |
| `s_node`, `mean_d`, `cap` | S_node = 1 − mean(d)/C over read calls, for tasks with a target node |
| `k_node` | first call whose read shows a target node |
| `k_seen` | first call whose results show a target file or node, read or only seen |
| `created_target` | the agent wrote a file the patch creates (new-file-only tasks have no `s_file`) |

k counts every call in the trace, a sub-agent's included. d of a read is the hop count of the
closest graph node it shows (C if it shows none); a read of a target file whose changes map to no
node (module-level, async or added code, non-Python) gets 0, since that file has no finer
location. A call with several reads takes the smallest d. Reads of paths that are in neither the
snapshot nor the targets (the agent's scratch scripts) count toward neither focus nor S_node.
`tail -n N` is placed at the file's last located node (the ground truth has no file lengths).

`score.prefix_scores(score, n)` repeats S_file, focus and S_node over trace positions 1..n of the
window only. A run that never edits a target file has a longer window, with more room to stray, so
comparing runs over the same span keeps window length out of the comparison.

S_node measures how close the reads were, not whether the agent "found the code": a read one or
two hops from the fix scores high. On run 1's hand-labelled traces S_file and k_node agree with the
labels (κ = 0.85); S_node doesn't track them.

## 4. Outcomes

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

The categories use the same file-level test as S_file, so the score is never used to predict A
against the rest.

### Escaping check (R3)

Tool results reach the model JSON-encoded twice (Kaggle forum topic 744272). `navalign.escaping`
compares each failed `edit_file`'s `old_string` with the text the agent had for that file (earlier
reads, writes and successful edits): `stale` if it matches as given (the file changed since),
`escaping` if it only matches with one or two levels of escaping undone or its backslashes
doubled (both directions occur), else `no_match`. The tools are not patched.
