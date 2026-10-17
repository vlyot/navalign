# Trace collection cells for the Kaggle notebook `navalign-trace-collection`
# (a Copy & Edit of kaic87/getting-started-gemma-4-developer-agent, GPU L4 x4, internet off,
# with the kaic87/navalign-src dataset attached). Paste each cell where its header says and run
# the notebook with Save Version -> Save & Run All, never as a draft session: a Save Version
# keeps /kaggle/working and may run for 12 hours.
#
# Everything the run produces lands in /kaggle/working: manifest.json, results.jsonl and the
# harness's results/traces/ and results/logs/. Download the version output to
# nav-alignment-toolkit/results/traces/<RUN_ID>/ and check it with
#   python -m navalign.runs --run results/traces/<RUN_ID>

# %% [A] Paste as the FIRST cell: run settings, and the clock for the 12-hour cap.
import time

T0 = time.time()

RUN_ID = "run2"
# None = every task (split by SHARD). val1 used the 5 dev tasks:
# ["fastapi_14303", "fastapi_15588", "fastapi_15661", "requests_7427", "rich_3944"]
TASK_IDS = None
SHARD, N_SHARDS = 0, 1          # run 1: shards 0-2 of 3 (~43 tasks, ~7.5 h each); run 2: one Save Version (~7 h)
RESUME_FROM = None              # an earlier version's output attached as input, e.g. "/kaggle/input/<dataset>"

MAX_OUTPUT_TOKENS = 8192        # R2 context fix (was 16384); the thinking budget stays at 4096
# Run 2 changes only max_tool_calls (run 1: 50) to the shipped default of 10; the shipped 1-minute and
# 60-second limits would end most runs on time instead, so the time budgets stay as in run 1.
BUDGETS = {"timeout_seconds": 300, "max_tool_calls": 10, "max_time_minutes": 10, "max_turns": 100}

KERNEL_LIMIT_S = 12 * 3600
SAVE_MARGIN_S = 30 * 60         # left over at the end for Kaggle to save the output
MAX_TASK_S = BUDGETS["max_time_minutes"] * 60 + BUDGETS["timeout_seconds"] + 5 * 60  # agent + tests + slack

# %% [B] Paste right after the Getting Started cell that loads `tasks`, and BEFORE the model
# server starts: makes navalign importable and applies the sampling and evaluation settings.
import sys
from pathlib import Path

import yaml

NAVALIGN_SRC = next(Path("/kaggle/input").rglob("navalign/collect.py")).parents[1]
sys.path.insert(0, str(NAVALIGN_SRC))


def find_one(name: str) -> Path:
    """The single copy of a config file the notebook uses (working copy first, then inputs)."""
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        found = sorted(root.rglob(name))
        if found:
            assert len(found) == 1, f"several {name} under {root}: {found}; set the path by hand"
            return found[0]
    raise FileNotFoundError(name)


def set_key(node, key: str, value) -> int:
    """Set every `key` in a nested YAML structure; returns how many were set."""
    n = 0
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
                n += 1
            else:
                n += set_key(v, key, value)
    elif isinstance(node, list):
        n = sum(set_key(v, key, value) for v in node)
    return n


SAMPLING_YAML = find_one("sampling.yaml")
EVAL_CONFIG_YAML = find_one("eval_config.yaml")

sampling = yaml.safe_load(SAMPLING_YAML.read_text())
assert set_key(sampling, "max_output_tokens", MAX_OUTPUT_TOKENS) == 1, "max_output_tokens not found exactly once"
SAMPLING_YAML.write_text(yaml.safe_dump(sampling, sort_keys=False))

eval_config = yaml.safe_load(EVAL_CONFIG_YAML.read_text())
eval_config["evaluation"].update(BUDGETS)
EVAL_CONFIG_YAML.write_text(yaml.safe_dump(eval_config, sort_keys=False))

print(SAMPLING_YAML, sampling, EVAL_CONFIG_YAML, eval_config["evaluation"], sep="\n")

# %% [C] In the Getting Started evaluation cell, keep everything up to and including
# `evaluator = Evaluator(eval_config)` (it defines `run_sync` and reads eval_config.yaml, which
# cell B has already edited), and replace the rest (the `tasks[:2]` loop and `submission_df`)
# with this cell.
import shutil

from navalign.collect import collect_run, load_done, select_tasks, write_manifest
from navalign.runs import check_run, load_run, summarize_run

OUT = Path("/kaggle/working")
TRACE_DIR = OUT / "results" / "traces"
if RESUME_FROM:
    shutil.copytree(RESUME_FROM, OUT, dirs_exist_ok=True)

tasks_by_id = {t.instance_id: t for t in tasks}
shard_ids = select_tasks(TASK_IDS or tasks_by_id, SHARD, N_SHARDS)
todo = select_tasks(shard_ids, done=load_done(OUT / "results.jsonl"))
write_manifest(OUT, {
    "run_id": RUN_ID, "shard": SHARD, "n_shards": N_SHARDS, "task_ids": shard_ids,
    "resumed_from": RESUME_FROM, "started_at": T0, "navalign_src": str(NAVALIGN_SRC),
    "sampling": yaml.safe_load(SAMPLING_YAML.read_text()),
    "evaluation": yaml.safe_load(EVAL_CONFIG_YAML.read_text())["evaluation"],
})
print(f"{RUN_ID}: {len(todo)} of {len(shard_ids)} tasks to run")


def run_one(task):
    """One task through the harness, called the way the Getting Started loop calls it."""
    return run_sync(evaluator.evaluate_task, task=task,
                    task_index=todo.index(task.instance_id) + 1, total_tasks=len(todo))


collect_run(tasks_by_id, todo, run_one, OUT, deadline=T0 + KERNEL_LIMIT_S - SAVE_MARGIN_S,
            max_task_s=MAX_TASK_S, trace_dir=TRACE_DIR)

run = load_run(OUT)
print(summarize_run(run).drop(columns="error").to_string(index=False))
for problem in check_run(run):
    print("PROBLEM", problem)
