"""Fetch an external test set: SWE-bench Verified tasks from a few repositories, one SWE-agent run's
trajectories on them, its resolved list, and a snapshot of each repository at each task's base commit.

    python notebooks/fetch_swebench.py [--out data/external/swebench] [--run 20250511_sweagent_lm_32b]

Writes, under --out: `tasks.jsonl` (competition fields, ids renamed `requests_2931`, plus
`swe_bench_id`), `trajs/<swe_bench_id>.traj`, `results.json`, `snapshots/<id>.tgz` (`git archive`
of the base commit) and `repos/` (bare clones). Already-downloaded files are kept, so a rerun resumes.
Then:

    navalign graph --tasks OUT/tasks.jsonl --snapshots OUT/snapshots --out OUT/graphs
    navalign build --data OUT --out OUT/groundtruth
    navalign swe-agent --trajs OUT/trajs --results OUT/results.json --out OUT/run
    navalign analysis --run OUT/run --groundtruth OUT/groundtruth --grades OUT/run --gradable OUT/run/gradable_tasks.csv --out OUT/analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from navalign.swe_agent import navalign_id

REPOS = ("psf/requests", "pallets/flask", "pytest-dev/pytest", "pylint-dev/pylint", "pydata/xarray")
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Verified"
            "&config=default&split=test&offset={offset}&length=100")
TRAJ_URL = "https://swe-bench-submissions.s3.amazonaws.com/verified/{run}/trajs/{swe_id}.traj"
RESULTS_URL = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified/{run}/results/results.json"
FIELDS = ("repo", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "created_at")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def verified_rows(repos: tuple[str, ...]) -> list[dict]:
    rows, offset = [], 0
    while True:
        page = json.loads(get(ROWS_URL.format(offset=offset)))
        rows += [r["row"] for r in page["rows"] if r["row"]["repo"] in repos]
        offset += len(page["rows"])
        if not page["rows"] or offset >= page["num_rows_total"]:
            return rows


def snapshot(repo: str, commit: str, repos_dir: Path, target: Path) -> None:
    bare = repos_dir / (repo.replace("/", "__") + ".git")
    if not bare.exists():
        subprocess.run(["git", "clone", "--bare", "--quiet", f"https://github.com/{repo}.git", str(bare)], check=True)
    subprocess.run(["git", "-C", str(bare), "archive", "--format=tar.gz", "-o", str(target.resolve()), commit], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch SWE-bench Verified tasks, SWE-agent trajectories and snapshots.")
    parser.add_argument("--out", default="data/external/swebench")
    parser.add_argument("--run", default="20250511_sweagent_lm_32b", help="SWE-bench experiments submission")
    parser.add_argument("--repo", action="append", help="owner/name (repeatable); default: " + ", ".join(REPOS))
    args = parser.parse_args(argv)

    out = Path(args.out)
    for sub in ("trajs", "snapshots", "repos"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    results = out / "results.json"
    if not results.exists():
        results.write_bytes(get(RESULTS_URL.format(run=args.run)))

    rows = verified_rows(tuple(args.repo or REPOS))
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"instance_id": navalign_id(row["instance_id"]), "swe_bench_id": row["instance_id"]}
                               | {k: row[k] for k in FIELDS}) + "\n")
    print(f"{len(rows)} tasks", file=sys.stderr)

    missing = []
    for n, row in enumerate(rows, 1):
        swe_id, tid = row["instance_id"], navalign_id(row["instance_id"])
        traj = out / "trajs" / f"{swe_id}.traj"
        if not traj.exists():
            try:
                traj.write_bytes(get(TRAJ_URL.format(run=args.run, swe_id=swe_id)))
            except OSError as exc:
                missing.append(f"{swe_id}: {exc}")
        tgz = out / "snapshots" / f"{tid}.tgz"
        if not tgz.exists():
            snapshot(row["repo"], row["base_commit"], out / "repos", tgz)
        print(f"[{n}/{len(rows)}] {tid}", file=sys.stderr)
    for problem in missing:
        print(f"no trajectory: {problem}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
