"""Loaders for the competition data: tasks.jsonl, code graphs, and repository snapshots."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


def load_tasks(path: str | Path) -> dict[str, dict]:
    """Read tasks.jsonl into a dict keyed by instance_id."""
    return load_jsonl(path)


def load_jsonl(path: str | Path, key: str = "instance_id") -> dict[str, dict]:
    """Read a JSON-lines file into a dict keyed by `key` (later rows win)."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row[key]] = row
    return rows


def graph_path(graphs_dir: str | Path, task: dict) -> Path:
    """Commit-named graph file for a task (<repo>_<base_commit>.json).

    Task-named copies can be 0-byte in the Kaggle download, so they are never used.
    """
    repo = task["repo"].split("/")[-1]
    return Path(graphs_dir) / f"{repo}_{task['base_commit']}.json"


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    """Load a NetworkX node-link graph whose edge list is under the key `edges`."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data, edges="edges")


def read_snapshot_files(snapshot: str | Path, paths: set[str] | list[str]) -> dict[str, str]:
    """Return {repo-relative path: text} for the requested files.

    `snapshot` is either a `.tgz` from the competition data or an extracted directory.
    Files that don't exist in the snapshot are left out of the result.
    """
    wanted = {p.removeprefix("./") for p in paths}
    snapshot = Path(snapshot)
    found: dict[str, str] = {}
    if snapshot.is_dir():
        for p in wanted:
            f = snapshot / p
            if f.is_file():
                found[p] = f.read_text(encoding="utf-8")
        return found
    with tarfile.open(snapshot, "r:gz") as tf:
        for member in tf:
            name = member.name[2:] if member.name.startswith("./") else member.name
            if name in wanted and member.isfile():
                found[name] = tf.extractfile(member).read().decode("utf-8")
                if len(found) == len(wanted):
                    break
    return found


def snapshot_files(snapshot: str | Path) -> list[str]:
    """Every file in a `.tgz` snapshot or a directory, as sorted repo-relative POSIX paths."""
    snapshot = Path(snapshot)
    if snapshot.is_dir():
        return sorted(f.relative_to(snapshot).as_posix() for f in snapshot.rglob("*") if f.is_file())
    with tarfile.open(snapshot, "r:gz") as tf:
        return sorted(m.name.removeprefix("./") for m in tf if m.isfile())


def iter_python_files(snapshot: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (repo-relative path, text) for every `.py` file in a `.tgz` snapshot or a directory.

    Text is decoded from the raw bytes in both cases, so line endings are left as stored.
    Files that aren't valid UTF-8 are skipped.
    """
    snapshot = Path(snapshot)
    if snapshot.is_dir():
        for f in sorted(snapshot.rglob("*.py")):
            if f.is_file():
                try:
                    yield f.relative_to(snapshot).as_posix(), f.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    continue
        return
    with tarfile.open(snapshot, "r:gz") as tf:
        for member in tf:
            name = member.name.removeprefix("./")
            if member.isfile() and name.endswith(".py"):
                try:
                    yield name, tf.extractfile(member).read().decode("utf-8")
                except UnicodeDecodeError:
                    continue
