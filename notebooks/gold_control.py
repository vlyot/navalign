# Gold-patch control (R1) and regrading cells for the Kaggle notebook `navalign-gold-control`
# (a Copy & Edit of the Getting Started notebook, CPU only, keeping only the cells up to and
# including the one that loads `tasks`; the kaic87/navalign-src and kaic87/navalign-repair-wheels
# datasets attached). Run with Save Version -> Save & Run All. Internet stays off: Kaggle does not
# allow it for notebooks attached to this competition.
#
# The grading calls swegemma's own Phase 2 test step, `harness.verification.verify_task`, with a
# given patch and no model (read from the harness source in D0, swegemma 0.2.7). In the notebook
# the sandbox is `SubprocessManager`: a fresh venv per task that sees the host's site-packages
# through `_host_env.pth`, with `/workspace` on PYTHONPATH. Its install steps
# (`install_test_dependencies`, `install_editable_package`) do nothing in subprocess mode, so
# the tests run on whatever the notebook's Python has installed. The repairs therefore add one
# more `.pth` file to each sandbox venv, read before `_host_env.pth`, that puts the workspace's
# `src/` and the repair directories ahead of the host's packages.
#
# Output in /kaggle/working: manifest.json and grades.jsonl (one row per task and patch variant).
# Download it and summarise it with
#   python -m navalign.grade --grades results/gold/<ENV> [--csv results/gold/gradable_tasks.csv]

# %% [D0] Harness discovery (read-only). Already run as Version 1; kept here as the record.
import importlib, inspect, pkgutil, sys
from pathlib import Path
import swegemma
from swegemma.config import EvalConfig
from swegemma.evaluate import Evaluator
out = Path("/kaggle/working/d0"); out.mkdir(exist_ok=True)
lines = [f"python {sys.version}", f"swegemma at {swegemma.__file__}",
         f"EvalConfig{inspect.signature(EvalConfig)}",
         "Evaluator methods: " + ", ".join(m for m in dir(Evaluator) if not m.startswith('__')),
         "wheelhouse: " + ", ".join(sorted(p.name for p in WHEELHOUSE_DIR.glob('*.whl')))]
for m in pkgutil.walk_packages(swegemma.__path__, "swegemma."):
    try:
        src = inspect.getsource(importlib.import_module(m.name))
    except Exception as e:
        lines.append(f"{m.name}: could not read ({type(e).__name__}: {e})"); continue
    (out / f"{m.name}.py").write_text(src)
    lines.append(f"{m.name}: {len(src.splitlines())} lines")
(out / "index.txt").write_text("\n".join(lines)); print("\n".join(lines))

# %% [D1] Paste after D0: settings, navalign on sys.path, and the repairs when ENV = "repaired".
import importlib.metadata
import re
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()

ENV = "shipped"           # "shipped": the harness as is | "repaired": the R1 repairs below | "hosts": a host-fixed setup
VARIANT = "reference"     # "reference": gold and empty patch | "agent": the patches of the runs in RUNS | "all": both
# Run 1's shard results (manifest.json + results.jsonl per shard) from the private dataset kaic87/navalign-run1-results.
RUNS = sorted(str(p.parent) for p in Path("/kaggle/input").rglob("manifest.json") if "navalign-run1-results" in str(p))
TASK_IDS = None           # None = all tasks
RESUME_FROM = None        # an earlier version of this notebook attached as input, e.g. "/kaggle/input/<dataset>"
TIMEOUT_S = 300           # per test run, as in the trace runs
KERNEL_LIMIT_S, SAVE_MARGIN_S, MAX_PAIR_S = 12 * 3600, 30 * 60, 15 * 60

# Run 1's traces come from swegemma 0.2.7 (wheelhouse v28). Later versions (0.2.10+, wheelhouse v29+)
# change how agent patches are applied in the test step, so grading must use the same version.
# The wheelhouse input has to be pinned to Version 28 in the notebook's input settings.
EXPECTED_SWEGEMMA = "0.2.7"
assert importlib.metadata.version("swegemma") == EXPECTED_SWEGEMMA, (
    f"swegemma {importlib.metadata.version('swegemma')} installed; pin the wheelhouse input to Version 28")
assert VARIANT == "reference" or len(RUNS) == 3, f"expected 3 shard dirs from navalign-run1-results, found {RUNS}"

# Kaggle allows no Internet in notebooks attached to this competition, so the repairs install from
# the private dataset kaic87/navalign-repair-wheels (built by notebooks/build_repair_wheels.sh).
# Each repo gets only the test packages its own tests need (from the shipped control: missing
# modules and fixtures), and only those the host lacks, in a dir per repo; nothing the host already
# has is shadowed, and rich and httpx run as shipped. PIN_PACKAGES are installed per task at the
# range its snapshot declares. (rich's lexer-guess tests flip between runs whatever the repairs:
# Pygments picks `python` or IPython's plugin lexer; grade.py marks such tasks `unstable`.)
REPAIR_BY_REPO = {
    "fastapi": ("inline-snapshot", "dirty-equals", "executing", "asttokens"),
    "requests": ("pytest-httpbin", "httpbin", "brotlicffi"),
}
PIN_PACKAGES = ("starlette",)

NAVALIGN_SRC = next(Path("/kaggle/input").rglob("navalign/grade.py")).parents[1]
sys.path.insert(0, str(NAVALIGN_SRC))
REPAIR_WHEELS = next((p.parent for p in Path("/kaggle/input").rglob("NAVALIGN_REPAIR_WHEELS.md")), None)
REPAIR_ROOT = Path("/tmp/navalign_repair")  # outside /kaggle/working, so it isn't saved with the output


def host_has(dist: str) -> bool:
    try:
        importlib.metadata.distribution(dist)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def pip_target(target: Path, *requirements: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-index",
                    "--find-links", str(REPAIR_WHEELS), "--target", str(target), *requirements], check=True)


INSTALLED: dict[str, list[str]] = {}  # repo -> distributions installed into REPAIR_ROOT / repo
if ENV == "repaired":
    assert REPAIR_WHEELS is not None, "attach the kaic87/navalign-repair-wheels dataset"
    for repo, dists in REPAIR_BY_REPO.items():
        INSTALLED[repo] = sorted(d for d in dists if not host_has(d))
        if INSTALLED[repo]:
            pip_target(REPAIR_ROOT / repo, *INSTALLED[repo])
print(f"ENV={ENV} VARIANT={VARIANT}; installed per repo: {INSTALLED or 'nothing'}")

# %% [D2] Paste after D1: grades every task under ENV / VARIANT. Run with Save Version.
import asyncio
import concurrent.futures
import shutil

from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.harness.verification import verify_task
from swegemma.sandbox import SubprocessManager

from navalign.collect import write_manifest
from navalign.data import read_snapshot_files
from navalign.grade import agent_patches, grade_tasks, main as grade_summary, reference_variants, snapshot_requirements
from navalign.runs import load_runs


def run_sync(coro_or_fn, *args, **kwargs):
    """Run an async coroutine synchronously inside the notebook's event loop (as in Getting Started)."""
    fn = (lambda: coro_or_fn(*args, **kwargs)) if callable(coro_or_fn) else (lambda: coro_or_fn)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(fn())).result()
    return asyncio.run(fn())


REPAIR_PATHS: list[str] = []  # set per task by verify_one


def add_repairs(sandbox: SubprocessManager) -> None:
    """Make each new sandbox venv carry `_aa_navalign_repair.pth`: the workspace's src/ (src-layout
    repos such as requests, so the patched code shadows the installed release) and REPAIR_PATHS.

    Wraps the instance's `start` instead of subclassing: the harness picks its behaviour by
    `type(sandbox).__name__`, and any other class name makes it run the Docker install steps
    (`pip install -e`, sandbox/setup.py; ~400 s per task, and they broke the imports in Run B).
    """
    start = sandbox.start

    def start_with_repairs() -> str:
        sid = start()
        paths = sandbox.sandboxes[sid]
        lines = [str(paths["workspace"] / "src"), *REPAIR_PATHS]
        for site_packages in paths["venv"].glob("lib/python*/site-packages"):
            (site_packages / "_aa_navalign_repair.pth").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return sid

    sandbox.start = start_with_repairs


def pins_for(snapshot: Path) -> Path | None:
    """A directory with PIN_PACKAGES at the versions the snapshot declares (shared between equal specs)."""
    text = read_snapshot_files(snapshot, ["pyproject.toml"]).get("pyproject.toml")
    reqs = sorted(snapshot_requirements(text, PIN_PACKAGES).values()) if text else []
    if not reqs:
        return None
    target = REPAIR_ROOT / "pins" / re.sub(r"[^A-Za-z0-9.]+", "_", "+".join(reqs))
    if not target.exists():
        pip_target(target, *reqs)
    return target


OUT = Path("/kaggle/working")
if RESUME_FROM:
    shutil.copytree(RESUME_FROM, OUT, dirs_exist_ok=True)

eval_config = EvalConfig(
    tasks_path=TASKS_PATH,
    snapshots_dir=DATA_DIR / "snapshots",
    results_dir=OUT / "verify",
    submission_dir=AGENT_DIR,
    models=None,  # verify_task never calls a model
    sandbox="subprocess",
    timeout_seconds=TIMEOUT_S,
    wheels_dir=DATA_DIR / "wheels",
    verbose=False,
)
sandbox = SubprocessManager(timeout_seconds=TIMEOUT_S)
if ENV == "repaired":
    add_repairs(sandbox)


def verify_one(task, patch: str):
    snapshot, base, patch_path = resolve_task_snapshot_paths(eval_config.snapshots_dir, task.instance_id, task.repo)
    if ENV == "repaired":
        repo = task.repo.split("/")[-1]
        repo_dir = REPAIR_ROOT / repo if INSTALLED.get(repo) else None
        REPAIR_PATHS[:] = [str(p) for p in (pins_for(snapshot), repo_dir) if p is not None]
    return run_sync(verify_task, sandbox, eval_config, task, snapshot, base_snapshot_path=base,
                    patch_path=patch_path, agent_patch=patch, start_time=time.perf_counter())


tasks_by_id = {t.instance_id: t for t in tasks}
ids = sorted(TASK_IDS or tasks_by_id)
variants = {}
if VARIANT in ("reference", "all"):
    variants |= reference_variants()
if VARIANT in ("agent", "all"):
    variants["agent"] = agent_patches(load_runs(RUNS))
write_manifest(OUT, {
    "env": ENV, "variant": VARIANT, "runs": RUNS, "task_ids": ids, "resumed_from": RESUME_FROM,
    "repair_installed": INSTALLED, "repair_wheels": str(REPAIR_WHEELS), "pin_packages": list(PIN_PACKAGES),
    "timeout_s": TIMEOUT_S,
    "python": sys.version, "swegemma": importlib.metadata.version("swegemma"),
    "navalign_src": str(NAVALIGN_SRC), "started_at": T0,
})
grade_tasks(tasks_by_id, ids, verify_one, OUT, variants,
            deadline=T0 + KERNEL_LIMIT_S - SAVE_MARGIN_S, max_task_s=MAX_PAIR_S)
grade_summary(["--grades", str(OUT)])
