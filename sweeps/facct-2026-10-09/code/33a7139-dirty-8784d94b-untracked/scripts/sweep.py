#!/usr/bin/env python
"""Run a sharded sweep of `run_experiments.py` from one declared grid.

A sweep is a directory, `sweeps/<batch>/`, whose `sweep.json` is the whole
definition of what should exist: datasets x methods x seeds x epsilons and the
downstream classifier. Everything else follows from that file:

    sweep.json          the grid (written by `init`, edited by hand to extend)
    db/<shard>.db       one SQLite file per shard -- one writer each, no locks
    synth/<shard>/      every synthetic table, as run_<run_id>.parquet
    logs/<shard>.log    the shard's stdout/stderr
    locks/<shard>.lock  held while a shard is running
    code/               the uncommitted diff behind each `git_commit` fingerprint
    scheduler-*.log     one per `run` invocation

A shard is one (dataset, method, seed). That is the unit `run_experiments.py`
caches a trained torch model over, and small enough (36 runs for a marginal
method) that losing one to a reboot costs minutes.

    python scripts/sweep.py init   --batch B --datasets ... --methods ... --seeds 0,1,2,3,4
    python scripts/sweep.py run    --batch B --workers 12 --cpus 32-47 [--methods ...] [--gpu 1]
    python scripts/sweep.py status --batch B

`run` is safe to start more than once: schedulers skip finished shards and
shards another scheduler holds the lock for, so freeing up cores (or a GPU)
later means starting a second `run`, not restarting the first. To grow a
sweep, add to the lists in `sweep.json` and `run` again; existing shards keep
their names and their results. Fold the databases into one CSV with
`scripts/merge_sweep_dbs.py sweeps/<batch>/db -o report/data/<batch>.csv`.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEPS_DIR = REPO_ROOT / "sweeps"
RUNNER = REPO_ROOT / "scripts" / "run_experiments.py"

#: Threads per shard for numpy/torch pools. Parallelism comes from running
#: shards side by side; a shard that also fans out just oversubscribes `--cpus`.
THREADS_PER_SHARD = "1"
THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS",
)


@dataclass(frozen=True)
class Shard:
    dataset: str
    method: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.dataset}-{self.method}-s{self.seed}"


def _csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _sweep_dir(batch: str) -> Path:
    return SWEEPS_DIR / batch


def load_spec(batch: str) -> Dict:
    path = _sweep_dir(batch) / "sweep.json"
    if not path.exists():
        raise SystemExit(f"no sweep at {path}; create it with `sweep.py init`")
    return json.loads(path.read_text())


def shards_of(spec: Dict, methods: Optional[Sequence[str]] = None,
              datasets: Optional[Sequence[str]] = None) -> List[Shard]:
    """Every shard of the sweep, seed-major: all of seed 0 is scheduled before
    any of seed 1, so the full grid has one replicate early and an interrupted
    sweep leaves complete seeds rather than complete datasets."""
    return [
        Shard(dataset, method, seed)
        for seed in spec["seeds"]
        for dataset in spec["datasets"]
        for method in spec["methods"]
        if (methods is None or method in methods)
        and (datasets is None or dataset in datasets)
    ]


def _expected_runs(spec: Dict, shard: Shard) -> int:
    """How many runs `run_experiments.py` will schedule for this shard."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_experiments import build_configs  # noqa: E402 - heavy import, deferred

    return len(build_configs(
        spec["batch"], [shard.seed], [shard.method], [shard.dataset],
        epsilons=spec["epsilons"], classifier=spec["classifier"],
    ))


def _status_counts(db_path: Path) -> Dict[str, int]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        # A cell retried after a failure has several rows; count it once, by
        # its best outcome, so "finished" means every cell has a usable row.
        rows = conn.execute(
            "SELECT fairness_mechanism, epsilon, json_extract(extra_params, '$.role_config'), "
            "MAX(status IN ('done', 'partial')), MAX(status = 'partial'), MAX(status = 'failed') "
            "FROM runs GROUP BY 1, 2, 3"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    return {
        "finished": sum(r[3] for r in rows),
        "partial": sum(r[3] and r[4] for r in rows),
        "failed": sum((not r[3]) and r[5] for r in rows),
    }


def _snapshot_code(sweep_dir: Path) -> str:
    """Save the uncommitted diff under the same fingerprint `db._git_commit`
    writes on every run row, so a `-dirty:<hash>` row can be traced to source."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=REPO_ROOT
        ).stdout

    head = git("rev-parse", "--short", "HEAD").strip()
    diff = git("diff", "HEAD")
    code_dir = sweep_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    if not diff:
        return head
    digest = hashlib.sha1(diff.encode("utf-8", "replace")).hexdigest()[:8]
    fingerprint = f"{head}-dirty:{digest}"
    (code_dir / f"{head}-dirty-{digest}.diff").write_text(diff)
    # `git diff` omits untracked files, so new modules are copied alongside.
    untracked = _csv(git("ls-files", "--others", "--exclude-standard", "src", "scripts", "tests")
                     .replace("\n", ","))
    for rel in untracked:
        target = code_dir / f"{head}-dirty-{digest}-untracked" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / rel).read_bytes())
    return fingerprint


def _mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("inf")


def _try_lock(path: Path):
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _spawn(spec: Dict, shard: Shard, sweep_dir: Path, args) -> subprocess.Popen:
    command = [
        "nice", "-n", str(args.nice), "taskset", "-c", args.cpus,
        sys.executable, str(RUNNER),
        "--batch", spec["batch"],
        "--datasets", shard.dataset, "--methods", shard.method,
        "--seeds", str(shard.seed),
        "--epsilons", ",".join(str(e) for e in spec["epsilons"]),
        "--classifier", spec["classifier"],
        "--db-path", str(sweep_dir / "db" / f"{shard.name}.db"),
        "--synth-dir", str(sweep_dir / "synth" / shard.name),
        "--results-dir", str(sweep_dir / "results"),
        "--no-report",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    env.update({var: THREADS_PER_SHARD for var in THREAD_VARS})
    if args.gpu is not None:
        # Hide every other GPU, so the one we were given is cuda:0 and the
        # rest are unreachable even to a code path that ignores --device.
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        command += ["--device", "cuda:0"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    log = open(sweep_dir / "logs" / f"{shard.name}.log", "a")
    return subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env=env, cwd=REPO_ROOT,
    )


def cmd_init(args) -> int:
    sweep_dir = _sweep_dir(args.batch)
    path = sweep_dir / "sweep.json"
    if path.exists():
        raise SystemExit(f"{path} exists; edit it to extend the sweep")
    spec = {
        "batch": args.batch,
        "created": datetime.now().isoformat(timespec="seconds"),
        "datasets": _csv(args.datasets),
        "methods": _csv(args.methods),
        "seeds": [int(s) for s in _csv(args.seeds)],
        "epsilons": [float(e) for e in _csv(args.epsilons)],
        "classifier": args.classifier,
        "notes": args.notes,
    }
    sweep_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {path}: {len(shards_of(spec))} shards")
    return 0


def cmd_run(args) -> int:
    spec = load_spec(args.batch)
    sweep_dir = _sweep_dir(args.batch)
    for sub in ("db", "synth", "logs", "locks", "results"):
        (sweep_dir / sub).mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        print(f"[{datetime.now():%m-%d %H:%M:%S}] {message}", flush=True)

    queue = shards_of(spec, args.methods and _csv(args.methods),
                      args.datasets and _csv(args.datasets))
    expected = {shard: _expected_runs(spec, shard) for shard in queue}
    say(f"batch={args.batch} code={_snapshot_code(sweep_dir)} shards={len(queue)} "
        f"workers={args.workers} cpus={args.cpus} gpu={args.gpu}")

    running: Dict[Shard, tuple] = {}
    attempts: Dict[Shard, int] = {}
    while queue or running:
        for shard, (process, lock) in list(running.items()):
            if process.poll() is None:
                continue
            counts = _status_counts(sweep_dir / "db" / f"{shard.name}.db")
            say(f"exit {shard.name} rc={process.returncode} "
                f"finished={counts.get('finished', 0)}/{expected[shard]} "
                f"failed={counts.get('failed', 0)}")
            lock.close()
            del running[shard]
            if counts.get("finished", 0) < expected[shard]:
                queue.append(shard)  # retried last, up to --max-attempts

        while queue and len(running) < args.workers:
            if _mem_available_gb() < args.min_free_gb:
                say(f"holding: {_mem_available_gb():.0f} GB available < {args.min_free_gb}")
                break
            shard = queue.pop(0)
            counts = _status_counts(sweep_dir / "db" / f"{shard.name}.db")
            if counts.get("finished", 0) >= expected[shard]:
                continue
            if attempts.get(shard, 0) >= args.max_attempts:
                say(f"giving up on {shard.name} after {attempts[shard]} attempts")
                continue
            lock = _try_lock(sweep_dir / "locks" / f"{shard.name}.lock")
            if lock is None:
                continue  # another scheduler has it
            attempts[shard] = attempts.get(shard, 0) + 1
            running[shard] = (_spawn(spec, shard, sweep_dir, args), lock)
            say(f"start {shard.name} pid={running[shard][0].pid} "
                f"({counts.get('finished', 0)}/{expected[shard]} already finished)")
        time.sleep(args.poll_seconds)
    say("nothing left to schedule")
    return 0


def cmd_status(args) -> int:
    spec = load_spec(args.batch)
    sweep_dir = _sweep_dir(args.batch)
    totals: Dict[tuple, Dict[str, int]] = {}
    active = []
    for shard in shards_of(spec):
        counts = _status_counts(sweep_dir / "db" / f"{shard.name}.db")
        cell = totals.setdefault(
            (shard.dataset, shard.method),
            {"expected": 0, "finished": 0, "partial": 0, "failed": 0, "shards_done": 0},
        )
        expected = _expected_runs(spec, shard)
        cell["expected"] += expected
        for key in ("finished", "partial", "failed"):
            cell[key] += counts.get(key, 0)
        cell["shards_done"] += counts.get("finished", 0) >= expected
        lock_path = sweep_dir / "locks" / f"{shard.name}.lock"
        if lock_path.exists():
            lock = _try_lock(lock_path)
            if lock is None:
                active.append(f"{shard.name} ({counts.get('finished', 0)}/{expected})")
            else:
                lock.close()

    n_seeds = len(spec["seeds"])
    print(f"sweep {args.batch}: eps={spec['epsilons']} classifier={spec['classifier']} "
          f"seeds={spec['seeds']}")
    print(f"{'dataset':14s} {'method':18s} {'finished':>12s} {'partial':>8s} "
          f"{'failed':>7s} {'seeds done':>11s}")
    grand = {"expected": 0, "finished": 0}
    for (dataset, method), cell in totals.items():
        print(f"{dataset:14s} {method:18s} "
              f"{cell['finished']:>5d}/{cell['expected']:<6d} {cell['partial']:>8d} "
              f"{cell['failed']:>7d} {cell['shards_done']:>8d}/{n_seeds}")
        grand["expected"] += cell["expected"]
        grand["finished"] += cell["finished"]
    print(f"total {grand['finished']}/{grand['expected']} runs; "
          f"{len(active)} shards running" + (": " + ", ".join(active) if active else ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="declare a new sweep")
    init.add_argument("--batch", required=True)
    init.add_argument("--datasets", required=True)
    init.add_argument("--methods", required=True)
    init.add_argument("--seeds", default="0,1,2,3,4")
    init.add_argument("--epsilons", default="1,10,1000")
    init.add_argument("--classifier", default="rf", choices=["mlp", "rf", "lr"])
    init.add_argument("--notes", default="")
    init.set_defaults(func=cmd_init)

    run = sub.add_parser("run", help="schedule the sweep's unfinished shards")
    run.add_argument("--batch", required=True)
    run.add_argument("--workers", type=int, required=True,
                     help="shards run side by side by this scheduler")
    run.add_argument("--cpus", required=True,
                     help="taskset core list every shard is confined to, e.g. 32-47. "
                          "A hard ceiling: nice alone only yields under contention.")
    run.add_argument("--gpu", type=int, default=None,
                     help="physical GPU index for torch methods; omitted, shards run on CPU")
    run.add_argument("--methods", default=None, help="restrict to these methods")
    run.add_argument("--datasets", default=None, help="restrict to these datasets")
    run.add_argument("--nice", type=int, default=19)
    run.add_argument("--min-free-gb", type=float, default=24.0,
                     help="do not start another shard below this much available RAM")
    run.add_argument("--max-attempts", type=int, default=2,
                     help="times one scheduler will (re)start a shard that exits unfinished")
    run.add_argument("--poll-seconds", type=float, default=20.0)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="finished/expected runs per dataset x method")
    status.add_argument("--batch", required=True)
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
