#!/usr/bin/env python
"""MAMA-MIA membership inference on every synthetic table a sweep has saved.

MAMA-MIA (Golob et al., "Privacy vulnerabilities in marginals-based synthetic
data"; code in `~/SyntheticData_MIA`) attacks a graphical generator through
the statistics it chose to preserve, its *focal points*. For a target record
t it sums, over the focal points, the ratio of t's probability under the
synthetic table to its probability under auxiliary population data; a record
the generator trained on pulls the synthetic statistics towards itself, so a
high ratio is evidence of membership. Which focal points to use is found by
shadow modelling: run the generator on other samples of the auxiliary data
and count how often each statistic is selected.

Two flavours, both from that code base (`conduct_attacks.py`), run here on
every generator whether or not it is the one the flavour was written for:

  mst        focal points are marginals: every 1-way and each selected pair.
             Score: sum of weight x P_synth(t) / P_aux(t), divided by the
             number of focal points used (`custom_mst_attack`).
  privbayes  focal points are conditionals: child given its parent set.
             Score: sum of P_synth(child | parents) / P_aux(child | parents),
             unweighted, same normalisation (`custom_privbayes_attack`).

What the focal points are for each generator:

  mst, privsyn      the pairs the run selected (its recorded graph), plus all
                    1-ways. For the conditional flavour MST's forest is
                    oriented away from its first column in table order;
                    PrivSyn's graph has cycles and no orientation, so the
                    conditional flavour is not run on it.
  privbayes         the Bayesian network the run selected.
  causal families   the DAG the generator was given, minus the edges the
                    fairness mechanism cut (a cut parent is permuted, so the
                    synthetic child no longer depends on it). A partial cut
                    (`cut_strength` < 1) keeps the edge.
  tabfairgan        no structure to read: every 1- and 2-way marginal, with
                    every shadow agreeing on all of them. Marginal flavour
                    only.

Two focal-point sources are scored for every run:

  shadow   what the paper's adversary has. The other seeds of the same cell
           (same generator, mechanism, epsilon, role split, settings) are
           runs of the same algorithm on other 70% samples of the same
           population, i.e. shadow models that cost nothing extra. Weight =
           number of those runs that selected the focal point; a focal point
           is used when its weight is at least `threshold(eps)` x the largest
           weight, with the paper's thresholds. At most four shadow runs
           here, against the paper's many.
  oracle   the target run's own recorded graph, weight 1. An upper bound on
           what better shadow modelling could reach.

For a fixed public graph (the hand-drawn DAGs) the two coincide.

Threat model, as in the paper's main experiments: the auxiliary data is the
whole dataset the training split was drawn from (training rows included,
"overlapping aux"). Members are training rows, non-members the holdout rows
of the run's own 70/30 split; all holdout rows are targets, with an equal
number of training rows drawn with the run's seed.

Output: `sweeps/<batch>/mama_mia/<shard>.csv`, one row per run x flavour x
focal-point source: AUC, TPR at 10% / 1% / 0.1% FPR and the paper's
membership advantage. Resumable per shard. Nothing is trained or generated.

    python scripts/mama_mia.py --batches facct-2026-10-09 --workers 14
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_fairness_sdg.data.datasets import DATASETS  # noqa: E402
from causal_fairness_sdg.fairness.base import AttributeRoles  # noqa: E402
from causal_fairness_sdg.fairness.registry import get_fairness_mechanism  # noqa: E402

SWEEPS_DIR = REPO_ROOT / "sweeps"
TEST_SIZE = 0.3  # must match `experiments.runner.RunConfig.test_size`
#: `Config.fp_weight_thresholds` in SyntheticData_MIA/util.py: epsilon -> share
#: of the largest focal-point weight a focal point needs to be used.
FP_WEIGHT_THRESHOLDS = {0.01: 0.5, 1: 0.6, 10: 0.7, 100: 0.8, 1000: 0.85}
DEFAULT_RATIO_VALUE = 1e-10  # `default_val` in the original attack code
UNDIRECTED_METHODS = ("mst", "privsyn", "aim")
#: Generators that select no structure (TabFairGAN). Without a graph to read
#: the attacker falls back to every pair: the marginal flavour over all 1- and
#: 2-way marginals, and no conditional flavour.
GRAPH_FREE_METHODS = ("tabfairgan",)
#: Keys of `extra_params` that describe the machine or the batch, not the
#: algorithm; everything else must match for two runs to be shadows of each other.
NOT_ALGORITHM_KEYS = (
    "batch", "classifier", "device", "noise_multiplier", "spent_epsilon", "steps",
    "sample_rate", "torch_version", "tabpfn_version",
)

Conditional = Tuple[str, Tuple[str, ...]]  # (child, sorted parents)
Marginal = FrozenSet[str]


# --------------------------------------------------------------------------
# Focal points
# --------------------------------------------------------------------------

def orient_forest(columns: Sequence[str], edges: Iterable[Tuple[str, str]]) -> Optional[List[Tuple[str, str]]]:
    """Direct an undirected forest away from the earliest column of each tree.
    None if the graph has a cycle (no tree-structured factorisation)."""
    graph = nx.Graph()
    graph.add_nodes_from(columns)
    graph.add_edges_from(edges)
    if not nx.is_forest(graph):
        return None
    order = {c: i for i, c in enumerate(columns)}
    directed: List[Tuple[str, str]] = []
    for component in nx.connected_components(graph):
        root = min(component, key=order.__getitem__)
        directed.extend(nx.bfs_edges(graph, root))
    return directed


def focal_points(
    method: str, mechanism: str, roles: AttributeRoles, columns: Sequence[str],
    edges: Sequence[Tuple[str, str]], cut_strength: float,
) -> Tuple[List[Marginal], Optional[List[Conditional]]]:
    """(marginal focal points, conditional focal points) of one recorded run."""
    edges = [(a, b) for a, b in edges if a in columns and b in columns]
    if method in GRAPH_FREE_METHODS:
        pairs = [frozenset(p) for p in itertools.combinations(columns, 2)]
        return [frozenset([c]) for c in columns] + pairs, None
    if method in UNDIRECTED_METHODS:
        directed = orient_forest(columns, edges)
    else:
        directed = list(edges)
        if method != "privbayes" and cut_strength >= 1.0:
            # Causal family: the recorded graph is the DAG it was given; drop
            # what the mechanism cut at generation time.
            dag = nx.DiGraph()
            dag.add_nodes_from(columns)
            dag.add_edges_from(edges)
            cut = get_fairness_mechanism(mechanism).select_biased_edges(dag, roles)
            removed = {(p, child) for child, parents in cut.items() for p in parents}
            directed = [e for e in directed if e not in removed]
            edges = directed

    marginals = [frozenset([c]) for c in columns]
    marginals += sorted({frozenset(e) for e in edges if e[0] != e[1]}, key=sorted)

    conditionals: Optional[List[Conditional]] = None
    if directed is not None:
        parents: Dict[str, List[str]] = {c: [] for c in columns}
        for parent, child in directed:
            parents[child].append(parent)
        conditionals = [(c, tuple(sorted(parents[c]))) for c in columns]
    return marginals, conditionals


def weight_threshold(epsilon: Optional[float]) -> float:
    eps = 1000.0 if epsilon is None else float(epsilon)
    return max(t for e, t in FP_WEIGHT_THRESHOLDS.items() if eps >= e)


def select(weights: Counter, epsilon: Optional[float]) -> Dict:
    """Focal points whose shadow frequency clears the paper's threshold."""
    if not weights:
        return {}
    floor = max(weights.values()) * weight_threshold(epsilon)
    return {fp: w for fp, w in weights.items() if w >= floor}


# --------------------------------------------------------------------------
# Probabilities
# --------------------------------------------------------------------------

class Tables:
    """Probability lookups for one (auxiliary data, targets) pair, with the
    synthetic table swapped per run. Rows are encoded as one integer per
    column subset (mixed radix over the column domains)."""

    def __init__(self, aux: pd.DataFrame, targets: pd.DataFrame, domain: Dict[str, int]):
        self.columns = list(aux.columns)
        self.size = {c: int(max(domain.get(c, 0), aux[c].max() + 1)) for c in self.columns}
        self.aux = {c: aux[c].to_numpy(dtype=np.int64) for c in self.columns}
        self.targets = {c: targets[c].to_numpy(dtype=np.int64) for c in self.columns}
        self.n_aux = len(aux)
        self._aux_cache: Dict[Tuple[str, ...], Tuple[np.ndarray, np.ndarray]] = {}
        self._target_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self.synth: Dict[str, np.ndarray] = {}
        self.n_synth = 0
        self._synth_cache: Dict[Tuple[str, ...], Tuple[np.ndarray, np.ndarray]] = {}

    def set_synth(self, synth: pd.DataFrame) -> None:
        self.synth = {
            c: np.clip(synth[c].to_numpy(dtype=np.int64), 0, self.size[c] - 1)
            for c in self.columns
        }
        self.n_synth = len(synth)
        self._synth_cache = {}

    def _key(self, source: Dict[str, np.ndarray], cols: Tuple[str, ...]) -> np.ndarray:
        radix = 1
        for c in cols:
            radix *= self.size[c]
        if radix >= 2 ** 62:
            raise OverflowError(f"focal point over {cols} too wide to encode")
        key = np.zeros(len(next(iter(source.values()))), dtype=np.int64)
        for c in cols:
            key = key * self.size[c] + source[c]
        return key

    def _counts(self, which: str, cols: Tuple[str, ...]) -> np.ndarray:
        """Count, for each target, of rows in `which` equal to it on `cols`."""
        if not cols:
            return np.full(len(next(iter(self.targets.values()))),
                           self.n_aux if which == "aux" else self.n_synth, dtype=np.float64)
        if cols not in self._target_cache:
            self._target_cache[cols] = self._key(self.targets, cols)
        cache = self._aux_cache if which == "aux" else self._synth_cache
        if cols not in cache:
            source = self.aux if which == "aux" else self.synth
            cache[cols] = np.unique(self._key(source, cols), return_counts=True)
        values, counts = cache[cols]
        keys = self._target_cache[cols]
        pos = np.clip(np.searchsorted(values, keys), 0, max(len(values) - 1, 0))
        found = (values[pos] == keys) if len(values) else np.zeros(len(keys), bool)
        return np.where(found, counts[pos], 0).astype(np.float64)

    def marginal_ratio(self, cols: Tuple[str, ...]) -> np.ndarray:
        p_synth = self._counts("synth", cols) / max(self.n_synth, 1)
        p_aux = self._counts("aux", cols) / self.n_aux
        p_synth = np.where(p_synth > 0, p_synth, DEFAULT_RATIO_VALUE)
        p_aux = np.where(p_aux > 0, p_aux, DEFAULT_RATIO_VALUE)
        return p_synth / p_aux

    def conditional_ratio(self, child: str, parents: Tuple[str, ...]) -> np.ndarray:
        joint = tuple(sorted((child,) + parents))
        def conditional(which: str) -> np.ndarray:
            num = self._counts(which, joint)
            den = self._counts(which, parents)
            return np.where(num > 0, num / np.maximum(den, 1.0), DEFAULT_RATIO_VALUE)
        return conditional("synth") / np.maximum(conditional("aux"), DEFAULT_RATIO_VALUE)


# --------------------------------------------------------------------------
# Scoring (util.py: activate_3, membership_advantage, area_under_curve)
# --------------------------------------------------------------------------

def activate(p_rel: np.ndarray) -> np.ndarray:
    logs = np.log(np.maximum(p_rel, 1e-300))
    spread = logs.std()
    z = (logs - logs.mean()) / spread if spread > 0 else np.zeros_like(logs)
    return 1.0 / (1.0 + np.exp(-(z - np.median(z))))


def score(membership: np.ndarray, raw: np.ndarray) -> Dict[str, float]:
    scores = activate(raw)
    out: Dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(membership, scores))
        fpr, tpr, _ = roc_curve(membership, scores)
        for name, level in (("tpr_at_fpr10", 0.1), ("tpr_at_fpr1", 0.01), ("tpr_at_fpr01", 0.001)):
            out[name] = float(np.interp(level, fpr, tpr))
    except ValueError:
        out.update(auc=np.nan, tpr_at_fpr10=np.nan, tpr_at_fpr1=np.nan, tpr_at_fpr01=np.nan)
    predicted = scores > 0.5
    confidence = 2 * np.abs(0.5 - scores)
    tp = confidence[predicted & (membership == 1)].sum()
    fn = confidence[~predicted & (membership == 1)].sum()
    fp = confidence[predicted & (membership == 0)].sum()
    tn = confidence[~predicted & (membership == 0)].sum()
    tpr_w = tp / (tp + fn) if tp + fn > 0 else 0.0
    fpr_w = fp / (fp + tn) if fp + tn > 0 else 0.0
    out["membership_advantage"] = float((tpr_w - fpr_w + 1) / 2)
    return out


def attack_marginals(tables: Tables, weights: Dict[Marginal, float]) -> np.ndarray:
    total = 0.0
    for fp, weight in weights.items():
        total = total + weight * tables.marginal_ratio(tuple(sorted(fp)))
    return total / max(len(weights), 1)


def attack_conditionals(tables: Tables, weights: Dict[Conditional, float]) -> np.ndarray:
    total = 0.0
    for child, parents in weights:
        total = total + tables.conditional_ratio(child, parents)
    return total / max(len(weights), 1)


# --------------------------------------------------------------------------
# Sweep plumbing
# --------------------------------------------------------------------------

def read_shard(db_path: Path) -> Tuple[pd.DataFrame, Dict[int, List[Tuple[str, str]]]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        runs = pd.read_sql("SELECT * FROM runs WHERE status = 'done'", conn)
        edge_rows = conn.execute("SELECT run_id, node_a, node_b FROM graph_edges").fetchall()
    finally:
        conn.close()
    edges: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for run_id, a, b in edge_rows:
        edges[run_id].append((a, b))
    return runs, edges


def cell_key(run: pd.Series) -> Tuple:
    extra = json.loads(run["extra_params"] or "{}")
    algo = tuple(sorted((k, json.dumps(v)) for k, v in extra.items() if k not in NOT_ALGORITHM_KEYS))
    eps = None if pd.isna(run["epsilon"]) else round(float(run["epsilon"]), 6)
    return (run["dataset"], run["sdg_method"], run["fairness_mechanism"], eps,
            run["protected_attrs"], run["admissible_attrs"], algo)


def run_focal_points(run: pd.Series, edges, columns) -> Tuple[List[Marginal], Optional[List[Conditional]]]:
    extra = json.loads(run["extra_params"] or "{}")
    roles = AttributeRoles.create(
        protected=json.loads(run["protected_attrs"]),
        admissible=json.loads(run["admissible_attrs"] or "[]"),
        outcome=[run["outcome_attr"]],
    )
    return focal_points(run["sdg_method"], run["fairness_mechanism"], roles, columns,
                        edges, float(extra.get("cut_strength", 1.0)))


def attack_shard(batch: str, shard: str) -> str:
    sweep_dir = SWEEPS_DIR / batch
    out_path = sweep_dir / "mama_mia" / f"{shard}.csv"
    runs, edges = read_shard(sweep_dir / "db" / f"{shard}.db")
    if runs.empty:
        return f"{shard}: no finished runs"
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["run_id"])["run_id"])
    todo = runs[~runs["run_id"].isin(done)]
    if todo.empty:
        return f"{shard}: up to date ({len(runs)} runs)"

    dataset, seed = runs["dataset"].iloc[0], int(runs["seed"].iloc[0])
    data, domain, _roles = DATASETS[dataset]()
    columns = list(data.columns)
    train, holdout = train_test_split(data, test_size=TEST_SIZE, random_state=seed)
    members = train.sample(n=min(len(holdout), len(train)), random_state=seed)
    targets = pd.concat([members, holdout])
    membership = np.r_[np.ones(len(members), int), np.zeros(len(holdout), int)]
    tables = Tables(data, targets, domain)

    # Shadow focal points: the same cell in the other seeds' shards.
    method = runs["sdg_method"].iloc[0]
    shadow_m: Dict[Tuple, Counter] = defaultdict(Counter)
    shadow_c: Dict[Tuple, Counter] = defaultdict(Counter)
    n_shadows: Counter = Counter()
    for other in sorted((sweep_dir / "db").glob(f"{dataset}-{method}-s*.db")):
        if other.stem == shard:
            continue
        other_runs, other_edges = read_shard(other)
        for _, run in other_runs.iterrows():
            key = cell_key(run)
            marginals, conditionals = run_focal_points(run, other_edges[run["run_id"]], columns)
            shadow_m[key].update(marginals)
            if conditionals is not None:
                shadow_c[key].update(conditionals)
            n_shadows[key] += 1

    rows = []
    started = time.time()
    for _, run in todo.iterrows():
        synth_path = sweep_dir / "synth" / shard / f"run_{run['run_id']}.parquet"
        if not synth_path.exists():
            continue
        tables.set_synth(pd.read_parquet(synth_path)[columns])
        key = cell_key(run)
        eps = None if pd.isna(run["epsilon"]) else float(run["epsilon"])
        own_m, own_c = run_focal_points(run, edges[run["run_id"]], columns)
        sources = {
            "oracle": ({fp: 1.0 for fp in own_m},
                       None if own_c is None else {fp: 1.0 for fp in own_c}),
        }
        if n_shadows[key]:
            sources["shadow"] = (select(shadow_m[key], eps),
                                 select(shadow_c[key], eps) if shadow_c[key] else None)
        base = {
            "batch": batch, "shard": shard, "run_id": int(run["run_id"]),
            "dataset": dataset, "sdg_method": method,
            "fairness_mechanism": run["fairness_mechanism"], "epsilon": eps, "seed": seed,
            "role_config": json.loads(run["extra_params"] or "{}").get("role_config"),
            "n_targets": len(targets), "n_shadows": int(n_shadows[key]),
        }
        for source, (marginals, conditionals) in sources.items():
            for flavour, weights in (("mst", marginals), ("privbayes", conditionals)):
                if not weights:
                    continue
                raw = (attack_marginals if flavour == "mst" else attack_conditionals)(tables, weights)
                rows.append({**base, "attack": flavour, "focal_points": source,
                             "n_focal_points": len(weights), **score(membership, raw)})
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    return f"{shard}: {len(todo)} runs, {len(rows)} rows, {time.time() - started:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batches", required=True, help="comma-separated sweep names")
    parser.add_argument("--shards", default=None, help="comma-separated shard names (default: all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--collect-to", default=None,
                        help="afterwards, write every batch's rows here as one CSV")
    args = parser.parse_args()

    jobs = []
    for batch in [b.strip() for b in args.batches.split(",") if b.strip()]:
        names = sorted(p.stem for p in (SWEEPS_DIR / batch / "db").glob("*.db"))
        if args.shards:
            names = [n for n in names if n in set(args.shards.split(","))]
        jobs += [(batch, name) for name in names]
    print(f"{len(jobs)} shards", flush=True)
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {pool.submit(attack_shard, batch, name): (batch, name) for batch, name in jobs}
        for future in as_completed(futures):
            batch, name = futures[future]
            try:
                print(f"[{batch}] {future.result()}", flush=True)
            except Exception as error:  # keep the other shards going
                print(f"[{batch}] {name}: FAILED {type(error).__name__}: {error}", flush=True)
    if args.collect_to:
        files = sorted(f for b in {b for b, _ in jobs} for f in (SWEEPS_DIR / b / "mama_mia").glob("*.csv"))
        if files:
            frame = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
            Path(args.collect_to).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.collect_to, index=False)
            print(f"{len(frame)} rows -> {args.collect_to}", flush=True)


if __name__ == "__main__":
    main()
