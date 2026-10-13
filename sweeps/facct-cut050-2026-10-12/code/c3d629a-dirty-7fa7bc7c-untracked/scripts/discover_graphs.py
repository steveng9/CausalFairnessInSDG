"""Discover each dataset's causal graph the way the DECAF paper does.

DECAF (van Breugel et al. 2021, arXiv 2110.12884) does not take its real-data
graphs as given. Section 6: "In both experiments, the ground-truth DAG is
unknown. We use causal discovery to uncover the underlying DAG." For Credit
Approval (App. D.3) the graph "was found using the Fast Greedy Equivalence
Search (FGES) with the pycausal library. We provide the prior knowledge that
age and ethnicity are root nodes to the FGES algorithm." For Adult it reuses
the graph Zhang et al. (2017) discovered with the PC algorithm in Tetrad.

This script follows the Credit Approval recipe, which is the one the paper
spells out:

  1. FGES from Tetrad (the engine behind pycausal), run through Tetrad's
     command-line jar, with the BDeu score (every column here is categorical)
     and Tetrad's default parameters.
  2. Prior knowledge: age and the protected demographic attributes are root
     nodes (`causal_graphs.DISCOVERY_ROOTS`).
  3. FGES returns a Markov equivalence class (CPDAG). DECAF needs a DAG, and
     the paper notes "any graph G in the equivalence class is compatible with
     the data and can thus be used" (Section 5.1), so the undirected edges are
     oriented by a consistent extension (Dor & Tarsi 1992), deterministically.

Two things here are ours, not stated in the paper:

  * The outcome is declared a sink (last knowledge tier). Both graphs the
    paper shows (Figs. 6 and 7) have the target as a sink, Zhang et al. put
    income in the last tier, and the edge cuts in `fairness/` act on edges
    *into* the outcome, which matches DECAF's Corollaries 1-3 only when the
    outcome has no children.
  * Discovery sees the training rows only, once per (dataset, seed), so the
    holdout never informs the graph. DECAF discovered on the whole table.

Output: one JSON per dataset in `src/causal_fairness_sdg/data/discovered_graphs/`,
with one graph per training split keyed by a hash of the training table
(`causal_graphs.table_key`), which is how the generators find the graph that
belongs to the rows they were handed.

    python scripts/discover_graphs.py                      # every dataset, seeds 0-4
    python scripts/discover_graphs.py --datasets compas --seeds 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_fairness_sdg.data.causal_graphs import (  # noqa: E402
    DISCOVERED_GRAPH_DIR,
    DISCOVERY_ROOTS,
    table_key,
)
from causal_fairness_sdg.data.datasets import DATASETS  # noqa: E402

#: Tetrad's command-line build. Kept outside the repository; fetched from
#: Maven Central (io.github.cmu-phil:causal-cmd).
TETRAD_JAR = Path.home() / "external/tetrad/causal-cmd-1.4.1-jar-with-dependencies.jar"
TEST_SIZE = 0.3  # must match `experiments.runner.RunConfig.test_size`
SEEDS = (0, 1, 2, 3, 4)

Edge = Tuple[str, str]


def knowledge_text(roots: Sequence[str], middle: Sequence[str], outcome: str) -> str:
    """Tetrad knowledge file: three tiers, no edges among the roots."""
    return (
        "/knowledge\naddtemporal\n"
        f"1* {' '.join(roots)}\n"
        f"2 {' '.join(middle)}\n"
        f"3 {outcome}\n\n"
        "forbiddirect\n\nrequiredirect\n"
    )


def run_fges(train, knowledge: str, workdir: Path, cpus: str) -> Tuple[List[Edge], List[Edge], str]:
    """Returns (directed edges, undirected edges, raw Tetrad output)."""
    data_path = workdir / "train.tsv"
    know_path = workdir / "knowledge.txt"
    train.to_csv(data_path, sep="\t", index=False)
    know_path.write_text(knowledge)
    cmd = [
        "taskset", "-c", cpus, "nice", "-n", "19",
        "java", "-Xmx16g", "-jar", str(TETRAD_JAR),
        "--algorithm", "fges",
        "--data-type", "discrete",
        "--score", "bdeu-score",
        "--dataset", str(data_path),
        "--delimiter", "tab",
        "--knowledge", str(know_path),
        "--out", str(workdir),
        "--prefix", "fges",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out_file = workdir / "fges.txt"
    if proc.returncode != 0 or not out_file.exists():
        raise RuntimeError(f"Tetrad failed:\n{proc.stdout}\n{proc.stderr}")
    raw = out_file.read_text()

    directed: List[Edge] = []
    undirected: List[Edge] = []
    in_edges = False
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Graph Edges:"):
            in_edges = True
            continue
        if not in_edges:
            continue
        if not line:
            if directed or undirected:
                break
            continue
        parts = line.split()
        if len(parts) < 4 or not parts[0].rstrip(".").isdigit():
            break
        a, kind, b = parts[1], parts[2], parts[3]
        if kind == "-->":
            directed.append((a, b))
        elif kind == "<--":
            directed.append((b, a))
        elif kind == "---":
            undirected.append((a, b))
        else:
            raise RuntimeError(f"unexpected edge type in Tetrad output: {line!r}")
    return directed, undirected, raw


def consistent_extension(
    nodes: Sequence[str], directed: Sequence[Edge], undirected: Sequence[Edge]
) -> List[Edge]:
    """Orient a CPDAG's undirected edges into a DAG of the same equivalence
    class (Dor & Tarsi 1992). Deterministic: among eligible nodes the one
    latest in `nodes` is removed first."""
    succ: Dict[str, Set[str]] = {n: set() for n in nodes}
    pred: Dict[str, Set[str]] = {n: set() for n in nodes}
    und: Dict[str, Set[str]] = {n: set() for n in nodes}
    for a, b in directed:
        succ[a].add(b)
        pred[b].add(a)
    for a, b in undirected:
        und[a].add(b)
        und[b].add(a)

    def adjacent(x: str, y: str) -> bool:
        return y in succ[x] or y in pred[x] or y in und[x]

    result: List[Edge] = list(directed)
    remaining = list(nodes)
    while remaining:
        for x in reversed(remaining):
            if succ[x]:
                continue  # not a sink among the directed edges
            neighbours = pred[x] | und[x]
            if all(
                all(z == y or adjacent(y, z) for z in neighbours) for y in und[x]
            ):
                break
        else:
            raise RuntimeError("graph has no consistent DAG extension")
        for y in und[x]:
            result.append((y, x))
            und[y].discard(x)
        for y in pred[x]:
            succ[y].discard(x)
        und[x], pred[x] = set(), set()
        remaining.remove(x)
    return result


def v_structures(nodes: Sequence[str], edges: Sequence[Edge]) -> Set[Tuple[str, str, str]]:
    parents: Dict[str, Set[str]] = {n: set() for n in nodes}
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        parents[b].add(a)
        adj[a].add(b)
        adj[b].add(a)
    return {
        (min(a, b), c, max(a, b))
        for c in nodes
        for a in parents[c]
        for b in parents[c]
        if a < b and b not in adj[a]
    }


def discover(dataset: str, seed: int, cpus: str) -> Dict[str, object]:
    data, _domain, roles = DATASETS[dataset]()
    columns = list(data.columns)
    outcome = next(iter(roles.outcome))
    roots = [c for c in columns if c in DISCOVERY_ROOTS[dataset]]
    missing = set(DISCOVERY_ROOTS[dataset]) - set(columns)
    if missing:
        raise ValueError(f"{dataset}: unknown root columns {sorted(missing)}")
    middle = [c for c in columns if c not in roots and c != outcome]
    train, _ = train_test_split(data, test_size=TEST_SIZE, random_state=seed)

    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"fges-{dataset}-s{seed}-") as tmp:
        directed, undirected, _raw = run_fges(
            train, knowledge_text(roots, middle, outcome), Path(tmp), cpus
        )
    edges = consistent_extension(columns, directed, undirected)

    dag = nx.DiGraph()
    dag.add_nodes_from(columns)
    dag.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(dag), "extension is cyclic"
    assert len(edges) == len(directed) + len(undirected)
    # Same equivalence class: no collider created by orienting.
    assert v_structures(columns, edges) == v_structures(columns, directed), (
        "extension changed the v-structures"
    )
    assert all(dag.in_degree(r) == 0 for r in roots), "a declared root has a parent"
    assert dag.out_degree(outcome) == 0, "the outcome has a child"

    order = {c: i for i, c in enumerate(columns)}
    edges = sorted(edges, key=lambda e: (order[e[1]], order[e[0]]))
    return {
        "seed": seed,
        "n_rows": int(len(train)),
        "n_directed": len(directed),
        "n_undirected": len(undirected),
        "undirected_in_cpdag": sorted([sorted(e) for e in undirected]),
        "edges": [list(e) for e in edges],
        "seconds": round(time.time() - started, 1),
        "key": table_key(train),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=sorted(DISCOVERY_ROOTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--cpus", default="32-47")
    args = parser.parse_args()

    jar_sha1 = hashlib.sha1(TETRAD_JAR.read_bytes()).hexdigest()
    DISCOVERED_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        path = DISCOVERED_GRAPH_DIR / f"fges_{dataset}.json"
        _data, _domain, roles = DATASETS[dataset]()
        record = json.loads(path.read_text()) if path.exists() else {"graphs": {}}
        record.update({
            "dataset": dataset,
            "algorithm": "FGES (Tetrad, causal-cmd 1.4.1)",
            "tetrad_jar_sha1": jar_sha1,
            "score": "bdeu-score",
            "parameters": {
                "priorEquivalentSampleSize": 10.0, "structurePrior": 1.0,
                "maxDegree": 1000, "faithfulnessAssumed": False,
                "symmetricFirstStep": False,
                "note": "Tetrad defaults for this build, as printed in its output",
            },
            "knowledge": {
                "roots": sorted(DISCOVERY_ROOTS[dataset]),
                "sink": next(iter(roles.outcome)),
            },
            "rows": "training split only (test_size=0.3, random_state=seed)",
            "dag_from_cpdag": "Dor-Tarsi consistent extension, last column first",
        })
        for seed in args.seeds:
            graph = discover(dataset, seed, args.cpus)
            key = graph.pop("key")
            record["graphs"][key] = graph
            path.write_text(json.dumps(record, indent=1) + "\n")
            print(
                f"{dataset} seed={seed}: {len(graph['edges'])} edges "
                f"({graph['n_undirected']} oriented by extension), "
                f"{graph['seconds']}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
