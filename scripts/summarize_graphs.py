"""Tabulate the hand-drawn and discovered graphs side by side.

One row per (dataset, seed, graph source, role split): edge count, the
outcome's parents, and the outcome parents each mechanism cuts. Written to
`report/data/facct/graphs_summary.csv`.

    python scripts/summarize_graphs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_experiments import ROLE_CONFIGS  # noqa: E402

from causal_fairness_sdg.data.causal_graphs import (  # noqa: E402
    GRAPH_SOURCES,
    as_digraph,
    resolve_graph,
)
from causal_fairness_sdg.data.datasets import DATASETS  # noqa: E402
from causal_fairness_sdg.fairness.base import AttributeRoles  # noqa: E402
from causal_fairness_sdg.fairness.registry import get_fairness_mechanism  # noqa: E402

OUT = REPO_ROOT / "report" / "data" / "facct" / "graphs_summary.csv"
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    rows = []
    for dataset, role_configs in ROLE_CONFIGS.items():
        data, _domain, roles = DATASETS[dataset]()
        columns = list(data.columns)
        outcome = next(iter(roles.outcome))
        for seed in SEEDS:
            train, _ = train_test_split(data, test_size=0.3, random_state=seed)
            for source in GRAPH_SOURCES:
                dag = as_digraph(resolve_graph(dataset, train, source), columns)
                parents = sorted(dag.predecessors(outcome))
                for role_name, protected, admissible in role_configs:
                    split = AttributeRoles.create(protected, admissible, [outcome])
                    row = {
                        "dataset": dataset, "seed": seed, "graph": source,
                        "role_config": role_name,
                        "n_columns": len(columns),
                        "n_edges": dag.number_of_edges(),
                        "n_outcome_parents": len(parents),
                        "outcome_parents": " ".join(parents),
                        "protected_with_path_to_outcome": " ".join(sorted(
                            p for p in protected
                            if outcome in nx.descendants(dag, p)
                        )),
                    }
                    for mechanism in ("ftu", "cf", "dp"):
                        cut = get_fairness_mechanism(mechanism).select_biased_edges(
                            dag, split
                        ).get(outcome, [])
                        row[f"n_cut_{mechanism}"] = len(cut)
                        row[f"cut_{mechanism}"] = " ".join(cut)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(frame)} rows)")
    summary = frame.groupby(["dataset", "graph"])[
        ["n_edges", "n_outcome_parents", "n_cut_ftu", "n_cut_cf", "n_cut_dp"]
    ].mean().round(1)
    print(summary.to_string())
    nothing = frame[frame.graph != "hand"].groupby("dataset")[
        ["n_cut_ftu", "n_cut_cf", "n_cut_dp"]
    ].agg(lambda s: f"{int((s == 0).sum())}/{len(s)}")
    print("\ncells (seed x role split) where the mechanism cuts nothing, discovered graphs:")
    print(nothing.to_string())


if __name__ == "__main__":
    main()
