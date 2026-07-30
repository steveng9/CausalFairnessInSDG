"""Ground-truth causal DAGs for DECAF (`sdg/decaf.py`).

Unlike MST/PrivBayes/AIM, DECAF does not discover a graph privately from
data -- it needs the *true* causal structure as an input (`dag_seed`), and
applies fairness by selectively shuffling parent contributions at generation
time (see `fairness/base.py::select_biased_edges`). Neither DECAF's own repo
nor PreFair's paper ships a DAG for Adult or COMPAS with this project's exact
attribute set, so these are our own constructions -- documented here rather
than treated as ground truth, consistent with standard practice in this
literature (DECAF's and Kusner et al.'s own real-data DAGs are author-
asserted, not derived from causal discovery).

Edges are `(parent, child)` pairs over attribute *names*; `as_digraph`
resolves them against a concrete column list (so index positions match
whatever `data/datasets.py` produced for a given run).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

Edge = Tuple[str, str]

# COMPAS: grounded in the SCM used by Kusner et al. 2017 ("Counterfactual
# Fairness") for recidivism-prediction data -- race/sex as root causes of
# juvenile and prior criminal history, which in turn (along with charge
# degree) drive the recidivism outcome. A direct race -> outcome edge is
# included to represent the literature-flagged residual direct effect
# distinct from the priors_count-mediated pathway (e.g. disparate policing/
# charging not captured by prior record alone) -- this is the one edge in
# this graph that isn't directly from Kusner et al., added so FTU (which
# only ever cuts direct protected->outcome edges) has something to act on.
COMPAS_DAG: List[Edge] = [
    ("sex", "priors_count"),
    ("race", "priors_count"),
    ("sex", "juv_fel_count"),
    ("race", "juv_fel_count"),
    ("sex", "juv_misd_count"),
    ("race", "juv_misd_count"),
    ("age_cat", "priors_count"),
    ("juv_fel_count", "priors_count"),
    ("juv_misd_count", "priors_count"),
    ("priors_count", "c_charge_degree"),
    ("race", "c_charge_degree"),
    ("sex", "c_charge_degree"),
    ("priors_count", "two_year_recid"),
    ("c_charge_degree", "two_year_recid"),
    ("age_cat", "two_year_recid"),
    ("race", "two_year_recid"),
]

# Adult: no equally standard reference graph exists for this exact
# attribute set. Constructed to be consistent with the protected/admissible/
# outcome role split already established in `data/datasets.py` (protected
# attributes as roots feeding admissible attributes, admissible attributes
# feeding `income`), with a direct race -> income edge for the same reason
# as COMPAS above (gives FTU a target; represents an asserted residual
# direct effect, not derived from data). This is the most speculative part
# of the DECAF integration -- a modeling assumption open to revision, not a
# ground-truth claim about how the Census Adult data was actually generated.
ADULT_DAG: List[Edge] = [
    ("sex", "marital-status"),
    ("race", "marital-status"),
    ("age", "marital-status"),
    ("sex", "education"),
    ("race", "education"),
    ("native-country", "education"),
    ("education", "education-num"),
    ("sex", "occupation"),
    ("race", "occupation"),
    ("education", "occupation"),
    ("occupation", "workclass"),
    ("marital-status", "relationship"),
    ("sex", "relationship"),
    ("occupation", "hours-per-week"),
    ("occupation", "capital-gain"),
    ("education", "capital-gain"),
    ("occupation", "capital-loss"),
    ("workclass", "income"),
    ("education", "income"),
    ("occupation", "income"),
    ("capital-gain", "income"),
    ("capital-loss", "income"),
    ("hours-per-week", "income"),
    ("relationship", "income"),
    ("marital-status", "income"),
    ("race", "income"),
]

CAUSAL_GRAPHS: Dict[str, List[Edge]] = {"adult": ADULT_DAG, "compas": COMPAS_DAG}


def as_digraph(edges: List[Edge], columns: List[str]) -> nx.DiGraph:
    """Build an `nx.DiGraph` over exactly `columns` (so isolated/no-edge
    columns are still present as nodes), validating every edge endpoint is a
    known column."""
    unknown = {n for e in edges for n in e} - set(columns)
    if unknown:
        raise ValueError(f"causal graph references unknown columns: {sorted(unknown)}")
    dag = nx.DiGraph()
    dag.add_nodes_from(columns)
    dag.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("causal graph is not acyclic")
    return dag
