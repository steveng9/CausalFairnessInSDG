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

# SNAKE (CPS extract): the standard status-attainment chain from the
# stratification literature -- ascribed characteristics (sex, race,
# citizenship, age) shape educational attainment, education sorts people into
# occupations and industries, and occupation plus hours worked determine
# earnings. Family structure enters as a parallel path: age/sex/marriage drive
# the presence and number of children, which suppress hours worked. State is a
# root affecting industry mix and earnings directly (cost of living, local
# labour market). Direct `female`/`wbhaom` -> outcome edges represent the
# residual wage gap that the occupation- and hours-mediated paths do not
# explain -- the empirically robust part of that literature, and the thing FTU
# is allowed to cut.
SNAKE_DAG: List[Edge] = [
    ("female", "gradeatn"),
    ("wbhaom", "gradeatn"),
    ("citistat", "gradeatn"),
    ("age", "gradeatn"),
    ("age", "married"),
    ("female", "married"),
    ("wbhaom", "married"),
    ("age", "agechild"),
    ("married", "agechild"),
    ("female", "agechild"),
    ("agechild", "ownchild"),
    ("married", "ownchild"),
    ("gradeatn", "mocc10"),
    ("female", "mocc10"),
    ("wbhaom", "mocc10"),
    ("citistat", "mocc10"),
    ("mocc10", "mind16"),
    ("gradeatn", "mind16"),
    ("statefips", "mind16"),
    ("mind16", "cow1"),
    ("mocc10", "cow1"),
    ("mocc10", "hoursut"),
    ("ownchild", "hoursut"),
    ("female", "hoursut"),
    ("married", "hoursut"),
    ("hoursut", "ftptstat"),
    ("mocc10", "ftptstat"),
    ("ownchild", "ftptstat"),
    ("gradeatn", "high_income"),
    ("mocc10", "high_income"),
    ("mind16", "high_income"),
    ("cow1", "high_income"),
    ("hoursut", "high_income"),
    ("ftptstat", "high_income"),
    ("married", "high_income"),
    ("age", "high_income"),
    ("statefips", "high_income"),
    ("female", "high_income"),
    ("wbhaom", "high_income"),
]

# SBO: 25 nodes, the largest graph here, and structurally different from the
# other three -- the protected attributes describe the business *owner* while
# the outcome describes the *business*, so every protected -> outcome path runs
# through the firm rather than through the person. Four layers: owner
# demographics -> owner characteristics (education, hours, whether this is
# their primary income) -> firm structure (sector, age of firm, home-based,
# franchise, ownership) -> firm practice and scale -> receipts.
#
# This is the graph the fairness mechanisms have the most room to act on: on
# COMPAS almost every protected -> outcome path is one or two edges, so DP and
# CF have little to distinguish them. Here the paths are four and five edges
# long and most of them run through admissible attributes, which is exactly
# the regime where CF should separate from DP.
SBO_DAG: List[Edge] = [
    ("ETH1", "BORNUS1"),
    ("RACE1", "BORNUS1"),
    ("SEX1", "EDUC1"),
    ("RACE1", "EDUC1"),
    ("ETH1", "EDUC1"),
    ("BORNUS1", "EDUC1"),
    ("AGE1", "EDUC1"),
    ("VET1", "EDUC1"),
    ("AGE1", "VET1"),
    ("SEX1", "VET1"),
    ("EDUC1", "SECTOR"),
    ("SEX1", "SECTOR"),
    ("RACE1", "SECTOR"),
    ("ETH1", "SECTOR"),
    ("AGE1", "ESTABLISHED"),
    ("SECTOR", "ESTABLISHED"),
    ("SEX1", "HOURS1"),
    ("AGE1", "HOURS1"),
    ("SECTOR", "HOURS1"),
    ("HOURS1", "PRMINC1"),
    ("SECTOR", "PRMINC1"),
    ("EDUC1", "PRMINC1"),
    ("PRMINC1", "SELFEMP1"),
    ("SECTOR", "SELFEMP1"),
    ("SEX1", "FAMILYBUS"),
    ("ETH1", "FAMILYBUS"),
    ("FAMILYBUS", "NUMOWNERS"),
    ("SECTOR", "NUMOWNERS"),
    ("SECTOR", "HOMEBASED"),
    ("ESTABLISHED", "HOMEBASED"),
    ("PRMINC1", "HOMEBASED"),
    ("SECTOR", "FRANCHISE"),
    ("ESTABLISHED", "FRANCHISE"),
    ("SECTOR", "WEBSITE"),
    ("EDUC1", "WEBSITE"),
    ("ESTABLISHED", "WEBSITE"),
    ("WEBSITE", "ECOMMERCE"),
    ("SECTOR", "ECOMMERCE"),
    ("SECTOR", "EXPORTS"),
    ("ECOMMERCE", "EXPORTS"),
    ("FIPST", "EXPORTS"),
    ("SECTOR", "EMPLOYMENT_NOISY"),
    ("ESTABLISHED", "EMPLOYMENT_NOISY"),
    ("HOMEBASED", "EMPLOYMENT_NOISY"),
    ("NUMOWNERS", "EMPLOYMENT_NOISY"),
    ("FIPST", "EMPLOYMENT_NOISY"),
    ("EMPLOYMENT_NOISY", "PAYROLL_NOISY"),
    ("SECTOR", "PAYROLL_NOISY"),
    ("FIPST", "PAYROLL_NOISY"),
    ("EMPLOYMENT_NOISY", "HEALTHINS"),
    ("SECTOR", "HEALTHINS"),
    ("EMPLOYMENT_NOISY", "RETIREMENT"),
    ("HEALTHINS", "RETIREMENT"),
    ("EMPLOYMENT_NOISY", "high_receipts"),
    ("PAYROLL_NOISY", "high_receipts"),
    ("SECTOR", "high_receipts"),
    ("ESTABLISHED", "high_receipts"),
    ("HOMEBASED", "high_receipts"),
    ("FRANCHISE", "high_receipts"),
    ("WEBSITE", "high_receipts"),
    ("ECOMMERCE", "high_receipts"),
    ("EXPORTS", "high_receipts"),
    ("HOURS1", "high_receipts"),
    ("PRMINC1", "high_receipts"),
    ("NUMOWNERS", "high_receipts"),
    ("FIPST", "high_receipts"),
    ("SEX1", "high_receipts"),
    ("RACE1", "high_receipts"),
]

CAUSAL_GRAPHS: Dict[str, List[Edge]] = {
    "adult": ADULT_DAG,
    "compas": COMPAS_DAG,
    "snake": SNAKE_DAG,
    "sbo": SBO_DAG,
}


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
