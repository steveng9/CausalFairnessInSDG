"""Causal DAGs for the DECAF-style generators (`sdg/decaf*.py`).

Unlike MST/PrivBayes/AIM, these generators do not select a structure
privately -- they take a DAG as an input (`dag_seed`) and apply fairness by
shuffling parent contributions at generation time (see
`fairness/base.py::select_biased_edges`). Two sources of graph are available,
chosen by the generators' `graph` setting (`resolve_graph`):

  "hand"  (default) The hand-drawn graphs in this file, one per dataset, from
          domain knowledge. Our own constructions, not ground truth.
  "fges"  Graphs discovered from the training rows the way the DECAF paper
          does it for Credit Approval (FGES in Tetrad, with age and the
          protected demographic attributes declared root nodes); see
          `scripts/discover_graphs.py`. The DECAF paper's real-data graphs
          are all discovered (Section 6: "the ground-truth DAG is unknown. We
          use causal discovery"): Adult's is the PC-algorithm graph of Zhang
          et al. 2017, Credit Approval's is FGES.

Edges are `(parent, child)` pairs over attribute *names*; `as_digraph`
resolves them against a concrete column list (so index positions match
whatever `data/datasets.py` produced for a given run).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

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

# ACSIncome (`data/acs.py`): the same status-attainment chain as SNAKE, on the
# attributes folktables keeps. Ascribed characteristics (sex, race, Hispanic
# origin, world area of birth, age) are roots; they shape schooling, schooling
# and the ascribed characteristics sort people into occupations, occupation
# fixes class of worker and hours, and all of those set earnings. Household
# position (`MAR`, `RELP`) is the parallel family path, entering earnings both
# directly and through hours. Direct `SEX`/`RAC1P`/`HISP` -> outcome edges are
# the residual gap the mediated paths do not explain, as in the other graphs,
# and are what FTU is allowed to cut. `WAOB` has no direct edge: nativity is
# taken to act through schooling and occupational sorting only.
#
# One graph for every state on purpose -- whether a graph specified once
# transfers across populations is one of the things the two states test.
ACSINCOME_DAG: List[Edge] = [
    ("SEX", "SCHL"),
    ("RAC1P", "SCHL"),
    ("HISP", "SCHL"),
    ("WAOB", "SCHL"),
    ("AGEP", "SCHL"),
    ("AGEP", "MAR"),
    ("SEX", "MAR"),
    ("RAC1P", "MAR"),
    ("HISP", "MAR"),
    ("MAR", "RELP"),
    ("AGEP", "RELP"),
    ("SEX", "RELP"),
    ("SCHL", "OCCP"),
    ("SEX", "OCCP"),
    ("RAC1P", "OCCP"),
    ("HISP", "OCCP"),
    ("WAOB", "OCCP"),
    ("OCCP", "COW"),
    ("SCHL", "COW"),
    ("OCCP", "WKHP"),
    ("COW", "WKHP"),
    ("SEX", "WKHP"),
    ("MAR", "WKHP"),
    ("AGEP", "WKHP"),
    ("SCHL", "high_income"),
    ("OCCP", "high_income"),
    ("COW", "high_income"),
    ("WKHP", "high_income"),
    ("MAR", "high_income"),
    ("RELP", "high_income"),
    ("AGEP", "high_income"),
    ("SEX", "high_income"),
    ("RAC1P", "high_income"),
    ("HISP", "high_income"),
]

# ACSPublicCoverage (`data/acs.py`): 20 nodes. Demographics (age, sex, race,
# Hispanic origin, nativity) are roots. Nativity fixes citizenship; age drives
# the three functional difficulties, which define the overall disability flag
# (`DIS` is the survey's own "any difficulty" recode). Schooling, marriage,
# military service, a recent birth and migration follow from demographics;
# employment status follows from all of those; income from employment and
# schooling. Coverage then depends on the things programmes are written in
# terms of -- income, employment, disability, age, a recent birth,
# citizenship, veteran status, and (for minors) the parents' employment --
# plus direct `SEX`/`RAC1P`/`HISP` edges for the residual gap, as elsewhere.
# `NATIVITY` has no direct edge: it reaches coverage through citizenship.
ACSPUBCOV_DAG: List[Edge] = [
    ("RAC1P", "ANC"),
    ("HISP", "ANC"),
    ("NATIVITY", "ANC"),
    ("NATIVITY", "CIT"),
    ("AGEP", "CIT"),
    ("HISP", "CIT"),
    ("AGEP", "DEAR"),
    ("AGEP", "DEYE"),
    ("AGEP", "DREM"),
    ("SEX", "DREM"),
    ("DEAR", "DIS"),
    ("DEYE", "DIS"),
    ("DREM", "DIS"),
    ("AGEP", "DIS"),
    ("AGEP", "SCHL"),
    ("SEX", "SCHL"),
    ("RAC1P", "SCHL"),
    ("HISP", "SCHL"),
    ("NATIVITY", "SCHL"),
    ("DREM", "SCHL"),
    ("AGEP", "MAR"),
    ("SEX", "MAR"),
    ("RAC1P", "MAR"),
    ("HISP", "MAR"),
    ("AGEP", "ESP"),
    ("MAR", "ESP"),
    ("AGEP", "MIL"),
    ("SEX", "MIL"),
    ("CIT", "MIL"),
    ("AGEP", "MIG"),
    ("MAR", "MIG"),
    ("NATIVITY", "MIG"),
    ("AGEP", "FER"),
    ("SEX", "FER"),
    ("MAR", "FER"),
    ("AGEP", "ESR"),
    ("SEX", "ESR"),
    ("SCHL", "ESR"),
    ("DIS", "ESR"),
    ("MIL", "ESR"),
    ("FER", "ESR"),
    ("MAR", "ESR"),
    ("ESR", "PINCP"),
    ("SCHL", "PINCP"),
    ("AGEP", "PINCP"),
    ("SEX", "PINCP"),
    ("DIS", "PINCP"),
    ("MAR", "PINCP"),
    ("PINCP", "public_coverage"),
    ("ESR", "public_coverage"),
    ("DIS", "public_coverage"),
    ("AGEP", "public_coverage"),
    ("FER", "public_coverage"),
    ("CIT", "public_coverage"),
    ("MIL", "public_coverage"),
    ("MAR", "public_coverage"),
    ("ESP", "public_coverage"),
    ("SEX", "public_coverage"),
    ("RAC1P", "public_coverage"),
    ("HISP", "public_coverage"),
]

# CDC Diabetes (`data/cdc_diabetes.py`): 22 nodes, the usual social-
# determinants layering. Sex and age are roots; they and each other shape
# education and income; socioeconomic status shapes access to care and health
# behaviours; behaviours shape BMI; BMI and behaviours shape blood pressure
# and cholesterol, which shape heart disease and stroke; general health,
# days unwell and difficulty walking summarise all of that.
#
# The outcome is kept a sink, as in every other graph here, because the task
# is predicting it. Clinically some arrows also run the other way (diabetes
# causes heart disease and poor self-rated health); those are drawn as
# predictors of the outcome, which is a modelling choice and not a claim.
# Direct `Sex`/`Income`/`Education` -> outcome edges are the residual effect.
CDC_DIABETES_DAG: List[Edge] = [
    ("Age", "Education"),
    ("Sex", "Education"),
    ("Education", "Income"),
    ("Age", "Income"),
    ("Sex", "Income"),
    ("Income", "AnyHealthcare"),
    ("Education", "AnyHealthcare"),
    ("Age", "AnyHealthcare"),
    ("Income", "NoDocbcCost"),
    ("AnyHealthcare", "NoDocbcCost"),
    ("AnyHealthcare", "CholCheck"),
    ("NoDocbcCost", "CholCheck"),
    ("Age", "CholCheck"),
    ("Sex", "Smoker"),
    ("Age", "Smoker"),
    ("Education", "Smoker"),
    ("Income", "Smoker"),
    ("Sex", "PhysActivity"),
    ("Age", "PhysActivity"),
    ("Education", "PhysActivity"),
    ("Income", "PhysActivity"),
    ("Sex", "Fruits"),
    ("Education", "Fruits"),
    ("Income", "Fruits"),
    ("Sex", "Veggies"),
    ("Education", "Veggies"),
    ("Income", "Veggies"),
    ("Sex", "HvyAlcoholConsump"),
    ("Age", "HvyAlcoholConsump"),
    ("Income", "HvyAlcoholConsump"),
    ("PhysActivity", "BMI"),
    ("Fruits", "BMI"),
    ("Veggies", "BMI"),
    ("Age", "BMI"),
    ("Sex", "BMI"),
    ("Income", "BMI"),
    ("Education", "BMI"),
    ("BMI", "HighBP"),
    ("Age", "HighBP"),
    ("Sex", "HighBP"),
    ("Smoker", "HighBP"),
    ("HvyAlcoholConsump", "HighBP"),
    ("PhysActivity", "HighBP"),
    ("BMI", "HighChol"),
    ("Age", "HighChol"),
    ("Sex", "HighChol"),
    ("CholCheck", "HighChol"),
    ("HighBP", "HeartDiseaseorAttack"),
    ("HighChol", "HeartDiseaseorAttack"),
    ("Smoker", "HeartDiseaseorAttack"),
    ("Age", "HeartDiseaseorAttack"),
    ("Sex", "HeartDiseaseorAttack"),
    ("BMI", "HeartDiseaseorAttack"),
    ("HighBP", "Stroke"),
    ("HighChol", "Stroke"),
    ("Smoker", "Stroke"),
    ("Age", "Stroke"),
    ("HeartDiseaseorAttack", "Stroke"),
    ("Age", "PhysHlth"),
    ("BMI", "PhysHlth"),
    ("HeartDiseaseorAttack", "PhysHlth"),
    ("Stroke", "PhysHlth"),
    ("Income", "MentHlth"),
    ("Sex", "MentHlth"),
    ("PhysHlth", "MentHlth"),
    ("Age", "DiffWalk"),
    ("BMI", "DiffWalk"),
    ("Stroke", "DiffWalk"),
    ("PhysHlth", "DiffWalk"),
    ("BMI", "GenHlth"),
    ("HighBP", "GenHlth"),
    ("HeartDiseaseorAttack", "GenHlth"),
    ("Stroke", "GenHlth"),
    ("PhysActivity", "GenHlth"),
    ("Income", "GenHlth"),
    ("Age", "GenHlth"),
    ("PhysHlth", "GenHlth"),
    ("MentHlth", "GenHlth"),
    ("BMI", "Diabetes_binary"),
    ("HighBP", "Diabetes_binary"),
    ("HighChol", "Diabetes_binary"),
    ("Age", "Diabetes_binary"),
    ("PhysActivity", "Diabetes_binary"),
    ("HeartDiseaseorAttack", "Diabetes_binary"),
    ("Smoker", "Diabetes_binary"),
    ("HvyAlcoholConsump", "Diabetes_binary"),
    ("AnyHealthcare", "Diabetes_binary"),
    ("CholCheck", "Diabetes_binary"),
    ("GenHlth", "Diabetes_binary"),
    ("DiffWalk", "Diabetes_binary"),
    ("Sex", "Diabetes_binary"),
    ("Income", "Diabetes_binary"),
    ("Education", "Diabetes_binary"),
]

# HMDA mortgage applications (`data/hmda.py`): 17 nodes, the order in which an
# application comes to exist. The applicant's sex, race, ethnicity and age are
# roots. They shape whether there is a co-applicant and what the household
# earns; race, ethnicity and income shape where the property is (the tract's
# minority share and relative income -- residential segregation, and the
# channel redlining acts through). The loan follows: its purpose, its size,
# whether that size is over the conforming limit, the programme (FHA, VA and
# USDA loans are where low-down-payment and minority borrowers are
# concentrated), the term, and the two underwriting ratios, which are
# arithmetic on income, loan size and programme. The lender's automated
# underwriting system follows from the programme. The decision depends on all
# of it.
#
# The direct protected -> `approved` edges are the residual disparity that the
# reported underwriting variables do not explain, as in the other graphs, and
# are what FTU cuts. They are not a claim of disparate treatment: the public
# file has no credit score, so part of that residual is an unobserved
# underwriting input (see the module docstring of `data/hmda.py`).
HMDA_DAG: List[Edge] = [
    ("sex", "coapplicant"),
    ("race", "coapplicant"),
    ("ethnicity", "coapplicant"),
    ("age", "coapplicant"),
    ("sex", "income"),
    ("race", "income"),
    ("ethnicity", "income"),
    ("age", "income"),
    ("coapplicant", "income"),
    ("race", "tract_minority"),
    ("ethnicity", "tract_minority"),
    ("income", "tract_minority"),
    ("race", "tract_income"),
    ("ethnicity", "tract_income"),
    ("income", "tract_income"),
    ("tract_minority", "tract_income"),
    ("age", "loan_purpose"),
    ("income", "loan_purpose"),
    ("income", "loan_amount"),
    ("coapplicant", "loan_amount"),
    ("age", "loan_amount"),
    ("tract_income", "loan_amount"),
    ("loan_purpose", "loan_amount"),
    ("loan_amount", "jumbo"),
    ("sex", "loan_type"),
    ("race", "loan_type"),
    ("ethnicity", "loan_type"),
    ("age", "loan_type"),
    ("income", "loan_type"),
    ("loan_purpose", "loan_type"),
    ("loan_amount", "loan_type"),
    ("jumbo", "loan_type"),
    ("age", "loan_term"),
    ("income", "loan_term"),
    ("loan_purpose", "loan_term"),
    ("loan_type", "loan_term"),
    ("age", "ltv"),
    ("income", "ltv"),
    ("loan_purpose", "ltv"),
    ("loan_amount", "ltv"),
    ("loan_type", "ltv"),
    ("income", "dti"),
    ("coapplicant", "dti"),
    ("loan_amount", "dti"),
    ("loan_term", "dti"),
    ("loan_type", "dti"),
    ("loan_type", "aus"),
    ("jumbo", "aus"),
    ("loan_purpose", "aus"),
    ("dti", "approved"),
    ("ltv", "approved"),
    ("income", "approved"),
    ("loan_amount", "approved"),
    ("jumbo", "approved"),
    ("loan_type", "approved"),
    ("loan_purpose", "approved"),
    ("loan_term", "approved"),
    ("aus", "approved"),
    ("coapplicant", "approved"),
    ("tract_minority", "approved"),
    ("tract_income", "approved"),
    ("sex", "approved"),
    ("race", "approved"),
    ("ethnicity", "approved"),
    ("age", "approved"),
]

CAUSAL_GRAPHS: Dict[str, List[Edge]] = {
    "adult": ADULT_DAG,
    "compas": COMPAS_DAG,
    "snake": SNAKE_DAG,
    "sbo": SBO_DAG,
    "acsincome_ca": ACSINCOME_DAG,
    "acsincome_tx": ACSINCOME_DAG,
    "acspubcov_ca": ACSPUBCOV_DAG,
    "acspubcov_tx": ACSPUBCOV_DAG,
    "cdc_diabetes": CDC_DIABETES_DAG,
    "hmda_ca": HMDA_DAG,
    "hmda_tx": HMDA_DAG,
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


GRAPH_SOURCES = ("hand", "fges")

#: Where `scripts/discover_graphs.py` writes, one file per dataset.
DISCOVERED_GRAPH_DIR = Path(__file__).resolve().parent / "discovered_graphs"

#: Prior knowledge given to the discovery algorithm: these attributes are root
#: nodes. DECAF's Credit Approval graph was discovered with "the prior
#: knowledge that age and ethnicity are root nodes" (App. D.3) -- age plus the
#: protected attribute. The same rule here: age, and the protected demographic
#: attributes a person is born with (sex, race, ethnicity, birthplace).
#: Protected attributes acquired later in some role splits (disability,
#: veteran status, education, income) are left to the algorithm.
DISCOVERY_ROOTS: Dict[str, Tuple[str, ...]] = {
    "adult": ("age", "sex", "race", "native-country"),
    "compas": ("age_cat", "sex", "race"),
    "snake": ("age", "female", "wbhaom", "citistat"),
    "sbo": ("AGE1", "SEX1", "RACE1", "ETH1", "BORNUS1"),
    "acsincome_ca": ("AGEP", "SEX", "RAC1P", "HISP", "WAOB"),
    "acsincome_tx": ("AGEP", "SEX", "RAC1P", "HISP", "WAOB"),
    "acspubcov_ca": ("AGEP", "SEX", "RAC1P", "HISP", "NATIVITY"),
    "acspubcov_tx": ("AGEP", "SEX", "RAC1P", "HISP", "NATIVITY"),
    "cdc_diabetes": ("Age", "Sex"),
    "hmda_ca": ("age", "sex", "race", "ethnicity"),
    "hmda_tx": ("age", "sex", "race", "ethnicity"),
}


def table_key(data: pd.DataFrame) -> str:
    """Identifies a training table: a discovered graph belongs to the exact
    rows it was discovered on."""
    payload = "\t".join(map(str, data.columns)).encode()
    payload += np.ascontiguousarray(data.to_numpy(dtype=np.int64)).tobytes()
    return hashlib.sha1(payload).hexdigest()


@lru_cache(maxsize=None)
def _discovered(source: str, dataset_name: str) -> Dict[str, List[Edge]]:
    path = DISCOVERED_GRAPH_DIR / f"{source}_{dataset_name}.json"
    if not path.exists():
        raise ValueError(
            f"No {source!r} graph discovered for dataset {dataset_name!r} "
            f"({path} is missing); run scripts/discover_graphs.py"
        )
    record = json.loads(path.read_text())
    return {
        key: [(p, c) for p, c in graph["edges"]]
        for key, graph in record["graphs"].items()
    }


def resolve_graph(dataset_name: str, data: pd.DataFrame, source: str = "hand") -> List[Edge]:
    """The edge list a DECAF-style generator should use for `data`."""
    if source == "hand":
        if dataset_name not in CAUSAL_GRAPHS:
            raise ValueError(
                f"No causal graph registered for dataset_name={dataset_name!r}; "
                f"available: {sorted(CAUSAL_GRAPHS)}"
            )
        return CAUSAL_GRAPHS[dataset_name]
    if source not in GRAPH_SOURCES:
        raise ValueError(f"unknown graph source {source!r}; one of {GRAPH_SOURCES}")
    graphs = _discovered(source, dataset_name)
    key = table_key(data)
    if key not in graphs:
        raise ValueError(
            f"No {source!r} graph for this training table of {dataset_name!r} "
            f"(key {key[:12]}); run scripts/discover_graphs.py for this seed"
        )
    return graphs[key]
