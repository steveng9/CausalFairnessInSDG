import itertools

from causal_fairness_sdg.fairness import AttributeRoles
from causal_fairness_sdg.fairness.ftu import FTUFairness

ROLES = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
ATTRS = ["A", "R", "Y", "B", "C"]


def test_ftu_removes_only_direct_protected_outcome_edge():
    candidates = list(itertools.combinations(ATTRS, 2))
    filtered = FTUFairness().filter_candidates(candidates, ROLES)

    assert ("A", "Y") not in filtered and ("Y", "A") not in filtered
    assert len(filtered) == len(candidates) - 1
    # every other pair, including indirect protected/outcome-adjacent ones, survives
    for e in candidates:
        if set(e) != {"A", "Y"}:
            assert e in filtered


def test_ftu_allow_edge_is_always_true():
    # FTU only needs the static filter; allow_edge should be a no-op default
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(ATTRS)
    assert FTUFairness().allow_edge(("A", "Y"), g, ROLES) is True
