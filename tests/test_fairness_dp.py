import networkx as nx

from causal_fairness_sdg.fairness import AttributeRoles
from causal_fairness_sdg.fairness.cf import CFFairness
from causal_fairness_sdg.fairness.dp import DPFairness

ATTRS = ["A", "B", "Y"]


def _empty_graph():
    g = nx.Graph()
    g.add_nodes_from(ATTRS)
    return g


def test_dp_forbids_direct_protected_outcome_edge():
    roles = AttributeRoles.create(protected={"A"}, admissible=set(), outcome={"Y"})
    g = _empty_graph()
    assert DPFairness().allow_edge(("A", "Y"), g, roles) is False


def test_dp_forbids_indirect_merge_even_through_a_neutral_attribute():
    roles = AttributeRoles.create(protected={"A"}, admissible=set(), outcome={"Y"})
    g = _empty_graph()
    g.add_edge("A", "B")
    # merging {A,B} with {Y} would create one component containing both -> forbidden
    assert DPFairness().allow_edge(("B", "Y"), g, roles) is False


def test_dp_is_strictly_stronger_than_cf_on_the_same_scenario():
    # Same graph, but B is declared admissible: CF should allow the merge
    # (path is blocked by an admissible attribute) while DP still forbids it,
    # since DP ignores admissibility entirely (R = empty set by definition).
    roles = AttributeRoles.create(protected={"A"}, admissible={"B"}, outcome={"Y"})
    g = _empty_graph()
    g.add_edge("A", "B")

    assert CFFairness().allow_edge(("B", "Y"), g, roles) is True
    assert DPFairness().allow_edge(("B", "Y"), g, roles) is False


def test_dp_allows_edges_among_purely_neutral_attributes():
    roles = AttributeRoles.create(protected={"A"}, admissible=set(), outcome={"Y"})
    g = nx.Graph()
    g.add_nodes_from(["A", "Y", "B", "C"])
    assert DPFairness().allow_edge(("B", "C"), g, roles) is True
