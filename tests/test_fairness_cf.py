import networkx as nx

from causal_fairness_sdg.fairness import AttributeRoles
from causal_fairness_sdg.fairness.cf import CFFairness

ATTRS = ["A", "R", "Y", "B"]


def _empty_graph():
    g = nx.Graph()
    g.add_nodes_from(ATTRS)
    return g


def test_cf_forbids_direct_protected_outcome_edge():
    roles = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
    g = _empty_graph()
    assert CFFairness().allow_edge(("A", "Y"), g, roles) is False


def test_cf_allows_edge_when_no_protected_or_outcome_involved():
    roles = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
    g = _empty_graph()
    assert CFFairness().allow_edge(("R", "B"), g, roles) is True


def test_cf_allows_path_blocked_by_admissible_attribute():
    roles = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
    g = _empty_graph()
    g.add_edge("A", "R")
    # A-R-Y: interior is {R}, which is admissible -> allowed
    assert CFFairness().allow_edge(("R", "Y"), g, roles) is True


def test_cf_forbids_path_not_blocked_by_admissible_attribute():
    roles = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
    g = _empty_graph()
    g.add_edge("A", "B")
    g.add_edge("Y", "R")  # separate component; R is admissible but not on this path
    # merging (A,B)-component with (Y,R)-component via edge (B,Y) gives path
    # A-B-Y whose interior is {B}, not admissible -> forbidden
    assert CFFairness().allow_edge(("B", "Y"), g, roles) is False


def test_cf_handles_multiple_protected_and_outcome_attributes():
    roles = AttributeRoles.create(
        protected={"A1", "A2"}, admissible={"R"}, outcome={"Y1", "Y2"}
    )
    g = nx.Graph()
    g.add_nodes_from(["A1", "A2", "R", "Y1", "Y2"])
    g.add_edge("A1", "R")
    g.add_edge("A2", "R")
    g.add_edge("Y1", "R")
    # every existing protected/outcome node already routes through R; adding
    # a second outcome via R should still be allowed
    assert CFFairness().allow_edge(("R", "Y2"), g, roles) is True
