import networkx as nx

from causal_fairness_sdg.fairness import AttributeRoles
from causal_fairness_sdg.fairness.base import edges_allowed
from causal_fairness_sdg.fairness.cf import CFFairness
from causal_fairness_sdg.fairness.dp import DPFairness
from causal_fairness_sdg.fairness.ftu import FTUFairness
from causal_fairness_sdg.fairness.none import NoFairness

# A (protected) has three ways to reach Y (outcome): a direct edge, a path
# blocked by the admissible attribute R, and an unblocked path through the
# neutral attribute B. Exercises all three definitions' select_biased_edges.
ROLES = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})


def _dag():
    g = nx.DiGraph()
    g.add_nodes_from(["A", "R", "B", "Y"])
    g.add_edge("A", "R")
    g.add_edge("R", "Y")
    g.add_edge("A", "B")
    g.add_edge("B", "Y")
    g.add_edge("A", "Y")
    return g


def test_no_fairness_biases_nothing():
    assert NoFairness().select_biased_edges(_dag(), ROLES) == {}


def test_ftu_biases_only_the_direct_edge():
    assert FTUFairness().select_biased_edges(_dag(), ROLES) == {"Y": ["A"]}


def test_cf_biases_unblocked_paths_but_not_the_one_through_admissible_r():
    # A-R-Y is blocked (R admissible) -> R->Y is not cut.
    # A-B-Y is unblocked (B neutral) -> B->Y is cut.
    # A-Y direct edge has an empty interior -> unblocked -> cut.
    assert CFFairness().select_biased_edges(_dag(), ROLES) == {"Y": ["A", "B"]}


def test_dp_biases_every_path_including_the_one_through_admissible_r():
    # DP ignores admissibility entirely (R = empty set), so all three paths
    # are cut -- strictly more aggressive than CF on the identical DAG.
    assert DPFairness().select_biased_edges(_dag(), ROLES) == {"Y": ["A", "B", "R"]}


def test_select_biased_edges_ignores_protected_or_outcome_not_in_dag():
    roles = AttributeRoles.create(protected={"A", "Z"}, admissible={"R"}, outcome={"Y"})
    # Z isn't a node in the DAG at all -- should be skipped, not raise.
    assert FTUFairness().select_biased_edges(_dag(), roles) == {"Y": ["A"]}


def test_edges_allowed_reuses_filter_candidates_and_allow_edge():
    g = nx.Graph()
    g.add_nodes_from(["A", "R", "B", "Y"])

    # FTU: filter_candidates statically drops (A, Y); a parent-set touching it
    # must be rejected as a whole.
    assert edges_allowed([("A", "R"), ("A", "Y")], g, ROLES, FTUFairness()) is False
    assert edges_allowed([("A", "R")], g, ROLES, FTUFairness()) is True

    # DP: allow_edge rejects merging a protected-containing component with an
    # outcome-containing one; a candidate set is rejected if any one of its
    # edges would be.
    roles_dp = AttributeRoles.create(protected={"A"}, admissible=set(), outcome={"Y"})
    assert edges_allowed([("A", "R"), ("A", "Y")], g, roles_dp, DPFairness()) is False
    assert edges_allowed([("A", "R")], g, roles_dp, DPFairness()) is True


def test_cf_cuts_a_feature_tied_to_protected_through_an_admissible_collider():
    # A -> R <- C -> Y (COMPAS: sex -> priors_count <- age_cat -> outcome).
    # No directed path runs from A through C, but among people with the same R,
    # C carries information about A, so CF (DECAF Corollary 1) must cut C -> Y.
    g = nx.DiGraph([("A", "R"), ("C", "R"), ("R", "Y"), ("C", "Y")])
    assert CFFairness().select_biased_edges(g, ROLES) == {"Y": ["C"]}


def test_cf_keeps_a_feature_independent_of_protected_given_r():
    g = nx.DiGraph([("A", "R"), ("R", "Y"), ("C", "Y")])
    assert CFFairness().select_biased_edges(g, ROLES) == {}
